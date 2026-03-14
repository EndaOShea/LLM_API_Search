"""LLM API Search — discover models, providers, and get connection snippets."""

import dataclasses

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.providers.base import TextModelInfo
from llm_api_search.selector import select_provider

DESCRIPTION = "Discover LLM API providers, models, and get ready-to-use code snippets"
MOUNT_PATH = "/llm-api-search"

# DNS rebinding protection is disabled because the server runs behind an nginx
# reverse proxy which is the security boundary.
mcp = FastMCP(
    "llm-api-search",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


@mcp.tool()
def llm_list_providers() -> list[str]:
    """List all supported LLM API providers (e.g. anthropic, google, openai)."""
    return list_providers()


@mcp.tool()
def llm_discover_all(live: bool = False) -> str:
    """Discover the latest API versions, models, and connection details for all LLM providers.

    Args:
        live: If True, fetch live model lists from provider APIs (requires
              API keys in environment). Defaults to False for static data.

    Returns:
        A formatted summary of all providers with their models and connection info.
    """
    results = discover(live=live)
    parts = []
    for key, info in results.items():
        parts.append(info.summary())
        parts.append("")
    return "\n".join(parts)


@mcp.tool()
def llm_discover_provider(provider: str, live: bool = False) -> str:
    """Discover API info for a single LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "google", or "openai".
        live: If True, fetch live model lists (requires API key in environment).

    Returns:
        A formatted summary with models, auth details, SDK install, and docs URL.
    """
    info = discover_provider(provider, live=live)
    return info.summary()


@mcp.tool()
def llm_get_connection_snippet(
    provider: str,
    model_id: str | None = None,
    language: str = "python",
) -> str:
    """Get a ready-to-use code snippet for connecting to an LLM API.

    Args:
        provider: Provider key — one of "anthropic", "google", or "openai".
        model_id: Optional specific model ID. Defaults to the provider's recommended model.
        language: Programming language for the snippet. One of "python", "typescript",
                  "javascript", "java", or "cpp". Defaults to "python".

    Returns:
        A code snippet showing how to install the SDK and make a basic API call.
    """
    sel = select_provider(provider, model_id=model_id, live=False, language=language)
    install = sel.provider_info.sdk_install_for(language)
    header = (
        f"# Provider: {sel.provider_info.name}\n"
        f"# Model:    {sel.model.model_id}\n"
        f"# Language: {language}\n"
        f"# Install:  {install}\n"
        f"# Auth:     Set {sel.provider_info.auth_env_var} environment variable\n\n"
    )
    return header + sel.connection_snippet


@mcp.tool()
def llm_list_models(
    provider: str,
    live: bool = False,
    model_type: str | None = None,
) -> list[dict]:
    """List available models for a specific LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", or "inception".
        live: If True, fetch live model lists (requires API key in environment).
        model_type: Optional filter — one of "text", "image", "audio_tts",
                    "audio_transcription", "embedding". Returns all types if omitted.

    Returns:
        A list of model details. Fields vary by model type.
    """
    info = discover_provider(provider, live=live)
    models = info.models
    if model_type:
        models = [m for m in models if m.model_type.value == model_type]
    return [dataclasses.asdict(m) for m in models]


@mcp.tool()
def llm_compare_providers(live: bool = False) -> str:
    """Compare all LLM providers side-by-side: pricing tiers, model counts, context windows, and SDK info.

    Args:
        live: If True, fetch live model lists (requires API keys in environment).

    Returns:
        A formatted comparison table of all providers.
    """
    results = discover(live=live)
    lines = ["LLM Provider Comparison", "=" * 60, ""]

    for key, info in results.items():
        text_models = [m for m in info.models if isinstance(m, TextModelInfo)]
        max_ctx = max((m.context_window or 0) for m in text_models) if text_models else 0
        max_out = max((m.max_output_tokens or 0) for m in text_models) if text_models else 0
        priced = [m for m in text_models if m.input_cost_per_mtok is not None]
        model_types = sorted(set(m.model_type.value for m in info.models))

        lines.append(f"{info.name}")
        lines.append(f"  Models available:    {len(info.models)}")
        lines.append(f"  Model types:         {', '.join(model_types)}")
        if text_models:
            lines.append(f"  Max context window:  {max_ctx:,} tokens")
            lines.append(f"  Max output tokens:   {max_out:,} tokens")
        if priced:
            cheapest = min(priced, key=lambda m: m.input_cost_per_mtok)
            priciest = max(priced, key=lambda m: m.input_cost_per_mtok)
            lines.append(
                f"  Text price range:    "
                f"${cheapest.input_cost_per_mtok:.2f}/${cheapest.output_cost_per_mtok:.2f} "
                f"— ${priciest.input_cost_per_mtok:.2f}/${priciest.output_cost_per_mtok:.2f} per 1M tok"
            )
            lines.append(f"  Per-model pricing (text):")
            for m in priced:
                lines.append(
                    f"    {m.model_id:40s} ${m.input_cost_per_mtok:>7.2f} in / ${m.output_cost_per_mtok:>7.2f} out"
                )
        lines.append(f"  SDK:                 {info.sdk_install}")
        lines.append(f"  Auth env var:        {info.auth_env_var}")
        lines.append(f"  Docs:                {info.documentation_url}")
        lines.append("")

    return "\n".join(lines)
