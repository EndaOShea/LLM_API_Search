"""LLM API Search — discover models, providers, and get connection snippets."""

from mcp.server.fastmcp import FastMCP

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.selector import select_provider

DESCRIPTION = "Discover LLM API providers, models, and get ready-to-use code snippets"
MOUNT_PATH = "/llm-api-search"

mcp = FastMCP("llm-api-search")


@mcp.tool()
def llm_list_providers() -> list[str]:
    """List all supported LLM API providers (e.g. anthropic, gemini, openai)."""
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
        provider: Provider key — one of "anthropic", "gemini", or "openai".
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
        provider: Provider key — one of "anthropic", "gemini", or "openai".
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
def llm_list_models(provider: str, live: bool = False) -> list[dict]:
    """List available models for a specific LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "gemini", or "openai".
        live: If True, fetch live model lists (requires API key in environment).

    Returns:
        A list of model details including ID, name, context window, and capabilities.
    """
    info = discover_provider(provider, live=live)
    return [
        {
            "model_id": m.model_id,
            "display_name": m.display_name,
            "description": m.description,
            "context_window": m.context_window,
            "max_output_tokens": m.max_output_tokens,
            "supports_vision": m.supports_vision,
            "supports_tool_use": m.supports_tool_use,
            "input_cost_per_mtok": m.input_cost_per_mtok,
            "output_cost_per_mtok": m.output_cost_per_mtok,
        }
        for m in info.models
    ]


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
        max_ctx = max((m.context_window or 0) for m in info.models)
        max_out = max((m.max_output_tokens or 0) for m in info.models)
        priced = [m for m in info.models if m.input_cost_per_mtok is not None]
        lines.append(f"{info.name}")
        lines.append(f"  Models available:    {len(info.models)}")
        lines.append(f"  Max context window:  {max_ctx:,} tokens")
        lines.append(f"  Max output tokens:   {max_out:,} tokens")
        if priced:
            cheapest = min(priced, key=lambda m: m.input_cost_per_mtok)
            priciest = max(priced, key=lambda m: m.input_cost_per_mtok)
            lines.append(
                f"  Price range:         "
                f"${cheapest.input_cost_per_mtok:.2f}/${cheapest.output_cost_per_mtok:.2f} "
                f"— ${priciest.input_cost_per_mtok:.2f}/${priciest.output_cost_per_mtok:.2f} per 1M tok"
            )
            lines.append(f"  Per-model pricing:")
            for m in priced:
                lines.append(
                    f"    {m.model_id:40s} ${m.input_cost_per_mtok:>7.2f} in / ${m.output_cost_per_mtok:>7.2f} out"
                )
        lines.append(f"  SDK:                 {info.sdk_install}")
        lines.append(f"  Auth env var:        {info.auth_env_var}")
        lines.append(f"  Docs:                {info.documentation_url}")
        lines.append("")

    return "\n".join(lines)
