"""LLM API Search — discover models, providers, and get connection snippets."""

import dataclasses

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.providers import filter_models, get_rate_limits, get_thinking_config
from llm_api_search.providers.base import SUPPORTED_LANGUAGES, TextModelInfo
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
    """List all supported LLM API providers (e.g. anthropic, google, openai, inception, deepseek, zai, minimax)."""
    return list_providers()


@mcp.tool()
def llm_discover_all(live: bool = False, include_all: bool = False) -> str:
    """Discover the latest API versions, models, and connection details for all LLM providers.

    Args:
        live: If True, fetch live model lists from provider APIs (requires
              API keys in environment). Defaults to False for static data.
        include_all: If True, include dated snapshots and legacy models.

    Returns:
        A formatted summary of all providers with their models and connection info.
    """
    results = discover(live=live)
    parts = []
    for key, info in results.items():
        if not include_all:
            info = dataclasses.replace(info, models=filter_models(info.models, key))
        parts.append(info.summary())
        parts.append("")
    return "\n".join(parts)


@mcp.tool()
def llm_discover_provider(provider: str, live: bool = False, include_all: bool = False) -> str:
    """Discover API info for a single LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", "inception", "deepseek", "zai", or "minimax".
        live: If True, fetch live model lists (requires API key in environment).
        include_all: If True, include dated snapshots and legacy models.

    Returns:
        A formatted summary with models, auth details, SDK install, and docs URL.
    """
    info = discover_provider(provider, live=live)
    if not include_all:
        info = dataclasses.replace(info, models=filter_models(info.models, provider))
    return info.summary()


@mcp.tool()
def llm_get_connection_snippet(
    provider: str,
    model_id: str | None = None,
    language: str | None = None,
) -> str:
    """Get a ready-to-use code snippet for connecting to an LLM API.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", "inception", "deepseek", "zai", or "minimax".
        model_id: Optional specific model ID. Defaults to the provider's recommended model.
        language: Programming language for the snippet. One of "python", "typescript",
                  "javascript", "java", or "cpp". If omitted, returns snippets for all
                  supported languages. If the caller knows the user's project language,
                  pass it here for a targeted response.

    Returns:
        A code snippet showing how to install the SDK and make a basic API call.
    """
    if language is not None and language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        return (
            f"Language \"{language}\" is not currently supported.\n\n"
            f"Supported languages: {supported}\n\n"
            f"To request support for \"{language}\", please open an issue at:\n"
            f"https://github.com/EndaOShea/LLM_API_Search/issues"
        )

    languages = (language,) if language else SUPPORTED_LANGUAGES

    parts: list[str] = []
    for lang in languages:
        sel = select_provider(provider, model_id=model_id, live=False, language=lang)
        install = sel.provider_info.sdk_install_for(lang)
        header = (
            f"# Provider: {sel.provider_info.name}\n"
            f"# Model:    {sel.model.model_id}\n"
            f"# Language: {lang}\n"
            f"# Install:  {install}\n"
            f"# Auth:     Set {sel.provider_info.auth_env_var} environment variable\n\n"
        )
        parts.append(header + sel.connection_snippet)

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def llm_list_models(
    provider: str,
    live: bool = False,
    model_type: str | None = None,
    include_all: bool = False,
) -> list[dict]:
    """List available models for a specific LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", "inception", "deepseek", "zai", or "minimax".
        live: If True, fetch live model lists (requires API key in environment).
        model_type: Optional filter — one of "text", "image", "audio_tts",
                    "audio_transcription", "embedding". Returns all types if omitted.
        include_all: If True, include dated snapshots and legacy models that are
                     hidden by default. Defaults to False.

    Returns:
        A list of model details. Fields vary by model type.
    """
    info = discover_provider(provider, live=live)
    models = info.models
    if not include_all:
        models = filter_models(models, provider)
    if model_type:
        models = [m for m in models if m.model_type.value == model_type]
    return [dataclasses.asdict(m) for m in models]


@mcp.tool()
def llm_compare_providers(live: bool = False, include_all: bool = False) -> str:
    """Compare all LLM providers side-by-side: pricing tiers, model counts, context windows, and SDK info.

    Args:
        live: If True, fetch live model lists (requires API keys in environment).
        include_all: If True, include dated snapshots and legacy models.

    Returns:
        A formatted comparison table of all providers.
    """
    results = discover(live=live)
    lines = ["LLM Provider Comparison", "=" * 60, ""]

    for key, info in results.items():
        if not include_all:
            info = dataclasses.replace(info, models=filter_models(info.models, key))
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


def _rl_to_dict(rl) -> dict:
    """Convert a RateLimit to a dict, omitting None fields."""
    return {k: v for k, v in dataclasses.asdict(rl).items() if v is not None}


def _tc_to_dict(tc) -> dict:
    """Convert a ThinkingConfig to a dict, omitting None/empty fields.

    For non-capable models (the default config) the False booleans
    ``supports_dynamic``/``can_disable`` are also dropped to keep the output
    lean. For *capable* models they are kept, because ``can_disable=False``
    ("thinking cannot be turned off", e.g. gemini-3-pro-preview, o3) is a
    meaningful fact that callers need.
    """
    out = {}
    for k, v in dataclasses.asdict(tc).items():
        if v is None or v == [] or v == "":
            continue
        if (
            isinstance(v, bool)
            and v is False
            and k in ("supports_dynamic", "can_disable")
            and not tc.supported
        ):
            continue
        out[k] = v.value if hasattr(v, "value") else v
    return out


@mcp.tool()
def llm_get_rate_limits(
    provider: str,
    model: str | None = None,
    tier: str | None = None,
) -> dict:
    """Get rate limits for an LLM provider, optionally for a specific model.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", "inception", "deepseek", "zai", or "minimax".
        model: Optional model ID.  If provided, returns limits for that model
               (falls back to the base alias for dated snapshots).  If omitted,
               returns limits for all models.
        tier: Optional tier name.  Tier names are provider-specific:
              Anthropic uses "start", "build", "scale" (no free tier; a fourth
              negotiated "Custom" tier publishes no fixed numbers).
              Google uses "free", "tier-1" through "tier-3".
              OpenAI uses "tier-1" (baseline only).
              Inception uses "free", "paid", "enterprise".
              DeepSeek publishes no numeric rate limits (limits are dynamic
              by server load), so this tool returns no data for DeepSeek models.
              When omitted, returns all tiers per model.

    Returns:
        A dict mapping model ID to its rate limits (requests_per_minute,
        input_tokens_per_minute, output_tokens_per_minute, etc.).
        Fields that are None are omitted.
    """
    from llm_api_search.providers.base import RateLimit

    limits = get_rate_limits(provider, model_id=model, tier=tier)
    result = {}
    for mid, entry in limits.items():
        if isinstance(entry, RateLimit):
            result[mid] = _rl_to_dict(entry)
        else:
            result[mid] = {t: _rl_to_dict(rl) for t, rl in entry.items()}
    return result


@mcp.tool()
def llm_get_thinking_config(provider: str, model: str | None = None) -> dict:
    """Get thinking / reasoning-control configuration for an LLM provider.

    Tells you whether a model supports extended thinking, the parameter that
    enables it, and valid values:
      - Anthropic: output_config.effort (low/medium/high/xhigh/max), adaptive thinking.
      - OpenAI: reasoning.effort (minimal/low/medium/high, model-dependent).
      - Google: thinkingLevel (Gemini 3) or thinkingBudget tokens (Gemini 2.5).
      - DeepSeek: reasoning_effort (high/max) + thinking enable/disable toggle.
      - Inception/Mercury: reasoning_effort (instant/low/medium/high).

    Args:
        provider: One of "anthropic", "google", "openai", "inception", "deepseek", "zai", "minimax".
        model: Optional model ID. A model with no thinking support returns
               {"supported": false, "mode": "none"}.

    Returns:
        A dict mapping model ID to its thinking config (supported, mode,
        parameter, levels, default_level, budget ranges, can_disable, notes).
        Fields that are None/empty are omitted.
    """
    configs = get_thinking_config(provider, model_id=model)
    return {mid: _tc_to_dict(tc) for mid, tc in configs.items()}
