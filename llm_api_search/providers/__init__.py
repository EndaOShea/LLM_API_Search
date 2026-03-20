from __future__ import annotations

import re

from llm_api_search.providers.base import (
    Provider, ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, MusicModelInfo, VideoModelInfo, ModelType,
    ProviderInfo, RateLimit, SUPPORTED_LANGUAGES,
)
from llm_api_search.providers.anthropic import AnthropicProvider
from llm_api_search.providers.google import GeminiProvider
from llm_api_search.providers.openai import OpenAIProvider
from llm_api_search.providers.inception import InceptionProvider

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "google": GeminiProvider,
    "openai": OpenAIProvider,
    "inception": InceptionProvider,
}

# ---------------------------------------------------------------------------
# Model filtering — hide dated snapshots and legacy models from default output
# ---------------------------------------------------------------------------

# Matches date suffixes: -YYYY-MM-DD, -YYYYMMDD, -MM-YYYY
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8}|\d{2}-\d{4})$")

# Explicitly superseded / legacy models per provider.
# Dated snapshots are handled automatically by pattern matching (see filter_models).
LEGACY_MODELS: dict[str, set[str]] = {
    "openai": {
        # GPT-4 base (superseded by GPT-4o, then GPT-5.x)
        "gpt-4", "gpt-4-0125-preview", "gpt-4-0613", "gpt-4-1106-preview",
        "gpt-4-turbo", "gpt-4-turbo-preview",
        # GPT-4.1 family (superseded by GPT-5.x)
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        # Superseded reasoning model
        "o3-mini",
        # Legacy embedding
        "text-embedding-ada-002",
        # TTS dated snapshots (non-standard date format)
        "tts-1-1106", "tts-1-hd-1106",
    },
    "anthropic": {
        # Claude 3.x (superseded by 4.x)
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        # Claude 4.0 (superseded by 4.5, then 4.6)
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        # Claude 4.1 (superseded by 4.5, then 4.6)
        "claude-opus-4-1-20250805",
        # Claude 4.5 (superseded by 4.6)
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
        "claude-opus-4-5-20251124",
    },
    "google": {
        # Versioned snapshots (non-date suffix format)
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite-001",
        # Specialized/niche
        "aqa",
    },
    "inception": {
        "mercury", "mercury-coder",
    },
}


def filter_models(models: list[ModelInfo], provider: str) -> list[ModelInfo]:
    """Filter out dated snapshots and legacy models.

    Dated snapshots (e.g. ``gpt-4o-2024-05-13``) are removed when the
    non-dated alias (``gpt-4o``) also exists in the model list.  Models
    in ``LEGACY_MODELS`` are always removed.
    """
    all_ids = {m.model_id for m in models}
    legacy = LEGACY_MODELS.get(provider, set())
    result = []
    for m in models:
        if m.model_id in legacy:
            continue
        date_match = _DATE_SUFFIX_RE.search(m.model_id)
        if date_match:
            base_id = m.model_id[: date_match.start()]
            if base_id in all_ids:
                continue
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# Rate limit lookup
# ---------------------------------------------------------------------------

def get_rate_limits(
    provider: str, model_id: str | None = None,
) -> dict[str, RateLimit]:
    """Look up rate limits for a provider, optionally narrowed to a model.

    Args:
        provider: Provider key (e.g. ``"openai"``).
        model_id: Optional model ID.  If the exact ID isn't found, falls back
                  to the base alias by stripping date suffixes.

    Returns:
        A dict mapping model ID → ``RateLimit``.  If *model_id* is given,
        the dict contains at most one entry.

    Raises:
        KeyError: If *provider* is unknown.
    """
    from llm_api_search.providers.rate_limits import PROVIDER_RATE_LIMITS

    provider_lower = provider.lower()
    if provider_lower not in PROVIDER_RATE_LIMITS:
        raise KeyError(
            f"Unknown provider {provider!r}. "
            f"Available: {', '.join(PROVIDER_RATE_LIMITS)}"
        )

    limits = PROVIDER_RATE_LIMITS[provider_lower]

    if model_id is None:
        return dict(limits)

    # Exact match first.
    if model_id in limits:
        return {model_id: limits[model_id]}

    # Fall back to base alias by stripping date suffix.
    date_match = _DATE_SUFFIX_RE.search(model_id)
    if date_match:
        base_id = model_id[: date_match.start()]
        if base_id in limits:
            return {model_id: limits[base_id]}

    return {}


__all__ = [
    "Provider",
    "ModelInfo",
    "TextModelInfo",
    "ImageModelInfo",
    "AudioTTSModelInfo",
    "AudioTranscriptionModelInfo",
    "EmbeddingModelInfo",
    "MusicModelInfo",
    "VideoModelInfo",
    "ModelType",
    "ProviderInfo",
    "SUPPORTED_LANGUAGES",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "InceptionProvider",
    "PROVIDERS",
    "LEGACY_MODELS",
    "filter_models",
    "RateLimit",
    "get_rate_limits",
]
