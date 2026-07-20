"""Tests for rate limit lookup."""

from llm_api_search.providers import get_rate_limits
from llm_api_search.providers.base import RateLimit

import pytest


def _assert_rate_limit_valid(rl: RateLimit, label: str) -> None:
    """Assert that a RateLimit has at least one non-None field."""
    fields = [
        rl.requests_per_minute,
        rl.tokens_per_minute,
        rl.input_tokens_per_minute,
        rl.output_tokens_per_minute,
        rl.requests_per_day,
        rl.tokens_per_day,
        rl.images_per_minute,
        rl.batch_queue_limit,
    ]
    assert any(f is not None for f in fields), (
        f"{label}: rate limit has no fields set"
    )


def test_get_all_rate_limits():
    """Each provider should return a non-empty dict of tiered rate limits."""
    for provider in ("anthropic", "google", "openai", "inception"):
        limits = get_rate_limits(provider)
        assert len(limits) > 0, f"{provider}: no rate limits defined"
        for model_id, entry in limits.items():
            assert isinstance(entry, dict), (
                f"{provider}/{model_id}: expected tier dict"
            )
            assert len(entry) > 0, (
                f"{provider}/{model_id}: tier dict is empty"
            )
            for tier_name, rl in entry.items():
                assert isinstance(rl, RateLimit), (
                    f"{provider}/{model_id}/{tier_name}: not a RateLimit"
                )


def test_get_rate_limits_with_tier():
    """Passing tier= should return flat RateLimits, skipping models without that tier."""
    # Anthropic uses start/build/scale, others use free/paid
    for provider, tier in [
        ("anthropic", "start"),
        ("openai", "tier-1"),
        ("google", "free"),
        ("inception", "free"),
    ]:
        limits = get_rate_limits(provider, tier=tier)
        assert len(limits) > 0, f"{provider}: no models with tier={tier}"
        for model_id, rl in limits.items():
            assert isinstance(rl, RateLimit), (
                f"{provider}/{model_id}: tier={tier} did not return RateLimit"
            )


def test_anthropic_has_three_tiers():
    """Anthropic models should have start/build/scale (Custom is unpublished)."""
    limits = get_rate_limits("anthropic")
    for model_id, entry in limits.items():
        for t in ("start", "build", "scale"):
            assert t in entry, f"anthropic/{model_id}: missing {t}"


def test_anthropic_tiers_increase():
    """Higher Anthropic tiers should have higher limits."""
    limits = get_rate_limits("anthropic", "claude-sonnet-4-6")
    entry = limits["claude-sonnet-4-6"]
    for lower, higher in [("start", "build"), ("build", "scale")]:
        assert entry[lower].requests_per_minute < entry[higher].requests_per_minute, (
            f"anthropic/claude-sonnet-4-6: {lower} RPM not less than {higher}"
        )


def test_get_rate_limit_for_specific_model():
    """Querying a specific model with tier= should return one entry."""
    limits = get_rate_limits("anthropic", "claude-sonnet-4-6", tier="start")
    assert "claude-sonnet-4-6" in limits
    rl = limits["claude-sonnet-4-6"]
    assert isinstance(rl, RateLimit)
    assert rl.requests_per_minute == 1_000
    assert rl.input_tokens_per_minute == 2_000_000
    assert rl.output_tokens_per_minute == 400_000


def test_dated_snapshot_fallback():
    """A dated snapshot should fall back to the base alias."""
    limits = get_rate_limits("openai", "gpt-4o-2024-08-06", tier="tier-1")
    assert "gpt-4o-2024-08-06" in limits
    rl = limits["gpt-4o-2024-08-06"]
    assert isinstance(rl, RateLimit)
    assert rl.requests_per_minute is not None


def test_unknown_model_returns_empty():
    """An unknown model should return an empty dict, not raise."""
    limits = get_rate_limits("openai", "nonexistent-model-xyz")
    assert limits == {}


def test_unknown_provider_raises():
    """An unknown provider should raise KeyError."""
    with pytest.raises(KeyError):
        get_rate_limits("nonexistent-provider")


def test_rate_limit_has_at_least_one_field():
    """Every rate limit entry should have at least one non-None field."""
    for provider in ("anthropic", "google", "openai", "inception"):
        limits = get_rate_limits(provider)
        for model_id, entry in limits.items():
            for tier_name, rl in entry.items():
                _assert_rate_limit_valid(rl, f"{provider}/{model_id}/{tier_name}")


# Pre-existing rate-limit gaps — models that landed in _STATIC_MODELS before
# this coverage test existed. New additions should NOT be added here; instead,
# fill in real rate limits in the appropriate providers/rate_limits/ module.
_RATE_LIMIT_COVERAGE_EXEMPT: dict[str, set[str]] = {
    "google": {
        "gemma-3n-e4b-it",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
        "aqa",
    },
    # DeepSeek publishes no numeric rate limits — the docs state limits are
    # adjusted dynamically based on server load (HTTP 429 on overflow).  This
    # is a publisher policy, not a coverage gap from the auto-update flow.
    "deepseek": {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    },
    # Z.ai does not publish per-model tier rate limits (RPM/TPM).  The rate-
    # limit reference page describes general service limits only.  This is a
    # publisher policy, not a coverage gap.
    "zai": {
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-4.6",
        "glm-4.5-air",
        "glm-4.5-flash",
        "glm-5v-turbo",
        "glm-4.6v-flash",
        "glm-4.5",
        "glm-4.7",
        "glm-5-turbo",
    },
}


def test_all_nonpreview_text_models_have_rate_limits():
    """Every MCP-visible, non-preview text model must have a rate-limit entry.

    Without this, ``update_models.py`` can silently grow ``_STATIC_MODELS``
    past what ``providers/rate_limits/`` covers, and ``llm_get_rate_limits``
    starts returning empty results for new models.
    """
    from llm_api_search.providers import PROVIDERS, filter_models
    from llm_api_search.providers.base import TextModelInfo

    for key in PROVIDERS:
        provider = PROVIDERS[key]()
        info = provider.get_static_info()
        filtered = filter_models(info.models, key)
        exempt = _RATE_LIMIT_COVERAGE_EXEMPT.get(key, set())

        for m in filtered:
            if not isinstance(m, TextModelInfo):
                continue
            if "preview" in m.model_id:
                continue
            if m.model_id in exempt:
                continue
            rates = get_rate_limits(key, model_id=m.model_id)
            assert rates, (
                f"{key}/{m.model_id}: no rate limit entry — add one to "
                f"providers/rate_limits/{key}.py"
            )


def test_zai_rate_limits_registered_empty():
    from llm_api_search.providers.rate_limits import PROVIDER_RATE_LIMITS
    assert "zai" in PROVIDER_RATE_LIMITS
    assert PROVIDER_RATE_LIMITS["zai"] == {}


def test_kimi_rate_limits_shared_across_tiers():
    """Kimi's spend tiers apply identically to every curated model."""
    limits = get_rate_limits("kimi")
    assert set(limits.keys()) == {"kimi-k3", "kimi-k2.6"}
    for model_id, entry in limits.items():
        assert set(entry.keys()) == {"tier0", "tier1", "tier2", "tier3", "tier4", "tier5"}
    k3 = get_rate_limits("kimi", "kimi-k3", tier="tier0")["kimi-k3"]
    assert k3.requests_per_minute == 3
    assert k3.tokens_per_minute == 500_000
    assert k3.tokens_per_day == 1_500_000
    tier5 = get_rate_limits("kimi", "kimi-k3", tier="tier5")["kimi-k3"]
    assert tier5.requests_per_minute == 10_000
    assert tier5.tokens_per_minute == 5_000_000
