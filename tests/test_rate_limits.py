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
    # Anthropic uses tier-1, others use free/paid
    for provider, tier in [
        ("anthropic", "tier-1"),
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


def test_anthropic_has_four_tiers():
    """Anthropic models should have tier-1 through tier-4."""
    limits = get_rate_limits("anthropic")
    for model_id, entry in limits.items():
        for t in ("tier-1", "tier-2", "tier-3", "tier-4"):
            assert t in entry, f"anthropic/{model_id}: missing {t}"


def test_anthropic_tiers_increase():
    """Higher Anthropic tiers should have higher limits."""
    limits = get_rate_limits("anthropic", "claude-sonnet-4-6")
    entry = limits["claude-sonnet-4-6"]
    for lower, higher in [("tier-1", "tier-2"), ("tier-2", "tier-3"), ("tier-3", "tier-4")]:
        assert entry[lower].requests_per_minute < entry[higher].requests_per_minute, (
            f"anthropic/claude-sonnet-4-6: {lower} RPM not less than {higher}"
        )


def test_get_rate_limit_for_specific_model():
    """Querying a specific model with tier= should return one entry."""
    limits = get_rate_limits("anthropic", "claude-sonnet-4-6", tier="tier-1")
    assert "claude-sonnet-4-6" in limits
    rl = limits["claude-sonnet-4-6"]
    assert isinstance(rl, RateLimit)
    assert rl.requests_per_minute == 50
    assert rl.input_tokens_per_minute == 30_000
    assert rl.output_tokens_per_minute == 8_000


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
