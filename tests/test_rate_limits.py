"""Tests for rate limit lookup."""

from llm_api_search.providers import get_rate_limits
from llm_api_search.providers.base import RateLimit

import pytest


def test_get_all_rate_limits():
    """Each provider should return a non-empty dict of rate limits."""
    for provider in ("anthropic", "google", "openai", "inception"):
        limits = get_rate_limits(provider)
        assert len(limits) > 0, f"{provider}: no rate limits defined"
        for model_id, rl in limits.items():
            assert isinstance(rl, RateLimit), f"{provider}/{model_id}: not a RateLimit"


def test_get_rate_limit_for_specific_model():
    """Querying a specific model should return at most one entry."""
    limits = get_rate_limits("openai", "gpt-4o")
    assert "gpt-4o" in limits
    rl = limits["gpt-4o"]
    assert rl.requests_per_minute is not None
    assert rl.requests_per_minute > 0


def test_dated_snapshot_fallback():
    """A dated snapshot should fall back to the base alias."""
    limits = get_rate_limits("openai", "gpt-4o-2024-08-06")
    assert "gpt-4o-2024-08-06" in limits
    rl = limits["gpt-4o-2024-08-06"]
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
        for model_id, rl in limits.items():
            fields = [
                rl.requests_per_minute,
                rl.tokens_per_minute,
                rl.requests_per_day,
                rl.tokens_per_day,
                rl.images_per_minute,
                rl.batch_queue_limit,
            ]
            assert any(f is not None for f in fields), (
                f"{provider}/{model_id}: rate limit has no fields set"
            )
