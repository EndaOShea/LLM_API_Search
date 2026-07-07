"""Anthropic rate limits — published API limits per model.

Source: https://platform.claude.com/docs/en/api/rate-limits
Verified: 2026-07-07

Anthropic has no free tier. Usage tiers are named Start / Build / Scale /
Custom (renamed from the older "Tier 1".."Tier 4" scheme). Custom is
negotiated per-account and publishes no fixed numbers, so only three tiers
are modeled here. Limits are per model class: the Opus 4.x limit is shared
across Opus 4.8/4.7/4.6/4.5, and the Sonnet 4.x limit is shared across
Sonnet 4.6/4.5 — Claude Sonnet 5 has its own separate limit, not part of
that combined bucket. Claude Fable 5 also has its own dedicated limits.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "claude-fable-5": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=500_000,
            output_tokens_per_minute=100_000,
        ),
        "build": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=1_500_000,
            output_tokens_per_minute=300_000,
        ),
        "scale": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=4_000_000,
            output_tokens_per_minute=800_000,
        ),
    },
    "claude-opus-4-8": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
    "claude-opus-4-7": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
    "claude-sonnet-5": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
    "claude-sonnet-4-6": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
    "claude-opus-4-6": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
    "claude-haiku-4-5-20251001": {
        "start": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
        "build": RateLimit(
            requests_per_minute=5_000,
            input_tokens_per_minute=5_000_000,
            output_tokens_per_minute=1_000_000,
        ),
        "scale": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=2_000_000,
        ),
    },
}
