"""Anthropic rate limits — published API limits per model.

Source: https://docs.anthropic.com/en/api/rate-limits
Verified: 2026-03-20

Anthropic has no free tier.  Tier 1 requires a $5 credit purchase.
Limits are per model class — Sonnet 4.x applies to all Sonnet 4.x models,
Opus 4.x applies to all Opus 4.x models.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "claude-fable-5": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=100_000,
            output_tokens_per_minute=20_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=500_000,
            output_tokens_per_minute=100_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=1_500_000,
            output_tokens_per_minute=300_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=4_000_000,
            output_tokens_per_minute=800_000,
        ),
    },
    "claude-opus-4-8": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=30_000,
            output_tokens_per_minute=8_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=450_000,
            output_tokens_per_minute=90_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=800_000,
            output_tokens_per_minute=160_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
    },
    "claude-opus-4-7": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=30_000,
            output_tokens_per_minute=8_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=450_000,
            output_tokens_per_minute=90_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=800_000,
            output_tokens_per_minute=160_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
    },
    "claude-sonnet-4-6": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=30_000,
            output_tokens_per_minute=8_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=450_000,
            output_tokens_per_minute=90_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=800_000,
            output_tokens_per_minute=160_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
    },
    "claude-opus-4-6": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=30_000,
            output_tokens_per_minute=8_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=450_000,
            output_tokens_per_minute=90_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=800_000,
            output_tokens_per_minute=160_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=2_000_000,
            output_tokens_per_minute=400_000,
        ),
    },
    "claude-haiku-4-5-20251001": {
        "tier-1": RateLimit(
            requests_per_minute=50,
            input_tokens_per_minute=50_000,
            output_tokens_per_minute=10_000,
        ),
        "tier-2": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=450_000,
            output_tokens_per_minute=90_000,
        ),
        "tier-3": RateLimit(
            requests_per_minute=2_000,
            input_tokens_per_minute=1_000_000,
            output_tokens_per_minute=200_000,
        ),
        "tier-4": RateLimit(
            requests_per_minute=4_000,
            input_tokens_per_minute=4_000_000,
            output_tokens_per_minute=800_000,
        ),
    },
}
