"""Anthropic rate limits — published API limits per model."""

from llm_api_search.providers.base import RateLimit

# Source: https://docs.anthropic.com/en/api/rate-limits
RATE_LIMITS: dict[str, RateLimit] = {
    "claude-opus-4-6": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=80_000,
        tokens_per_day=2_500_000,
    ),
    "claude-sonnet-4-6": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=80_000,
        tokens_per_day=2_500_000,
    ),
    "claude-haiku-4-5-20251001": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=100_000,
        tokens_per_day=5_000_000,
    ),
}
