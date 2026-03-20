"""Inception Labs (Mercury) rate limits — published API limits per model."""

from llm_api_search.providers.base import RateLimit

# Source: https://docs.inceptionlabs.ai/get-started/rate-limits#rate-limits
RATE_LIMITS: dict[str, RateLimit] = {
    "mercury-2": RateLimit(
        requests_per_minute=60,
        tokens_per_minute=100_000,
    ),
}
