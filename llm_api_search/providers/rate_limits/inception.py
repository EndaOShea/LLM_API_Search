"""Inception Labs (Mercury) rate limits — published API limits per model.

Source: https://docs.inceptionlabs.ai/get-started/rate-limits#rate-limits
Verified: 2026-03-20
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "mercury-2": {
        "free": RateLimit(
            requests_per_minute=100,
            input_tokens_per_minute=100_000,
            output_tokens_per_minute=10_000,
        ),
        "paid": RateLimit(
            requests_per_minute=1_000,
            input_tokens_per_minute=1_000_000,
            output_tokens_per_minute=100_000,
        ),
        "enterprise": RateLimit(
            requests_per_minute=10_000,
            input_tokens_per_minute=10_000_000,
            output_tokens_per_minute=1_000_000,
        ),
    },
}
