"""Inception Labs (Mercury) rate limits — published API limits per tier.

Source: https://docs.inceptionlabs.ai/get-started/rate-limits#rate-limits
Verified: 2026-09-15

Inception publishes one per-minute limit table per account tier (Free, Pay As
You Go, Enterprise), not per model, so every model carries the same tier dict.
"paid" is the Pay As You Go tier. The Enterprise tier is "Custom" (negotiated,
no published numbers), so it isn't modeled — same convention as Anthropic's
Custom tier.
"""

from llm_api_search.providers.base import RateLimit

_TIERS: dict[str, RateLimit] = {
    "free": RateLimit(
        requests_per_minute=1_000,
        input_tokens_per_minute=1_000_000,
        output_tokens_per_minute=100_000,
    ),
    "paid": RateLimit(
        requests_per_minute=3_000,
        input_tokens_per_minute=3_000_000,
        output_tokens_per_minute=300_000,
    ),
}

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "mercury-2.5": _TIERS,
    "mercury-2": _TIERS,
    "mercury-edit-2": _TIERS,
    "mercury-edit": _TIERS,
}
