"""Kimi (Moonshot AI) rate limits — account-wide spend tiers.

Source: https://platform.kimi.ai/docs/pricing/limits
Verified: 2026-07-20

Kimi's limits are account-wide (shared across all models by the same spend
tier), not per-model — both curated models get the same tier dict. Tier
names follow Moonshot's own Tier0..Tier5 labels (by cumulative recharge).
Unlisted TPD means Moonshot documents it as unlimited at that tier.
"""

from llm_api_search.providers.base import RateLimit

_KIMI_TIERS: dict[str, RateLimit] = {
    "tier0": RateLimit(requests_per_minute=3, tokens_per_minute=500_000, tokens_per_day=1_500_000),
    "tier1": RateLimit(requests_per_minute=200, tokens_per_minute=2_000_000),
    "tier2": RateLimit(requests_per_minute=500, tokens_per_minute=3_000_000),
    "tier3": RateLimit(requests_per_minute=5_000, tokens_per_minute=3_000_000),
    "tier4": RateLimit(requests_per_minute=5_000, tokens_per_minute=4_000_000),
    "tier5": RateLimit(requests_per_minute=10_000, tokens_per_minute=5_000_000),
}

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "kimi-k3": dict(_KIMI_TIERS),
    "kimi-k2.6": dict(_KIMI_TIERS),
}
