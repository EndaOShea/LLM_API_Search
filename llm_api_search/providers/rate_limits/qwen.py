"""Qwen (Alibaba Model Studio) rate limits — Singapore/international defaults.

Source: https://www.alibabacloud.com/help/en/model-studio/rate-limit
Verified: 2026-07-20

DashScope publishes flat per-model defaults for the international region
(not spend-tiered like Kimi), so both models are exposed under a single
"default" tier, matching the MiniMax convention.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "qwen3.7-max": {
        "default": RateLimit(requests_per_minute=600, tokens_per_minute=1_000_000),
    },
    "qwen3.7-plus": {
        "default": RateLimit(requests_per_minute=15_000, tokens_per_minute=5_000_000),
    },
}
