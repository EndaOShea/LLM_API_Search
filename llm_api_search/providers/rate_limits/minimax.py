"""MiniMax rate limits — published per-model API limits.

Source: https://platform.minimax.io/docs/guides/rate-limits
Verified: 2026-07-01

MiniMax publishes flat per-model limits without a named tier system (the
Standard/Priority split on the pricing page is a billing tier, not a rate
tier), so all limits are exposed under a single "default" tier. RPM = requests
per minute; TPM = combined input+output tokens per minute.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    # --- Text: M3 200 RPM / 10M TPM; M2.7 family 500 RPM / 20M TPM ---
    "MiniMax-M3": {
        "default": RateLimit(requests_per_minute=200, tokens_per_minute=10_000_000),
    },
    "MiniMax-M2.7": {
        "default": RateLimit(requests_per_minute=500, tokens_per_minute=20_000_000),
    },
    "MiniMax-M2.7-highspeed": {
        "default": RateLimit(requests_per_minute=500, tokens_per_minute=20_000_000),
    },
    # --- Image: 10 RPM ---
    "image-01": {
        "default": RateLimit(requests_per_minute=10),
    },
    # --- Video: 5 RPM ---
    "MiniMax-Hailuo-2.3": {
        "default": RateLimit(requests_per_minute=5),
    },
    "MiniMax-Hailuo-2.3-Fast": {
        "default": RateLimit(requests_per_minute=5),
    },
    # --- TTS (T2A): 60 RPM ---
    "speech-2.8-hd": {
        "default": RateLimit(requests_per_minute=60),
    },
    "speech-2.8-turbo": {
        "default": RateLimit(requests_per_minute=60),
    },
    # --- Music: 120 RPM ---
    "Music-2.6": {
        "default": RateLimit(requests_per_minute=120),
    },
}
