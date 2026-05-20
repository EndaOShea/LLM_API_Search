"""Google (Gemini) rate limits — published API limits per model.

Source: Google AI Studio rate limits page (https://aistudio.google.com/rate-limit)
Verified: 2026-03-20

Models with 0/0/0 on a tier have no access on that tier and are omitted.
"Unlimited" values are represented as None.
Google has: free, tier-1, tier-2, tier-3.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    # --- Text models ---
    "gemini-2.5-flash": {
        "free": RateLimit(requests_per_minute=5, tokens_per_minute=250_000, requests_per_day=20),
        "tier-1": RateLimit(requests_per_minute=1_000, tokens_per_minute=1_000_000, requests_per_day=10_000),
        "tier-2": RateLimit(requests_per_minute=2_000, tokens_per_minute=3_000_000, requests_per_day=100_000),
        "tier-3": RateLimit(requests_per_minute=20_000, tokens_per_minute=20_000_000),
    },
    "gemini-2.5-pro": {
        "tier-1": RateLimit(requests_per_minute=150, tokens_per_minute=2_000_000, requests_per_day=1_000),
        "tier-2": RateLimit(requests_per_minute=1_000, tokens_per_minute=5_000_000, requests_per_day=50_000),
        "tier-3": RateLimit(requests_per_minute=2_000, tokens_per_minute=8_000_000),
    },
    "gemini-2.5-flash-lite": {
        "free": RateLimit(requests_per_minute=10, tokens_per_minute=250_000, requests_per_day=20),
        "tier-1": RateLimit(requests_per_minute=4_000, tokens_per_minute=4_000_000),
        "tier-2": RateLimit(requests_per_minute=10_000, tokens_per_minute=10_000_000),
        "tier-3": RateLimit(requests_per_minute=30_000, tokens_per_minute=30_000_000),
    },
    "gemini-2.0-flash": {
        "tier-1": RateLimit(requests_per_minute=2_000, tokens_per_minute=4_000_000),
        "tier-2": RateLimit(requests_per_minute=10_000, tokens_per_minute=10_000_000),
        "tier-3": RateLimit(requests_per_minute=30_000, tokens_per_minute=30_000_000),
    },
    "gemini-2.0-flash-lite": {
        "tier-1": RateLimit(requests_per_minute=4_000, tokens_per_minute=4_000_000),
        "tier-2": RateLimit(requests_per_minute=20_000, tokens_per_minute=10_000_000),
        "tier-3": RateLimit(requests_per_minute=30_000, tokens_per_minute=30_000_000),
    },
    "gemini-3-flash-preview": {
        "free": RateLimit(requests_per_minute=5, tokens_per_minute=250_000, requests_per_day=20),
        "tier-1": RateLimit(requests_per_minute=1_000, tokens_per_minute=2_000_000, requests_per_day=10_000),
        "tier-2": RateLimit(requests_per_minute=2_000, tokens_per_minute=3_000_000, requests_per_day=100_000),
        "tier-3": RateLimit(requests_per_minute=20_000, tokens_per_minute=20_000_000),
    },
    # Mirrors gemini-2.5-flash (closest GA Flash analog); Google's public docs
    # do not yet publish per-tier numbers for gemini-3.5-flash. Refine when
    # AI Studio rate-limit dashboard surfaces verified figures.
    "gemini-3.5-flash": {
        "free": RateLimit(requests_per_minute=5, tokens_per_minute=250_000, requests_per_day=20),
        "tier-1": RateLimit(requests_per_minute=1_000, tokens_per_minute=1_000_000, requests_per_day=10_000),
        "tier-2": RateLimit(requests_per_minute=2_000, tokens_per_minute=3_000_000, requests_per_day=100_000),
        "tier-3": RateLimit(requests_per_minute=20_000, tokens_per_minute=20_000_000),
    },
    "gemini-3.1-pro-preview": {
        "tier-1": RateLimit(requests_per_minute=25, tokens_per_minute=2_000_000, requests_per_day=250),
        "tier-2": RateLimit(requests_per_minute=1_000, tokens_per_minute=5_000_000, requests_per_day=50_000),
        "tier-3": RateLimit(requests_per_minute=2_000, tokens_per_minute=8_000_000),
    },
    "gemini-3.1-flash-lite-preview": {
        "free": RateLimit(requests_per_minute=15, tokens_per_minute=250_000, requests_per_day=500),
        "tier-1": RateLimit(requests_per_minute=4_000, tokens_per_minute=4_000_000, requests_per_day=150_000),
        "tier-2": RateLimit(requests_per_minute=10_000, tokens_per_minute=10_000_000, requests_per_day=350_000),
        "tier-3": RateLimit(requests_per_minute=30_000, tokens_per_minute=30_000_000),
    },
    "gemini-3.1-flash-lite": {
        "free": RateLimit(requests_per_minute=15, tokens_per_minute=250_000, requests_per_day=500),
        "tier-1": RateLimit(requests_per_minute=4_000, tokens_per_minute=4_000_000, requests_per_day=150_000),
        "tier-2": RateLimit(requests_per_minute=10_000, tokens_per_minute=10_000_000, requests_per_day=350_000),
        "tier-3": RateLimit(requests_per_minute=30_000, tokens_per_minute=30_000_000),
    },
    # --- Experimental (no tier changes) ---
    "gemini-2.0-flash-exp": {
        "tier-1": RateLimit(requests_per_minute=10, tokens_per_minute=250_000, requests_per_day=500),
        "tier-2": RateLimit(requests_per_minute=10, tokens_per_minute=250_000, requests_per_day=500),
        "tier-3": RateLimit(requests_per_minute=10, tokens_per_minute=250_000, requests_per_day=500),
    },
    # --- TTS models ---
    "gemini-2.5-flash-preview-tts": {
        "free": RateLimit(requests_per_minute=3, tokens_per_minute=10_000, requests_per_day=10),
        "tier-1": RateLimit(requests_per_minute=10, tokens_per_minute=10_000, requests_per_day=100),
        "tier-2": RateLimit(requests_per_minute=1_000, tokens_per_minute=100_000, requests_per_day=10_000),
        "tier-3": RateLimit(requests_per_minute=1_000, tokens_per_minute=1_000_000),
    },
    "gemini-2.5-pro-preview-tts": {
        "tier-1": RateLimit(requests_per_minute=10, tokens_per_minute=10_000, requests_per_day=50),
        "tier-2": RateLimit(requests_per_minute=250, tokens_per_minute=25_000, requests_per_day=2_500),
        "tier-3": RateLimit(requests_per_minute=250, tokens_per_minute=1_000_000),
    },
    # --- Gemma open models (same across all tiers) ---
    "gemma-3-1b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-3-4b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-3-12b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-3-27b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-3n-e2b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-4-26b-a4b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    "gemma-4-31b-it": {
        "free": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-1": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
        "tier-3": RateLimit(requests_per_minute=30, tokens_per_minute=15_000, requests_per_day=14_400),
    },
    # --- Image models ---
    "imagen-4.0-generate-001": {
        "free": RateLimit(requests_per_day=25),
        "tier-1": RateLimit(requests_per_minute=10, requests_per_day=70),
        "tier-2": RateLimit(requests_per_minute=15, requests_per_day=1_000),
        "tier-3": RateLimit(requests_per_minute=20, requests_per_day=15_000),
    },
    "imagen-4.0-ultra-generate-001": {
        "free": RateLimit(requests_per_day=25),
        "tier-1": RateLimit(requests_per_minute=5, requests_per_day=30),
        "tier-2": RateLimit(requests_per_minute=10, requests_per_day=400),
        "tier-3": RateLimit(requests_per_minute=15, requests_per_day=5_000),
    },
    "imagen-4.0-fast-generate-001": {
        "free": RateLimit(requests_per_day=25),
        "tier-1": RateLimit(requests_per_minute=10, requests_per_day=70),
        "tier-2": RateLimit(requests_per_minute=15, requests_per_day=1_000),
        "tier-3": RateLimit(requests_per_minute=20, requests_per_day=15_000),
    },
    # --- Multimodal image generation (Nano Banana family) ---
    "gemini-2.5-flash-image": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=2_000),
        "tier-2": RateLimit(requests_per_minute=2_000, tokens_per_minute=1_500_000, requests_per_day=50_000),
        "tier-3": RateLimit(requests_per_minute=5_000, tokens_per_minute=5_000_000),
    },
    "nano-banana-pro-preview": {
        "tier-1": RateLimit(requests_per_minute=20, tokens_per_minute=100_000, requests_per_day=250),
        "tier-2": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=15_000),
        "tier-3": RateLimit(requests_per_minute=2_000, tokens_per_minute=5_000_000),
    },
    "gemini-3.1-flash-image-preview": {
        "tier-1": RateLimit(requests_per_minute=100, tokens_per_minute=200_000, requests_per_day=1_000),
        "tier-2": RateLimit(requests_per_minute=500, tokens_per_minute=1_000_000, requests_per_day=10_000),
        "tier-3": RateLimit(requests_per_minute=5_000, tokens_per_minute=10_000_000, requests_per_day=50_000),
    },
    # --- Video models ---
    "veo-3.0-generate-001": {
        "tier-1": RateLimit(requests_per_minute=2, requests_per_day=10),
        "tier-2": RateLimit(requests_per_minute=4, requests_per_day=50),
        "tier-3": RateLimit(requests_per_minute=10, requests_per_day=500),
    },
    "veo-3.0-fast-generate-001": {
        "tier-1": RateLimit(requests_per_minute=2, requests_per_day=10),
        "tier-2": RateLimit(requests_per_minute=4, requests_per_day=50),
        "tier-3": RateLimit(requests_per_minute=10, requests_per_day=500),
    },
    # --- Embedding models ---
    "gemini-embedding-001": {
        "free": RateLimit(requests_per_minute=100, tokens_per_minute=30_000, requests_per_day=1_000),
        "tier-1": RateLimit(requests_per_minute=3_000, tokens_per_minute=1_000_000),
        "tier-2": RateLimit(requests_per_minute=5_000, tokens_per_minute=5_000_000),
        "tier-3": RateLimit(requests_per_minute=20_000, tokens_per_minute=20_000_000),
    },
    "gemini-embedding-2-preview": {
        "free": RateLimit(requests_per_minute=100, tokens_per_minute=30_000, requests_per_day=1_000),
        "tier-1": RateLimit(requests_per_minute=3_000, tokens_per_minute=1_000_000),
        "tier-2": RateLimit(requests_per_minute=5_000, tokens_per_minute=5_000_000),
        "tier-3": RateLimit(requests_per_minute=20_000, tokens_per_minute=20_000_000),
    },
    # --- Specialized models ---
    "gemini-robotics-er-1.5-preview": {
        "free": RateLimit(requests_per_minute=10, tokens_per_minute=250_000, requests_per_day=20),
        "tier-1": RateLimit(requests_per_minute=1_000, tokens_per_minute=2_000_000, requests_per_day=14_400),
        "tier-2": RateLimit(requests_per_minute=2_000, tokens_per_minute=3_000_000, requests_per_day=100_000),
        "tier-3": RateLimit(requests_per_minute=2_000, tokens_per_minute=8_000_000),
    },
    "gemini-2.5-computer-use-preview-10-2025": {
        "tier-1": RateLimit(requests_per_minute=150, tokens_per_minute=2_000_000, requests_per_day=10_000),
        "tier-2": RateLimit(requests_per_minute=1_000, tokens_per_minute=5_000_000, requests_per_day=50_000),
        "tier-3": RateLimit(requests_per_minute=2_000, tokens_per_minute=8_000_000),
    },
    "deep-research-pro-preview-12-2025": {
        "tier-1": RateLimit(requests_per_minute=1, tokens_per_minute=500_000, requests_per_day=1_440),
        "tier-2": RateLimit(requests_per_minute=5, tokens_per_minute=5_000_000, requests_per_day=7_200),
        "tier-3": RateLimit(requests_per_minute=10, tokens_per_minute=10_000_000),
    },
    # --- Live API (native audio) ---
    "gemini-2.5-flash-native-audio-latest": {
        "free": RateLimit(tokens_per_minute=1_000_000),
        "tier-1": RateLimit(tokens_per_minute=1_000_000),
        "tier-2": RateLimit(tokens_per_minute=10_000_000),
        "tier-3": RateLimit(tokens_per_minute=10_000_000),
    },
}
