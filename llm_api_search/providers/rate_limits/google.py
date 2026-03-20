"""Google (Gemini) rate limits — published API limits per model."""

from llm_api_search.providers.base import RateLimit

# Source: https://ai.google.dev/gemini-api/docs/rate-limits
RATE_LIMITS: dict[str, RateLimit] = {
    # Gemini 2.5
    "gemini-2.5-flash": RateLimit(
        requests_per_minute=1_000,
        tokens_per_minute=250_000,
        requests_per_day=14_400,
    ),
    "gemini-2.5-pro": RateLimit(
        requests_per_minute=150,
        tokens_per_minute=1_000_000,
        requests_per_day=1_000,
    ),
    "gemini-2.5-flash-lite": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=500_000,
        requests_per_day=28_800,
    ),
    # Gemini 2.0
    "gemini-2.0-flash": RateLimit(
        requests_per_minute=1_000,
        tokens_per_minute=250_000,
        requests_per_day=14_400,
    ),
    "gemini-2.0-flash-lite": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=500_000,
        requests_per_day=28_800,
    ),
    # Gemini 3.x previews
    "gemini-3-pro-preview": RateLimit(
        requests_per_minute=150,
        tokens_per_minute=1_000_000,
        requests_per_day=1_000,
    ),
    "gemini-3-flash-preview": RateLimit(
        requests_per_minute=1_000,
        tokens_per_minute=250_000,
        requests_per_day=14_400,
    ),
    "gemini-3.1-pro-preview": RateLimit(
        requests_per_minute=150,
        tokens_per_minute=1_000_000,
        requests_per_day=1_000,
    ),
    "gemini-3.1-flash-lite-preview": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=500_000,
        requests_per_day=28_800,
    ),
    # Latest aliases
    "gemini-flash-latest": RateLimit(
        requests_per_minute=1_000,
        tokens_per_minute=250_000,
        requests_per_day=14_400,
    ),
    "gemini-flash-lite-latest": RateLimit(
        requests_per_minute=2_000,
        tokens_per_minute=500_000,
        requests_per_day=28_800,
    ),
    "gemini-pro-latest": RateLimit(
        requests_per_minute=150,
        tokens_per_minute=1_000_000,
        requests_per_day=1_000,
    ),
    # Gemma open models
    "gemma-3-27b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    "gemma-3-12b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    "gemma-3-4b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    "gemma-3-1b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    "gemma-3n-e4b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    "gemma-3n-e2b-it": RateLimit(
        requests_per_minute=30,
        tokens_per_minute=250_000,
    ),
    # TTS
    "gemini-2.5-flash-preview-tts": RateLimit(
        requests_per_minute=1_000,
        tokens_per_minute=250_000,
    ),
    "gemini-2.5-pro-preview-tts": RateLimit(
        requests_per_minute=150,
        tokens_per_minute=1_000_000,
    ),
    # Image models
    "imagen-4.0-generate-001": RateLimit(
        requests_per_minute=10,
        images_per_minute=10,
    ),
    "imagen-4.0-ultra-generate-001": RateLimit(
        requests_per_minute=5,
        images_per_minute=5,
    ),
    "imagen-4.0-fast-generate-001": RateLimit(
        requests_per_minute=20,
        images_per_minute=20,
    ),
    # Video models
    "veo-2.0-generate-001": RateLimit(
        requests_per_minute=5,
    ),
    "veo-3.0-generate-001": RateLimit(
        requests_per_minute=5,
    ),
    "veo-3.0-fast-generate-001": RateLimit(
        requests_per_minute=10,
    ),
    "veo-3.1-generate-preview": RateLimit(
        requests_per_minute=5,
    ),
    "veo-3.1-fast-generate-preview": RateLimit(
        requests_per_minute=10,
    ),
    # Embedding models
    "gemini-embedding-001": RateLimit(
        requests_per_minute=1_500,
        tokens_per_minute=1_000_000,
    ),
    "gemini-embedding-2-preview": RateLimit(
        requests_per_minute=1_500,
        tokens_per_minute=1_000_000,
    ),
}
