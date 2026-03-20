"""OpenAI rate limits — published API limits per model."""

from llm_api_search.providers.base import RateLimit

# Source: https://platform.openai.com/docs/guides/rate-limits
RATE_LIMITS: dict[str, RateLimit] = {
    # GPT-4o family
    "gpt-4o": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
        requests_per_day=10_000,
    ),
    "gpt-4o-mini": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
        requests_per_day=10_000,
    ),
    "gpt-4o-audio-preview": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=20_000,
    ),
    "gpt-4o-mini-audio-preview": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=20_000,
    ),
    "gpt-4o-realtime-preview": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=20_000,
    ),
    "gpt-4o-mini-realtime-preview": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=20_000,
    ),
    "gpt-4o-search-preview": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-4o-mini-search-preview": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
    ),
    # GPT-5.x family
    "gpt-5": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
        requests_per_day=10_000,
    ),
    "gpt-5-mini": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
        requests_per_day=10_000,
    ),
    "gpt-5-nano": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
        requests_per_day=10_000,
    ),
    "gpt-5-pro": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=10_000,
    ),
    "gpt-5-codex": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5-search-api": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.1": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.1-codex": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.1-codex-max": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=10_000,
    ),
    "gpt-5.1-codex-mini": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
    ),
    "gpt-5.2": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.2-codex": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.2-pro": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=10_000,
    ),
    "gpt-5.3-codex": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.4": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.4-pro": RateLimit(
        requests_per_minute=100,
        tokens_per_minute=10_000,
    ),
    # Chat-latest aliases (same as base model)
    "gpt-5-chat-latest": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.1-chat-latest": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.2-chat-latest": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    "gpt-5.3-chat-latest": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=30_000,
    ),
    # Reasoning models
    "o3": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=100_000,
        requests_per_day=10_000,
    ),
    "o3-pro": RateLimit(
        requests_per_minute=20,
        tokens_per_minute=10_000,
    ),
    "o4-mini": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
        requests_per_day=10_000,
    ),
    # Image models
    "gpt-image-1": RateLimit(
        requests_per_minute=7,
        images_per_minute=7,
    ),
    "gpt-image-1-mini": RateLimit(
        requests_per_minute=15,
        images_per_minute=15,
    ),
    "gpt-image-1.5": RateLimit(
        requests_per_minute=7,
        images_per_minute=7,
    ),
    # Embedding models
    "text-embedding-3-large": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=1_000_000,
    ),
    "text-embedding-3-small": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=1_000_000,
    ),
    # TTS models
    "tts-1": RateLimit(
        requests_per_minute=50,
    ),
    "tts-1-hd": RateLimit(
        requests_per_minute=7,
    ),
    "gpt-4o-mini-tts": RateLimit(
        requests_per_minute=500,
        tokens_per_minute=200_000,
    ),
    # Transcription models
    "whisper-1": RateLimit(
        requests_per_minute=50,
    ),
    "gpt-4o-transcribe": RateLimit(
        requests_per_minute=100,
    ),
    "gpt-4o-mini-transcribe": RateLimit(
        requests_per_minute=100,
    ),
    "gpt-4o-transcribe-diarize": RateLimit(
        requests_per_minute=100,
    ),
}
