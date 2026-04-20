"""OpenAI rate limits — published API limits per model.

Source: https://platform.openai.com/docs/guides/rate-limits
Verified: 2026-03-20 (Tier 1 baseline, approximate ranges)

Values are conservative lower bounds from published Tier 1 ranges.
OpenAI has additional tiers (free, tier-2 through tier-5) not yet captured.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    # --- GPT-4o family: 500 RPM, 30K TPM, 10K RPD ---
    "gpt-4o": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "gpt-4o-audio-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "gpt-4o-realtime-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "gpt-4o-search-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    # --- GPT-4o-mini family: 500 RPM, 200K TPM, 20K RPD ---
    "gpt-4o-mini": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-4o-mini-audio-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-4o-mini-realtime-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-4o-mini-search-preview": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    # --- Standard GPT-5 (5, 5.1–5.4 base, chat-latest, search-api): 500 RPM, 500K TPM, 20K RPD ---
    "gpt-5": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5-chat-latest": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5-search-api": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.1": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.1-chat-latest": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.2": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.2-chat-latest": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.3-chat-latest": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    "gpt-5.4": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=20_000),
    },
    # --- GPT-5 mini/nano: 500 RPM, 500K TPM, 50K RPD ---
    "gpt-5-mini": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=50_000),
    },
    "gpt-5-nano": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=50_000),
    },
    "gpt-5.4-mini": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=50_000),
    },
    "gpt-5.4-nano": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=500_000, requests_per_day=50_000),
    },
    # --- GPT-5 pro (heavy): 500 RPM, 30K TPM, 10K RPD ---
    "gpt-5-pro": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "gpt-5.2-pro": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "gpt-5.4-pro": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    # --- Codex (all generations): 500 RPM, 200K TPM, 20K RPD ---
    "gpt-5-codex": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-5.1-codex": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-5.1-codex-max": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-5.1-codex-mini": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-5.2-codex": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    "gpt-5.3-codex": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    # --- Reasoning models ---
    "o3": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "o3-pro": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=30_000, requests_per_day=10_000),
    },
    "o4-mini": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=200_000, requests_per_day=20_000),
    },
    # --- Image models: 50 RPM, 1K RPD ---
    "gpt-image-1": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=1_000),
    },
    "gpt-image-1-mini": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=1_000),
    },
    "gpt-image-1.5": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=1_000),
    },
    # --- TTS: 500 RPM, 20K RPD ---
    "gpt-4o-mini-tts": {
        "tier-1": RateLimit(requests_per_minute=500, requests_per_day=20_000),
    },
    "tts-1": {
        "tier-1": RateLimit(requests_per_minute=500, requests_per_day=20_000),
    },
    "tts-1-hd": {
        "tier-1": RateLimit(requests_per_minute=500, requests_per_day=20_000),
    },
    # --- Transcription: 50 RPM, 2K RPD ---
    "gpt-4o-transcribe": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=2_000),
    },
    "gpt-4o-mini-transcribe": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=2_000),
    },
    "gpt-4o-transcribe-diarize": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=2_000),
    },
    "whisper-1": {
        "tier-1": RateLimit(requests_per_minute=50, requests_per_day=2_000),
    },
    # --- Embeddings: 500 RPM, 1M TPM, 50K RPD ---
    "text-embedding-3-large": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=1_000_000, requests_per_day=50_000),
    },
    "text-embedding-3-small": {
        "tier-1": RateLimit(requests_per_minute=500, tokens_per_minute=1_000_000, requests_per_day=50_000),
    },
}
