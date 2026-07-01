"""Rate limit data registry — maps provider keys to their rate limit dicts."""

from llm_api_search.providers.base import RateLimit
from llm_api_search.providers.rate_limits.anthropic import RATE_LIMITS as _ANTHROPIC
from llm_api_search.providers.rate_limits.google import RATE_LIMITS as _GOOGLE
from llm_api_search.providers.rate_limits.openai import RATE_LIMITS as _OPENAI
from llm_api_search.providers.rate_limits.inception import RATE_LIMITS as _INCEPTION
from llm_api_search.providers.rate_limits.deepseek import RATE_LIMITS as _DEEPSEEK
from llm_api_search.providers.rate_limits.zai import RATE_LIMITS as _ZAI

# Each provider maps model_id → {tier_name: RateLimit, ...}.
# Tier names are provider-specific (e.g. "tier-1" for Anthropic, "free" for OpenAI).
PROVIDER_RATE_LIMITS: dict[str, dict[str, dict[str, RateLimit]]] = {
    "anthropic": _ANTHROPIC,
    "google": _GOOGLE,
    "openai": _OPENAI,
    "inception": _INCEPTION,
    "deepseek": _DEEPSEEK,
    "zai": _ZAI,
}
