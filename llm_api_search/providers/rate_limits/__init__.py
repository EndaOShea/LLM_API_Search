"""Rate limit data registry — maps provider keys to their rate limit dicts."""

from llm_api_search.providers.base import RateLimit
from llm_api_search.providers.rate_limits.anthropic import RATE_LIMITS as _ANTHROPIC
from llm_api_search.providers.rate_limits.google import RATE_LIMITS as _GOOGLE
from llm_api_search.providers.rate_limits.openai import RATE_LIMITS as _OPENAI
from llm_api_search.providers.rate_limits.inception import RATE_LIMITS as _INCEPTION

PROVIDER_RATE_LIMITS: dict[str, dict[str, RateLimit]] = {
    "anthropic": _ANTHROPIC,
    "google": _GOOGLE,
    "openai": _OPENAI,
    "inception": _INCEPTION,
}
