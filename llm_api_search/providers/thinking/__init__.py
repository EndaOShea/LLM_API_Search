"""Thinking-config registry — maps provider keys to their thinking config dicts."""

from llm_api_search.providers.base import ThinkingConfig
from llm_api_search.providers.thinking.anthropic import THINKING_CONFIGS as _ANTHROPIC
from llm_api_search.providers.thinking.openai import THINKING_CONFIGS as _OPENAI
from llm_api_search.providers.thinking.google import THINKING_CONFIGS as _GOOGLE
from llm_api_search.providers.thinking.deepseek import THINKING_CONFIGS as _DEEPSEEK
from llm_api_search.providers.thinking.inception import THINKING_CONFIGS as _INCEPTION

PROVIDER_THINKING_CONFIGS: dict[str, dict[str, ThinkingConfig]] = {
    "anthropic": _ANTHROPIC,
    "google": _GOOGLE,
    "openai": _OPENAI,
    "inception": _INCEPTION,
    "deepseek": _DEEPSEEK,
}
