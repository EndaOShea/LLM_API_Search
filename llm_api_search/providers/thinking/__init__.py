"""Thinking-config registry — maps provider keys to their thinking config dicts."""

from llm_api_search.providers.base import ThinkingConfig
from llm_api_search.providers.thinking.anthropic import THINKING_CONFIGS as _ANTHROPIC

# Modules added in later tasks; import lazily as they are created.
try:
    from llm_api_search.providers.thinking.openai import THINKING_CONFIGS as _OPENAI
except ImportError:  # pragma: no cover
    _OPENAI = {}
try:
    from llm_api_search.providers.thinking.google import THINKING_CONFIGS as _GOOGLE
except ImportError:  # pragma: no cover
    _GOOGLE = {}
try:
    from llm_api_search.providers.thinking.deepseek import THINKING_CONFIGS as _DEEPSEEK
except ImportError:  # pragma: no cover
    _DEEPSEEK = {}
try:
    from llm_api_search.providers.thinking.inception import THINKING_CONFIGS as _INCEPTION
except ImportError:  # pragma: no cover
    _INCEPTION = {}

PROVIDER_THINKING_CONFIGS: dict[str, dict[str, ThinkingConfig]] = {
    "anthropic": _ANTHROPIC,
    "google": _GOOGLE,
    "openai": _OPENAI,
    "inception": _INCEPTION,
    "deepseek": _DEEPSEEK,
}
