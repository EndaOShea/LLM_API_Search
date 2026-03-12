from llm_api_search.providers.base import Provider, ModelInfo, SUPPORTED_LANGUAGES
from llm_api_search.providers.anthropic import AnthropicProvider
from llm_api_search.providers.google import GeminiProvider
from llm_api_search.providers.openai import OpenAIProvider

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "google": GeminiProvider,
    "openai": OpenAIProvider,
}

__all__ = [
    "Provider",
    "ModelInfo",
    "SUPPORTED_LANGUAGES",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "PROVIDERS",
]
