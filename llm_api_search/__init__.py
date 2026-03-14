"""
LLM API Search - Discover and connect to the latest LLM API versions.

A Python library that discovers the latest API versions for Claude, Gemini,
and OpenAI, and provides connection helpers for each.
"""

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.selector import select_provider
from llm_api_search.providers.base import (
    ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, VideoModelInfo, ModelType,
)

__version__ = "0.1.0"
__all__ = [
    "discover", "discover_provider", "list_providers", "select_provider",
    "ModelInfo", "TextModelInfo", "ImageModelInfo", "AudioTTSModelInfo",
    "AudioTranscriptionModelInfo", "EmbeddingModelInfo", "VideoModelInfo", "ModelType",
]
