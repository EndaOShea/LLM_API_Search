"""Base provider interface and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

SUPPORTED_LANGUAGES = ("python", "typescript", "javascript", "java", "cpp")


class ModelType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO_TTS = "audio_tts"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    EMBEDDING = "embedding"
    VIDEO = "video"
    MUSIC = "music"


@dataclass
class ModelInfo:
    """Base class for all model types."""

    model_id: str
    display_name: str
    model_type: ModelType = ModelType.TEXT
    description: str = ""


@dataclass
class TextModelInfo(ModelInfo):
    """Text/chat LLM model."""
    model_type: ModelType = field(default=ModelType.TEXT, init=False)
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_tool_use: bool = False
    supports_image_generation: bool = False
    supports_computer_use: bool = False
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


@dataclass
class ImageModelInfo(ModelInfo):
    """Image generation model."""
    model_type: ModelType = field(default=ModelType.IMAGE, init=False)
    supported_sizes: list[str] = field(default_factory=list)
    supported_qualities: list[str] = field(default_factory=list)
    max_images_per_request: int | None = None
    cost_per_image: float | None = None


@dataclass
class AudioTTSModelInfo(ModelInfo):
    """Text-to-speech model."""
    model_type: ModelType = field(default=ModelType.AUDIO_TTS, init=False)
    supported_voices: list[str] = field(default_factory=list)
    supported_output_formats: list[str] = field(default_factory=list)
    cost_per_mchars: float | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


@dataclass
class AudioTranscriptionModelInfo(ModelInfo):
    """Speech-to-text model."""
    model_type: ModelType = field(default=ModelType.AUDIO_TRANSCRIPTION, init=False)
    supported_input_formats: list[str] = field(default_factory=list)
    max_file_size_mb: int | None = None
    cost_per_minute: float | None = None


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Embedding/vector model."""
    model_type: ModelType = field(default=ModelType.EMBEDDING, init=False)
    dimensions: int | None = None
    max_input_tokens: int | None = None
    supports_multimodal: bool = False
    input_cost_per_mtok: float | None = None


@dataclass
class MusicModelInfo(ModelInfo):
    """Music generation model."""
    model_type: ModelType = field(default=ModelType.MUSIC, init=False)
    cost_per_second: float | None = None


@dataclass
class VideoModelInfo(ModelInfo):
    """Video generation model."""
    model_type: ModelType = field(default=ModelType.VIDEO, init=False)
    supported_resolutions: list[str] = field(default_factory=list)
    supports_audio: bool = False
    cost_per_second: float | None = None


@dataclass
class RateLimit:
    """Rate limits for a model."""

    # Core (universal)
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None
    # Optional
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    images_per_minute: int | None = None
    batch_queue_limit: int | None = None


@dataclass
class ProviderInfo:
    """Discovered information about an LLM provider."""

    name: str
    api_base_url: str
    api_version: str | None
    auth_env_var: str
    auth_header: str
    sdk_packages: dict[str, str] = field(default_factory=dict)
    sdk_installs: dict[str, str] = field(default_factory=dict)
    models: list[ModelInfo] = field(default_factory=list)
    documentation_url: str = ""
    rate_limits_url: str = ""

    @property
    def sdk_package(self) -> str:
        """Return the Python SDK package name (backward compat)."""
        return self.sdk_packages.get("python", "")

    @property
    def sdk_install(self) -> str:
        """Return the Python SDK install command (backward compat)."""
        return self.sdk_installs.get("python", "")

    def sdk_install_for(self, language: str) -> str:
        """Return the SDK install command for a given language."""
        return self.sdk_installs.get(language, self.sdk_installs.get("python", ""))

    def summary(self) -> str:
        """Return a human-readable summary of this provider."""
        lines = [
            f"Provider: {self.name}",
            f"  API Base URL:    {self.api_base_url}",
        ]
        if self.api_version:
            lines.append(f"  API Version:     {self.api_version}")
        lines += [
            f"  Auth Env Var:    {self.auth_env_var}",
            f"  Auth Header:     {self.auth_header}",
        ]
        for lang, install in self.sdk_installs.items():
            lines.append(f"  SDK ({lang:>10}): {install}")
        lines.append(f"  Docs:            {self.documentation_url}")
        if self.rate_limits_url:
            lines.append(f"  Rate Limits:     {self.rate_limits_url}")

        # Group models by type.
        from collections import defaultdict
        by_type: dict[ModelType, list[ModelInfo]] = defaultdict(list)
        for m in self.models:
            by_type[m.model_type].append(m)

        for mtype in ModelType:
            models = by_type.get(mtype, [])
            if not models:
                continue
            lines.append(f"  Models ({mtype.value}): {len(models)}")
            for m in models:
                lines.append(f"    - {m.model_id}{_format_model_cost(m)}")

        return "\n".join(lines)


def _format_model_cost(m: ModelInfo) -> str:
    """Format the cost string for a model based on its type."""
    if isinstance(m, TextModelInfo):
        parts = []
        if m.context_window:
            parts.append(f"ctx: {m.context_window:,}")
        if m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None:
            parts.append(f"${m.input_cost_per_mtok:.2f}/${m.output_cost_per_mtok:.2f} per 1M tok")
        return f" | {' | '.join(parts)}" if parts else ""
    elif isinstance(m, ImageModelInfo):
        if m.cost_per_image is not None:
            return f" | ${m.cost_per_image:.3f}/image"
        return ""
    elif isinstance(m, AudioTTSModelInfo):
        if m.cost_per_mchars is not None:
            return f" | ${m.cost_per_mchars:.2f}/1M chars"
        if m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None:
            return f" | ${m.input_cost_per_mtok:.2f}/${m.output_cost_per_mtok:.2f} per 1M tok"
        return ""
    elif isinstance(m, AudioTranscriptionModelInfo):
        if m.cost_per_minute is not None:
            return f" | ${m.cost_per_minute:.3f}/min"
        return ""
    elif isinstance(m, EmbeddingModelInfo):
        parts = []
        if m.dimensions:
            parts.append(f"{m.dimensions}d")
        if m.input_cost_per_mtok is not None:
            parts.append(f"${m.input_cost_per_mtok:.2f}/1M tok")
        return f" | {' | '.join(parts)}" if parts else ""
    elif isinstance(m, MusicModelInfo):
        if m.cost_per_second is not None:
            return f" | ${m.cost_per_second:.3f}/sec"
        return ""
    elif isinstance(m, VideoModelInfo):
        if m.cost_per_second is not None:
            return f" | ${m.cost_per_second:.2f}/sec"
        return ""
    return ""


class Provider(ABC):
    """Abstract base for an LLM API provider."""

    @abstractmethod
    def get_static_info(self) -> ProviderInfo:
        """Return statically-known provider metadata and models."""
        ...

    @abstractmethod
    def fetch_live_models(self) -> ProviderInfo:
        """Fetch live model information from the provider's API.

        Falls back to static info if the API call fails or no API key is set.
        """
        ...

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        """Return a code snippet showing how to connect."""
        ...
