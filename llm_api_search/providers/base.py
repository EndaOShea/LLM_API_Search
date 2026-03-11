"""Base provider interface and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

SUPPORTED_LANGUAGES = ("python", "typescript", "javascript", "java", "cpp")


@dataclass
class ModelInfo:
    """Information about a single model available from a provider."""

    model_id: str
    display_name: str
    description: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_tool_use: bool = False
    input_cost_per_mtok: float | None = None   # USD per 1M input tokens
    output_cost_per_mtok: float | None = None  # USD per 1M output tokens


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
        lines += [
            f"  Docs:            {self.documentation_url}",
            f"  Models ({len(self.models)}):",
        ]
        for m in self.models:
            ctx = f" | ctx: {m.context_window}" if m.context_window else ""
            cost = ""
            if m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None:
                cost = f" | ${m.input_cost_per_mtok:.2f}/${m.output_cost_per_mtok:.2f} per 1M tok"
            lines.append(f"    - {m.model_id}{ctx}{cost}")
        return "\n".join(lines)


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
