"""Base provider interface and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


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


@dataclass
class ProviderInfo:
    """Discovered information about an LLM provider."""

    name: str
    api_base_url: str
    api_version: str | None
    auth_env_var: str
    auth_header: str
    sdk_package: str
    sdk_install: str
    models: list[ModelInfo] = field(default_factory=list)
    documentation_url: str = ""

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
            f"  SDK Package:     {self.sdk_package}",
            f"  Install:         {self.sdk_install}",
            f"  Docs:            {self.documentation_url}",
            f"  Models ({len(self.models)}):",
        ]
        for m in self.models:
            ctx = f" | ctx: {m.context_window}" if m.context_window else ""
            lines.append(f"    - {m.model_id}{ctx}")
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

    def get_connection_snippet(self, model_id: str | None = None) -> str:
        """Return a Python code snippet showing how to connect."""
        ...
