"""Anthropic / Claude API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo

# Known models — kept as a fallback when the live API is unavailable.
_STATIC_MODELS = [
    ModelInfo(
        model_id="claude-opus-4-6",
        display_name="Claude Opus 4.6",
        description="Most capable model for complex tasks",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        description="Balanced performance and speed",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        description="Fastest and most compact model",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
    ),
]

_API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    """Anthropic (Claude) provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Anthropic (Claude)",
            api_base_url="https://api.anthropic.com",
            api_version=_API_VERSION,
            auth_env_var="ANTHROPIC_API_KEY",
            auth_header="x-api-key",
            sdk_package="anthropic",
            sdk_install="pip install anthropic",
            models=list(_STATIC_MODELS),
            documentation_url="https://docs.anthropic.com/en/api",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Attempt to list models via the Anthropic API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return info

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": _API_VERSION,
                    "content-type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                live_models.append(
                    ModelInfo(
                        model_id=model_id,
                        display_name=m.get("display_name", model_id),
                        description=m.get("description", ""),
                    )
                )
            if live_models:
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass  # fall back to static

        return info

    def get_connection_snippet(self, model_id: str | None = None) -> str:
        model = model_id or "claude-sonnet-4-6"
        return (
            'import anthropic\n\n'
            'client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var\n\n'
            'message = client.messages.create(\n'
            f'    model="{model}",\n'
            '    max_tokens=1024,\n'
            '    messages=[{"role": "user", "content": "Hello!"}],\n'
            ')\n'
            'print(message.content[0].text)\n'
        )
