"""Google Gemini API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo

_STATIC_MODELS = [
    ModelInfo(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        description="Most capable Gemini model with thinking",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="Fast model with thinking",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        description="Next generation fast model",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
    ),
]


class GeminiProvider(Provider):
    """Google Gemini provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Google (Gemini)",
            api_base_url="https://generativelanguage.googleapis.com",
            api_version="v1beta",
            auth_env_var="GOOGLE_API_KEY",
            auth_header="x-goog-api-key",
            sdk_package="google-genai",
            sdk_install="pip install google-genai",
            models=list(_STATIC_MODELS),
            documentation_url="https://ai.google.dev/gemini-api/docs",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the Gemini API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return info

        try:
            url = (
                f"{info.api_base_url}/{info.api_version}/models"
                f"?key={api_key}"
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("models", []):
                model_id = m.get("name", "").removeprefix("models/")
                if not model_id or model_id.startswith("embedding"):
                    continue
                live_models.append(
                    ModelInfo(
                        model_id=model_id,
                        display_name=m.get("displayName", model_id),
                        description=m.get("description", ""),
                        context_window=m.get("inputTokenLimit"),
                        max_output_tokens=m.get("outputTokenLimit"),
                    )
                )
            if live_models:
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def get_connection_snippet(self, model_id: str | None = None) -> str:
        model = model_id or "gemini-2.5-flash"
        return (
            'from google import genai\n\n'
            'client = genai.Client()  # uses GOOGLE_API_KEY env var\n\n'
            'response = client.models.generate_content(\n'
            f'    model="{model}",\n'
            '    contents="Hello!",\n'
            ')\n'
            'print(response.text)\n'
        )
