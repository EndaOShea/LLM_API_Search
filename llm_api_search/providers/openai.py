"""OpenAI API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo

_STATIC_MODELS = [
    ModelInfo(
        model_id="gpt-5.4",
        display_name="GPT-5.4",
        description="Current top recommendation for reasoning and coding",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        description="More compute for harder problems, supports reasoning effort levels",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        description="Lower latency and cost GPT-5 variant",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="o3",
        display_name="o3",
        description="Reasoning model",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="o4-mini",
        display_name="o4-mini",
        description="Fast, affordable reasoning model",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        description="Previous generation GPT model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        description="Previous generation balanced model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
    ModelInfo(
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        description="Previous generation fastest model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
    ),
]


class OpenAIProvider(Provider):
    """OpenAI provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="OpenAI",
            api_base_url="https://api.openai.com",
            api_version=None,
            auth_env_var="OPENAI_API_KEY",
            auth_header="Authorization",
            sdk_package="openai",
            sdk_install="pip install openai",
            models=list(_STATIC_MODELS),
            documentation_url="https://platform.openai.com/docs/api-reference",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the OpenAI API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return info

        # Well-known model prefixes we care about (skip fine-tunes, embeddings, etc.)
        _PREFIXES = ("gpt-5", "gpt-4", "o3", "o4")

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if any(mid.startswith(p) for p in _PREFIXES):
                    live_models.append(
                        ModelInfo(model_id=mid, display_name=mid)
                    )
            if live_models:
                live_models.sort(key=lambda x: x.model_id)
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def get_connection_snippet(self, model_id: str | None = None) -> str:
        model = model_id or "gpt-5.4"
        return (
            'from openai import OpenAI\n\n'
            'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
            'response = client.chat.completions.create(\n'
            f'    model="{model}",\n'
            '    messages=[{{"role": "user", "content": "Hello!"}}],\n'
            ')\n'
            'print(response.choices[0].message.content)\n'
        )
