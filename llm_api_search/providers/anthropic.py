"""Anthropic / Claude API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, TextModelInfo, Provider, ProviderInfo

# Known models — kept as a fallback when the live API is unavailable.
_STATIC_MODELS = [
    TextModelInfo(
        model_id='claude-opus-4-7',
        display_name='Claude Opus 4.7',
        description='',
        context_window=None,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=None,  # TODO: add pricing
        output_cost_per_mtok=None,  # TODO: add pricing
    ),
    TextModelInfo(
        model_id='claude-sonnet-4-6',
        display_name='Claude Sonnet 4.6',
        description='Balanced performance and speed',
        context_window=1_000_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='claude-opus-4-6',
        display_name='Claude Opus 4.6',
        description='Most capable model for complex tasks',
        context_window=1_000_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=25.0,
    ),
    TextModelInfo(
        model_id='claude-opus-4-5-20251101',
        display_name='Claude Opus 4.5',
        description='Previous generation most capable model',
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=25.0,
    ),
    TextModelInfo(
        model_id='claude-haiku-4-5-20251001',
        display_name='Claude Haiku 4.5',
        description='Fastest and most compact model',
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=5.0,
    ),
    TextModelInfo(
        model_id='claude-sonnet-4-5-20250929',
        display_name='Claude Sonnet 4.5',
        description='Previous generation balanced model',
        context_window=1_000_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='claude-opus-4-1-20250805',
        display_name='Claude Opus 4.1',
        description='Previous generation high-capability model',
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
    ),
    TextModelInfo(
        model_id='claude-opus-4-20250514',
        display_name='Claude Opus 4',
        description='First Claude 4 generation flagship model',
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
    ),
    TextModelInfo(
        model_id='claude-sonnet-4-20250514',
        display_name='Claude Sonnet 4',
        description='First Claude 4 generation balanced model',
        context_window=200_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=True,
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='claude-3-haiku-20240307',
        display_name='Claude Haiku 3',
        description='Legacy fast and compact model',
        context_window=200_000,
        max_output_tokens=4_096,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.25,
    ),
    TextModelInfo(
        model_id='claude-3-5-haiku-20241022',
        display_name='Claude Haiku 3.5',
        description='',
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.8,
        output_cost_per_mtok=4.0,
    ),
    TextModelInfo(
        model_id='claude-opus-4-5-20251124',
        display_name='Claude Opus 4.5',
        description='Previous generation most capable model',
        context_window=1_000_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=5.0,
        output_cost_per_mtok=25.0,
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
            sdk_packages={
                "python": "anthropic",
                "typescript": "@anthropic-ai/sdk",
                "javascript": "@anthropic-ai/sdk",
                "java": "com.anthropic:anthropic-java",
            },
            sdk_installs={
                "python": "pip install anthropic",
                "typescript": "npm install @anthropic-ai/sdk",
                "javascript": "npm install @anthropic-ai/sdk",
                "java": "Maven/Gradle: com.anthropic:anthropic-java",
                "cpp": "No official SDK — use REST API via libcurl",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://docs.anthropic.com/en/api",
            rate_limits_url="https://docs.anthropic.com/en/api/rate-limits",
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

            live_models: list[TextModelInfo] = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                live_models.append(
                    TextModelInfo(
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

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "claude-sonnet-4-6"
        snippets = {
            "python": (
                'import anthropic\n\n'
                'client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var\n\n'
                'message = client.messages.create(\n'
                f'    model="{model}",\n'
                '    max_tokens=1024,\n'
                '    messages=[{"role": "user", "content": "Hello!"}],\n'
                ')\n'
                'print(message.content[0].text)\n'
            ),
            "typescript": (
                'import Anthropic from "@anthropic-ai/sdk";\n\n'
                'const client = new Anthropic(); // uses ANTHROPIC_API_KEY env var\n\n'
                'const message = await client.messages.create({\n'
                f'  model: "{model}",\n'
                '  max_tokens: 1024,\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(message.content[0].text);\n'
            ),
            "javascript": (
                'const Anthropic = require("@anthropic-ai/sdk");\n\n'
                'const client = new Anthropic(); // uses ANTHROPIC_API_KEY env var\n\n'
                'const message = await client.messages.create({\n'
                f'  model: "{model}",\n'
                '  max_tokens: 1024,\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(message.content[0].text);\n'
            ),
            "java": (
                'import com.anthropic.client.AnthropicClient;\n'
                'import com.anthropic.client.okhttp.AnthropicOkHttpClient;\n'
                'import com.anthropic.models.messages.*;\n\n'
                '// Uses ANTHROPIC_API_KEY env var\n'
                'AnthropicClient client = AnthropicOkHttpClient.builder().build();\n\n'
                'Message message = client.messages().create(\n'
                '    MessageCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .maxTokens(1024)\n'
                '        .addUserMessage("Hello!")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(message.content().get(0).text());\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <curl/curl.h>\n'
                '#include <nlohmann/json.hpp>\n\n'
                'using json = nlohmann::json;\n\n'
                'static size_t WriteCallback(void* contents, size_t size,\n'
                '                            size_t nmemb, std::string* out) {\n'
                '    out->append((char*)contents, size * nmemb);\n'
                '    return size * nmemb;\n'
                '}\n\n'
                'int main() {\n'
                '    const char* api_key = std::getenv("ANTHROPIC_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"max_tokens", 1024},\n'
                '        {"messages", {{{"role", "user"}, {"content", "Hello!"}}}}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, ("x-api-key: " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");\n'
                '    headers = curl_slist_append(headers, "content-type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["content"][0]["text"].get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
