"""Kimi (Moonshot AI) API provider.

Kimi's international API is OpenAI-compatible (see
https://platform.kimi.ai/docs/api/overview). Snippets use the ``openai`` SDK
pointed at ``https://api.moonshot.ai/v1``.

Pricing source: https://platform.kimi.ai/docs/pricing/chat-k3 and
.../chat-k26 (USD, cache-miss input rate; cache-hit rates noted in each
model's description, not recorded as separate fields).
Verified: 2026-07-20.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    TextModelInfo, Provider, ProviderInfo,
)

_BASE_URL = "https://api.moonshot.ai/v1"
_DEFAULT_MODEL = "kimi-k3"

_STATIC_MODELS = [
    TextModelInfo(
        model_id='kimi-k3',
        display_name='Kimi K3',
        description="Moonshot's flagship model with native visual understanding. 1,048,576 token context, output settable up to the full context window (default 131,072). Tool calling. Thinking is always on via reasoning_effort=low|high|max (default max); cannot be disabled. Cache-hit input rate $0.30/Mtok.",
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=3.00,
        output_cost_per_mtok=15.00,
    ),
    TextModelInfo(
        model_id='kimi-k2.6',
        display_name='Kimi K2.6',
        description="General-purpose Kimi model with text, image, and video input. 262,144 token context; output is bounded by the context window (Moonshot doesn't publish a separate ceiling for this model). Tool calling. Thinking togglable via thinking={type:'enabled'|'disabled'}, enabled by default. Cache-hit input rate $0.16/Mtok.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.95,
        output_cost_per_mtok=4.00,
    ),
]


class KimiProvider(Provider):
    """Kimi (Moonshot AI) provider (OpenAI-compatible API)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Kimi (Moonshot AI)",
            api_base_url=_BASE_URL,
            api_version=None,
            auth_env_var="MOONSHOT_API_KEY",
            auth_header="Authorization: Bearer",
            sdk_packages={
                "python": "openai",
                "typescript": "openai",
                "javascript": "openai",
                "java": "com.openai:openai-java",
                "cpp": "libcurl + nlohmann/json",
            },
            sdk_installs={
                "python": "pip install openai",
                "typescript": "npm install openai",
                "javascript": "npm install openai",
                "java": "Maven/Gradle: com.openai:openai-java",
                "cpp": "Install via vcpkg/Conan: libcurl, nlohmann-json",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://platform.kimi.ai/docs/api/overview",
            rate_limits_url="https://platform.kimi.ai/docs/pricing/limits",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """List models via the OpenAI-compatible /models endpoint; fall back to static.

        Returns every live ID (as bare TextModelInfo) so the weekly update can
        auto-add genuinely-new Kimi models. Kimi IDs are versioned (not
        generic aliases), so no unrecognized_live_model_ids() override is
        needed.
        """
        info = self.get_static_info()
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            return info

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[TextModelInfo] = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                if not model_id:
                    continue
                live_models.append(
                    TextModelInfo(
                        model_id=model_id,
                        display_name=model_id,
                        description="",
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
        model = model_id or _DEFAULT_MODEL
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["MOONSHOT_API_KEY"],\n'
                f'    base_url="{_BASE_URL}",\n'
                ')\n\n'
                'response = client.chat.completions.create(\n'
                f'    model="{model}",\n'
                '    messages=[{"role": "user", "content": "Hello!"}],\n'
                ')\n'
                'print(response.choices[0].message.content)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MOONSHOT_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const response = await client.chat.completions.create({\n'
                f'  model: "{model}",\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(response.choices[0].message.content);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MOONSHOT_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const response = await client.chat.completions.create({\n'
                f'  model: "{model}",\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(response.choices[0].message.content);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.chat.*;\n\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder()\n'
                '    .apiKey(System.getenv("MOONSHOT_API_KEY"))\n'
                f'    .baseUrl("{_BASE_URL}")\n'
                '    .build();\n\n'
                'ChatCompletion response = client.chat().completions().create(\n'
                '    ChatCompletionCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .addUserMessage("Hello!")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.choices().get(0).message().content());\n'
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
                '    const char* api_key = std::getenv("MOONSHOT_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"messages", {{{"role", "user"}, {"content", "Hello!"}}}}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                f'    curl_easy_setopt(curl, CURLOPT_URL,\n'
                f'        "{_BASE_URL}/chat/completions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["choices"][0]["message"]["content"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
