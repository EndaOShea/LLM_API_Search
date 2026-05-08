"""DeepSeek API provider.

DeepSeek's API is OpenAI-compatible (see https://api-docs.deepseek.com/).
Snippets use the ``openai`` SDK pointed at ``https://api.deepseek.com``,
which is the integration path DeepSeek's own docs recommend.

Pricing notes:
- Input pricing recorded here is the *cache miss* rate. DeepSeek charges
  significantly less on cache hits ($0.0028/Mtok flash, $0.0145/Mtok pro)
  but most callers should plan for the cache-miss case.
- V4-Pro currently runs at a 75% promotional discount through 2026-05-31.
  We record the regular post-promo price ($1.74 in / $3.48 out) so the
  numbers don't silently shift when the promo ends.
- ``deepseek-chat`` and ``deepseek-reasoner`` are scheduled for deprecation
  on 2026-07-24 and are intentionally not listed.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    ModelInfo, TextModelInfo, Provider, ProviderInfo,
)

_STATIC_MODELS = [
    TextModelInfo(
        model_id='deepseek-v4-pro',
        display_name='DeepSeek V4 Pro',
        description=(
            'DeepSeek\'s flagship reasoning model with configurable thinking '
            'mode and reasoning effort. 1M token context, up to 384K output. '
            'Supports tool calls, JSON output, chat prefix completion, and '
            'FIM completion. Currently at a 75% promotional discount through '
            '2026-05-31; pricing here reflects the regular post-promo rate.'
        ),
        context_window=1_000_000,
        max_output_tokens=384_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.74,
        output_cost_per_mtok=3.48,
    ),
    TextModelInfo(
        model_id='deepseek-v4-flash',
        display_name='DeepSeek V4 Flash',
        description=(
            'Fast, low-cost DeepSeek model with configurable thinking mode. '
            '1M token context, up to 384K output. Supports tool calls, JSON '
            'output, chat prefix completion, and FIM completion.'
        ),
        context_window=1_000_000,
        max_output_tokens=384_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.14,
        output_cost_per_mtok=0.28,
    ),
]


class DeepSeekProvider(Provider):
    """DeepSeek provider (OpenAI-compatible API)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="DeepSeek",
            api_base_url="https://api.deepseek.com",
            api_version=None,
            auth_env_var="DEEPSEEK_API_KEY",
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
            documentation_url="https://api-docs.deepseek.com/",
            rate_limits_url="https://api-docs.deepseek.com/quick_start/rate_limit",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Attempt to list models via the DeepSeek API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
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

            live_ids = {m.get("id", "") for m in data.get("data", [])}
            if live_ids:
                # Keep only static models that the live API still serves;
                # this filters out anything DeepSeek has retired without
                # losing curated descriptions/pricing.
                info.models = [m for m in _STATIC_MODELS if m.model_id in live_ids]
                if not info.models:
                    info.models = list(_STATIC_MODELS)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "deepseek-v4-pro"
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["DEEPSEEK_API_KEY"],\n'
                '    base_url="https://api.deepseek.com",\n'
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
                '  apiKey: process.env.DEEPSEEK_API_KEY,\n'
                '  baseURL: "https://api.deepseek.com",\n'
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
                '  apiKey: process.env.DEEPSEEK_API_KEY,\n'
                '  baseURL: "https://api.deepseek.com",\n'
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
                '    .apiKey(System.getenv("DEEPSEEK_API_KEY"))\n'
                '    .baseUrl("https://api.deepseek.com")\n'
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
                '    const char* api_key = std::getenv("DEEPSEEK_API_KEY");\n'
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
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.deepseek.com/chat/completions");\n'
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
