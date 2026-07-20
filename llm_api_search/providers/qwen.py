"""Qwen (Alibaba Cloud Model Studio) API provider.

Qwen's international OpenAI-compatible endpoint is DashScope's
"compatible-mode" API (see
https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope).
Snippets use the ``openai`` SDK pointed at
``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``.

DashScope's live /models catalog spans hundreds of Qwen variants (sizes, VL,
audio, coder, older generations), so ``fetch_live_models()`` stays
curated-only (like deepseek.py) rather than surfacing every live ID.
``unrecognized_live_model_ids()`` flags a genuinely-new frontier generation
(``qwen{N}.{M}-max``/``-plus`` not yet curated) without also flagging the
rest of DashScope's catalog every week.

Pricing source: https://www.alibabacloud.com/help/en/model-studio/model-pricing
(list price; time-limited promos noted per-model in the description, not
recorded as the static number since neither promo has a stated expiry).
Verified: 2026-07-20.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    TextModelInfo, Provider, ProviderInfo,
)

_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3.7-max"

# Matches a frontier-generation chat model ID, e.g. "qwen3.7-max",
# "qwen3.8-plus". Used to pick genuinely-new frontier releases out of
# DashScope's much larger catalog (qwen-vl-*, qwen-audio-*, qwen-coder-*,
# fixed-size open-weight variants, ...) without flagging all of it.
_FRONTIER_ID_RE = re.compile(r"^qwen\d+\.\d+-(?:max|plus)$")


def _classify_unrecognized(live_ids: set[str], static_ids: set[str]) -> set[str]:
    """Return live frontier-shaped IDs that aren't already curated.

    Pure (no network) so the classification can be unit-tested directly.
    """
    return {
        mid for mid in live_ids
        if mid not in static_ids and _FRONTIER_ID_RE.match(mid)
    }


_STATIC_MODELS = [
    TextModelInfo(
        model_id='qwen3.7-max',
        display_name='Qwen3.7 Max',
        description="Alibaba's text-only flagship. 1,000,000 token context, 65,536 max output. Tool calling. Thinking via enable_thinking (bool) + thinking_budget (token cap) + preserve_thinking, enabled by default. List price shown; a 50%-off promo with no stated expiry currently brings this to $1.25/$3.75 per Mtok.",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=7.50,
    ),
    TextModelInfo(
        model_id='qwen3.7-plus',
        display_name='Qwen3.7 Plus',
        description="Multimodal agent model (text/image/video input up to 16MP/image). 1,000,000 token context, 65,536 max output. Tool calling plus built-in web search and code execution. Pricing shown is the <=256K-context tier; above 256K it's $1.20/$4.80 per Mtok. Same thinking mechanism as qwen3.7-max. A 20%-off promo with no stated expiry is currently active on top of the list price shown.",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.40,
        output_cost_per_mtok=1.60,
    ),
]


class QwenProvider(Provider):
    """Qwen (Alibaba Cloud Model Studio) provider (OpenAI-compatible API)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Qwen (Alibaba Model Studio)",
            api_base_url=_BASE_URL,
            api_version=None,
            auth_env_var="DASHSCOPE_API_KEY",
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
            documentation_url="https://www.alibabacloud.com/help/en/model-studio/models",
            rate_limits_url="https://www.alibabacloud.com/help/en/model-studio/rate-limit",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Attempt to list models via the DashScope API, fall back to static.

        Kept curated-only (like DeepSeek): DashScope's catalog spans hundreds
        of Qwen variants, so only live IDs already in _STATIC_MODELS are kept.
        """
        info = self.get_static_info()
        api_key = os.environ.get("DASHSCOPE_API_KEY")
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
                info.models = [m for m in _STATIC_MODELS if m.model_id in live_ids]
                if not info.models:
                    info.models = list(_STATIC_MODELS)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def unrecognized_live_model_ids(self) -> set[str]:
        """Live frontier-shaped IDs (qwen{N}.{M}-max/-plus) not yet curated.

        Scoped to the frontier-generation ID shape so this doesn't flood the
        weekly update with DashScope's hundreds of non-frontier model IDs.
        Returns an empty set if no API key is set or any API call fails.
        """
        info = self.get_static_info()
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return set()

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
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            return set()

        live_ids = {m.get("id", "") for m in data.get("data", [])} - {""}
        static_ids = {m.model_id for m in _STATIC_MODELS}
        return _classify_unrecognized(live_ids, static_ids)

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or _DEFAULT_MODEL
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["DASHSCOPE_API_KEY"],\n'
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
                '  apiKey: process.env.DASHSCOPE_API_KEY,\n'
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
                '  apiKey: process.env.DASHSCOPE_API_KEY,\n'
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
                '    .apiKey(System.getenv("DASHSCOPE_API_KEY"))\n'
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
                '    const char* api_key = std::getenv("DASHSCOPE_API_KEY");\n'
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
