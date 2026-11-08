"""Kimi (Moonshot AI) API provider.

Kimi's international API is OpenAI-compatible (see
https://platform.kimi.ai/docs/api/overview). Snippets use the ``openai`` SDK
pointed at ``https://api.moonshot.ai/v1``.

Pricing source: https://platform.kimi.ai/docs/pricing/chat (USD, cache-miss
input rate; cache-hit and cache-write rates are noted in each model's
description, not recorded as separate fields). Spec source: the per-model
quickstarts under https://platform.kimi.ai/docs/guide/ and the parameter
reference at https://platform.kimi.ai/docs/api/models-overview.
Verified: 2026-11-08.

Default: Moonshot's quickstart (https://platform.kimi.ai/docs/overview) says
"we recommend starting with Kimi K3.1 Pro", so ``kimi-k3.1-pro`` is
``_STATIC_MODELS[0]``.

The K2 series (kimi-k2.6, kimi-k2.7-code, kimi-k2.7-code-highspeed) is listed
under "Deprecated models" on https://platform.kimi.ai/docs/models: closed to
newly registered users since 2026-10-31, retirement expected late December
2026. They stay curated (existing accounts can still call them) but are kept
out of default listings via ``LEGACY_MODELS`` in ``providers/__init__.py``.
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

_STATIC_MODELS = [
    TextModelInfo(
        model_id='kimi-k3.1-pro',
        display_name='Kimi K3.1 Pro',
        description="Kimi's most capable flagship and the first of the K3.1 series: a post-trained Kimi K3 with native visual understanding (text, image, and video input). 1,048,576 token context; max_completion_tokens defaults to 131,072 and can be set up to 1,048,576. Tool calling (tool_choice auto/none/required). Thinking is always on via reasoning_effort=low|high|max (default max); cannot be disabled. Moonshot's recommended starting model; unlocked after a first top-up (minimum $1). Cache-hit input rate $0.20/Mtok; cache writes $2.00/Mtok (5m TTL) or $4.00/Mtok (1h). Batch inference is billed at 60% of the standard price.",
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=10.0,
    ),
    TextModelInfo(
        model_id='kimi-k3.1-pro-highspeed',
        display_name='Kimi K3.1 Pro HighSpeed',
        description='High-speed variant of Kimi K3.1 Pro: the same model at about 2-3x the output speed and double the base token price. Same limits and parameters as kimi-k3.1-pro (1,048,576 token context; output settable up to the full window; tool calling; always-on reasoning_effort=low|high|max). Cache-hit input rate $0.20/Mtok; cache writes $4.00/Mtok (5m TTL) or $8.00/Mtok (1h). Curated from the pricing page and quickstart; it was not in the live /models listing behind the 2026-11-06 update.',
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=4.0,
        output_cost_per_mtok=20.0,
    ),
    TextModelInfo(
        model_id='kimi-k3',
        display_name='Kimi K3',
        description="Moonshot's previous-generation flagship (superseded by Kimi K3.1 Pro) with native visual understanding. 1,048,576 token context, output settable up to the full context window (default 131,072). Tool calling. Thinking is always on via reasoning_effort=low|high|max (default max); cannot be disabled. Cache-hit input rate $0.30/Mtok; cache writes $3.00/Mtok (5m TTL) or $6.00/Mtok (1h). Batch inference is billed at 60% of the standard price.",
        context_window=1_048_576,
        max_output_tokens=1_048_576,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    TextModelInfo(
        model_id='kimi-k2.6',
        display_name='Kimi K2.6',
        description="General-purpose Kimi model with text, image, and video input. 262,144 token context; output is bounded by the context window (Moonshot doesn't publish a separate ceiling for this model). Tool calling. Thinking togglable via thinking={type:'enabled'|'disabled'}, enabled by default. Cache-hit input rate $0.16/Mtok. K2 series: closed to newly registered users since 2026-10-31 and scheduled for retirement by the end of December 2026.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.95,
        output_cost_per_mtok=4.0,
    ),
    TextModelInfo(
        model_id='kimi-k2.7-code',
        display_name='Kimi K2.7 Code',
        description="Kimi's dedicated coding model, with text, image, and video input. 262,144 token context; output is bounded by the context window (Moonshot doesn't publish a separate ceiling for this model). Tool calling. Cache-hit input rate $0.19/Mtok. K2 series: closed to newly registered users since 2026-10-31 and scheduled for retirement by the end of December 2026.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.95,
        output_cost_per_mtok=4.0,
    ),
    TextModelInfo(
        model_id='kimi-k2.7-code-highspeed',
        display_name='Kimi K2.7 Code HighSpeed',
        description="High-speed variant of Kimi K2.7 Code: the same dedicated coding model with faster output (about 180 tokens/s, up to 260 tokens/s in short-context scenarios) at double the base token price. Text, image, and video input. 262,144 token context; output is bounded by the context window (max_tokens defaults to 32,768; Moonshot doesn't publish a separate ceiling). Tool calling (tool_choice auto/none only). Thinking is always on: thinking may be omitted or set to {type:'enabled', keep:'all'}; 'disabled' errors. Cache-hit input rate $0.38/Mtok. K2 series: closed to newly registered users since 2026-10-31 and scheduled for retirement by the end of December 2026.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.9,
        output_cost_per_mtok=8.0,
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
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as exc:
            self._note_fetch_failure(exc)  # report-only; fall back to static

        return info

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or _STATIC_MODELS[0].model_id
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
