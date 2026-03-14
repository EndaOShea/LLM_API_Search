"""OpenAI API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, TextModelInfo, Provider, ProviderInfo

_STATIC_MODELS = [
    # --- GPT-5 family (current) ---
    TextModelInfo(
        model_id="gpt-5.4",
        display_name="GPT-5.4",
        description="Best intelligence at scale for agentic, coding, and professional workflows",
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=15.00,
    ),
    TextModelInfo(
        model_id="gpt-5.4-pro",
        display_name="GPT-5.4 Pro",
        description="More compute for harder problems, supports reasoning effort levels",
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=30.00,
        output_cost_per_mtok=180.00,
    ),
    TextModelInfo(
        model_id="gpt-5.3-codex",
        display_name="GPT-5.3 Codex",
        description="Most capable agentic coding model, optimized for Codex environments",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.00,
    ),
    TextModelInfo(
        model_id="gpt-5.2",
        display_name="GPT-5.2",
        description="Previous frontier model for complex professional work",
        context_window=1_050_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.75,
        output_cost_per_mtok=14.00,
    ),
    TextModelInfo(
        model_id="gpt-5.1",
        display_name="GPT-5.1",
        description="Earlier frontier iteration",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.00,
    ),
    TextModelInfo(
        model_id="gpt-5",
        display_name="GPT-5",
        description="Reasoning model with configurable reasoning effort",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.00,
    ),
    TextModelInfo(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        description="Near-frontier intelligence optimized for cost-sensitive workloads",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=2.00,
    ),
    TextModelInfo(
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        description="Fastest, most cost-efficient GPT-5 variant",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.05,
        output_cost_per_mtok=0.40,
    ),
    # --- Reasoning models ---
    TextModelInfo(
        model_id="o3",
        display_name="o3",
        description="Reasoning model for complex tasks",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=8.00,
    ),
    TextModelInfo(
        model_id="o3-pro",
        display_name="o3 Pro",
        description="o3 with increased compute for harder reasoning problems",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=20.00,
        output_cost_per_mtok=80.00,
    ),
    TextModelInfo(
        model_id="o4-mini",
        display_name="o4-mini",
        description="Fast, affordable reasoning model",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.10,
        output_cost_per_mtok=4.40,
    ),
    # --- Previous generation (GPT-4.1) ---
    TextModelInfo(
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        description="Previous generation GPT model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=8.00,
    ),
    TextModelInfo(
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        description="Previous generation balanced model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.40,
        output_cost_per_mtok=1.60,
    ),
    TextModelInfo(
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        description="Previous generation fastest model (retired from ChatGPT, still in API)",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.10,
        output_cost_per_mtok=0.40,
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
            sdk_packages={
                "python": "openai",
                "typescript": "openai",
                "javascript": "openai",
                "java": "com.openai:openai-java",
            },
            sdk_installs={
                "python": "pip install openai",
                "typescript": "npm install openai",
                "javascript": "npm install openai",
                "java": "Maven/Gradle: com.openai:openai-java",
                "cpp": "No official SDK — use REST API via libcurl",
            },
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

            live_models: list[TextModelInfo] = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if any(mid.startswith(p) for p in _PREFIXES):
                    live_models.append(
                        TextModelInfo(model_id=mid, display_name=mid)
                    )
            if live_models:
                live_models.sort(key=lambda x: x.model_id)
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gpt-5.4"
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.chat.completions.create(\n'
                f'    model="{model}",\n'
                '    messages=[{"role": "user", "content": "Hello!"}],\n'
                ')\n'
                'print(response.choices[0].message.content)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.chat.completions.create({\n'
                f'  model: "{model}",\n'
                '  messages: [{ role: "user", content: "Hello!" }],\n'
                '});\n'
                'console.log(response.choices[0].message.content);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
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
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
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
                '    const char* api_key = std::getenv("OPENAI_API_KEY");\n'
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
                '        "https://api.openai.com/v1/chat/completions");\n'
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
