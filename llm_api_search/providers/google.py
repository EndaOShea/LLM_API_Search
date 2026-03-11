"""Google Gemini API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo

_STATIC_MODELS = [
    ModelInfo(
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro Preview",
        description="Most advanced model; reasoning-first for agentic workflows and coding",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=12.00,
    ),
    ModelInfo(
        model_id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash Preview",
        description="Fast frontier-class; strong visual/spatial reasoning and agentic coding",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.50,
        output_cost_per_mtok=3.00,
    ),
    ModelInfo(
        model_id="gemini-3.1-flash-lite-preview",
        display_name="Gemini 3.1 Flash Lite Preview",
        description="Most cost-efficient; low latency, high volume",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.50,
    ),
    ModelInfo(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        description="Complex reasoning with adaptive thinking",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=1.25,
        output_cost_per_mtok=10.00,
    ),
    ModelInfo(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        description="Balance of speed and intelligence with controllable thinking budgets",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.30,
        output_cost_per_mtok=2.50,
    ),
    ModelInfo(
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
        description="Massive scale, cost-optimized",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_tool_use=True,
        input_cost_per_mtok=0.10,
        output_cost_per_mtok=0.40,
    ),
]


class GeminiProvider(Provider):
    """Google Gemini provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Google (Gemini)",
            api_base_url="https://generativelanguage.googleapis.com",
            api_version="v1beta",
            auth_env_var="GEMINI_API_KEY",
            auth_header="x-goog-api-key",
            sdk_packages={
                "python": "google-genai",
                "typescript": "@google/genai",
                "javascript": "@google/genai",
                "java": "com.google.genai:google-genai",
            },
            sdk_installs={
                "python": "pip install google-genai",
                "typescript": "npm install @google/genai",
                "javascript": "npm install @google/genai",
                "java": "Maven/Gradle: com.google.genai:google-genai",
                "cpp": "No official SDK — use REST API via libcurl",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://ai.google.dev/gemini-api/docs",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the Gemini API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("GEMINI_API_KEY")
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

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gemini-2.5-flash"
        snippets = {
            "python": (
                'from google import genai\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello!",\n'
                ')\n'
                'print(response.text)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello!",\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello!",\n'
                '});\n'
                'console.log(response.text);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "Hello!"\n'
                ');\n'
                'System.out.println(response.text());\n'
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
                '    const char* api_key = std::getenv("GEMINI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                '        {"contents", {{{"parts", {{{"text", "Hello!"}}}}}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:generateContent?key=" + std::string(api_key);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["candidates"][0]["content"]["parts"][0]["text"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
