"""Inception Labs / Mercury API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo

# Known models — kept as a fallback when the live API is unavailable.
# NOTE: Pricing is in EUR, not USD.
_STATIC_MODELS = [
    ModelInfo(
        model_id="mercury-2",
        display_name="Mercury 2",
        description="Fastest reasoning LLM — diffusion-based with adjustable reasoning effort",
        context_window=128_000,
        max_output_tokens=50_000,
        supports_vision=False,
        supports_tool_use=True,
        input_cost_per_mtok=0.25,   # EUR
        output_cost_per_mtok=0.75,  # EUR
    ),
    ModelInfo(
        model_id="mercury-edit",
        display_name="Mercury Edit",
        description="Code editing LLM for autocomplete (FIM), apply-edit, and next-edit",
        context_window=32_000,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_tool_use=False,
        input_cost_per_mtok=0.25,   # EUR
        output_cost_per_mtok=0.75,  # EUR
    ),
]


class InceptionProvider(Provider):
    """Inception Labs (Mercury) provider."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Inception Labs (Mercury)",
            api_base_url="https://api.inceptionlabs.ai",
            api_version=None,
            auth_env_var="INCEPTION_API_KEY",
            auth_header="Authorization: Bearer",
            sdk_packages={
                "python": "requests",
                "javascript": "fetch (built-in)",
            },
            sdk_installs={
                "python": "pip install requests",
                "typescript": "No official SDK — use fetch API (Node.js 18+)",
                "javascript": "No install needed — uses native fetch API",
                "java": "No official SDK — use java.net.http.HttpClient",
                "cpp": "No official SDK — use REST API via libcurl",
            },
            models=list(_STATIC_MODELS),
            documentation_url="https://docs.inceptionlabs.ai",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Attempt to list models via the Inception API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("INCEPTION_API_KEY")
        if not api_key:
            return info

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
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

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "mercury-2"

        if model == "mercury-edit":
            return self._fim_snippet(model, language)
        return self._chat_snippet(model, language)

    def _chat_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'import requests\n\n'
                'response = requests.post(\n'
                '    "https://api.inceptionlabs.ai/v1/chat/completions",\n'
                '    headers={\n'
                '        "Content-Type": "application/json",\n'
                '        "Authorization": f"Bearer {os.environ[\'INCEPTION_API_KEY\']}",\n'
                '    },\n'
                '    json={\n'
                f'        "model": "{model}",\n'
                '        "messages": [{"role": "user", "content": "Hello!"}],\n'
                '        "max_tokens": 10000,\n'
                '    },\n'
                ')\n\n'
                'print(response.json()["choices"][0]["message"]["content"])\n'
            ),
            "typescript": (
                '// No official SDK — using native fetch API\n\n'
                'const response = await fetch(\n'
                '  "https://api.inceptionlabs.ai/v1/chat/completions",\n'
                '  {\n'
                '    method: "POST",\n'
                '    headers: {\n'
                '      "Content-Type": "application/json",\n'
                '      Authorization: `Bearer ${process.env.INCEPTION_API_KEY}`,\n'
                '    },\n'
                '    body: JSON.stringify({\n'
                f'      model: "{model}",\n'
                '      messages: [{{ role: "user", content: "Hello!" }}],\n'
                '      max_tokens: 10000,\n'
                '    }),\n'
                '  }\n'
                ');\n\n'
                'const data = await response.json();\n'
                'console.log(data.choices[0].message.content);\n'
            ),
            "javascript": (
                '// No official SDK — using native fetch API\n\n'
                'const response = await fetch(\n'
                '  "https://api.inceptionlabs.ai/v1/chat/completions",\n'
                '  {\n'
                '    method: "POST",\n'
                '    headers: {\n'
                '      "Content-Type": "application/json",\n'
                '      Authorization: `Bearer ${process.env.INCEPTION_API_KEY}`,\n'
                '    },\n'
                '    body: JSON.stringify({\n'
                f'      model: "{model}",\n'
                '      messages: [{{ role: "user", content: "Hello!" }}],\n'
                '      max_tokens: 10000,\n'
                '    }),\n'
                '  }\n'
                ');\n\n'
                'const data = await response.json();\n'
                'console.log(data.choices[0].message.content);\n'
            ),
            "java": (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n\n'
                '// No official SDK — using java.net.http.HttpClient\n'
                'HttpClient client = HttpClient.newHttpClient();\n\n'
                'String json = """\n'
                '    {\n'
                f'        "model": "{model}",\n'
                '        "messages": [{{"role": "user", "content": "Hello!"}}],\n'
                '        "max_tokens": 10000\n'
                '    }""";\n\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                '    .uri(URI.create("https://api.inceptionlabs.ai/v1/chat/completions"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .header("Authorization", "Bearer " + System.getenv("INCEPTION_API_KEY"))\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(json))\n'
                '    .build();\n\n'
                'HttpResponse<String> response = client.send(request,\n'
                '    HttpResponse.BodyHandlers.ofString());\n'
                'System.out.println(response.body());\n'
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
                '    const char* api_key = std::getenv("INCEPTION_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"max_tokens", 10000},\n'
                '        {"messages", {{{"role", "user"}, {"content", "Hello!"}}}}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.inceptionlabs.ai/v1/chat/completions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["choices"][0]["message"]["content"]'
                '.get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _fim_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'import requests\n\n'
                'response = requests.post(\n'
                '    "https://api.inceptionlabs.ai/v1/fim/completions",\n'
                '    headers={\n'
                '        "Content-Type": "application/json",\n'
                '        "Authorization": f"Bearer {os.environ[\'INCEPTION_API_KEY\']}",\n'
                '    },\n'
                '    json={\n'
                f'        "model": "{model}",\n'
                '        "prompt": "def fibonacci(",\n'
                '        "suffix": "return a + b",\n'
                '    },\n'
                ')\n\n'
                'print(response.json()["choices"][0]["text"])\n'
            ),
            "typescript": (
                '// No official SDK — using native fetch API\n\n'
                'const response = await fetch(\n'
                '  "https://api.inceptionlabs.ai/v1/fim/completions",\n'
                '  {\n'
                '    method: "POST",\n'
                '    headers: {\n'
                '      "Content-Type": "application/json",\n'
                '      Authorization: `Bearer ${process.env.INCEPTION_API_KEY}`,\n'
                '    },\n'
                '    body: JSON.stringify({\n'
                f'      model: "{model}",\n'
                '      prompt: "def fibonacci(",\n'
                '      suffix: "return a + b",\n'
                '    }),\n'
                '  }\n'
                ');\n\n'
                'const data = await response.json();\n'
                'console.log(data.choices[0].text);\n'
            ),
            "javascript": (
                '// No official SDK — using native fetch API\n\n'
                'const response = await fetch(\n'
                '  "https://api.inceptionlabs.ai/v1/fim/completions",\n'
                '  {\n'
                '    method: "POST",\n'
                '    headers: {\n'
                '      "Content-Type": "application/json",\n'
                '      Authorization: `Bearer ${process.env.INCEPTION_API_KEY}`,\n'
                '    },\n'
                '    body: JSON.stringify({\n'
                f'      model: "{model}",\n'
                '      prompt: "def fibonacci(",\n'
                '      suffix: "return a + b",\n'
                '    }),\n'
                '  }\n'
                ');\n\n'
                'const data = await response.json();\n'
                'console.log(data.choices[0].text);\n'
            ),
            "java": (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n\n'
                '// No official SDK — using java.net.http.HttpClient\n'
                'HttpClient client = HttpClient.newHttpClient();\n\n'
                'String json = """\n'
                '    {\n'
                f'        "model": "{model}",\n'
                '        "prompt": "def fibonacci(",\n'
                '        "suffix": "return a + b"\n'
                '    }""";\n\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                '    .uri(URI.create("https://api.inceptionlabs.ai/v1/fim/completions"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .header("Authorization", "Bearer " + System.getenv("INCEPTION_API_KEY"))\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(json))\n'
                '    .build();\n\n'
                'HttpResponse<String> response = client.send(request,\n'
                '    HttpResponse.BodyHandlers.ofString());\n'
                'System.out.println(response.body());\n'
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
                '    const char* api_key = std::getenv("INCEPTION_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"prompt", "def fibonacci("},\n'
                '        {"suffix", "return a + b"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers, ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.inceptionlabs.ai/v1/fim/completions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["choices"][0]["text"]'
                '.get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
