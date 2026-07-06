"""Z.ai (Zhipu GLM) API provider.

Z.ai's API is OpenAI-compatible (see https://docs.z.ai/api-reference/introduction).
Text/vision snippets use the ``openai`` SDK pointed at
``https://api.z.ai/api/paas/v4``. Image (CogView) and video (CogVideoX)
generation use Z.ai's REST endpoints directly.

Pricing source: https://docs.z.ai/guides/overview/pricing (USD).
Free tiers (glm-4.5-flash, glm-4.6v-flash) are recorded as 0.0.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    TextModelInfo, ImageModelInfo, VideoModelInfo,
    ModelType, Provider, ProviderInfo,
)

_BASE_URL = "https://api.z.ai/api/paas/v4"
_DEFAULT_MODEL = "glm-5.2"

_STATIC_MODELS = [
    TextModelInfo(
        model_id='glm-5.2',
        display_name='GLM-5.2',
        description="Z.ai's flagship GLM model. 1M token context, up to 128K output. Tool calling, structured output (JSON), context caching, MCP, and configurable thinking (reasoning_effort, default max).",
        context_window=1_000_000,
        max_output_tokens=128_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.4,
        output_cost_per_mtok=4.4,
    ),
    TextModelInfo(
        model_id='glm-5.1',
        display_name='GLM-5.1',
        description='GLM-5.1 text model. 200K token context, up to 128K output. Tool calling, structured output, and configurable thinking.',
        context_window=200_000,
        max_output_tokens=128_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.4,
        output_cost_per_mtok=4.4,
    ),
    TextModelInfo(
        model_id='glm-5',
        display_name='GLM-5',
        description='GLM-5 text model. 200K token context, up to 128K output. Strong tool invocation and configurable thinking.',
        context_window=200_000,
        max_output_tokens=128_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=3.2,
    ),
    TextModelInfo(
        model_id='glm-4.6',
        display_name='GLM-4.6',
        description='GLM-4.6 text model, strong in coding and agentic use. 200K token context, up to 128K output. Tool calling and configurable thinking.',
        context_window=200_000,
        max_output_tokens=128_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=2.2,
    ),
    TextModelInfo(
        model_id='glm-4.5-air',
        display_name='GLM-4.5-Air',
        description='Lightweight GLM-4.5 variant (106B/12B active). 128K token context, up to 96K output. Tool calling and configurable thinking.',
        context_window=128_000,
        max_output_tokens=96_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.2,
        output_cost_per_mtok=1.1,
    ),
    TextModelInfo(
        model_id='glm-4.5-flash',
        display_name='GLM-4.5-Flash',
        description='Free fast GLM-4.5 variant. 128K token context, up to 96K output. Tool calling and structured output.',
        context_window=128_000,
        max_output_tokens=96_000,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    TextModelInfo(
        model_id='glm-5v-turbo',
        display_name='GLM-5V-Turbo',
        description='GLM-5V-Turbo multimodal model (image/video/text input). 200K token context, up to 128K output. Function calling and configurable thinking.',
        context_window=200_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.2,
        output_cost_per_mtok=4.0,
    ),
    TextModelInfo(
        model_id='glm-4.6v-flash',
        display_name='GLM-4.6V-Flash',
        description='Free GLM-4.6V vision variant (image/video/text/file input). 128K token context. Native multimodal function calling.',
        context_window=128_000,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
    ),
    ImageModelInfo(
        model_id='cogview-4',
        display_name='CogView-4',
        description='CogView-4 image generation (pinned snapshot cogview-4-250304). Generates images across a range of resolutions; per-image billing.',
        supported_sizes=['1024x1024', '1728x2304', '2048x2048'],
        supported_qualities=[],
        max_images_per_request=None,
        cost_per_image=0.01,
    ),
    VideoModelInfo(
        model_id='cogvideox-3',
        display_name='CogVideoX-3',
        description='CogVideoX-3 video generation. Async task API (submit, then poll by id). Text/image/start-end-frame inputs, up to 4K, optional audio. Per-video billing.',
        supported_resolutions=['1920x1080', '3840x2160'],
        supports_audio=True,
        cost_per_second=None,
        cost_per_video=0.2,
    ),
    TextModelInfo(
        model_id='glm-4.5',
        display_name='glm-4.5',
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
        model_id='glm-4.7',
        display_name='glm-4.7',
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
        model_id='glm-5-turbo',
        display_name='glm-5-turbo',
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
]


class ZaiProvider(Provider):
    """Z.ai / Zhipu GLM provider (OpenAI-compatible API)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Z.ai (GLM)",
            api_base_url=_BASE_URL,
            api_version=None,
            auth_env_var="ZAI_API_KEY",
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
            documentation_url="https://docs.z.ai/",
            rate_limits_url="https://docs.z.ai/api-reference/rate-limit",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """List models via the OpenAI-compatible /models endpoint; fall back to static.

        Returns every live ID (as bare TextModelInfo) so the weekly update can
        auto-add genuinely-new GLM models. GLM IDs are versioned (not generic
        aliases), so no unrecognized_live_model_ids() override is needed.
        """
        info = self.get_static_info()
        api_key = os.environ.get("ZAI_API_KEY")
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

    def _get_model_type(self, model_id: str) -> ModelType:
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or _DEFAULT_MODEL
        mtype = self._get_model_type(model)
        if mtype == ModelType.IMAGE:
            return self._image_snippet(model, language)
        elif mtype == ModelType.VIDEO:
            return self._video_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["ZAI_API_KEY"],\n'
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
                '  apiKey: process.env.ZAI_API_KEY,\n'
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
                '  apiKey: process.env.ZAI_API_KEY,\n'
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
                '    .apiKey(System.getenv("ZAI_API_KEY"))\n'
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
                '    const char* api_key = std::getenv("ZAI_API_KEY");\n'
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

    def _image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["ZAI_API_KEY"],\n'
                f'    base_url="{_BASE_URL}",\n'
                ')\n\n'
                'response = client.images.generate(\n'
                f'    model="{model}",\n'
                '    prompt="A cute kitten on a sunny windowsill",\n'
                '    size="1024x1024",\n'
                ')\n'
                'print(response.data[0].url)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.ZAI_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cute kitten on a sunny windowsill",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.ZAI_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cute kitten on a sunny windowsill",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "java": (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n\n'
                'HttpClient client = HttpClient.newHttpClient();\n'
                'String body = """\n'
                '    {\\n'
                f'      "model": "{model}",\\n'
                '      "prompt": "A cute kitten on a sunny windowsill",\\n'
                '      "size": "1024x1024"\\n'
                '    }""";\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                f'    .uri(URI.create("{_BASE_URL}/images/generations"))\n'
                '    .header("Authorization", "Bearer " + System.getenv("ZAI_API_KEY"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(body))\n'
                '    .build();\n'
                'HttpResponse<String> response =\n'
                '    client.send(request, HttpResponse.BodyHandlers.ofString());\n'
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
                '    const char* api_key = std::getenv("ZAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"prompt", "A cute kitten on a sunny windowsill"},\n'
                '        {"size", "1024x1024"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                f'    curl_easy_setopt(curl, CURLOPT_URL, "{_BASE_URL}/images/generations");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    std::cout << response << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _video_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os, json, urllib.request\n\n'
                'body = json.dumps({\n'
                f'    "model": "{model}",\n'
                '    "prompt": "A cat playing with a ball of yarn",\n'
                '    "quality": "quality",\n'
                '    "size": "1920x1080",\n'
                '    "with_audio": True,\n'
                '}).encode()\n'
                'req = urllib.request.Request(\n'
                f'    "{_BASE_URL}/videos/generations",\n'
                '    data=body,\n'
                '    headers={\n'
                '        "Authorization": f"Bearer {os.environ[\'ZAI_API_KEY\']}",\n'
                '        "Content-Type": "application/json",\n'
                '    },\n'
                ')\n'
                'task = json.loads(urllib.request.urlopen(req).read())\n'
                '# Async: poll the result by task id until the video URL is ready.\n'
                'print(task["id"])\n'
            ),
            "typescript": (
                'const body = JSON.stringify({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cat playing with a ball of yarn",\n'
                '  quality: "quality",\n'
                '  size: "1920x1080",\n'
                '  with_audio: true,\n'
                '});\n'
                f'const res = await fetch("{_BASE_URL}/videos/generations", {{\n'
                '  method: "POST",\n'
                '  headers: {\n'
                '    Authorization: `Bearer ${process.env.ZAI_API_KEY}`,\n'
                '    "Content-Type": "application/json",\n'
                '  },\n'
                '  body,\n'
                '});\n'
                'const task = await res.json();\n'
                '// Async: poll the result by task.id until the video URL is ready.\n'
                'console.log(task.id);\n'
            ),
            "javascript": (
                'const body = JSON.stringify({\n'
                f'  model: "{model}",\n'
                '  prompt: "A cat playing with a ball of yarn",\n'
                '  quality: "quality",\n'
                '  size: "1920x1080",\n'
                '  with_audio: true,\n'
                '});\n'
                f'const res = await fetch("{_BASE_URL}/videos/generations", {{\n'
                '  method: "POST",\n'
                '  headers: {\n'
                '    Authorization: `Bearer ${process.env.ZAI_API_KEY}`,\n'
                '    "Content-Type": "application/json",\n'
                '  },\n'
                '  body,\n'
                '});\n'
                'const task = await res.json();\n'
                '// Async: poll the result by task.id until the video URL is ready.\n'
                'console.log(task.id);\n'
            ),
            "java": (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n\n'
                'HttpClient client = HttpClient.newHttpClient();\n'
                'String body = """\n'
                '    {\\n'
                f'      "model": "{model}",\\n'
                '      "prompt": "A cat playing with a ball of yarn",\\n'
                '      "quality": "quality",\\n'
                '      "size": "1920x1080",\\n'
                '      "with_audio": true\\n'
                '    }""";\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                f'    .uri(URI.create("{_BASE_URL}/videos/generations"))\n'
                '    .header("Authorization", "Bearer " + System.getenv("ZAI_API_KEY"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(body))\n'
                '    .build();\n'
                'HttpResponse<String> response =\n'
                '    client.send(request, HttpResponse.BodyHandlers.ofString());\n'
                '// Async: poll the result by the returned task id.\n'
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
                '    const char* api_key = std::getenv("ZAI_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                '    json body = {\n'
                f'        {{"model", "{model}"}},\n'
                '        {"prompt", "A cat playing with a ball of yarn"},\n'
                '        {"quality", "quality"},\n'
                '        {"size", "1920x1080"},\n'
                '        {"with_audio", true}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                f'    curl_easy_setopt(curl, CURLOPT_URL, "{_BASE_URL}/videos/generations");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    // Async: poll the result by the returned task id.\n'
                '    std::cout << response << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
