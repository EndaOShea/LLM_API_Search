"""MiniMax API provider.

MiniMax's chat API is OpenAI-compatible (see
https://platform.minimax.io/docs/api-reference/text-chat-openai): text snippets
use the ``openai`` SDK pointed at ``https://api.minimax.io/v1``. Image, video,
TTS, and music generation use MiniMax's own REST endpoints via stdlib urllib.

Pricing source: https://platform.minimax.io/docs/guides/pricing-paygo (USD).
Text pricing encodes the standard tier at the <=512k context band; the >512k
band, the Priority tier, and prompt-caching rates are noted in each description.
MiniMax exposes no model-listing endpoint, so discovery is curated-only.
"""

from __future__ import annotations

from llm_api_search.providers.base import (
    TextModelInfo, ImageModelInfo, VideoModelInfo, AudioTTSModelInfo,
    MusicModelInfo, ModelType, Provider, ProviderInfo,
)

_BASE_URL = "https://api.minimax.io/v1"
_DEFAULT_MODEL = "MiniMax-M3"

_STATIC_MODELS = [
    TextModelInfo(
        model_id='MiniMax-M3',
        display_name='MiniMax-M3',
        description="MiniMax's flagship natively-multimodal coding/agentic model (image + video input). 1M token context, function calling, and adaptive thinking (thinking={type:'adaptive'|'disabled'}). Standard tier: $0.30/$1.20 per 1M tok at <=512k context, $0.60/$2.40 above 512k; Priority tier is 1.5x. Prompt-cache read $0.06/1M.",
        context_window=1_000_000,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=1.2,
    ),
    TextModelInfo(
        model_id='MiniMax-M2.7',
        display_name='MiniMax-M2.7',
        description='MiniMax-M2.7 text/agentic model. 200K token context, function calling, and adaptive thinking. $0.30/$1.20 per 1M tok; prompt-cache read $0.06/1M, write $0.375/1M.',
        context_window=200_000,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=1.2,
    ),
    TextModelInfo(
        model_id='MiniMax-M2.7-highspeed',
        display_name='MiniMax-M2.7-highspeed',
        description='Latency-optimized MiniMax-M2.7. 200K token context, function calling, and adaptive thinking. $0.60/$2.40 per 1M tok; prompt-cache read $0.06/1M, write $0.375/1M.',
        context_window=200_000,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.6,
        output_cost_per_mtok=2.4,
    ),
    ImageModelInfo(
        model_id='image-01',
        display_name='image-01',
        description='MiniMax image generation. Prompt-to-image with selectable aspect ratios; per-image billing.',
        supported_sizes=['1:1', '16:9', '4:3', '3:2', '2:3', '3:4', '9:16', '21:9'],
        supported_qualities=[],
        max_images_per_request=9,
        cost_per_image=0.0035,
    ),
    VideoModelInfo(
        model_id='MiniMax-Hailuo-2.3',
        display_name='MiniMax-Hailuo-2.3',
        description='MiniMax Hailuo 2.3 text/image-to-video (up to 1080p, 24fps). Async task API (create, then poll by id). Per-video billing, priced by resolution x duration: 768P/6s $0.28, 768P/10s $0.56, 1080P/6s $0.49 (cost_per_video encodes 768P/6s).',
        supported_resolutions=['768P', '1080P'],
        supports_audio=False,
        cost_per_second=None,
        cost_per_video=0.28,
    ),
    VideoModelInfo(
        model_id='MiniMax-Hailuo-2.3-Fast',
        display_name='MiniMax-Hailuo-2.3-Fast',
        description='Efficiency-focused MiniMax Hailuo 2.3. Async task API. Per-video billing: 768P/6s $0.19, 768P/10s $0.32, 1080P/6s $0.33 (cost_per_video encodes 768P/6s).',
        supported_resolutions=['768P', '1080P'],
        supports_audio=False,
        cost_per_second=None,
        cost_per_video=0.19,
    ),
    AudioTTSModelInfo(
        model_id='speech-2.8-hd',
        display_name='speech-2.8-hd',
        description='MiniMax high-fidelity text-to-speech (T2A v2). 40 languages, emotional tones, up to 10k chars/request. Per-character billing.',
        supported_voices=['male-qn-qingse', 'female-shaonv', 'audiobook_male_1', 'audiobook_female_1', 'presenter_male'],
        supported_output_formats=['mp3', 'pcm', 'flac', 'wav'],
        cost_per_mchars=100.0,
        input_cost_per_mtok=None,
        output_cost_per_mtok=None,
    ),
    AudioTTSModelInfo(
        model_id='speech-2.8-turbo',
        display_name='speech-2.8-turbo',
        description='Speed-optimized MiniMax text-to-speech (T2A v2). 40 languages, up to 10k chars/request. Per-character billing.',
        supported_voices=['male-qn-qingse', 'female-shaonv', 'audiobook_male_1', 'audiobook_female_1', 'presenter_male'],
        supported_output_formats=['mp3', 'pcm', 'flac', 'wav'],
        cost_per_mchars=60.0,
        input_cost_per_mtok=None,
        output_cost_per_mtok=None,
    ),
    MusicModelInfo(
        model_id='Music-2.6',
        display_name='Music-2.6',
        description='MiniMax text-to-music generation (prompt + lyrics). Billed $0.15 per generation of up to 5 minutes (cost_per_second encodes $0.15 / 300s).',
        cost_per_second=0.0005,
    ),
]


class MiniMaxProvider(Provider):
    """MiniMax provider (OpenAI-compatible chat + REST media endpoints)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="MiniMax",
            api_base_url=_BASE_URL,
            api_version=None,
            auth_env_var="MINIMAX_API_KEY",
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
            documentation_url="https://platform.minimax.io/docs/",
            rate_limits_url="https://platform.minimax.io/docs/guides/rate-limits",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """MiniMax exposes no model-listing endpoint, so discovery is curated-only.

        Returns the static catalog unchanged. New MiniMax models are added by a
        human tracking the pricing/models pages (there is no live list to diff,
        so ``unrecognized_live_model_ids`` is left at the base default).
        """
        return self.get_static_info()

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
        # NOTE: _BASE_URL already ends in "/v1"; media paths are relative to it
        # (do NOT prefix another "/v1"), so the full URL is e.g.
        # https://api.minimax.io/v1/image_generation.
        if mtype == ModelType.IMAGE:
            return self._rest_snippet(
                model, language, "/image_generation",
                f'{{"model": "{model}", "prompt": "A cute kitten on a sunny '
                f'windowsill", "aspect_ratio": "1:1", "n": 1}}',
            )
        if mtype == ModelType.VIDEO:
            return self._rest_snippet(
                model, language, "/video_generation",
                f'{{"model": "{model}", "prompt": "A cat playing with a ball of yarn"}}',
                note="Async: poll GET /v1/query/video_generation?task_id=<id> "
                     "until the video is ready.",
            )
        if mtype == ModelType.AUDIO_TTS:
            return self._rest_snippet(
                model, language, "/t2a_v2",
                f'{{"model": "{model}", "text": "Hello from MiniMax!", '
                f'"voice_setting": {{"voice_id": "male-qn-qingse", "speed": 1.0}}, '
                f'"audio_setting": {{"format": "mp3"}}}}',
            )
        if mtype == ModelType.MUSIC:
            return self._rest_snippet(
                model, language, "/music_generation",
                f'{{"model": "{model}", "prompt": "An upbeat summer pop song", '
                f'"lyrics": "##Sunlight on my face##"}}',
            )
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["MINIMAX_API_KEY"],\n'
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
                '  apiKey: process.env.MINIMAX_API_KEY,\n'
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
                '  apiKey: process.env.MINIMAX_API_KEY,\n'
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
                '    .apiKey(System.getenv("MINIMAX_API_KEY"))\n'
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
                '    const char* api_key = std::getenv("MINIMAX_API_KEY");\n'
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
                f'    curl_easy_setopt(curl, CURLOPT_URL, "{_BASE_URL}/chat/completions");\n'
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

    def _rest_snippet(
        self, model: str, language: str, path: str, body_literal: str, note: str = ""
    ) -> str:
        """Render a urllib/fetch/HttpClient/libcurl REST snippet for a media endpoint.

        ``body_literal`` is a JSON object literal (quoted keys, no booleans/None)
        that is simultaneously valid as a Python dict, a JS object, and JSON, so
        it embeds directly in every language.
        """
        url = f"{_BASE_URL}{path}"
        if language in ("typescript", "javascript"):
            js_note = f'  // {note}\n' if note else ''
            return (
                f'const res = await fetch("{url}", {{\n'
                '  method: "POST",\n'
                '  headers: {\n'
                '    Authorization: `Bearer ${process.env.MINIMAX_API_KEY}`,\n'
                '    "Content-Type": "application/json",\n'
                '  },\n'
                f'  body: JSON.stringify({body_literal}),\n'
                '});\n'
                'const data = await res.json();\n'
                f'{js_note}'
                'console.log(data);\n'
            )
        if language == "java":
            java_note = f'// {note}\n' if note else ''
            return (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n\n'
                'HttpClient client = HttpClient.newHttpClient();\n'
                f'String body = """\n    {body_literal}""";\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                f'    .uri(URI.create("{url}"))\n'
                '    .header("Authorization", "Bearer " + System.getenv("MINIMAX_API_KEY"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(body))\n'
                '    .build();\n'
                'HttpResponse<String> response =\n'
                '    client.send(request, HttpResponse.BodyHandlers.ofString());\n'
                f'{java_note}'
                'System.out.println(response.body());\n'
            )
        if language == "cpp":
            cpp_note = f'    // {note}\n' if note else ''
            return (
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
                '    const char* api_key = std::getenv("MINIMAX_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                f'    json body = json::parse(R"({body_literal})");\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                f'    curl_easy_setopt(curl, CURLOPT_URL, "{url}");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                f'{cpp_note}'
                '    std::cout << response << std::endl;\n'
                '    return 0;\n'
                '}\n'
            )
        # python (default / fallback)
        py_note = f'# {note}\n' if note else ''
        return (
            'import os, json, urllib.request\n\n'
            f'body = json.dumps({body_literal}).encode()\n'
            'req = urllib.request.Request(\n'
            f'    "{url}",\n'
            '    data=body,\n'
            '    headers={\n'
            '        "Authorization": f"Bearer {os.environ[\'MINIMAX_API_KEY\']}",\n'
            '        "Content-Type": "application/json",\n'
            '    },\n'
            ')\n'
            'resp = json.loads(urllib.request.urlopen(req).read())\n'
            f'{py_note}'
            'print(resp)\n'
        )
