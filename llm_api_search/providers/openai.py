"""OpenAI API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    ModelInfo, ModelType, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, Provider, ProviderInfo,
)

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
    # --- Image generation ---
    ImageModelInfo(
        model_id="gpt-image-1.5",
        display_name="GPT Image 1.5",
        description="Flagship image generation model with transparent backgrounds and streaming",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.034,
    ),
    ImageModelInfo(
        model_id="gpt-image-1",
        display_name="GPT Image 1",
        description="Standard image generation model",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.042,
    ),
    ImageModelInfo(
        model_id="gpt-image-1-mini",
        display_name="GPT Image 1 Mini",
        description="Cost-effective image generation model",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.011,
    ),
    # --- Audio TTS ---
    AudioTTSModelInfo(
        model_id="gpt-4o-mini-tts",
        display_name="GPT-4o Mini TTS",
        description="Instruction-steerable text-to-speech model",
        supported_voices=["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        input_cost_per_mtok=0.60,
        output_cost_per_mtok=12.00,
    ),
    AudioTTSModelInfo(
        model_id="tts-1",
        display_name="TTS-1",
        description="Low-latency text-to-speech model",
        supported_voices=["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        cost_per_mchars=15.00,
    ),
    AudioTTSModelInfo(
        model_id="tts-1-hd",
        display_name="TTS-1 HD",
        description="High-quality text-to-speech model",
        supported_voices=["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        cost_per_mchars=30.00,
    ),
    # --- Audio transcription ---
    AudioTranscriptionModelInfo(
        model_id="gpt-4o-transcribe",
        display_name="GPT-4o Transcribe",
        description="High-accuracy speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    AudioTranscriptionModelInfo(
        model_id="gpt-4o-mini-transcribe",
        display_name="GPT-4o Mini Transcribe",
        description="Cost-effective speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.003,
    ),
    AudioTranscriptionModelInfo(
        model_id="whisper-1",
        display_name="Whisper",
        description="Open-source speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    # --- Embeddings ---
    EmbeddingModelInfo(
        model_id="text-embedding-3-large",
        display_name="Text Embedding 3 Large",
        description="Most capable embedding model",
        dimensions=3072,
        max_input_tokens=8192,
        input_cost_per_mtok=0.13,
    ),
    EmbeddingModelInfo(
        model_id="text-embedding-3-small",
        display_name="Text Embedding 3 Small",
        description="Cost-effective embedding model",
        dimensions=1536,
        max_input_tokens=8192,
        input_cost_per_mtok=0.02,
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

    @staticmethod
    def _model_class_for_id(model_id: str):
        """Return the appropriate ModelInfo subclass for a model ID."""
        # Order matters: more specific prefixes before less specific ones.
        if model_id.startswith("gpt-image-"):
            return ImageModelInfo
        elif model_id.startswith("gpt-4o-mini-tts") or model_id.startswith("tts-"):
            return AudioTTSModelInfo
        elif model_id.startswith("whisper-") or model_id.startswith("gpt-4o-transcribe") or model_id.startswith("gpt-4o-mini-transcribe"):
            return AudioTranscriptionModelInfo
        elif model_id.startswith("text-embedding-"):
            return EmbeddingModelInfo
        return TextModelInfo

    def fetch_live_models(self) -> ProviderInfo:
        """Fetch models from the OpenAI API, fall back to static."""
        info = self.get_static_info()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return info

        # Well-known model prefixes we care about.
        _PREFIXES = ("gpt-5", "gpt-4", "o3", "o4", "gpt-image-", "tts-", "whisper-", "text-embedding-")

        try:
            req = urllib.request.Request(
                f"{info.api_base_url}/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            live_models: list[ModelInfo] = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if any(mid.startswith(p) for p in _PREFIXES):
                    model_cls = self._model_class_for_id(mid)
                    live_models.append(model_cls(model_id=mid, display_name=mid))
            if live_models:
                live_models.sort(key=lambda x: x.model_id)
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def _get_model_type(self, model_id: str) -> ModelType:
        """Look up the model type from _STATIC_MODELS."""
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gpt-5.4"
        mtype = self._get_model_type(model)
        if mtype == ModelType.IMAGE:
            return self._image_snippet(model, language)
        elif mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        elif mtype == ModelType.AUDIO_TRANSCRIPTION:
            return self._transcription_snippet(model, language)
        elif mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
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

    def _image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.images.generate(\n'
                f'    model="{model}",\n'
                '    prompt="A white siamese cat",\n'
                '    size="1024x1024",\n'
                ')\n'
                'print(response.data[0].url)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.images.generate({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  size: "1024x1024",\n'
                '});\n'
                'console.log(response.data[0].url);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.images.*;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'ImagesResponse response = client.images().generate(\n'
                '    ImageGenerateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .prompt("A white siamese cat")\n'
                '        .size("1024x1024")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.data().get(0).url());\n'
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
                '        {"prompt", "A white siamese cat"},\n'
                '        {"size", "1024x1024"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/images/generations");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["data"][0]["url"]\n'
                '              .get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _tts_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.audio.speech.create(\n'
                f'    model="{model}",\n'
                '    voice="alloy",\n'
                '    input="Hello!",\n'
                ')\n'
                'response.stream_to_file("output.mp3")\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n'
                'import fs from "fs";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.audio.speech.create({\n'
                f'  model: "{model}",\n'
                '  voice: "alloy",\n'
                '  input: "Hello!",\n'
                '});\n'
                'const buffer = Buffer.from(await response.arrayBuffer());\n'
                'fs.writeFileSync("output.mp3", buffer);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n'
                'const fs = require("fs");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.audio.speech.create({\n'
                f'  model: "{model}",\n'
                '  voice: "alloy",\n'
                '  input: "Hello!",\n'
                '});\n'
                'const buffer = Buffer.from(await response.arrayBuffer());\n'
                'fs.writeFileSync("output.mp3", buffer);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.audio.*;\n'
                'import java.nio.file.Files;\n'
                'import java.nio.file.Path;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'byte[] response = client.audio().speech().create(\n'
                '    SpeechCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .voice("alloy")\n'
                '        .input("Hello!")\n'
                '        .build()\n'
                ');\n'
                'Files.write(Path.of("output.mp3"), response);\n'
                'System.out.println("Audio saved to output.mp3");\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <fstream>\n'
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
                '        {"voice", "alloy"},\n'
                '        {"input", "Hello!"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/audio/speech");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    std::ofstream out("output.mp3", std::ios::binary);\n'
                '    out.write(response.data(), response.size());\n'
                '    std::cout << "Audio saved to output.mp3" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _transcription_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'transcript = client.audio.transcriptions.create(\n'
                f'    model="{model}",\n'
                '    file=open("audio.mp3", "rb"),\n'
                ')\n'
                'print(transcript.text)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n'
                'import fs from "fs";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const transcript = await client.audio.transcriptions.create({\n'
                f'  model: "{model}",\n'
                '  file: fs.createReadStream("audio.mp3"),\n'
                '});\n'
                'console.log(transcript.text);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n'
                'const fs = require("fs");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const transcript = await client.audio.transcriptions.create({\n'
                f'  model: "{model}",\n'
                '  file: fs.createReadStream("audio.mp3"),\n'
                '});\n'
                'console.log(transcript.text);\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.audio.*;\n'
                'import java.nio.file.Path;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'Transcription transcript = client.audio().transcriptions().create(\n'
                '    TranscriptionCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .file(Path.of("audio.mp3"))\n'
                '        .build()\n'
                ');\n'
                'System.out.println(transcript.text());\n'
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
                '    curl_mime* mime = curl_mime_init(curl);\n'
                '    curl_mimepart* part = curl_mime_addpart(mime);\n'
                '    curl_mime_name(part, "file");\n'
                '    curl_mime_filedata(part, "audio.mp3");\n'
                '    part = curl_mime_addpart(mime);\n'
                '    curl_mime_name(part, "model");\n'
                f'    curl_mime_data(part, "{model}", CURL_ZERO_TERMINATED);\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n\n'
                '    std::string response;\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/audio/transcriptions");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_mime_free(mime);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["text"].get<std::string>() << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _embedding_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from openai import OpenAI\n\n'
                'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
                'response = client.embeddings.create(\n'
                f'    model="{model}",\n'
                '    input="Hello!",\n'
                ')\n'
                'print(response.data[0].embedding[:5])\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.embeddings.create({\n'
                f'  model: "{model}",\n'
                '  input: "Hello!",\n'
                '});\n'
                'console.log(response.data[0].embedding.slice(0, 5));\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI(); // uses OPENAI_API_KEY env var\n\n'
                'const response = await client.embeddings.create({\n'
                f'  model: "{model}",\n'
                '  input: "Hello!",\n'
                '});\n'
                'console.log(response.data[0].embedding.slice(0, 5));\n'
            ),
            "java": (
                'import com.openai.client.OpenAIClient;\n'
                'import com.openai.client.okhttp.OpenAIOkHttpClient;\n'
                'import com.openai.models.embeddings.*;\n\n'
                '// Uses OPENAI_API_KEY env var\n'
                'OpenAIClient client = OpenAIOkHttpClient.builder().build();\n\n'
                'EmbeddingResponse response = client.embeddings().create(\n'
                '    EmbeddingCreateParams.builder()\n'
                f'        .model("{model}")\n'
                '        .input("Hello!")\n'
                '        .build()\n'
                ');\n'
                'System.out.println(response.data().get(0).embedding());\n'
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
                '        {"input", "Hello!"}\n'
                '    };\n\n'
                '    struct curl_slist* headers = nullptr;\n'
                '    headers = curl_slist_append(headers,\n'
                '        ("Authorization: Bearer " + std::string(api_key)).c_str());\n'
                '    headers = curl_slist_append(headers, "Content-Type: application/json");\n\n'
                '    std::string response;\n'
                '    std::string payload = body.dump();\n'
                '    curl_easy_setopt(curl, CURLOPT_URL,\n'
                '        "https://api.openai.com/v1/embeddings");\n'
                '    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);\n'
                '    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);\n'
                '    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);\n\n'
                '    curl_easy_perform(curl);\n'
                '    curl_easy_cleanup(curl);\n'
                '    curl_slist_free_all(headers);\n\n'
                '    auto result = json::parse(response);\n'
                '    std::cout << result["data"][0]["embedding"] << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
