"""Google Gemini API provider."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    ModelInfo, ModelType, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    EmbeddingModelInfo, Provider, ProviderInfo,
)

_STATIC_MODELS = [
    TextModelInfo(
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
    TextModelInfo(
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
    TextModelInfo(
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
    TextModelInfo(
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
    TextModelInfo(
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
    TextModelInfo(
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
    # --- Image generation (Imagen 4) ---
    ImageModelInfo(
        model_id="imagen-4.0-generate-001",
        display_name="Imagen 4",
        description="Standard image generation model",
        supported_sizes=["1024x1024", "2048x2048"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.04,
    ),
    ImageModelInfo(
        model_id="imagen-4.0-fast-generate-001",
        display_name="Imagen 4 Fast",
        description="Fast image generation model",
        supported_sizes=["1024x1024"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.02,
    ),
    ImageModelInfo(
        model_id="imagen-4.0-ultra-generate-001",
        display_name="Imagen 4 Ultra",
        description="Highest quality image generation model",
        supported_sizes=["1024x1024", "2048x2048"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.06,
    ),
    # --- Audio TTS ---
    AudioTTSModelInfo(
        model_id="gemini-2.5-flash-preview-tts",
        display_name="Gemini 2.5 Flash TTS",
        description="Low-latency text-to-speech model",
        supported_voices=["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"],
        supported_output_formats=["wav"],
        input_cost_per_mtok=0.50,
        output_cost_per_mtok=10.00,
    ),
    AudioTTSModelInfo(
        model_id="gemini-2.5-pro-preview-tts",
        display_name="Gemini 2.5 Pro TTS",
        description="High-quality text-to-speech model",
        supported_voices=["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"],
        supported_output_formats=["wav"],
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=20.00,
    ),
    # --- Embeddings ---
    EmbeddingModelInfo(
        model_id="gemini-embedding-001",
        display_name="Gemini Embedding",
        description="Text embedding model",
        dimensions=3072,
        max_input_tokens=2048,
        input_cost_per_mtok=0.15,
    ),
    EmbeddingModelInfo(
        model_id="gemini-embedding-2-preview",
        display_name="Gemini Embedding 2",
        description="Multimodal embedding model (text, images, audio, video)",
        dimensions=3072,
        max_input_tokens=8192,
        supports_multimodal=True,
        input_cost_per_mtok=0.20,
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

    @staticmethod
    def _model_class_for_id(model_id: str):
        if model_id.startswith("imagen-"):
            return ImageModelInfo
        elif model_id.startswith("gemini-embedding") or model_id.startswith("text-embedding"):
            return EmbeddingModelInfo
        elif "tts" in model_id:
            return AudioTTSModelInfo
        return TextModelInfo

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
                if not model_id:
                    continue
                model_cls = self._model_class_for_id(model_id)
                live_models.append(
                    model_cls(
                        model_id=model_id,
                        display_name=m.get("displayName", model_id),
                        description=m.get("description", ""),
                    )
                )
            if live_models:
                info.models = live_models
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError):
            pass

        return info

    def _get_model_type(self, model_id: str) -> ModelType:
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gemini-2.5-flash"
        mtype = self._get_model_type(model)
        if mtype == ModelType.IMAGE:
            return self._image_snippet(model, language)
        elif mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        elif mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
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

    def _image_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_images(\n'
                f'    model="{model}",\n'
                '    prompt="A white siamese cat",\n'
                '    config=types.GenerateImagesConfig(number_of_images=1),\n'
                ')\n'
                'response.generated_images[0].image.save("output.png")\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateImages({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  config: { numberOfImages: 1 },\n'
                '});\n'
                'console.log(response.generatedImages[0]);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateImages({\n'
                f'  model: "{model}",\n'
                '  prompt: "A white siamese cat",\n'
                '  config: { numberOfImages: 1 },\n'
                '});\n'
                'console.log(response.generatedImages[0]);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateImagesConfig;\n'
                'import com.google.genai.types.GenerateImagesResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateImagesResponse response = client.models.generateImages(\n'
                f'    "{model}",\n'
                '    "A white siamese cat",\n'
                '    GenerateImagesConfig.builder().numberOfImages(1).build()\n'
                ');\n'
                'System.out.println(response.generatedImages().get(0));\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
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
                '        {"instances", {{{"prompt", "A white siamese cat"}}}},\n'
                '        {"parameters", {{"sampleCount", 1}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:predict?key=" + std::string(api_key);\n\n'
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
                '    std::cout << "Image generated successfully" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _tts_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n'
                'from google.genai import types\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.generate_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello, world!",\n'
                '    config=types.GenerateContentConfig(\n'
                '        response_modalities=["AUDIO"],\n'
                '        speech_config=types.SpeechConfig(\n'
                '            voice_config=types.VoiceConfig(\n'
                '                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")\n'
                '            )\n'
                '        ),\n'
                '    ),\n'
                ')\n'
                'with open("output.wav", "wb") as f:\n'
                '    f.write(response.candidates[0].content.parts[0].inline_data.data)\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '  config: {\n'
                '    responseModalities: ["AUDIO"],\n'
                '    speechConfig: {\n'
                '      voiceConfig: {\n'
                '        prebuiltVoiceConfig: { voiceName: "Kore" },\n'
                '      },\n'
                '    },\n'
                '  },\n'
                '});\n'
                'const audioData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.generateContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '  config: {\n'
                '    responseModalities: ["AUDIO"],\n'
                '    speechConfig: {\n'
                '      voiceConfig: {\n'
                '        prebuiltVoiceConfig: { voiceName: "Kore" },\n'
                '      },\n'
                '    },\n'
                '  },\n'
                '});\n'
                'const audioData = response.candidates[0].content.parts[0].inlineData.data;\n'
                'console.log("Audio generated, length:", audioData.length);\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.GenerateContentConfig;\n'
                'import com.google.genai.types.GenerateContentResponse;\n'
                'import com.google.genai.types.SpeechConfig;\n'
                'import com.google.genai.types.VoiceConfig;\n'
                'import com.google.genai.types.PrebuiltVoiceConfig;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'GenerateContentResponse response = client.models.generateContent(\n'
                f'    "{model}",\n'
                '    "Hello, world!",\n'
                '    GenerateContentConfig.builder()\n'
                '        .responseModalities(java.util.List.of("AUDIO"))\n'
                '        .speechConfig(SpeechConfig.builder()\n'
                '            .voiceConfig(VoiceConfig.builder()\n'
                '                .prebuiltVoiceConfig(PrebuiltVoiceConfig.builder()\n'
                '                    .voiceName("Kore").build()).build()).build())\n'
                '        .build()\n'
                ');\n'
                'System.out.println("Audio generated successfully");\n'
            ),
            "cpp": (
                '#include <iostream>\n'
                '#include <string>\n'
                '#include <fstream>\n'
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
                '        {"contents", {{{"parts", {{{"text", "Hello, world!"}}}}}}}},\n'
                '        {"generationConfig", {\n'
                '            {"responseModalities", {"AUDIO"}},\n'
                '            {"speechConfig", {{"voiceConfig", {{"prebuiltVoiceConfig",\n'
                '                {{"voiceName", "Kore"}}}}}}}\n'
                '        }}\n'
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
                '    std::cout << "Audio speech generated successfully" << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])

    def _embedding_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'from google import genai\n\n'
                'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
                'response = client.models.embed_content(\n'
                f'    model="{model}",\n'
                '    contents="Hello, world!",\n'
                ')\n'
                'print(response.embeddings[0].values[:5])\n'
            ),
            "typescript": (
                'import { GoogleGenAI } from "@google/genai";\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });\n\n'
                'const response = await ai.models.embedContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '});\n'
                'console.log(response.embeddings[0].values.slice(0, 5));\n'
            ),
            "javascript": (
                'const { GoogleGenAI } = require("@google/genai");\n\n'
                'const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });\n\n'
                'const response = await ai.models.embedContent({\n'
                f'  model: "{model}",\n'
                '  contents: "Hello, world!",\n'
                '});\n'
                'console.log(response.embeddings[0].values.slice(0, 5));\n'
            ),
            "java": (
                'import com.google.genai.Client;\n'
                'import com.google.genai.types.EmbedContentResponse;\n\n'
                '// Uses GEMINI_API_KEY env var\n'
                'Client client = Client.builder().build();\n\n'
                'EmbedContentResponse response = client.models.embedContent(\n'
                f'    "{model}",\n'
                '    "Hello, world!"\n'
                ');\n'
                'System.out.println(response.embeddings().get(0).values());\n'
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
                '        {"content", {{"parts", {{{"text", "Hello, world!"}}}}}}\n'
                '    };\n\n'
                '    std::string url =\n'
                '        "https://generativelanguage.googleapis.com/v1beta/models/"\n'
                f'        "{model}:embedContent?key=" + std::string(api_key);\n\n'
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
                '    std::cout << "Embedding: " << result["embedding"]["values"][0]\n'
                '              << std::endl;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
