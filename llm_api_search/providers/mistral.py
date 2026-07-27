"""Mistral (La Plateforme) API provider.

Mistral's chat, embeddings, and transcription endpoints are OpenAI-shaped, so
those snippets use the ``openai`` SDK pointed at ``https://api.mistral.ai/v1``.
Voxtral TTS (``/v1/audio/speech``) uses voice-cloning params that don't map to
OpenAI's ``audio.speech`` schema, so its snippet is REST (stdlib ``urllib``).

Mistral's live ``/v1/models`` catalog mixes every model type (text, embed,
audio, OCR, moderation) and every legacy version, so ``fetch_live_models()``
stays curated-only (like qwen.py) rather than surfacing every live ID.
``unrecognized_live_model_ids()`` flags a genuinely-new frontier *text* release
without also flagging the rest of the catalog every week. Family prefix alone
is not enough — Mistral keeps every legacy snapshot and a moving alias per
family live forever, so an ID is reported only when its trailing ``YYMM``
release stamp is strictly newer than the newest curated stamp in the same
family. That keeps aliases (``mistral-medium-latest``), superseded snapshots
(``mistral-medium-2505``), non-chat classes (``codestral-embed``), and
alternate spellings of a curated model (``mistral-medium-2604`` is the same
release as the curated ``mistral-medium-3-5-26-04``) all silent.

Pricing source: per-model model cards at
https://docs.mistral.ai/models/model-cards/<id> (USD).
Verified: 2026-07-21.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

from llm_api_search.providers.base import (
    TextModelInfo, EmbeddingModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, ModelType, Provider, ProviderInfo,
)

_BASE_URL = "https://api.mistral.ai/v1"
_DEFAULT_MODEL = "mistral-medium-3-5-26-04"

# The frontier *text* families Mistral ships. Matching this prefix alone is far
# too loose to use as the discovery filter on its own: Mistral keeps every
# legacy snapshot and a moving alias for each family live indefinitely, so a
# bare prefix match flagged 14 IDs in a single week — every one of them
# `-latest` aliases, superseded snapshots, or alternate spellings of a model
# already curated here. The real filter is the release-stamp comparison in
# _classify_unrecognized below; this only narrows the field to text families.
_FRONTIER_FAMILY_RE = re.compile(r"^(mistral-(?:medium|large|small)|codestral)(?:-|$)")

# Model classes that share a frontier family prefix but are not chat models
# (e.g. `codestral-embed` is an embedding model). Never flag these: their model
# classes have no home in the curated `ModelType` set for this provider.
_NON_TEXT_MARKERS = ("-embed", "-ocr", "-moderation", "-tts", "-transcribe")

# A release stamp in either form Mistral uses: a bare -YYMM ("mistral-small-2603",
# "codestral-2508") or a split -YY-MM ("mistral-medium-3-5-26-04"). Anchored to
# end-of-string so an interior version fragment — the "3-5" in
# `mistral-medium-3-5` — is never misread as a date.
_RELEASE_STAMP_RE = re.compile(r"-(\d{2})-?(\d{2})$")


def _family_of(model_id: str) -> str | None:
    """Return the frontier text family a model ID belongs to, if any."""
    match = _FRONTIER_FAMILY_RE.match(model_id)
    return match.group(1) if match else None


def _release_stamp(model_id: str) -> int | None:
    """Return a model ID's trailing YYMM release stamp as an int, or None.

    None means the ID carries no stamp at all — a moving alias
    (`mistral-medium-latest`) or a bare version name (`mistral-medium-3-5`).
    Both always point at a release we already curate under its dated ID, so
    neither can be a genuinely new model. YYMM compares correctly as an int
    across a year boundary (2512 < 2603).
    """
    match = _RELEASE_STAMP_RE.search(model_id)
    if not match:
        return None
    return int(match.group(1) + match.group(2))


def _classify_unrecognized(live_ids: set[str], static_ids: set[str]) -> set[str]:
    """Return live IDs that look like a genuinely *newer* frontier text release.

    Hardened well beyond a family-prefix match: an ID is flagged only when it
    carries a release stamp *and* that stamp is strictly newer than the newest
    curated stamp in the same family. This drops all three noise classes in
    Mistral's catalog at once:

    * moving aliases (`mistral-medium-latest`) — no stamp, never flagged;
    * superseded snapshots (`mistral-medium-2505`) — older stamp;
    * alternate spellings of a curated model (`mistral-medium-2604` is the same
      April-2026 release as the curated `mistral-medium-3-5-26-04`; both parse
      to 2604, so it is correctly treated as already known).

    A real new release still surfaces: `mistral-medium-2707` outranks 2604 and
    is flagged. A family with nothing curated yet has an implicit floor of 0,
    so its first dated model is flagged too.

    Pure (no network) so the classification can be unit-tested directly.
    """
    newest_curated: dict[str, int] = {}
    for mid in static_ids:
        family = _family_of(mid)
        stamp = _release_stamp(mid)
        if family and stamp is not None:
            newest_curated[family] = max(newest_curated.get(family, 0), stamp)

    flagged: set[str] = set()
    for mid in live_ids:
        if mid in static_ids:
            continue
        family = _family_of(mid)
        if family is None:
            continue
        if any(marker in mid for marker in _NON_TEXT_MARKERS):
            continue
        stamp = _release_stamp(mid)
        if stamp is None:
            continue
        if stamp > newest_curated.get(family, 0):
            flagged.add(mid)
    return flagged


_STATIC_MODELS = [
    TextModelInfo(
        model_id='mistral-medium-3-5-26-04',
        display_name='Mistral Medium 3.5',
        description="Mistral's frontier-class multimodal model, optimized for agentic and coding use cases. 256K token context. Vision + function calling. Adjustable reasoning via reasoning_effort (high enables a full thinking chunk; none disables). Aliases: mistral-medium-latest, mistral-medium-3-5.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=1.5,
        output_cost_per_mtok=7.5,
    ),
    TextModelInfo(
        model_id='mistral-small-2603',
        display_name='Mistral Small 4',
        description='Hybrid model unifying instruct, reasoning, and coding in a single efficient model, with document/image understanding. 256K token context. Function calling. Adjustable reasoning via reasoning_effort (high/none). Alias: mistral-small-latest.',
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.6,
    ),
    TextModelInfo(
        model_id='mistral-large-3-25-12',
        display_name='Mistral Large 3',
        description='State-of-the-art open-weight, general-purpose multimodal model (Mixture-of-Experts, 41B active params). 256K token context. Vision + function calling. Alias: mistral-large-latest.',
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=True,
        supports_tool_use=True,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.5,
        output_cost_per_mtok=1.5,
    ),
    TextModelInfo(
        model_id='codestral-2508',
        display_name='Codestral',
        description="Code-specialized model for low-latency, high-frequency tasks such as fill-in-the-middle (FIM) and code generation. 256K token context (Mistral's model card lists 128k; 256K per the model owner). Alias: codestral-latest.",
        context_window=262_144,
        max_output_tokens=None,
        supports_vision=False,
        supports_tool_use=False,
        supports_image_generation=False,
        supports_computer_use=False,
        input_cost_per_mtok=0.3,
        output_cost_per_mtok=0.9,
    ),
    EmbeddingModelInfo(
        model_id='mistral-embed',
        display_name='Mistral Embed',
        description='Semantic text-embedding model. 1024-dimensional vectors, 8K token max input.',
        dimensions=1_024,
        max_input_tokens=8_192,
        supports_multimodal=False,
        input_cost_per_mtok=0.1,
    ),
    AudioTTSModelInfo(
        model_id='voxtral-tts-26-03',
        display_name='Voxtral TTS',
        description='State-of-the-art text-to-speech with zero-shot voice cloning (via a short audio prompt), 9 languages, and streaming (~90ms time-to-first-audio). Per-character billing. REST endpoint /v1/audio/speech.',
        supported_voices=[],
        supported_output_formats=[],
        cost_per_mchars=16.0,
        input_cost_per_mtok=None,
        output_cost_per_mtok=None,
    ),
    AudioTranscriptionModelInfo(
        model_id='voxtral-mini-transcribe-realtime-26-02',
        display_name='Voxtral Mini Transcribe Realtime',
        description='Efficient audio-input model optimized for live transcription. OpenAI-compatible /v1/audio/transcriptions endpoint. Per-minute-of-audio billing.',
        supported_input_formats=[],
        max_file_size_mb=None,
        cost_per_minute=0.006,
    ),
]


class MistralProvider(Provider):
    """Mistral (La Plateforme) provider (OpenAI-compatible chat/embed/transcribe)."""

    def get_static_info(self) -> ProviderInfo:
        return ProviderInfo(
            name="Mistral",
            api_base_url=_BASE_URL,
            api_version=None,
            auth_env_var="MISTRAL_API_KEY",
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
            documentation_url="https://docs.mistral.ai/",
            rate_limits_url="https://help.mistral.ai/en/articles/698531-why-am-i-hitting-api-rate-limits-and-how-do-i-increase-them",
        )

    def fetch_live_models(self) -> ProviderInfo:
        """Attempt to list models via the Mistral API, fall back to static.

        Kept curated-only (like Qwen): Mistral's /v1/models catalog mixes every
        model type and legacy version, so only live IDs already in
        _STATIC_MODELS are kept.
        """
        info = self.get_static_info()
        api_key = os.environ.get("MISTRAL_API_KEY")
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
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, OSError) as exc:
            self._note_fetch_failure(exc)  # report-only; fall back to static

        return info

    def unrecognized_live_model_ids(self) -> set[str]:
        """Live frontier-text-shaped IDs not yet curated.

        Scoped to the frontier-text ID shape so this doesn't flood the weekly
        update with Mistral's embeddings/audio/OCR/moderation/legacy IDs.
        Returns an empty set if no API key is set or any API call fails.
        """
        info = self.get_static_info()
        api_key = os.environ.get("MISTRAL_API_KEY")
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
        if mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        if mtype == ModelType.AUDIO_TRANSCRIPTION:
            return self._transcription_snippet(model, language)
        if mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        return self._text_snippet(model, language)

    def _text_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["MISTRAL_API_KEY"],\n'
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
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
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
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
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
                '    .apiKey(System.getenv("MISTRAL_API_KEY"))\n'
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
                '    const char* api_key = std::getenv("MISTRAL_API_KEY");\n'
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

    def _embedding_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["MISTRAL_API_KEY"],\n'
                f'    base_url="{_BASE_URL}",\n'
                ')\n\n'
                'response = client.embeddings.create(\n'
                f'    model="{model}",\n'
                '    input="Hello!",\n'
                ')\n'
                'print(response.data[0].embedding[:5])\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const response = await client.embeddings.create({\n'
                f'  model: "{model}",\n'
                '  input: "Hello!",\n'
                '});\n'
                'console.log(response.data[0].embedding.slice(0, 5));\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
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
                'OpenAIClient client = OpenAIOkHttpClient.builder()\n'
                '    .apiKey(System.getenv("MISTRAL_API_KEY"))\n'
                f'    .baseUrl("{_BASE_URL}")\n'
                '    .build();\n\n'
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
                '    const char* api_key = std::getenv("MISTRAL_API_KEY");\n'
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
                f'    curl_easy_setopt(curl, CURLOPT_URL, "{_BASE_URL}/embeddings");\n'
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

    def _transcription_snippet(self, model: str, language: str) -> str:
        snippets = {
            "python": (
                'import os\n'
                'from openai import OpenAI\n\n'
                'client = OpenAI(\n'
                '    api_key=os.environ["MISTRAL_API_KEY"],\n'
                f'    base_url="{_BASE_URL}",\n'
                ')\n\n'
                'transcript = client.audio.transcriptions.create(\n'
                f'    model="{model}",\n'
                '    file=open("audio.mp3", "rb"),\n'
                ')\n'
                'print(transcript.text)\n'
            ),
            "typescript": (
                'import OpenAI from "openai";\n'
                'import fs from "fs";\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
                'const transcript = await client.audio.transcriptions.create({\n'
                f'  model: "{model}",\n'
                '  file: fs.createReadStream("audio.mp3"),\n'
                '});\n'
                'console.log(transcript.text);\n'
            ),
            "javascript": (
                'const OpenAI = require("openai");\n'
                'const fs = require("fs");\n\n'
                'const client = new OpenAI({\n'
                '  apiKey: process.env.MISTRAL_API_KEY,\n'
                f'  baseURL: "{_BASE_URL}",\n'
                '});\n\n'
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
                'OpenAIClient client = OpenAIOkHttpClient.builder()\n'
                '    .apiKey(System.getenv("MISTRAL_API_KEY"))\n'
                f'    .baseUrl("{_BASE_URL}")\n'
                '    .build();\n\n'
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
                '    const char* api_key = std::getenv("MISTRAL_API_KEY");\n'
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
                f'    curl_easy_setopt(curl, CURLOPT_URL,\n'
                f'        "{_BASE_URL}/audio/transcriptions");\n'
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

    def _tts_snippet(self, model: str, language: str) -> str:
        """Voxtral TTS is REST (POST /v1/audio/speech); voice cloning uses a
        short reference audio prompt, which OpenAI's audio.speech schema does
        not model — so the snippet shows Mistral's own endpoint directly."""
        url = f"{_BASE_URL}/audio/speech"
        # JSON object literal valid simultaneously as a Python dict, a JS
        # object, and JSON so it embeds directly in every language.
        body = (
            f'{{"model": "{model}", "input": "Hello from Mistral Voxtral TTS!"}}'
        )
        snippets = {
            "python": (
                'import os, json, urllib.request\n\n'
                f'body = json.dumps({body}).encode()\n'
                'req = urllib.request.Request(\n'
                f'    "{url}",\n'
                '    data=body,\n'
                '    headers={\n'
                '        "Authorization": f"Bearer {os.environ[\'MISTRAL_API_KEY\']}",\n'
                '        "Content-Type": "application/json",\n'
                '    },\n'
                ')\n'
                '# Returns audio bytes; add a "voice" reference for zero-shot cloning.\n'
                'audio = urllib.request.urlopen(req).read()\n'
                'open("speech.mp3", "wb").write(audio)\n'
            ),
            "typescript": (
                f'const res = await fetch("{url}", {{\n'
                '  method: "POST",\n'
                '  headers: {\n'
                '    Authorization: `Bearer ${process.env.MISTRAL_API_KEY}`,\n'
                '    "Content-Type": "application/json",\n'
                '  },\n'
                f'  body: JSON.stringify({body}),\n'
                '});\n'
                '// Returns audio bytes; add a "voice" reference for zero-shot cloning.\n'
                'const audio = Buffer.from(await res.arrayBuffer());\n'
                'require("fs").writeFileSync("speech.mp3", audio);\n'
            ),
            "javascript": (
                f'const res = await fetch("{url}", {{\n'
                '  method: "POST",\n'
                '  headers: {\n'
                '    Authorization: `Bearer ${process.env.MISTRAL_API_KEY}`,\n'
                '    "Content-Type": "application/json",\n'
                '  },\n'
                f'  body: JSON.stringify({body}),\n'
                '});\n'
                '// Returns audio bytes; add a "voice" reference for zero-shot cloning.\n'
                'const audio = Buffer.from(await res.arrayBuffer());\n'
                'require("fs").writeFileSync("speech.mp3", audio);\n'
            ),
            "java": (
                'import java.net.URI;\n'
                'import java.net.http.HttpClient;\n'
                'import java.net.http.HttpRequest;\n'
                'import java.net.http.HttpResponse;\n'
                'import java.nio.file.Files;\n'
                'import java.nio.file.Path;\n\n'
                'HttpClient client = HttpClient.newHttpClient();\n'
                f'String body = """\n    {body}""";\n'
                'HttpRequest request = HttpRequest.newBuilder()\n'
                f'    .uri(URI.create("{url}"))\n'
                '    .header("Authorization", "Bearer " + System.getenv("MISTRAL_API_KEY"))\n'
                '    .header("Content-Type", "application/json")\n'
                '    .POST(HttpRequest.BodyPublishers.ofString(body))\n'
                '    .build();\n'
                'HttpResponse<byte[]> response =\n'
                '    client.send(request, HttpResponse.BodyHandlers.ofByteArray());\n'
                '// Returns audio bytes; add a "voice" reference for zero-shot cloning.\n'
                'Files.write(Path.of("speech.mp3"), response.body());\n'
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
                '    const char* api_key = std::getenv("MISTRAL_API_KEY");\n'
                '    CURL* curl = curl_easy_init();\n\n'
                f'    json body = json::parse(R"({body})");\n\n'
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
                '    // Returns audio bytes; add a "voice" reference for zero-shot cloning.\n'
                '    std::ofstream("speech.mp3", std::ios::binary) << response;\n'
                '    return 0;\n'
                '}\n'
            ),
        }
        return snippets.get(language, snippets["python"])
