"""Tests for the discovery and selector modules."""

import dataclasses
import os
from unittest.mock import patch

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.selector import select_provider, Selection
from llm_api_search.providers.base import (
    ProviderInfo, SUPPORTED_LANGUAGES, ModelInfo, ModelType,
    TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, MusicModelInfo, VideoModelInfo,
)


def test_list_providers():
    names = list_providers()
    assert "anthropic" in names
    assert "google" in names
    assert "openai" in names


def test_discover_provider_static():
    info = discover_provider("anthropic", live=False)
    assert isinstance(info, ProviderInfo)
    assert info.name == "Anthropic (Claude)"
    assert len(info.models) > 0
    assert info.sdk_package == "anthropic"


def test_discover_all_static():
    results = discover(live=False)
    assert set(results.keys()) == {"anthropic", "google", "openai", "inception", "deepseek", "zai"}
    for key, info in results.items():
        assert isinstance(info, ProviderInfo)
        assert len(info.models) > 0


def test_discover_subset():
    results = discover(live=False, providers=["anthropic", "openai"])
    assert set(results.keys()) == {"anthropic", "openai"}


def test_discover_provider_unknown():
    try:
        discover_provider("unknown_provider", live=False)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_select_provider_programmatic():
    sel = select_provider("anthropic", live=False)
    assert isinstance(sel, Selection)
    assert sel.provider_key == "anthropic"
    assert sel.model.model_id == sel.provider_info.models[0].model_id
    assert "anthropic" in sel.connection_snippet.lower() or "import" in sel.connection_snippet


def test_select_provider_with_model():
    sel = select_provider("openai", model_id="gpt-5.4", live=False)
    assert sel.model.model_id == "gpt-5.4"
    assert "gpt-5.4" in sel.connection_snippet


def test_select_provider_unknown_model():
    try:
        select_provider("anthropic", model_id="nonexistent-model", live=False)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_provider_summaries():
    results = discover(live=False)
    for info in results.values():
        summary = info.summary()
        assert info.name in summary
        assert info.api_base_url in summary


def test_connection_snippets_all_providers():
    for key in list_providers():
        sel = select_provider(key, live=False)
        assert len(sel.connection_snippet) > 20
        assert sel.model.model_id in sel.connection_snippet


def test_provider_info_fields():
    """Verify all providers have required fields populated."""
    results = discover(live=False)
    for key, info in results.items():
        assert info.api_base_url.startswith("https://"), f"{key} missing https URL"
        assert info.auth_env_var, f"{key} missing auth_env_var"
        assert info.auth_header, f"{key} missing auth_header"
        assert info.sdk_package, f"{key} missing sdk_package"
        assert info.sdk_install.startswith("pip install"), f"{key} bad sdk_install"
        assert info.documentation_url.startswith("https://"), f"{key} missing docs URL"


def test_model_info_fields():
    """Verify static models have key fields set."""
    results = discover(live=False)
    for key, info in results.items():
        for m in info.models:
            assert m.model_id, f"{key}: model missing model_id"
            assert m.display_name, f"{key}: model missing display_name"


# --- Multi-language snippet tests ---


def test_connection_snippet_typescript():
    """Verify TypeScript snippets use TS syntax (import or fetch)."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="typescript")
        # Providers with SDKs use import; others use fetch API directly
        assert "import" in sel.connection_snippet or "fetch" in sel.connection_snippet
        # TS snippets should not use Python syntax
        assert "print(" not in sel.connection_snippet


def test_connection_snippet_javascript():
    """Verify JavaScript snippets use JS syntax (require or fetch)."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="javascript")
        # Providers with SDKs use require; others use fetch API directly
        assert "require(" in sel.connection_snippet or "fetch" in sel.connection_snippet
        assert "console.log" in sel.connection_snippet


def test_connection_snippet_java():
    """Verify Java snippets use Java import syntax."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="java")
        # Providers with SDKs use com.*; others use java.net.http
        assert "import com." in sel.connection_snippet or "import java." in sel.connection_snippet
        assert "System.out.println" in sel.connection_snippet or "System.getenv" in sel.connection_snippet


def test_connection_snippet_cpp():
    """Verify C++ snippets use libcurl and include directives."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="cpp")
        assert "#include" in sel.connection_snippet
        assert "curl" in sel.connection_snippet.lower()


def test_connection_snippet_all_languages_all_providers():
    """Every provider should return a non-empty snippet for every supported language."""
    for key in list_providers():
        for lang in SUPPORTED_LANGUAGES:
            sel = select_provider(key, live=False, language=lang)
            assert len(sel.connection_snippet) > 20, (
                f"{key}/{lang}: snippet too short"
            )


def test_connection_snippet_model_id_in_all_languages():
    """Model ID should appear in snippets for all languages."""
    model = "claude-sonnet-4-6"
    for lang in SUPPORTED_LANGUAGES:
        sel = select_provider("anthropic", model_id=model, live=False, language=lang)
        assert model in sel.connection_snippet, (
            f"Model ID missing from {lang} snippet"
        )


def test_sdk_installs_per_language():
    """Verify sdk_installs contains entries for multiple languages."""
    results = discover(live=False)
    for key, info in results.items():
        assert "python" in info.sdk_installs, f"{key} missing python sdk_install"
        assert "typescript" in info.sdk_installs, f"{key} missing typescript sdk_install"
        assert "java" in info.sdk_installs, f"{key} missing java sdk_install"
        assert "cpp" in info.sdk_installs, f"{key} missing cpp sdk_install"


def test_sdk_install_for():
    """Verify sdk_install_for returns the correct install command."""
    info = discover_provider("anthropic", live=False)
    assert info.sdk_install_for("python") == "pip install anthropic"
    assert "npm" in info.sdk_install_for("typescript")
    assert info.sdk_install_for("cpp").lower().count("libcurl") or "REST" in info.sdk_install_for("cpp")


def test_unknown_language_falls_back_to_python():
    """Unknown language should fall back to Python snippet."""
    sel = select_provider("anthropic", live=False, language="rust")
    python_sel = select_provider("anthropic", live=False, language="python")
    assert sel.connection_snippet == python_sel.connection_snippet


def test_summary_includes_pricing():
    """Provider summary should show pricing for each model."""
    results = discover(live=False)
    for key, info in results.items():
        summary = info.summary()
        for m in info.models:
            if isinstance(m, TextModelInfo) and m.input_cost_per_mtok is not None:
                assert f"${m.input_cost_per_mtok:.2f}" in summary, (
                    f"{key}/{m.model_id}: pricing missing from summary"
                )
            elif isinstance(m, ImageModelInfo) and m.cost_per_image is not None:
                assert f"${m.cost_per_image:.3f}" in summary, (
                    f"{key}/{m.model_id}: pricing missing from summary"
                )


def test_model_pricing_fields():
    """All static GA models should have type-appropriate pricing set.

    Preview models are exempted because providers routinely ship them with
    no public pricing; the live-update workflow lands them here and the
    pricing is filled in once it's announced.
    """
    results = discover(live=False)
    for key, info in results.items():
        for m in info.models:
            if "preview" in m.model_id:
                continue
            if isinstance(m, TextModelInfo):
                assert m.input_cost_per_mtok is not None, f"{key}/{m.model_id}: missing input_cost_per_mtok"
                assert m.output_cost_per_mtok is not None, f"{key}/{m.model_id}: missing output_cost_per_mtok"
                assert m.input_cost_per_mtok >= 0
                assert m.output_cost_per_mtok >= 0
                assert m.output_cost_per_mtok >= m.input_cost_per_mtok, f"{key}/{m.model_id}: output < input"
            elif isinstance(m, ImageModelInfo):
                assert m.cost_per_image is not None, f"{key}/{m.model_id}: missing cost_per_image"
                assert m.cost_per_image >= 0
            elif isinstance(m, AudioTTSModelInfo):
                has_char = m.cost_per_mchars is not None
                has_tok = m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None
                assert has_char or has_tok, f"{key}/{m.model_id}: missing TTS pricing"
            elif isinstance(m, AudioTranscriptionModelInfo):
                assert m.cost_per_minute is not None, f"{key}/{m.model_id}: missing cost_per_minute"
                assert m.cost_per_minute >= 0
            elif isinstance(m, EmbeddingModelInfo):
                assert m.input_cost_per_mtok is not None, f"{key}/{m.model_id}: missing input_cost_per_mtok"
                assert m.input_cost_per_mtok >= 0
            elif isinstance(m, MusicModelInfo):
                assert m.cost_per_second is not None, f"{key}/{m.model_id}: missing cost_per_second"
                assert m.cost_per_second >= 0
            elif isinstance(m, VideoModelInfo):
                has_sec = m.cost_per_second is not None
                has_vid = m.cost_per_video is not None
                assert has_sec or has_vid, f"{key}/{m.model_id}: missing video pricing"
                if has_sec:
                    assert m.cost_per_second >= 0
                if has_vid:
                    assert m.cost_per_video >= 0


def test_video_model_cost_per_video_formats():
    from llm_api_search.providers.base import VideoModelInfo, _format_model_cost
    m = VideoModelInfo(model_id="x", display_name="X", cost_per_video=0.20)
    assert _format_model_cost(m) == " | $0.20/video"
    # cost_per_second still works for models priced that way
    m2 = VideoModelInfo(model_id="y", display_name="Y", cost_per_second=0.35)
    assert _format_model_cost(m2) == " | $0.35/sec"


# --- ModelType enum and subclass tests ---


def test_model_type_enum():
    assert ModelType.TEXT.value == "text"
    assert ModelType.IMAGE.value == "image"
    assert ModelType.AUDIO_TTS.value == "audio_tts"
    assert ModelType.AUDIO_TRANSCRIPTION.value == "audio_transcription"
    assert ModelType.EMBEDDING.value == "embedding"


def test_text_model_info_has_model_type():
    m = TextModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.TEXT
    assert isinstance(m, ModelInfo)


def test_image_model_info_has_model_type():
    m = ImageModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.IMAGE
    assert isinstance(m, ModelInfo)


def test_audio_tts_model_info_has_model_type():
    m = AudioTTSModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.AUDIO_TTS
    assert isinstance(m, ModelInfo)


def test_audio_transcription_model_info_has_model_type():
    m = AudioTranscriptionModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.AUDIO_TRANSCRIPTION
    assert isinstance(m, ModelInfo)


def test_embedding_model_info_has_model_type():
    m = EmbeddingModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.EMBEDDING
    assert isinstance(m, ModelInfo)


def test_mcp_list_models_type_filter():
    """MCP llm_list_models tool should support model_type filtering."""
    from mcp_servers.llm_api_search import llm_list_models
    all_models = llm_list_models("openai")
    # All models should have model_type field
    assert all("model_type" in m for m in all_models)
    text_models = llm_list_models("openai", model_type="text")
    assert all(m["model_type"] == "text" for m in text_models)
    assert len(text_models) > 0
    # OpenAI has non-text models, so all_models should be more than text_models.
    assert len(all_models) > len(text_models)


def test_summary_groups_by_model_type():
    """Summary should group models by type when multiple types exist."""
    info = ProviderInfo(
        name="TestProvider",
        api_base_url="https://test.com",
        api_version=None,
        auth_env_var="TEST_KEY",
        auth_header="Authorization",
        models=[
            TextModelInfo(model_id="t1", display_name="T1", input_cost_per_mtok=1.0, output_cost_per_mtok=5.0, context_window=100_000),
            ImageModelInfo(model_id="i1", display_name="I1", cost_per_image=0.04),
        ],
    )
    summary = info.summary()
    assert "Models (text):" in summary
    assert "Models (image):" in summary
    assert "$1.00/$5.00 per 1M tok" in summary
    assert "$0.040/image" in summary


# --- OpenAI non-text model tests ---


def test_openai_has_image_models():
    info = discover_provider("openai", live=False)
    image_models = [m for m in info.models if isinstance(m, ImageModelInfo)]
    assert len(image_models) >= 3
    ids = [m.model_id for m in image_models]
    assert "gpt-image-1.5" in ids
    assert "gpt-image-1" in ids
    assert "gpt-image-1-mini" in ids
    for m in image_models:
        assert m.cost_per_image is not None
        assert len(m.supported_sizes) > 0


def test_openai_has_tts_models():
    info = discover_provider("openai", live=False)
    tts_models = [m for m in info.models if isinstance(m, AudioTTSModelInfo)]
    assert len(tts_models) >= 3
    ids = [m.model_id for m in tts_models]
    assert "gpt-4o-mini-tts" in ids
    assert "tts-1" in ids
    for m in tts_models:
        assert len(m.supported_voices) > 0
        assert len(m.supported_output_formats) > 0


def test_openai_has_transcription_models():
    info = discover_provider("openai", live=False)
    trans_models = [m for m in info.models if isinstance(m, AudioTranscriptionModelInfo)]
    assert len(trans_models) >= 3
    ids = [m.model_id for m in trans_models]
    assert "gpt-4o-transcribe" in ids
    assert "whisper-1" in ids
    for m in trans_models:
        assert m.cost_per_minute is not None
        assert len(m.supported_input_formats) > 0


def test_openai_has_embedding_models():
    info = discover_provider("openai", live=False)
    emb_models = [m for m in info.models if isinstance(m, EmbeddingModelInfo)]
    assert len(emb_models) >= 2
    ids = [m.model_id for m in emb_models]
    assert "text-embedding-3-large" in ids
    assert "text-embedding-3-small" in ids
    for m in emb_models:
        assert m.input_cost_per_mtok is not None
        assert m.dimensions is not None


def test_openai_image_snippet():
    sel = select_provider("openai", model_id="gpt-image-1", live=False)
    assert "images" in sel.connection_snippet.lower() or "generate" in sel.connection_snippet.lower()
    assert "gpt-image-1" in sel.connection_snippet


def test_openai_tts_snippet():
    sel = select_provider("openai", model_id="tts-1", live=False)
    assert "speech" in sel.connection_snippet.lower() or "audio" in sel.connection_snippet.lower()
    assert "tts-1" in sel.connection_snippet


def test_openai_transcription_snippet():
    sel = select_provider("openai", model_id="whisper-1", live=False)
    assert "transcri" in sel.connection_snippet.lower()
    assert "whisper-1" in sel.connection_snippet


def test_openai_embedding_snippet():
    sel = select_provider("openai", model_id="text-embedding-3-small", live=False)
    assert "embed" in sel.connection_snippet.lower()
    assert "text-embedding-3-small" in sel.connection_snippet


# --- Google non-text model tests ---


def test_google_has_image_models():
    info = discover_provider("google", live=False)
    image_models = [m for m in info.models if isinstance(m, ImageModelInfo)]
    assert len(image_models) >= 3
    ids = [m.model_id for m in image_models]
    assert "imagen-4.0-generate-001" in ids
    for m in image_models:
        assert m.cost_per_image is not None


def test_google_has_tts_models():
    info = discover_provider("google", live=False)
    tts_models = [m for m in info.models if isinstance(m, AudioTTSModelInfo)]
    assert len(tts_models) >= 2
    ids = [m.model_id for m in tts_models]
    assert "gemini-2.5-flash-preview-tts" in ids
    assert "gemini-2.5-pro-preview-tts" in ids
    for m in tts_models:
        if "preview" in m.model_id and m.input_cost_per_mtok is None:
            continue
        assert m.input_cost_per_mtok is not None
        assert len(m.supported_voices) > 0


def test_google_has_embedding_models():
    info = discover_provider("google", live=False)
    emb_models = [m for m in info.models if isinstance(m, EmbeddingModelInfo)]
    assert len(emb_models) >= 2
    ids = [m.model_id for m in emb_models]
    assert "gemini-embedding-001" in ids
    assert "gemini-embedding-2-preview" in ids
    for m in emb_models:
        assert m.input_cost_per_mtok is not None
        assert m.dimensions is not None


def test_google_image_snippet():
    sel = select_provider("google", model_id="imagen-4.0-generate-001", live=False)
    assert "imagen-4.0-generate-001" in sel.connection_snippet
    assert "image" in sel.connection_snippet.lower()


def test_google_tts_snippet():
    sel = select_provider("google", model_id="gemini-2.5-flash-preview-tts", live=False)
    assert "gemini-2.5-flash-preview-tts" in sel.connection_snippet
    assert "audio" in sel.connection_snippet.lower() or "speech" in sel.connection_snippet.lower()


def test_google_embedding_snippet():
    sel = select_provider("google", model_id="gemini-embedding-001", live=False)
    assert "gemini-embedding-001" in sel.connection_snippet
    assert "embed" in sel.connection_snippet.lower()


# --- Comprehensive multi-type cross-cutting tests ---


def test_list_models_filter_by_type():
    """model_type filter should return correct subset."""
    info = discover_provider("openai", live=False)
    all_models = info.models
    text_only = [m for m in all_models if m.model_type.value == "text"]
    image_only = [m for m in all_models if m.model_type.value == "image"]
    assert len(all_models) > len(text_only)
    assert len(text_only) > 0
    assert len(image_only) > 0


def test_list_models_no_filter_returns_all():
    """Without model_type filter, all models should be returned."""
    info = discover_provider("openai", live=False)
    # 14 text + 3 image + 3 tts + 3 transcription + 2 embedding = 25
    assert len(info.models) >= 25


def test_connection_snippet_all_types_all_languages():
    """Every model type should produce a non-empty snippet in every language."""
    test_cases = [
        ("openai", "gpt-5.4"),
        ("openai", "gpt-image-1"),
        ("openai", "tts-1"),
        ("openai", "whisper-1"),
        ("openai", "text-embedding-3-small"),
        ("google", "gemini-2.5-flash"),
        ("google", "imagen-4.0-generate-001"),
        ("google", "gemini-2.5-flash-preview-tts"),
        ("google", "gemini-embedding-001"),
        ("google", "veo-3.1"),
    ]
    for provider, model_id in test_cases:
        for lang in SUPPORTED_LANGUAGES:
            sel = select_provider(provider, model_id=model_id, live=False, language=lang)
            assert len(sel.connection_snippet) > 20, (
                f"{provider}/{model_id}/{lang}: snippet too short"
            )
            assert model_id in sel.connection_snippet, (
                f"{provider}/{model_id}/{lang}: model_id missing from snippet"
            )


def test_dataclasses_asdict_includes_type_specific_fields():
    """dataclasses.asdict should include all subclass-specific fields."""
    info = discover_provider("openai", live=False)
    for m in info.models:
        d = dataclasses.asdict(m)
        assert "model_type" in d
        assert "model_id" in d
        if isinstance(m, TextModelInfo):
            assert "context_window" in d
            assert "input_cost_per_mtok" in d
        elif isinstance(m, ImageModelInfo):
            assert "cost_per_image" in d
            assert "supported_sizes" in d
        elif isinstance(m, AudioTTSModelInfo):
            assert "supported_voices" in d
        elif isinstance(m, AudioTranscriptionModelInfo):
            assert "cost_per_minute" in d
            assert "supported_input_formats" in d
        elif isinstance(m, EmbeddingModelInfo):
            assert "dimensions" in d
            assert "input_cost_per_mtok" in d


def test_mcp_list_models_filter_multiple_types():
    """MCP tool should filter correctly across model types."""
    from mcp_servers.llm_api_search import llm_list_models
    all_models = llm_list_models("openai")
    text_models = llm_list_models("openai", model_type="text")
    image_models = llm_list_models("openai", model_type="image")
    embedding_models = llm_list_models("openai", model_type="embedding")
    assert len(all_models) == len(text_models) + len(image_models) + len(llm_list_models("openai", model_type="audio_tts")) + len(llm_list_models("openai", model_type="audio_transcription")) + len(embedding_models)
    assert len(image_models) >= 3
    assert len(embedding_models) >= 2


def test_google_has_video_models():
    info = discover_provider("google", live=False)
    video_models = [m for m in info.models if isinstance(m, VideoModelInfo)]
    assert len(video_models) >= 5
    ids = [m.model_id for m in video_models]
    assert "veo-3.1" in ids
    assert "veo-3" in ids
    assert "veo-2" in ids
    for m in video_models:
        if "preview" in m.model_id:
            continue
        assert m.cost_per_second is not None
        assert len(m.supported_resolutions) > 0


def test_google_video_snippet():
    sel = select_provider("google", model_id="veo-3.1", live=False)
    assert "veo-3.1" in sel.connection_snippet
    assert "video" in sel.connection_snippet.lower()


def test_google_has_music_model():
    info = discover_provider("google", live=False)
    music_models = [m for m in info.models if isinstance(m, MusicModelInfo)]
    assert len(music_models) >= 1
    ids = [m.model_id for m in music_models]
    assert "lyria-2" in ids
    lyria_2 = next(m for m in music_models if m.model_id == "lyria-2")
    assert lyria_2.cost_per_second == 0.002


def test_google_music_snippet():
    sel = select_provider("google", model_id="lyria-2", live=False)
    assert "lyria-2" in sel.connection_snippet
    assert "audio" in sel.connection_snippet.lower() or "music" in sel.connection_snippet.lower()


def test_google_has_opensource_embeddings():
    info = discover_provider("google", live=False)
    emb_models = [m for m in info.models if isinstance(m, EmbeddingModelInfo)]
    ids = [m.model_id for m in emb_models]
    assert "multilingual-e5-small" in ids
    assert "multilingual-e5-large" in ids


def test_zai_static_info():
    from llm_api_search.providers import PROVIDERS
    from llm_api_search.providers.base import (
        TextModelInfo, ImageModelInfo, VideoModelInfo,
    )
    info = PROVIDERS["zai"]().get_static_info()
    assert info.name == "Z.ai (GLM)"
    assert info.api_base_url == "https://api.z.ai/api/paas/v4"
    assert info.auth_env_var == "ZAI_API_KEY"
    # Text-first, glm-5.2 default.
    assert isinstance(info.models[0], TextModelInfo)
    assert info.models[0].model_id == "glm-5.2"
    ids = {m.model_id for m in info.models}
    assert {"glm-5.2", "glm-5v-turbo", "cogview-4", "cogvideox-3"} <= ids
    # Vision flag on the vision models.
    vision = {m.model_id for m in info.models
              if isinstance(m, TextModelInfo) and m.supports_vision}
    assert vision == {"glm-5v-turbo", "glm-4.6v-flash"}
    # Correct subtypes for image/video.
    by_id = {m.model_id: m for m in info.models}
    assert isinstance(by_id["cogview-4"], ImageModelInfo)
    assert by_id["cogview-4"].cost_per_image == 0.01
    assert isinstance(by_id["cogvideox-3"], VideoModelInfo)
    assert by_id["cogvideox-3"].cost_per_video == 0.20


def test_zai_image_snippet_all_languages():
    from llm_api_search.providers import PROVIDERS
    from llm_api_search.providers.base import SUPPORTED_LANGUAGES
    p = PROVIDERS["zai"]()
    for lang in SUPPORTED_LANGUAGES:
        snip = p.get_connection_snippet("cogview-4", lang)
        assert "cogview-4" in snip, f"{lang}: model id missing"
        assert "image" in snip.lower(), f"{lang}: no image reference"
        assert len(snip) > 20


def test_zai_video_snippet_all_languages():
    from llm_api_search.providers import PROVIDERS
    from llm_api_search.providers.base import SUPPORTED_LANGUAGES
    p = PROVIDERS["zai"]()
    for lang in SUPPORTED_LANGUAGES:
        snip = p.get_connection_snippet("cogvideox-3", lang)
        assert "cogvideox-3" in snip, f"{lang}: model id missing"
        assert "videos/generations" in snip, f"{lang}: no video endpoint"
        assert len(snip) > 20


def test_zai_video_snippet_no_python_syntax_in_ts():
    from llm_api_search.providers import PROVIDERS
    p = PROVIDERS["zai"]()
    ts = p.get_connection_snippet("cogvideox-3", "typescript")
    assert "print(" not in ts
