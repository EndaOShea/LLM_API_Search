"""Tests for the MiniMax provider (direct import; also runs post-registration)."""

from llm_api_search.providers.minimax import MiniMaxProvider, _STATIC_MODELS, _BASE_URL
from llm_api_search.providers.base import (
    SUPPORTED_LANGUAGES, TextModelInfo, ImageModelInfo, VideoModelInfo,
    AudioTTSModelInfo, MusicModelInfo,
)


def test_static_info_basics():
    info = MiniMaxProvider().get_static_info()
    assert info.name == "MiniMax"
    assert info.api_base_url == "https://api.minimax.io/v1"
    assert info.auth_env_var == "MINIMAX_API_KEY"
    assert info.sdk_installs["python"] == "pip install openai"
    # Text-first; default is MiniMax-M3.
    assert isinstance(info.models[0], TextModelInfo)
    assert info.models[0].model_id == "MiniMax-M3"


def test_covers_all_five_model_types():
    by_id = {m.model_id: m for m in _STATIC_MODELS}
    assert isinstance(by_id["MiniMax-M3"], TextModelInfo)
    assert isinstance(by_id["image-01"], ImageModelInfo)
    assert isinstance(by_id["MiniMax-Hailuo-2.3"], VideoModelInfo)
    assert isinstance(by_id["speech-2.8-hd"], AudioTTSModelInfo)
    assert isinstance(by_id["Music-2.6"], MusicModelInfo)
    # Pricing encoded per the plan's canonical table.
    assert by_id["MiniMax-M3"].input_cost_per_mtok == 0.30
    assert by_id["MiniMax-M3"].output_cost_per_mtok == 1.20
    assert by_id["image-01"].cost_per_image == 0.0035
    assert by_id["MiniMax-Hailuo-2.3"].cost_per_video == 0.28
    assert by_id["speech-2.8-hd"].cost_per_mchars == 100.0
    assert by_id["Music-2.6"].cost_per_second == 0.0005


def test_pricing_invariant_output_ge_input():
    for m in _STATIC_MODELS:
        if isinstance(m, TextModelInfo):
            assert m.output_cost_per_mtok >= m.input_cost_per_mtok


def test_text_snippet_all_languages():
    p = MiniMaxProvider()
    for lang in SUPPORTED_LANGUAGES:
        snip = p.get_connection_snippet("MiniMax-M3", lang)
        assert "MiniMax-M3" in snip
        assert _BASE_URL in snip
        assert len(snip) > 20
    assert "print(" not in p.get_connection_snippet("MiniMax-M3", "typescript")


def test_media_snippets_all_languages():
    p = MiniMaxProvider()
    cases = {
        "image-01": "/v1/image_generation",
        "MiniMax-Hailuo-2.3": "/v1/video_generation",
        "speech-2.8-hd": "/v1/t2a_v2",
        "Music-2.6": "/v1/music_generation",
    }
    for model_id, path in cases.items():
        for lang in SUPPORTED_LANGUAGES:
            snip = p.get_connection_snippet(model_id, lang)
            assert model_id in snip, f"{model_id}/{lang}: id missing"
            assert path in snip, f"{model_id}/{lang}: endpoint missing"
            assert len(snip) > 20


def test_fetch_live_returns_static_without_key():
    # No /v1/models endpoint + no key → curated static is returned unchanged.
    info = MiniMaxProvider().fetch_live_models()
    assert [m.model_id for m in info.models] == [m.model_id for m in _STATIC_MODELS]


def test_rate_limits_cover_every_static_model():
    from llm_api_search.providers.rate_limits.minimax import RATE_LIMITS
    static_ids = {m.model_id for m in _STATIC_MODELS}
    assert static_ids <= set(RATE_LIMITS), "every model needs a rate-limit entry"
    m3 = RATE_LIMITS["MiniMax-M3"]["default"]
    assert m3.requests_per_minute == 200
    assert m3.tokens_per_minute == 10_000_000
    assert RATE_LIMITS["MiniMax-M2.7"]["default"].requests_per_minute == 500
    assert RATE_LIMITS["MiniMax-Hailuo-2.3"]["default"].requests_per_minute == 5


def test_thinking_configs_are_toggle_for_text_models():
    from llm_api_search.providers.thinking.minimax import THINKING_CONFIGS
    from llm_api_search.providers.base import ThinkingMode
    for mid in ("MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"):
        tc = THINKING_CONFIGS[mid]
        assert tc.supported is True
        assert tc.mode is ThinkingMode.TOGGLE
        assert tc.parameter == "thinking"
        assert tc.can_disable is True
        assert tc.default_level is None
    # Keys must be real static model ids (guards typos).
    static_ids = {m.model_id for m in _STATIC_MODELS}
    assert set(THINKING_CONFIGS) <= static_ids
