"""Tests for per-model thinking configuration."""

import pytest

from llm_api_search.providers import PROVIDERS, get_thinking_config
from llm_api_search.providers.base import TextModelInfo, ThinkingConfig, ThinkingMode


def test_thinking_config_defaults_to_unsupported_none():
    tc = ThinkingConfig()
    assert tc.supported is False
    assert tc.mode is ThinkingMode.NONE
    assert tc.levels == []
    assert tc.parameter is None


def test_thinking_config_effort_levels():
    tc = ThinkingConfig(
        supported=True,
        mode=ThinkingMode.EFFORT_LEVELS,
        parameter="reasoning_effort",
        levels=["low", "medium", "high"],
        default_level="medium",
        can_disable=True,
    )
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.default_level in tc.levels


def test_get_thinking_config_unknown_provider_raises():
    with pytest.raises(KeyError):
        get_thinking_config("nope")


def test_get_thinking_config_miss_returns_default_none():
    # A model with no entry must resolve to the default (supported=False, NONE),
    # never {} or KeyError.
    result = get_thinking_config("anthropic", "gpt-4o")
    assert "gpt-4o" in result
    assert result["gpt-4o"].supported is False
    assert result["gpt-4o"].mode is ThinkingMode.NONE


def test_get_thinking_config_all_for_provider_is_dict():
    result = get_thinking_config("anthropic")
    assert isinstance(result, dict)


def test_anthropic_opus_48_effort_levels():
    tc = get_thinking_config("anthropic", "claude-opus-4-8")["claude-opus-4-8"]
    assert tc.supported is True
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "output_config.effort"
    assert tc.levels == ["low", "medium", "high", "xhigh", "max"]
    assert tc.default_level == "high"


def test_anthropic_sonnet_46_has_no_xhigh():
    tc = get_thinking_config("anthropic", "claude-sonnet-4-6")["claude-sonnet-4-6"]
    assert "xhigh" not in tc.levels
    assert "max" in tc.levels


def test_openai_gpt5_effort():
    tc = get_thinking_config("openai", "gpt-5.5")["gpt-5.5"]
    assert tc.supported is True
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "reasoning.effort"
    assert "medium" in tc.levels
    assert tc.default_level == "medium"


def test_openai_chat_latest_is_not_thinking():
    tc = get_thinking_config("openai", "gpt-5.1-chat-latest")["gpt-5.1-chat-latest"]
    assert tc.supported is False
    assert tc.mode is ThinkingMode.NONE


def test_openai_gpt4o_is_not_thinking():
    tc = get_thinking_config("openai", "gpt-4o")["gpt-4o"]
    assert tc.supported is False


def test_google_25_flash_token_budget():
    tc = get_thinking_config("google", "gemini-2.5-flash")["gemini-2.5-flash"]
    assert tc.supported is True
    assert tc.mode is ThinkingMode.TOKEN_BUDGET
    assert tc.parameter == "thinkingBudget"
    assert tc.max_budget == 24576
    assert tc.supports_dynamic is True
    assert tc.can_disable is True


def test_google_25_pro_cannot_disable():
    tc = get_thinking_config("google", "gemini-2.5-pro")["gemini-2.5-pro"]
    assert tc.can_disable is False
    assert tc.min_budget == 128
    assert tc.max_budget == 32768


def test_google_3_pro_effort_levels():
    tc = get_thinking_config("google", "gemini-3-pro-preview")["gemini-3-pro-preview"]
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "thinkingLevel"
    assert tc.default_level == "high"


def test_google_gemma_is_not_thinking():
    tc = get_thinking_config("google", "gemma-3-27b-it")["gemma-3-27b-it"]
    assert tc.supported is False


def test_deepseek_effort_and_toggle():
    tc = get_thinking_config("deepseek", "deepseek-v4-pro")["deepseek-v4-pro"]
    assert tc.supported is True
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "reasoning_effort"
    assert tc.levels == ["high", "max"]
    assert tc.can_disable is True


def test_inception_mercury_effort():
    tc = get_thinking_config("inception", "mercury-2")["mercury-2"]
    assert tc.supported is True
    assert tc.mode is ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "reasoning_effort"
    assert tc.levels == ["instant", "low", "medium", "high"]
    assert tc.default_level == "medium"


def test_zai_thinking_config():
    from llm_api_search.providers import get_thinking_config
    from llm_api_search.providers.base import ThinkingMode
    # glm-5.2 is the only GLM with graded reasoning_effort (z.ai docs: 5.2+).
    tc = get_thinking_config("zai", "glm-5.2")["glm-5.2"]
    assert tc.supported is True
    assert tc.mode == ThinkingMode.EFFORT_LEVELS
    assert tc.parameter == "reasoning_effort"
    assert tc.levels == ["high", "max"]
    assert tc.default_level == "max"
    assert tc.can_disable is True
    # Older thinking-capable GLMs expose only the on/off `thinking` toggle
    # (z.ai docs: `thinking` param is GLM-4.5+, reasoning_effort is 5.2+).
    tc46 = get_thinking_config("zai", "glm-4.6")["glm-4.6"]
    assert tc46.supported is True
    assert tc46.mode == ThinkingMode.TOGGLE
    assert tc46.parameter == "thinking"
    assert tc46.can_disable is True
    assert tc46.default_level is None
    # Non-thinking model resolves to the unsupported default.
    assert get_thinking_config("zai", "glm-4.5-flash")["glm-4.5-flash"].supported is False


# Models a human has confirmed are reasoning-capable. If one of these ever
# resolves to supported=False, its config was dropped/renamed — fail loudly.
_KNOWN_THINKING = {
    "anthropic": ["claude-fable-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6"],
    "openai": ["gpt-5", "gpt-5.5", "gpt-5.4", "o3", "o4-mini"],
    "google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview", "gemini-3.5-flash"],
    "deepseek": ["deepseek-v4-pro", "deepseek-v4-flash"],
    "inception": ["mercury-2", "mercury-edit", "mercury-edit-2"],
    "zai": ["glm-5.2", "glm-5.1", "glm-5", "glm-4.6", "glm-4.5-air", "glm-5v-turbo"],
}


def test_every_text_model_resolves_to_a_config():
    """Default-NONE contract: every text model gets a definitive ThinkingConfig."""
    for key, cls in PROVIDERS.items():
        info = cls().get_static_info()
        for m in info.models:
            if not isinstance(m, TextModelInfo):
                continue
            result = get_thinking_config(key, m.model_id)
            assert m.model_id in result, f"{key}/{m.model_id}: no config returned"
            assert isinstance(result[m.model_id], ThinkingConfig)


def test_known_thinking_models_are_supported():
    for provider, ids in _KNOWN_THINKING.items():
        for mid in ids:
            tc = get_thinking_config(provider, mid)[mid]
            assert tc.supported is True, f"{provider}/{mid}: expected supported=True"
            assert tc.mode is not ThinkingMode.NONE


def test_stored_configs_have_valid_fields():
    from llm_api_search.providers.thinking import PROVIDER_THINKING_CONFIGS
    for provider, configs in PROVIDER_THINKING_CONFIGS.items():
        for mid, tc in configs.items():
            if tc.mode is ThinkingMode.EFFORT_LEVELS:
                assert tc.levels, f"{provider}/{mid}: effort mode needs levels"
                assert tc.default_level in tc.levels, (
                    f"{provider}/{mid}: default_level {tc.default_level!r} not in levels"
                )
            elif tc.mode is ThinkingMode.TOKEN_BUDGET:
                assert tc.max_budget is not None, f"{provider}/{mid}: budget mode needs max_budget"


def test_mcp_thinking_tool_serializes_and_omits_empty():
    from mcp_servers.llm_api_search import llm_get_thinking_config
    result = llm_get_thinking_config("anthropic", "claude-opus-4-8")
    entry = result["claude-opus-4-8"]
    assert entry["supported"] is True
    assert entry["mode"] == "effort_levels"
    assert entry["levels"] == ["low", "medium", "high", "xhigh", "max"]
    # None / empty fields are omitted
    assert "min_budget" not in entry


def test_mcp_thinking_tool_non_thinking_model():
    from mcp_servers.llm_api_search import llm_get_thinking_config
    result = llm_get_thinking_config("openai", "gpt-4o")
    assert result["gpt-4o"]["supported"] is False
    assert result["gpt-4o"]["mode"] == "none"


def test_mcp_thinking_tool_keeps_can_disable_false_for_capable_models():
    # "thinking cannot be turned off" is a meaningful fact for always-on
    # models, so can_disable=False must survive serialization when supported.
    from mcp_servers.llm_api_search import llm_get_thinking_config
    for provider, mid in [("google", "gemini-3-pro-preview"), ("openai", "o3")]:
        entry = llm_get_thinking_config(provider, mid)[mid]
        assert entry["supported"] is True
        assert entry["can_disable"] is False


def test_no_orphan_thinking_config_keys():
    # Every key in the registry must be a real static model id (catches typos
    # and renames that would otherwise sit dead with no failure).
    from llm_api_search.providers.thinking import PROVIDER_THINKING_CONFIGS
    for provider, configs in PROVIDER_THINKING_CONFIGS.items():
        static_ids = {m.model_id for m in PROVIDERS[provider]().get_static_info().models}
        for mid in configs:
            assert mid in static_ids, f"{provider}/{mid}: not a known static model id"
