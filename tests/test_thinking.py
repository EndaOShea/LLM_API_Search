"""Tests for per-model thinking configuration."""

import pytest

from llm_api_search.providers import get_thinking_config
from llm_api_search.providers.base import ThinkingConfig, ThinkingMode


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
