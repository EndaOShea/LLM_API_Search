"""Tests for per-model thinking configuration."""

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
