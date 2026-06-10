"""Anthropic thinking configurations.

Source: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
Verified: 2026-06-04
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_ADAPTIVE_NOTE = (
    "Requires thinking={type:'adaptive'}. effort set via output_config.effort. "
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "claude-fable-5": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400). "
                               "display defaults to 'omitted'.",
    ),
    "claude-opus-4-8": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400). "
                               "display defaults to 'omitted'.",
    ),
    "claude-opus-4-7": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400).",
    ),
    "claude-opus-4-6": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "max"],
        default_level="high", can_disable=True,
        notes=_ADAPTIVE_NOTE + "Legacy thinking={type:'enabled',budget_tokens} still works but is deprecated.",
    ),
    "claude-sonnet-4-6": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "max"],
        default_level="high", can_disable=True,
        notes=_ADAPTIVE_NOTE + "Legacy budget_tokens deprecated. Manual-mode interleaved "
                               "thinking via interleaved-thinking-2025-05-14 beta header.",
    ),
}
