"""Google Gemini thinking configurations.

Source: https://ai.google.dev/gemini-api/docs/thinking
Verified: 2026-06-04

Gemini 3.x -> thinkingLevel (effort). Gemini 2.5 -> thinkingBudget (tokens).
Specialized models (Gemma, audio, computer-use, deep-research, robotics-er
1.x, etc.) are non-reasoning and default to ThinkingMode.NONE on lookup.
Robotics-ER 2 is the exception: it thinks via the Gemini 3 thinkingLevel
knob (https://ai.google.dev/gemini-api/docs/robotics-overview).
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_LEVEL_NOTE = "Response pricing = output tokens + thinking tokens."


def _gemini3(default_level: str, can_disable: bool) -> ThinkingConfig:
    return ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="thinkingLevel",
        levels=["minimal", "low", "medium", "high"],
        default_level=default_level, can_disable=can_disable,
        notes=_LEVEL_NOTE,
    )


def _budget(min_b: int, max_b: int, can_disable: bool) -> ThinkingConfig:
    return ThinkingConfig(
        supported=True, mode=ThinkingMode.TOKEN_BUDGET,
        parameter="thinkingBudget",
        min_budget=min_b, max_budget=max_b, default_budget=None,
        supports_dynamic=True, can_disable=can_disable,
        notes="thinkingBudget=-1 enables dynamic thinking. " + _LEVEL_NOTE,
    )


THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    # Gemini 2.5 (token budget)
    "gemini-2.5-pro": _budget(128, 32768, can_disable=False),
    "gemini-2.5-flash": _budget(0, 24576, can_disable=True),
    "gemini-2.5-flash-lite": _budget(512, 24576, can_disable=True),
    "gemini-2.5-flash-lite-preview-09-2025": _budget(512, 24576, can_disable=True),
    # Gemini 3.x (thinking levels)
    "gemini-3-pro-preview": _gemini3("high", can_disable=False),
    "gemini-3.1-pro-preview": _gemini3("high", can_disable=False),
    "gemini-3.1-pro-preview-customtools": _gemini3("high", can_disable=False),
    "gemini-3-flash-preview": _gemini3("medium", can_disable=False),
    "gemini-3.5-flash": _gemini3("medium", can_disable=False),
    "gemini-3.6-flash": _gemini3("medium", can_disable=False),
    "gemini-3.1-flash-lite": _gemini3("medium", can_disable=False),
    "gemini-3.5-flash-lite": _gemini3("medium", can_disable=False),
    # Robotics-ER 2 (thinkingLevel; the ER 1.x models are non-reasoning).
    # Docs don't publish its default level — "high" is the Gemini 3 family
    # default; docs recommend "medium" to balance latency vs performance.
    "gemini-robotics-er-2-preview": _gemini3("high", can_disable=False),
}
