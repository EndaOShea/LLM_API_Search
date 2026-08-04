"""Google Gemini thinking configurations.

Source: https://ai.google.dev/gemini-api/docs/thinking
Verified: 2026-06-04

Gemini 3.x -> thinkingLevel (effort). Gemini 2.5 -> thinkingBudget (tokens).
Specialized models (Gemma, audio, computer-use, deep-research, etc.) are
non-reasoning and default to ThinkingMode.NONE on lookup. The Robotics-ER
models are the exception — both generations think, via different knobs:
ER 2 uses the Gemini 3 thinkingLevel effort control, while ER 1.5 predates
thinkingLevel and uses the Gemini 2.5-era thinkingBudget token cap
(https://ai.google.dev/gemini-api/docs/robotics-overview and
https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/,
verified 2026-08-04).

ER 1.5's ``max_budget`` is an INFERENCE, not a quoted figure: Google
documents that thinking is on by default and tunable via ``thinking_config``
but publishes no budget ceiling for this model, and the general thinking
docs' budget table omits it. 65_536 is its documented output-token limit,
used here on the assumption that reasoning tokens draw from the same
ceiling — the same convention (and caveat) as
``providers/thinking/qwen.py``. Correct it if Google publishes a real
figure.
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
    # Robotics-ER 2 (thinkingLevel). Docs don't publish its default level —
    # "high" is the Gemini 3 family default; docs recommend "medium" to
    # balance latency vs performance.
    "gemini-robotics-er-2-preview": _gemini3("high", can_disable=False),
    # Robotics-ER 1.5 (Gemini 2.5-era thinkingBudget, not thinkingLevel).
    # "thinking is enabled by default ... you can set a thinking budget, or
    # even disable thinking, by including the thinking_config option".
    # min_budget/supports_dynamic are left unset rather than copied from the
    # 2.5 text models: Google publishes neither for this model, and the
    # thinking docs' budget table omits it entirely. See module docstring
    # for the max_budget inference.
    "gemini-robotics-er-1.5-preview": ThinkingConfig(
        supported=True, mode=ThinkingMode.TOKEN_BUDGET,
        parameter="thinkingBudget", max_budget=65_536, can_disable=True,
        notes="Thinking is on by default; set thinking_config.thinking_budget "
              "to cap reasoning tokens, or disable thinking entirely. Budget "
              "tunes the latency/accuracy trade-off — short budgets suffice "
              "for object detection, longer ones help complex spatial "
              "reasoning. max_budget is inferred from the model's output-token "
              "limit, not a published figure — see module docstring. "
              + _LEVEL_NOTE,
    ),
}
