"""OpenAI thinking (reasoning) configurations.

Source: https://developers.openai.com/api/docs/guides/reasoning
Verified: 2026-06-04

Only reasoning models are listed; chat-latest / gpt-4* / gpt-4o* / audio /
realtime models are non-reasoning and default to ThinkingMode.NONE on lookup.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_GPT5_NOTE = (
    "Valid effort values are model-dependent; newer models also accept "
    "'none' and 'xhigh'. Reasoning tokens are billed as output tokens."
)


def _gpt5() -> ThinkingConfig:
    return ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="reasoning.effort",
        levels=["minimal", "low", "medium", "high"],
        default_level="medium", can_disable=True,
        notes=_GPT5_NOTE,
    )


def _oseries() -> ThinkingConfig:
    return ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="reasoning.effort",
        levels=["low", "medium", "high"],
        default_level="medium", can_disable=False,
        notes="Always-on reasoning model. Reasoning tokens billed as output tokens.",
    )


# gpt-5.x reasoning models (excludes *-chat-latest, which are non-reasoning).
_GPT5_IDS = [
    "gpt-5", "gpt-5-codex", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5-search-api",
    "gpt-5.1", "gpt-5.1-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini",
    "gpt-5.2", "gpt-5.2-codex", "gpt-5.2-pro",
    "gpt-5.3-codex",
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro",
    "gpt-5.5", "gpt-5.5-pro",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
]
_OSERIES_IDS = ["o3", "o3-pro", "o4-mini"]


def _gpt6() -> ThinkingConfig:
    """GPT-6 widens the effort ladder past ``high`` to ``xhigh`` and ``max``.

    Kept separate from ``_gpt5`` rather than folded in: the GPT-5.x models
    reject the two new levels, so sharing one config would advertise knobs
    that 400 on most of the catalog.
    """
    return ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="reasoning.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="medium", can_disable=False,
        notes="Always-on reasoning model. Reasoning tokens billed as output "
              "tokens. The model page does not document a way to disable "
              "reasoning, so can_disable follows the rest of the family.",
    )


_GPT6_IDS = ["gpt-6-astra"]

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    **{mid: _gpt5() for mid in _GPT5_IDS},
    **{mid: _oseries() for mid in _OSERIES_IDS},
    **{mid: _gpt6() for mid in _GPT6_IDS},
}
