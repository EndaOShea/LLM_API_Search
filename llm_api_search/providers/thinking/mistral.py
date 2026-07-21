"""Mistral thinking configurations.

Source: https://docs.mistral.ai/capabilities/reasoning
Verified: 2026-07-21

Mistral folded its dedicated reasoning models (Magistral, now deprecated) into
its hybrid instruct models. Adjustable reasoning is exposed via
``reasoning_effort``: ``high`` generates a full thinking chunk before the
answer, ``none`` omits it (disable). Documented on Mistral Medium 3.5 and
Mistral Small 4.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_MISTRAL = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["high"], default_level="high", can_disable=True,
    notes="Adjustable reasoning: reasoning_effort='high' emits a full thinking "
          "chunk; 'none' disables it. Only high/none are documented (other "
          "OpenAI-style levels may be accepted but are not documented). Also "
          "available on the Agents/Conversations endpoints via completion_args.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "mistral-medium-3-5-26-04": _MISTRAL,
    "mistral-small-2603": _MISTRAL,
}
