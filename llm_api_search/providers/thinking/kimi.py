"""Kimi (Moonshot AI) thinking configurations.

Source: https://platform.kimi.ai/docs/pricing/chat-k3, .../chat-k26
Verified: 2026-07-20

kimi-k3 thinks always-on with a graded reasoning_effort (low/high/max,
default max) and cannot be disabled. kimi-k2.6 exposes only an on/off
thinking toggle (thinking={type:'enabled'|'disabled'}), enabled by default.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "kimi-k3": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="reasoning_effort",
        levels=["low", "high", "max"], default_level="max", can_disable=False,
        notes="Thinking is always on; reasoning_effort=low|high|max (default "
              "max) sets depth but cannot disable thinking entirely.",
    ),
    "kimi-k2.6": ThinkingConfig(
        supported=True, mode=ThinkingMode.TOGGLE,
        parameter="thinking", can_disable=True,
        notes="Thinking on/off via thinking={type:'enabled'|'disabled'}, "
              "enabled by default. No graded effort levels. thinking.keep "
              "controls whether reasoning is retained across turns "
              "(default: not kept).",
    ),
}
