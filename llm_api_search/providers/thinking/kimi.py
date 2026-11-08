"""Kimi (Moonshot AI) thinking configurations.

Sources:
  https://platform.kimi.ai/docs/api/models-overview   (per-model parameter table)
  https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart
  https://platform.kimi.ai/docs/guide/kimi-k2-7-code-quickstart
Verified: 2026-11-08 (kimi-k3 / kimi-k2.6 first verified 2026-07-20)

The K3 series (kimi-k3.1-pro, kimi-k3.1-pro-highspeed, kimi-k3) thinks
always-on with a graded top-level ``reasoning_effort`` (low/high/max, default
max) and cannot be disabled; ``thinking`` is a K2.x-only request parameter.
kimi-k2.6 exposes only an on/off thinking toggle
(thinking={type:'enabled'|'disabled'}), enabled by default. kimi-k2.7-code and
its HighSpeed variant use the same ``thinking`` parameter but accept only
type='enabled' — thinking is always on and passing 'disabled' errors.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

# K3 series: always-on, graded reasoning_effort.
_K3_EFFORT = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["low", "high", "max"], default_level="max", can_disable=False,
    notes="Thinking is always on; reasoning_effort=low|high|max (default "
          "max) sets depth but cannot disable thinking entirely.",
)

# K2.7 Code pair: always-on toggle (only type='enabled' accepted).
_K27_CODE_TOGGLE = ThinkingConfig(
    supported=True, mode=ThinkingMode.TOGGLE,
    parameter="thinking", can_disable=False,
    notes="Thinking is always on: thinking may be omitted (defaults to "
          "enabled) or set to {type:'enabled', keep:'all'}; passing "
          "'disabled' returns an error. No graded effort levels. "
          "thinking.keep controls whether reasoning is retained across "
          "turns (default: not kept).",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "kimi-k3.1-pro": _K3_EFFORT,
    "kimi-k3.1-pro-highspeed": _K3_EFFORT,
    "kimi-k3": _K3_EFFORT,
    "kimi-k2.6": ThinkingConfig(
        supported=True, mode=ThinkingMode.TOGGLE,
        parameter="thinking", can_disable=True,
        notes="Thinking on/off via thinking={type:'enabled'|'disabled'}, "
              "enabled by default. No graded effort levels. thinking.keep "
              "controls whether reasoning is retained across turns "
              "(default: not kept).",
    ),
    "kimi-k2.7-code": _K27_CODE_TOGGLE,
    "kimi-k2.7-code-highspeed": _K27_CODE_TOGGLE,
}
