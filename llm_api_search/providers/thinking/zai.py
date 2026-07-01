"""Z.ai (GLM) thinking configurations.

Source: https://docs.z.ai/guides/capabilities/thinking
Verified: 2026-07-01

GLM thinking models expose a ``reasoning_effort`` parameter whose
allowed values are none, minimal, low, medium, high, xhigh, max
(API-wide default: max). Internally low/medium collapse to high and
xhigh collapses to max, so only "high" and "max" are natively distinct.
Thinking can be disabled via ``thinking: {type: "disabled"}``.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_GLM = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["high", "max"], default_level="max", can_disable=True,
    notes="Thinking enabled by default; disable via thinking={type:'disabled'}. "
          "reasoning_effort allowed values: none, minimal, low, medium, high, "
          "xhigh, max (default max). Internally low/medium map to high and xhigh "
          "maps to max; none/minimal skip thinking.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "glm-5.2": _GLM, "glm-5.1": _GLM, "glm-5": _GLM,
    "glm-4.6": _GLM, "glm-4.5-air": _GLM, "glm-5v-turbo": _GLM,
}
