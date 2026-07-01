"""Z.ai (GLM) thinking configurations.

Sources:
  https://docs.z.ai/guides/overview/concept-param  (parameter support ranges)
  https://docs.z.ai/guides/capabilities/thinking     (reasoning_effort values)
Verified: 2026-07-01

GLM exposes two distinct thinking controls with different model ranges:

  * ``thinking={type: "enabled"|"disabled"}`` — on/off toggle, GLM-4.5 and above.
  * ``reasoning_effort`` — graded effort (none..max, default max), GLM-5.2 and
    above ONLY. Internally low/medium collapse to high and xhigh collapses to
    max, so only "high" and "max" are natively distinct.

So glm-5.2 exposes graded ``reasoning_effort``; the older thinking-capable GLM
models expose only the enable/disable toggle.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

# GLM-5.2 and above: graded reasoning_effort.
_GLM_EFFORT = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["high", "max"], default_level="max", can_disable=True,
    notes="Graded reasoning_effort (GLM-5.2+). Allowed values: none, minimal, "
          "low, medium, high, xhigh, max (default max). Internally low/medium "
          "map to high and xhigh maps to max; none/minimal skip thinking. "
          "Disable via thinking={type:'disabled'}.",
)

# GLM-4.5+ thinking-capable models without graded effort: on/off toggle only.
_GLM_TOGGLE = ThinkingConfig(
    supported=True, mode=ThinkingMode.TOGGLE,
    parameter="thinking", can_disable=True,
    notes="Thinking on/off only via thinking={type:'enabled'|'disabled'} "
          "(GLM-4.5+). Graded reasoning_effort is GLM-5.2+ and not available "
          "on this model.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "glm-5.2": _GLM_EFFORT,
    "glm-5.1": _GLM_TOGGLE,
    "glm-5": _GLM_TOGGLE,
    "glm-4.6": _GLM_TOGGLE,
    "glm-4.5-air": _GLM_TOGGLE,
    "glm-5v-turbo": _GLM_TOGGLE,
}
