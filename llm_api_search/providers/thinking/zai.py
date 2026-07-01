"""Z.ai (GLM) thinking configurations.

Source: https://docs.z.ai/guides/overview/pricing and
        https://docs.z.ai/api-reference/introduction
Verified: 2026-07-01

GLM thinking models expose a ``reasoning_effort`` parameter
(values: "low", "high", "max"). glm-5.2 defaults to "max";
other thinking-capable models default to "high". Thinking can
be disabled via ``thinking: {type: "disabled"}``.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_GLM_EFFORT = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["low", "high", "max"], default_level="high", can_disable=True,
    notes="Disable thinking via thinking={type:'disabled'}. "
          "Chain-of-thought returned in reasoning_content.",
)

_GLM_52 = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["low", "high", "max"], default_level="max", can_disable=True,
    notes="Flagship GLM model; defaults reasoning_effort to max. "
          "Disable thinking via thinking={type:'disabled'}. "
          "Chain-of-thought returned in reasoning_content.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "glm-5.2": _GLM_52,
    "glm-5.1": _GLM_EFFORT,
    "glm-5": _GLM_EFFORT,
    "glm-4.6": _GLM_EFFORT,
    "glm-4.5-air": _GLM_EFFORT,
    "glm-5v-turbo": _GLM_EFFORT,
}
