"""Inception Labs (Mercury) thinking configurations.

Source: https://docs.inceptionlabs.ai/capabilities/reasoning-efforts
Verified: 2026-09-15 (mercury-2.5 added; levels unchanged)
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_MERCURY = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["instant", "low", "medium", "high"], default_level="medium",
    can_disable=False,
    notes="instant/low = ultra-low latency; high = extended thinking. "
          "Recommended defaults: temperature=0.75, max_tokens=8192.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "mercury-2.5": _MERCURY,
    "mercury-2": _MERCURY,
    "mercury-edit": _MERCURY,
    "mercury-edit-2": _MERCURY,
}
