"""Inception Labs (Mercury) thinking configurations.

Source: https://docs.inceptionlabs.ai/capabilities/reasoning-efforts
Verified: 2026-10-09 (mercury-voice added from the models page; others unchanged)
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

# Mercury Voice: the models page and launch post list three settings only (no
# "instant"). Neither states a Voice-specific default; "medium" is the documented
# chat-completions default when reasoning_effort is omitted.
_MERCURY_VOICE = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["low", "medium", "high"], default_level="medium",
    can_disable=False,
    notes="Three settings (low, medium, high). The default is the documented "
          "chat-completions default; Inception states none specifically for Voice.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "mercury-2.5": _MERCURY,
    "mercury-voice": _MERCURY_VOICE,
    "mercury-2": _MERCURY,
    "mercury-edit": _MERCURY,
    "mercury-edit-2": _MERCURY,
}
