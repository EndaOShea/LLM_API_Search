"""DeepSeek thinking configurations.

Source: https://api-docs.deepseek.com/guides/thinking_mode
Verified: 2026-09-15
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_DEEPSEEK = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["low", "high", "max"], default_level="high", can_disable=True,
    notes="Thinking toggle defaults to enabled; disable via thinking={type:'disabled'} "
          "(Anthropic/Responses format: effort 'none'). "
          "Effort compat: minimal->low, medium/xhigh->high, ultra->max. Complex agent requests "
          "(Claude Code, OpenCode) auto-set max. Anthropic-format equivalent: "
          "output_config.effort. Chain-of-thought returned in reasoning_content.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "deepseek-flash": _DEEPSEEK,
    "deepseek-v4-pro": _DEEPSEEK,
    # Retired names, served by deepseek-flash (V4.1-Flash).
    "deepseek-v4-flash": _DEEPSEEK,
    "deepseek-v4-flash-vision-exp": _DEEPSEEK,
}
