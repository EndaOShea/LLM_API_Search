"""DeepSeek thinking configurations.

Source: https://api-docs.deepseek.com/guides/thinking_mode
Verified: 2026-06-04
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_DEEPSEEK = ThinkingConfig(
    supported=True, mode=ThinkingMode.EFFORT_LEVELS,
    parameter="reasoning_effort",
    levels=["high", "max"], default_level="high", can_disable=True,
    notes="Thinking toggle defaults to enabled; disable via thinking={type:'disabled'}. "
          "Effort compat: low/medium->high, xhigh->max. Complex agent requests "
          "(Claude Code, OpenCode) auto-set max. Anthropic-format equivalent: "
          "output_config.effort. Chain-of-thought returned in reasoning_content.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "deepseek-v4-pro": _DEEPSEEK,
    "deepseek-v4-flash": _DEEPSEEK,
}
