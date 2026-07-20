"""Qwen (Alibaba Model Studio) thinking configurations.

Source: https://www.alibabacloud.com/help/en/model-studio/deep-thinking
Verified: 2026-07-20

Both curated Qwen models use a hybrid thinking mode controlled by
enable_thinking (bool) plus a thinking_budget token cap, enabled by default.

max_budget is an INFERENCE, not a directly-quoted figure: Alibaba's docs say
thinking_budget "defaults to the model's maximum chain-of-thought length"
and point to the (account-gated) Model Studio console for the exact number,
which wasn't reachable to verify. 65_536 is this model's documented
max_output_tokens, used here on the assumption that reasoning tokens draw
from the same output-token ceiling (as in most token-budget thinking
implementations). Confirm against the console and correct if it differs.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_QWEN_BUDGET = ThinkingConfig(
    supported=True, mode=ThinkingMode.TOKEN_BUDGET,
    parameter="thinking_budget", can_disable=True,
    max_budget=65_536,
    notes="Thinking enabled by default; disable via enable_thinking=false. "
          "thinking_budget caps reasoning tokens; preserve_thinking carries "
          "reasoning across turns (recommended for agentic use). max_budget "
          "here is inferred from max_output_tokens, not a directly-quoted "
          "API-docs figure — see module docstring.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "qwen3.7-max": _QWEN_BUDGET,
    "qwen3.7-plus": _QWEN_BUDGET,
}
