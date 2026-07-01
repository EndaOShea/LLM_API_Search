"""MiniMax thinking configurations.

Source: https://platform.minimax.io/docs/api-reference/text-chat-openai
Verified: 2026-07-01

MiniMax's M-series reasoning models are controlled with a ``thinking`` object:
``thinking={type: "adaptive"|"disabled"}``. "adaptive" (the model decides how
much to think) is the default when the field is omitted; "disabled" turns
reasoning off. There are no graded effort levels, so this is a TOGGLE.
"""

from llm_api_search.providers.base import ThinkingConfig, ThinkingMode

_MINIMAX_TOGGLE = ThinkingConfig(
    supported=True, mode=ThinkingMode.TOGGLE,
    parameter="thinking", can_disable=True,
    notes="Thinking on/off via thinking={type:'adaptive'|'disabled'}. Adaptive "
          "(model chooses depth) is the default when omitted; set "
          "type='disabled' to turn reasoning off. No graded effort levels.",
)

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "MiniMax-M3": _MINIMAX_TOGGLE,
    "MiniMax-M2.7": _MINIMAX_TOGGLE,
    "MiniMax-M2.7-highspeed": _MINIMAX_TOGGLE,
}
