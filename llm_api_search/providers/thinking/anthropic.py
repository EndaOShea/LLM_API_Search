"""Anthropic thinking configurations.

Source: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
Verified: 2026-06-04

Sampling constraints source:
https://platform.claude.com/docs/en/build-with-claude/thinking
("Sampling parameters" section), verified 2026-08-04. Two generations:
the newest models (Fable 5, Opus 5, Opus 4.8/4.7, Sonnet 5) return a 400
for non-default temperature/top_p/top_k on EVERY request, thinking or
not; the 4.6 generation restricts sampling only while thinking is on
(temperature and top_k incompatible; top_p allowed in [0.95, 1]).
"""

from llm_api_search.providers.base import (
    SamplingConstraint, SamplingStatus, SamplingWhen,
    ThinkingConfig, ThinkingMode,
)

_ADAPTIVE_NOTE = (
    "Requires thinking={type:'adaptive'}. effort set via output_config.effort. "
)

# Newest generation: non-default sampling values 400 on every request.
_SAMPLING_LOCKED_ALWAYS = {
    "temperature": SamplingConstraint(status=SamplingStatus.DEFAULT_ONLY),
    "top_p": SamplingConstraint(status=SamplingStatus.DEFAULT_ONLY),
    "top_k": SamplingConstraint(status=SamplingStatus.DEFAULT_ONLY),
}

# 4.6 generation: restrictions bite only while thinking is enabled.
_SAMPLING_THINKING_ONLY = {
    "temperature": SamplingConstraint(
        status=SamplingStatus.FORBIDDEN, when=SamplingWhen.THINKING_ENABLED),
    "top_k": SamplingConstraint(
        status=SamplingStatus.FORBIDDEN, when=SamplingWhen.THINKING_ENABLED),
    "top_p": SamplingConstraint(
        status=SamplingStatus.RANGE, min=0.95, max=1.0,
        when=SamplingWhen.THINKING_ENABLED),
}

THINKING_CONFIGS: dict[str, ThinkingConfig] = {
    "claude-fable-5": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_LOCKED_ALWAYS,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400). "
                               "display defaults to 'omitted'.",
    ),
    "claude-opus-5": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_LOCKED_ALWAYS,
        notes=_ADAPTIVE_NOTE + "Thinking is ON by default (omitting the thinking field runs "
                               "adaptive, unlike Opus 4.8/4.7). thinking={type:'disabled'} is "
                               "accepted only at effort high or below — pairing it with xhigh/max "
                               "returns a 400. Manual budget_tokens is rejected (400). The raw "
                               "chain of thought is never returned; display defaults to 'omitted'.",
    ),
    "claude-opus-4-8": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_LOCKED_ALWAYS,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400). "
                               "display defaults to 'omitted'.",
    ),
    "claude-opus-4-7": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_LOCKED_ALWAYS,
        notes=_ADAPTIVE_NOTE + "Adaptive is the only mode; manual budget_tokens is rejected (400).",
    ),
    "claude-opus-4-6": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_THINKING_ONLY,
        notes=_ADAPTIVE_NOTE + "Legacy thinking={type:'enabled',budget_tokens} still works but is deprecated.",
    ),
    "claude-sonnet-4-6": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_THINKING_ONLY,
        notes=_ADAPTIVE_NOTE + "Legacy budget_tokens deprecated. Manual-mode interleaved "
                               "thinking via interleaved-thinking-2025-05-14 beta header.",
    ),
    "claude-sonnet-5": ThinkingConfig(
        supported=True, mode=ThinkingMode.EFFORT_LEVELS,
        parameter="output_config.effort",
        levels=["low", "medium", "high", "xhigh", "max"],
        default_level="high", can_disable=True,
        sampling_params_allowed=_SAMPLING_LOCKED_ALWAYS,
        notes=_ADAPTIVE_NOTE + "Adaptive thinking is on by default (unlike Sonnet 4.6, where "
                               "no thinking field means no thinking). Manual budget_tokens is "
                               "rejected (400), same as Opus 4.8/4.7.",
    ),
    "claude-haiku-4-5-20251001": ThinkingConfig(
        supported=True, mode=ThinkingMode.TOKEN_BUDGET,
        parameter="thinking.budget_tokens",
        min_budget=1024, max_budget=64000, supports_dynamic=False, can_disable=True,
        sampling_params_allowed=_SAMPLING_THINKING_ONLY,
        notes="Manual extended thinking only (thinking={type:'enabled',budget_tokens}); "
              "thinking={type:'adaptive'} returns a 400 on this model. budget_tokens must "
              "be less than max_tokens (64k output ceiling). No interleaved thinking — the "
              "interleaved-thinking-2025-05-14 beta header is accepted but ignored.",
    ),
}
