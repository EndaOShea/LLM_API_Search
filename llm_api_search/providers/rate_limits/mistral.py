"""Mistral rate limits.

Mistral publishes no fixed public per-model rate-limit numbers. Limits are
per-workspace and vary by billed-usage tier and model — the docs direct you to
Admin Panel > API > Limits to see the values for your account
(https://help.mistral.ai/en/articles/698531). This is a publisher policy, not a
coverage gap, so RATE_LIMITS is intentionally empty and Mistral's text model
IDs are listed in _RATE_LIMIT_COVERAGE_EXEMPT in tests/test_rate_limits.py.
"""

from llm_api_search.providers.base import RateLimit  # noqa: F401

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {}
