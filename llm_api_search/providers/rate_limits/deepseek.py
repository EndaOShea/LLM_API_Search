"""DeepSeek rate limits.

DeepSeek does not publish numeric rate limits.  Per the official docs
(https://api-docs.deepseek.com/quick_start/rate_limit), the API
"dynamically limits user concurrency based on server load" and surfaces
HTTP 429 when callers exceed the current threshold.  No tier structure
or per-model RPM/TPM numbers are available.

We expose an empty dict so that ``llm_get_rate_limits("deepseek", ...)``
returns ``{}`` rather than raising ``KeyError`` — accurate (we have no
data) and consistent with how other providers are registered.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {}
