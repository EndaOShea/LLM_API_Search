"""Z.ai (GLM) rate limits.

Z.ai does not publish detailed numeric rate limits per model tier.
The official rate-limit reference (https://docs.z.ai/api-reference/rate-limit)
describes general service-level limits but does not provide per-model RPM/TPM
figures comparable to Anthropic or Google tiers.

We expose an empty dict so that ``llm_get_rate_limits("zai", ...)``
returns ``{}`` rather than raising ``KeyError`` — accurate (we have no
tier data) and consistent with the DeepSeek convention.
"""

from llm_api_search.providers.base import RateLimit

RATE_LIMITS: dict[str, dict[str, RateLimit]] = {}
