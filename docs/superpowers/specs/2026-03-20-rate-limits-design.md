# Rate Limits Feature Design

**Date:** 2026-03-20

## Problem

The LLM API Search tool aims to be a one-stop discovery tool. Users shouldn't have to visit provider docs to find rate limit information. Rate limits need to be stored on the server, queryable per provider and model, and updatable independently of model data.

## Design

### Data Model

A `RateLimit` dataclass in `providers/base.py` with core universal fields and optional provider-specific fields:

```python
@dataclass
class RateLimit:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_day: int | None = None
    images_per_minute: int | None = None
    batch_queue_limit: int | None = None
```

### Rate Limit Data Storage

Separate modules under `providers/rate_limits/`, one per provider. Each exports a `RATE_LIMITS` dict keyed by model ID:

```python
# providers/rate_limits/openai.py
RATE_LIMITS: dict[str, RateLimit] = {
    "gpt-4o": RateLimit(requests_per_minute=500, tokens_per_minute=30_000),
    "gpt-4o-mini": RateLimit(requests_per_minute=500, tokens_per_minute=200_000),
    ...
}
```

Keys are model IDs (not family names). When a user queries a dated snapshot (e.g., `gpt-4o-2024-08-06`), lookup falls back to the base alias (`gpt-4o`) using the same date-stripping regex from `filter_models`.

A registry in `providers/rate_limits/__init__.py` maps provider keys to their rate limit dicts.

### Lookup Function

In `providers/__init__.py`:

```python
def get_rate_limits(provider: str, model_id: str | None = None) -> dict[str, RateLimit]
```

- No model → returns all rate limits for the provider
- With model → returns `{model_id: RateLimit}` for that model, falling back to date-stripped base ID
- Raises `KeyError` for unknown providers

### MCP Tool

```python
llm_get_rate_limits(provider: str, model: str | None = None) -> dict
```

- No model → returns all rate limits for the provider as a dict of model → limits
- With model → returns limits for that specific model

### Update Script

`scripts/update_rate_limits.py` — independent from `update_models.py`. Scrapes or manually updates rate limit data per provider.

## Files

**Create:**
- `providers/base.py` — add `RateLimit` dataclass
- `providers/rate_limits/__init__.py` — registry
- `providers/rate_limits/anthropic.py` — Anthropic rate limits
- `providers/rate_limits/openai.py` — OpenAI rate limits
- `providers/rate_limits/google.py` — Google rate limits
- `providers/rate_limits/inception.py` — Inception rate limits
- `scripts/update_rate_limits.py` — update script
- `tests/test_rate_limits.py` — tests

**Modify:**
- `providers/__init__.py` — add `get_rate_limits`, export `RateLimit`
- `mcp_servers/llm_api_search.py` — add `llm_get_rate_limits` tool
