# LLM API Search

A Python library that discovers the latest API versions for **Claude (Anthropic)**, **Gemini (Google)**, and **OpenAI**, and provides connection helpers for each.

## Install

```bash
pip install -e .          # core (no SDK deps)
pip install -e ".[all]"   # with all provider SDKs
```

## Quick Start

```python
from llm_api_search import discover, select_provider

# Discover all providers (static fallback if no API keys are set)
providers = discover()
for key, info in providers.items():
    print(info.summary())

# Programmatically select a provider and model
sel = select_provider("anthropic", model_id="claude-sonnet-4-6")
print(sel.connection_snippet)

# Interactive selection (prompts via stdin)
sel = select_provider(interactive=True)
```

## Live vs Static Discovery

When API keys are present in the environment (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`), the library fetches live model lists directly from each provider. Without keys it falls back to built-in static model data.

```python
# Force static-only (no network calls)
providers = discover(live=False)

# Query a single provider
from llm_api_search import discover_provider
info = discover_provider("gemini", live=True)
```

## API

| Function | Description |
|---|---|
| `discover(live=True, providers=None)` | Discover all (or selected) providers in parallel |
| `discover_provider(name, live=True)` | Discover a single provider |
| `list_providers()` | List available provider keys |
| `select_provider(provider_key, model_id, live, interactive)` | Select a provider/model and get a connection snippet |
