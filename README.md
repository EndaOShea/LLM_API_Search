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

When API keys are present in the environment (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`), the library fetches live model lists directly from each provider. Without keys it falls back to built-in static model data.

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

## MCP Server (Claude Code Integration)

This repo also ships as an **MCP server** so Claude Code can call the discovery tools directly — no Python imports needed.

### Setup

```bash
# Install with MCP support
pip install -e ".[mcp]"

# Register with Claude Code (user-wide)
claude mcp add --transport stdio --scope user llm-api-search \
  -- python /path/to/LLM_API_Search/mcp_server.py

# Or for project-scope, the .mcp.json is already included in this repo
```

After registering, restart Claude Code and verify with `/mcp`.

### Available MCP Tools

| Tool | Description |
|---|---|
| `llm_list_providers` | List all supported provider keys |
| `llm_discover_all` | Discover all providers with models, auth, and SDK info |
| `llm_discover_provider` | Discover a single provider's details |
| `llm_list_models` | List models for a specific provider |
| `llm_get_connection_snippet` | Get a ready-to-use Python code snippet for any provider/model |
| `llm_compare_providers` | Side-by-side comparison of all providers |
