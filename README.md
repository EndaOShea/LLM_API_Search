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

## MCP Server (Use from Claude Code — no install needed)

This repo runs as a remote **MCP server** so Claude Code can call the tools directly. The server supports hosting multiple MCP services — you connect only to the ones you need.

### Connect to a hosted server

Browse what's available:

```
curl http://YOUR_SERVER:8080/
```

Then connect the one(s) you want:

```bash
claude mcp add --transport url --scope user llm-api-search http://YOUR_SERVER:8080/llm-api-search/mcp
```

Restart Claude Code. That's it — no installs, no dependencies. Verify with `/mcp`.

### Deploy the server (VPS)

```bash
git clone https://github.com/EndaOShea/LLM_API_Search.git
cd LLM_API_Search

# Docker
docker build -t llm-api-search .
docker run -d -p 8080:8080 --name llm-api-search llm-api-search

# Or directly
pip install -e ".[mcp]"
python mcp_server.py --http --port 8080
```

### Adding more MCP servers

Drop a new Python file in `mcp_servers/` with a `mcp` instance, `MOUNT_PATH`, and `DESCRIPTION`. It will be auto-discovered and mounted at its own path. See `mcp_servers/llm_api_search.py` as a template.

### Available tools (llm-api-search)

| Tool | Description |
|---|---|
| `llm_list_providers` | List all supported provider keys |
| `llm_discover_all` | Discover all providers with models, auth, and SDK info |
| `llm_discover_provider` | Discover a single provider's details |
| `llm_list_models` | List models for a specific provider |
| `llm_get_connection_snippet` | Get a ready-to-use Python code snippet for any provider/model |
| `llm_compare_providers` | Side-by-side comparison of all providers |
