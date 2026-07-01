# LLM API Search

An MCP server and Python library that discovers the latest API versions, models, pricing, and rate limits for **Claude (Anthropic)**, **Gemini (Google)**, **OpenAI**, **Mercury (Inception Labs)**, **DeepSeek**, and **GLM (Z.ai)**, and provides ready-to-use connection snippets in **Python, TypeScript, JavaScript, Java, and C++**.

Covers all model types: **text/chat, image generation, audio TTS, audio transcription, embeddings, music generation, and video generation**. Also includes specialized models for **computer use, native audio (Live API), deep research, and robotics**.

Works with any MCP-compatible client — **Claude Code**, **Gemini CLI**, **OpenAI Codex CLI**, and more.

## MCP Server

This is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server. MCP is an open standard supported by multiple AI coding tools, so one server works with all of them.

### Connect to a hosted server

Browse what's available:

```
curl https://llm-mcp.cora-branch.com/
```

The hosted server also exposes two operational endpoints:

- `https://llm-mcp.cora-branch.com/health` — readiness check (200 if all providers' static catalogs parse, 503 otherwise)
- `https://llm-mcp.cora-branch.com/stats` — anonymous request counts (total, today, last 7 days, broken down by path)

Then connect from your preferred tool:

**Claude Code:**
```bash
claude mcp add --transport http --scope user llm-api-search https://llm-mcp.cora-branch.com/llm-api-search/mcp
```

**Gemini CLI:**
```bash
gemini mcp add llm-api-search -t http -s user https://llm-mcp.cora-branch.com/llm-api-search/mcp
```

**OpenAI Codex CLI:**
```bash
codex mcp add llm-api-search --url https://llm-mcp.cora-branch.com/llm-api-search/mcp
```

No installs, no dependencies. All three use the same URL, same protocol, same tools.

### Run locally (stdio mode)

If you prefer to run the server locally instead of connecting to a hosted instance:

```bash
pip install -e ".[mcp]"
python mcp_server.py
```

Then configure your tool to use the local server:

**Claude Code:**
```bash
claude mcp add llm-api-search -- python3 /path/to/LLM_API_Search/mcp_server.py
```

**Gemini CLI:**
```bash
gemini mcp add llm-api-search -t stdio -s user -- python3 /path/to/LLM_API_Search/mcp_server.py
```

**OpenAI Codex CLI:**
```bash
codex mcp add llm-api-search -- python3 /path/to/LLM_API_Search/mcp_server.py
```

### Available tools

| Tool | Description |
|---|---|
| `llm_list_providers` | List all supported provider keys |
| `llm_discover_all` | Discover all providers with models, auth, and SDK info |
| `llm_discover_provider` | Discover a single provider's details |
| `llm_list_models` | List models for a specific provider, with optional `model_type` filter |
| `llm_get_connection_snippet` | Get a ready-to-use code snippet — returns all 5 languages by default, or a single language if specified. Unsupported languages get a message with supported options and a link to request it. |
| `llm_get_rate_limits` | Get rate limits (RPM, TPM, etc.) for a provider or specific model |
| `llm_compare_providers` | Side-by-side comparison of all providers with pricing |

Legacy models and dated snapshots are filtered out by default. Pass `include_all=True` to any tool to see everything.

### Deploy your own server

The server has **no built-in authentication or TLS**, so don't expose it directly to the internet. Bind it to localhost and put a reverse proxy in front that terminates TLS (and adds auth / rate-limiting if you need it). Note: if you set provider API keys for live discovery, an open endpoint lets anyone trigger calls that spend them.

```bash
git clone https://github.com/EndaOShea/LLM_API_Search.git
cd LLM_API_Search

# Docker — bound to localhost; front it with a reverse proxy (TLS) for public access
docker build -t llm-api-search .
docker run -d -p 127.0.0.1:8080:8080 --name llm-api-search llm-api-search

# Or directly
pip install -e ".[mcp]"
python mcp_server.py --http --host 127.0.0.1 --port 8080
```

The hosted instance follows this pattern: the container binds `127.0.0.1` and the host nginx reverse proxy terminates TLS and forwards to it — nothing binds `0.0.0.0`.

### Adding more MCP servers

Drop a new Python file in `mcp_servers/` with a `mcp` instance, `MOUNT_PATH`, and `DESCRIPTION`. It will be auto-discovered and mounted at its own path. See `mcp_servers/llm_api_search.py` as a template.

## Automatic model updates

A GitHub Actions workflow runs weekly to fetch the latest model lists from each provider's API and open a PR with any changes. New models are added automatically; pricing is preserved for existing models and flagged for manual review on new ones.

DeepSeek is handled differently: its live `/models` endpoint returns only generic aliases, so its catalog is curated by hand to keep versioned IDs and pricing accurate. To avoid silently missing a genuinely new DeepSeek model, the update also reports any unrecognized upstream IDs — surfaced in the PR body, the workflow run summary, and a CI warning — for someone to add manually.

## Python library

The MCP tools are backed by a Python library you can also use directly.

### Install

```bash
pip install -e .          # core (no SDK deps)
pip install -e ".[all]"   # with all provider SDKs
```

### Quick start

```python
from llm_api_search import discover, select_provider

# Discover all providers (static fallback if no API keys are set)
providers = discover()
for key, info in providers.items():
    print(info.summary())

# Programmatically select a provider and model
sel = select_provider("anthropic", model_id="claude-sonnet-4-6")
print(sel.connection_snippet)

# Get a TypeScript snippet instead
sel = select_provider("openai", model_id="gpt-5.4", language="typescript")
print(sel.connection_snippet)

# Interactive selection (prompts via stdin)
sel = select_provider(interactive=True)
```

### Supported languages

Connection snippets are available in five languages:

| Language | SDK / Approach | Example Install |
|---|---|---|
| **Python** | Official SDK (`anthropic`, `openai`, `google-genai`) | `pip install anthropic` |
| **TypeScript** | Official SDK (ES module imports) | `npm install @anthropic-ai/sdk` |
| **JavaScript** | Official SDK (CommonJS `require`) | `npm install openai` |
| **Java** | Official SDK (Maven/Gradle) | `com.openai:openai-java` |
| **C++** | REST API via libcurl + nlohmann/json | No official SDK |

### Model types

Models are organized by type, each with type-specific fields and pricing:

| Type | Subclass | Pricing | Example |
|------|----------|---------|---------|
| `text` | `TextModelInfo` | per 1M tokens (in/out) | GPT-5.4, Claude Opus 4.6, Gemini Robotics ER |
| `image` | `ImageModelInfo` | per image | gpt-image-1.5, Imagen 4, Nano Banana |
| `audio_tts` | `AudioTTSModelInfo` | per 1M chars or tokens | tts-1, Gemini Flash TTS |
| `audio_transcription` | `AudioTranscriptionModelInfo` | per minute | Whisper, GPT-4o Transcribe |
| `embedding` | `EmbeddingModelInfo` | per 1M tokens (input) | text-embedding-3-large |
| `music` | `MusicModelInfo` | per second | Lyria 2 |
| `video` | `VideoModelInfo` | per second or per video | Veo 3.1, CogVideoX-3 |

Filter by type using the MCP tool or programmatically:

```python
from llm_api_search import discover_provider
from llm_api_search.providers.base import TextModelInfo, ImageModelInfo

info = discover_provider("openai", live=False)

# All models
for m in info.models:
    print(f"{m.model_id} ({m.model_type.value})")

# Just image models
for m in info.models:
    if isinstance(m, ImageModelInfo):
        print(f"{m.model_id}: ${m.cost_per_image}/image")
```

### Live vs static discovery

When API keys are present in the environment (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `INCEPTION_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`), the library fetches live model lists directly from each provider. Without keys it falls back to built-in static model data.

```python
# Force static-only (no network calls)
providers = discover(live=False)

# Query a single provider
info = discover_provider("google", live=True)
```

### Rate limits

Per-model rate limits are available for all providers, with provider-specific tiers:

```python
from llm_api_search.providers import get_rate_limits

# All rate limits for a provider (returns all tiers per model)
limits = get_rate_limits("google")

# Specific tier — Anthropic: tier-1 to tier-4, Google: free/tier-1/tier-2/tier-3, DeepSeek/Z.ai: none published
limits = get_rate_limits("anthropic", "claude-sonnet-4-6", tier="tier-1")
rl = limits["claude-sonnet-4-6"]
print(f"{rl.requests_per_minute} RPM, {rl.input_tokens_per_minute} ITPM")

# Google free vs paid
free = get_rate_limits("google", "gemini-2.5-flash", tier="free")   # 5 RPM
paid = get_rate_limits("google", "gemini-2.5-flash", tier="tier-1") # 1000 RPM

# Dated snapshots fall back to the base alias automatically
limits = get_rate_limits("openai", "gpt-4o-2024-08-06", tier="tier-1")
```

### API

| Function | Description |
|---|---|
| `discover(live=True, providers=None)` | Discover all (or selected) providers in parallel |
| `discover_provider(name, live=True)` | Discover a single provider |
| `list_providers()` | List available provider keys |
| `select_provider(provider_key, model_id, live, interactive, language)` | Select a provider/model and get a connection snippet |
| `get_rate_limits(provider, model_id=None)` | Get rate limits for a provider or specific model |
