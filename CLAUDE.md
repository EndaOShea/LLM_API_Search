# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP server and Python library that discovers LLM API versions, models, pricing, and rate limits for Anthropic, Google, OpenAI, Inception Labs (Mercury), and DeepSeek, and generates connection snippets in Python, TypeScript, JavaScript, Java, and C++. Covers all model types: text/chat, image generation, audio TTS, audio transcription, embeddings, music generation, and video generation. Also includes specialized Google models for computer use, native audio (Live API), deep research, and robotics.

Hosted at `https://llm-mcp.cora-branch.com/`. Deployed via Docker + nginx on VPS.

## Commands

```bash
# Install
pip install -e .            # core library only
pip install -e ".[mcp]"     # with MCP server support
pip install -e ".[all]"     # with all provider SDKs + MCP

# Tests
pytest tests/ -v            # all tests
pytest tests/test_discovery.py -v                    # discovery tests
pytest tests/test_discovery.py::test_name -v         # single test

# Run MCP server
python mcp_server.py                    # stdio mode (single server)
python mcp_server.py --http --port 8080 # HTTP mode (composite server)

# Update static model data from live APIs (requires API keys)
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=... python scripts/update_models.py
python scripts/update_models.py openai  # single provider
```

## Architecture

### Two-layer design

1. **Python library** (`llm_api_search/`): Core discovery and selection logic, no MCP dependency.
2. **MCP server layer** (`mcp_server.py` + `mcp_servers/`): Thin wrapper exposing the library as MCP tools.

### Provider system

Each provider in `llm_api_search/providers/` extends `Provider` (ABC in `base.py`) and must implement:
- `get_static_info()` — returns `ProviderInfo` from a hardcoded `_STATIC_MODELS` list
- `fetch_live_models()` — calls the provider's API, falls back to static on failure
- `get_connection_snippet(model_id, language)` — returns code snippets for 5 languages

`Provider` also has an optional `unrecognized_live_model_ids()` method (base default returns an empty set) — see the new-model discovery signal under "Static model data".

The `PROVIDERS` dict in `providers/__init__.py` maps string keys ("anthropic", "google", "openai", "inception", "deepseek") to provider classes. The Google provider key is `"google"`, not `"gemini"`. The Inception Labs provider key is `"inception"`. DeepSeek's API is OpenAI-compatible, so its connection snippets use the `openai` SDK with `base_url="https://api.deepseek.com"`.

### Model type system

`ModelInfo` is a base dataclass with subclasses for each model type: `TextModelInfo`, `ImageModelInfo`, `AudioTTSModelInfo`, `AudioTranscriptionModelInfo`, `EmbeddingModelInfo`, `MusicModelInfo`, `VideoModelInfo`. Each subclass has type-specific fields (e.g., `cost_per_image` for images, `cost_per_second` for video/music, `dimensions` for embeddings). The `ModelType` enum (`text`, `image`, `audio_tts`, `audio_transcription`, `embedding`, `music`, `video`) is set automatically via `field(default=..., init=False)` on each subclass. `TextModelInfo` has boolean capability flags: `supports_vision`, `supports_tool_use`, `supports_image_generation`, `supports_computer_use`.

### Static model data

Each provider file contains a `_STATIC_MODELS` list of model subclass instances at the top. Text models must come first (the selector defaults to `models[0]`). The `scripts/update_models.py` script rewrites this block by regex-replacing it with live API data while preserving existing pricing. New models get `pricing=None` and need manual review.

### New-model discovery signal

Most providers' `fetch_live_models()` returns *every* live ID, so the weekly update auto-adds anything new (with `pricing=None`). DeepSeek is the exception: its `fetch_live_models()` deliberately intersects live IDs with `_STATIC_MODELS` to keep runtime discovery curated-only (its `/models` returns generic aliases `deepseek-chat`/`deepseek-reasoner`, not the curated versioned IDs). That makes a genuinely-new DeepSeek model invisible to the auto-add path. To close that gap, `Provider.unrecognized_live_model_ids()` (base default empty; overridden in `deepseek.py`) returns live IDs that are neither curated nor a known alias (`_KNOWN_LIVE_ALIASES`). It is **report-only** — `fetch_live_models()` is unchanged, so runtime stays curated-only. `update_models.py` calls it per provider and surfaces any hits to stdout, `$GITHUB_STEP_SUMMARY`, a `::warning::` annotation, and `model_update_notes.md` (gitignored); the `update-models.yml` workflow folds that note into the auto-update PR body. A human then adds the model manually.

### Composite MCP server

`mcp_server.py` auto-discovers all modules in `mcp_servers/` that export `mcp` (FastMCP instance), `MOUNT_PATH`, and `DESCRIPTION`. Each gets mounted as a sub-app on its own HTTP path. In stdio mode, only the first discovered server runs.

In HTTP mode, the outer Starlette app also serves three non-MCP routes:
- `/` — index page listing registered MCP servers and connect commands.
- `/health` — `check_health()` in `mcp_server.py` returns 200 + JSON if every provider's `get_static_info()` succeeds and yields a non-empty model list; 503 otherwise. The deploy workflow polls this to gate the rebuild as healthy.
- `/stats` — request counts (total, today, last_7d, by_path) computed by `summarize_usage()` from the JSONL log.

### Usage tracking

`UsageLoggingMiddleware` in `usage_middleware.py` is registered on the Starlette app and appends one JSONL record `{ts, path}` per MCP request (paths ending in `/mcp` only — index/health/stats are excluded). No client identification. The log file lives at `/var/log/llm-mcp/usage.jsonl` inside the container; `docker-compose.yml` bind-mounts `./logs:/var/log/llm-mcp` so the data persists across rebuilds. On the VPS the host path is `/home/root/apps/LLM_API_Search/logs/`.

### Deployment

Production runs at `https://llm-mcp.cora-branch.com/` on a VPS with the repo at `/home/root/apps/LLM_API_Search`. The `.github/workflows/deploy.yml` action fires on `workflow_run` after CI succeeds on `main` (gated by `github.event.workflow_run.conclusion == 'success'`), SSHes in via secrets, runs `git reset --hard origin/main && docker compose up -d --build && docker image prune -f`, then polls `/health` for up to 30s. Required GH secrets: `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `VPS_REPO_PATH`. Manual re-runs available via `workflow_dispatch`.

### Model filtering

MCP tools filter out dated snapshots and legacy models by default. `filter_models()` in `providers/__init__.py` uses two mechanisms: pattern-based date suffix matching (`-YYYY-MM-DD`, `-YYYYMMDD`, `-MM-YYYY`) to remove snapshots when a non-dated alias exists, and an explicit `LEGACY_MODELS` dict for superseded models (e.g., `gpt-4`, `gpt-4.1`, `claude-3-haiku-20240307`). All MCP tools accept `include_all=True` to bypass filtering.

### Rate limits

Per-model rate limit data lives in `providers/rate_limits/` with one module per provider. Each exports a `RATE_LIMITS` dict mapping model ID → `{tier_name: RateLimit, …}`. Tier names are provider-specific: Anthropic uses `"tier-1"` through `"tier-4"` (no free tier), Google uses `"free"`, `"tier-1"`, `"tier-2"`, `"tier-3"`, OpenAI uses `"tier-1"` (baseline only), and Inception uses `"free"`, `"paid"`, `"enterprise"`. DeepSeek publishes no numeric rate limits (limits are dynamic by server load), so its `RATE_LIMITS` dict is empty by design — DeepSeek model IDs are listed in `_RATE_LIMIT_COVERAGE_EXEMPT` in `tests/test_rate_limits.py`. The `RateLimit` dataclass has fields: `requests_per_minute`, `tokens_per_minute`, `input_tokens_per_minute`, `output_tokens_per_minute`, `requests_per_day`, `tokens_per_day`, `images_per_minute`, `batch_queue_limit`. Anthropic uses separate `input_tokens_per_minute`/`output_tokens_per_minute`; others use combined `tokens_per_minute`. The `get_rate_limits(provider, model_id?, tier?)` function in `providers/__init__.py` handles lookup with date-suffix fallback for snapshots and gracefully skips models that lack the requested tier. The `llm_get_rate_limits` MCP tool exposes this.

### Discovery flow

`discover()` uses `ThreadPoolExecutor` to query all providers in parallel. `discover_provider()` instantiates the provider class and calls either `fetch_live_models()` or `get_static_info()` based on the `live` flag. Live discovery requires provider API keys in the environment.

## Adding a New Provider

1. Create `llm_api_search/providers/newprovider.py` with a class extending `Provider`
2. Include `_STATIC_MODELS` list with model subclass entries (text models first, must have pricing)
3. Implement `get_connection_snippet()` for all 5 languages in `SUPPORTED_LANGUAGES`, dispatching by model type via `_get_model_type()` helper
4. Register it in `providers/__init__.py` in the `PROVIDERS` dict
5. Add its source path to `_PROVIDER_FILES` in `scripts/update_models.py`
6. Create `providers/rate_limits/newprovider.py` with a `RATE_LIMITS` dict and register it in `providers/rate_limits/__init__.py`
7. Add any superseded models to `LEGACY_MODELS` in `providers/__init__.py`
8. Tests in `test_discovery.py` iterate all providers automatically — new providers are covered
9. If the provider's `fetch_live_models()` is curated (won't auto-surface new live IDs, like DeepSeek), also override `unrecognized_live_model_ids()` so new models still get flagged — see "New-model discovery signal"

## Adding a New MCP Server

Drop a Python file in `mcp_servers/` with a `mcp` (FastMCP instance), `MOUNT_PATH`, and `DESCRIPTION`. It will be auto-discovered and mounted.

## Key Conventions

- All tests use `live=False` to avoid network calls — static data must be complete
- Connection snippets must contain the model ID and language-appropriate syntax (e.g., `require()` for JS, `import` for TS)
- The `llm_get_connection_snippet` MCP tool returns all 5 languages when no language is specified, a single snippet when a valid language is given, or a message with supported languages and a GitHub issue link for unsupported ones
- Every static model must have non-None type-appropriate pricing (e.g., `input_cost_per_mtok`/`output_cost_per_mtok` for text, `cost_per_image` for image, `cost_per_second` for video/music, `cost_per_minute` for transcription)
- Provider live API calls use `urllib.request` (stdlib only), not third-party HTTP libraries
