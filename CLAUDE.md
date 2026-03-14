# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP server and Python library that discovers LLM API versions, models, and pricing for Anthropic, Google, OpenAI, and Inception Labs (Mercury), and generates connection snippets in Python, TypeScript, JavaScript, Java, and C++. Covers all model types: text/chat, image generation, audio TTS, audio transcription, embeddings, music generation, and video generation.

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
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... python scripts/update_models.py
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

The `PROVIDERS` dict in `providers/__init__.py` maps string keys ("anthropic", "google", "openai", "inception") to provider classes. The Google provider key is `"google"`, not `"gemini"`. The Inception Labs provider key is `"inception"`.

### Model type system

`ModelInfo` is a base dataclass with subclasses for each model type: `TextModelInfo`, `ImageModelInfo`, `AudioTTSModelInfo`, `AudioTranscriptionModelInfo`, `EmbeddingModelInfo`, `MusicModelInfo`, `VideoModelInfo`. Each subclass has type-specific fields (e.g., `cost_per_image` for images, `cost_per_second` for video/music, `dimensions` for embeddings). The `ModelType` enum (`text`, `image`, `audio_tts`, `audio_transcription`, `embedding`, `music`, `video`) is set automatically via `field(default=..., init=False)` on each subclass.

### Static model data

Each provider file contains a `_STATIC_MODELS` list of model subclass instances at the top. Text models must come first (the selector defaults to `models[0]`). The `scripts/update_models.py` script rewrites this block by regex-replacing it with live API data while preserving existing pricing. New models get `pricing=None` and need manual review.

### Composite MCP server

`mcp_server.py` auto-discovers all modules in `mcp_servers/` that export `mcp` (FastMCP instance), `MOUNT_PATH`, and `DESCRIPTION`. Each gets mounted as a sub-app on its own HTTP path. In stdio mode, only the first discovered server runs.

### Discovery flow

`discover()` uses `ThreadPoolExecutor` to query all providers in parallel. `discover_provider()` instantiates the provider class and calls either `fetch_live_models()` or `get_static_info()` based on the `live` flag. Live discovery requires provider API keys in the environment.

## Adding a New Provider

1. Create `llm_api_search/providers/newprovider.py` with a class extending `Provider`
2. Include `_STATIC_MODELS` list with model subclass entries (text models first, must have pricing)
3. Implement `get_connection_snippet()` for all 5 languages in `SUPPORTED_LANGUAGES`, dispatching by model type via `_get_model_type()` helper
4. Register it in `providers/__init__.py` in the `PROVIDERS` dict
5. Add its source path to `_PROVIDER_FILES` in `scripts/update_models.py`
6. Tests in `test_discovery.py` iterate all providers automatically — new providers are covered

## Adding a New MCP Server

Drop a Python file in `mcp_servers/` with a `mcp` (FastMCP instance), `MOUNT_PATH`, and `DESCRIPTION`. It will be auto-discovered and mounted.

## Key Conventions

- All tests use `live=False` to avoid network calls — static data must be complete
- Connection snippets must contain the model ID and language-appropriate syntax (e.g., `require()` for JS, `import` for TS)
- Every static model must have non-None type-appropriate pricing (e.g., `input_cost_per_mtok`/`output_cost_per_mtok` for text, `cost_per_image` for image, `cost_per_second` for video/music, `cost_per_minute` for transcription)
- Provider live API calls use `urllib.request` (stdlib only), not third-party HTTP libraries
