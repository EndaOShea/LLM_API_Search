# Multi-Model-Type Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the LLM API Search tool to support all model types from each provider (image generation, audio TTS, audio transcription, embeddings) — not just text LLMs.

**Architecture:** Add `ModelType` enum and subclasses of `ModelInfo` (`TextModelInfo`, `ImageModelInfo`, `AudioTTSModelInfo`, `AudioTranscriptionModelInfo`, `EmbeddingModelInfo`) to `base.py`. Migrate existing text models to `TextModelInfo`. Add new non-text models to OpenAI and Google providers with type-appropriate connection snippets for all 5 languages. Update MCP tools with `model_type` filter and `dataclasses.asdict()` serialization.

**Tech Stack:** Python 3.12+, dataclasses, FastMCP, pytest

**Spec:** `docs/superpowers/specs/2026-03-14-multi-model-type-support-design.md`

---

## Chunk 1: Data Model & Exports

### Task 1: Add ModelType enum and model subclasses to base.py

**Files:**
- Modify: `llm_api_search/providers/base.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test for ModelType and TextModelInfo**

Add to `tests/test_discovery.py`:

```python
from llm_api_search.providers.base import ModelType, TextModelInfo

def test_model_type_enum():
    assert ModelType.TEXT.value == "text"
    assert ModelType.IMAGE.value == "image"
    assert ModelType.AUDIO_TTS.value == "audio_tts"
    assert ModelType.AUDIO_TRANSCRIPTION.value == "audio_transcription"
    assert ModelType.EMBEDDING.value == "embedding"

def test_text_model_info_has_model_type():
    m = TextModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.TEXT
    assert isinstance(m, ModelInfo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py::test_model_type_enum tests/test_discovery.py::test_text_model_info_has_model_type -v`
Expected: FAIL with `ImportError: cannot import name 'ModelType'`

- [ ] **Step 3: Implement ModelType enum and all model subclasses in base.py**

In `llm_api_search/providers/base.py`, add after the imports:

```python
from enum import Enum

class ModelType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO_TTS = "audio_tts"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    EMBEDDING = "embedding"
```

Replace the existing `ModelInfo` dataclass with:

```python
@dataclass
class ModelInfo:
    """Base class for all model types."""
    model_id: str
    display_name: str
    model_type: ModelType = ModelType.TEXT
    description: str = ""


@dataclass
class TextModelInfo(ModelInfo):
    """Text/chat LLM model."""
    model_type: ModelType = field(default=ModelType.TEXT, init=False)
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_tool_use: bool = False
    supports_image_generation: bool = False
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


@dataclass
class ImageModelInfo(ModelInfo):
    """Image generation model."""
    model_type: ModelType = field(default=ModelType.IMAGE, init=False)
    supported_sizes: list[str] = field(default_factory=list)
    supported_qualities: list[str] = field(default_factory=list)
    max_images_per_request: int | None = None
    cost_per_image: float | None = None


@dataclass
class AudioTTSModelInfo(ModelInfo):
    """Text-to-speech model."""
    model_type: ModelType = field(default=ModelType.AUDIO_TTS, init=False)
    supported_voices: list[str] = field(default_factory=list)
    supported_output_formats: list[str] = field(default_factory=list)
    cost_per_mchars: float | None = None
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None


@dataclass
class AudioTranscriptionModelInfo(ModelInfo):
    """Speech-to-text model."""
    model_type: ModelType = field(default=ModelType.AUDIO_TRANSCRIPTION, init=False)
    supported_input_formats: list[str] = field(default_factory=list)
    max_file_size_mb: int | None = None
    cost_per_minute: float | None = None


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Embedding/vector model."""
    model_type: ModelType = field(default=ModelType.EMBEDDING, init=False)
    dimensions: int | None = None
    max_input_tokens: int | None = None
    supports_multimodal: bool = False
    input_cost_per_mtok: float | None = None
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `pytest tests/test_discovery.py::test_model_type_enum tests/test_discovery.py::test_text_model_info_has_model_type -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for other subclasses**

Add to `tests/test_discovery.py`:

```python
from llm_api_search.providers.base import (
    ImageModelInfo, AudioTTSModelInfo, AudioTranscriptionModelInfo,
    EmbeddingModelInfo,
)

def test_image_model_info_has_model_type():
    m = ImageModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.IMAGE
    assert isinstance(m, ModelInfo)

def test_audio_tts_model_info_has_model_type():
    m = AudioTTSModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.AUDIO_TTS
    assert isinstance(m, ModelInfo)

def test_audio_transcription_model_info_has_model_type():
    m = AudioTranscriptionModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.AUDIO_TRANSCRIPTION
    assert isinstance(m, ModelInfo)

def test_embedding_model_info_has_model_type():
    m = EmbeddingModelInfo(model_id="test", display_name="Test")
    assert m.model_type == ModelType.EMBEDDING
    assert isinstance(m, ModelInfo)
```

- [ ] **Step 6: Run all new subclass tests**

Run: `pytest tests/test_discovery.py -k "model_type" -v`
Expected: PASS for all

- [ ] **Step 7: Commit**

```bash
git add llm_api_search/providers/base.py tests/test_discovery.py
git commit -m "feat: add ModelType enum and model subclasses"
```

### Task 2: Update ProviderInfo.summary() for multi-type models

**Files:**
- Modify: `llm_api_search/providers/base.py:54-78`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test for type-grouped summary**

Add to `tests/test_discovery.py`:

```python
def test_summary_groups_by_model_type():
    """Summary should group models by type when multiple types exist."""
    from llm_api_search.providers.base import ProviderInfo, TextModelInfo, ImageModelInfo
    info = ProviderInfo(
        name="TestProvider",
        api_base_url="https://test.com",
        api_version=None,
        auth_env_var="TEST_KEY",
        auth_header="Authorization",
        models=[
            TextModelInfo(model_id="t1", display_name="T1", input_cost_per_mtok=1.0, output_cost_per_mtok=5.0, context_window=100_000),
            ImageModelInfo(model_id="i1", display_name="I1", cost_per_image=0.04),
        ],
    )
    summary = info.summary()
    assert "Models (text):" in summary
    assert "Models (image):" in summary
    assert "$1.00/$5.00 per 1M tok" in summary
    assert "$0.04/image" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py::test_summary_groups_by_model_type -v`
Expected: FAIL — summary doesn't group by type yet

- [ ] **Step 3: Rewrite ProviderInfo.summary() in base.py**

Replace the `summary()` method (lines 54-78) with the code below. **Important:** The `_format_model_cost` helper is a module-level function (not a method) and must be placed *after* all the model subclass definitions in `base.py` (after `EmbeddingModelInfo`, before or after `ProviderInfo`). It uses `isinstance()` checks which need the actual class objects at runtime.

```python
    def summary(self) -> str:
        """Return a human-readable summary of this provider."""
        lines = [
            f"Provider: {self.name}",
            f"  API Base URL:    {self.api_base_url}",
        ]
        if self.api_version:
            lines.append(f"  API Version:     {self.api_version}")
        lines += [
            f"  Auth Env Var:    {self.auth_env_var}",
            f"  Auth Header:     {self.auth_header}",
        ]
        for lang, install in self.sdk_installs.items():
            lines.append(f"  SDK ({lang:>10}): {install}")
        lines.append(f"  Docs:            {self.documentation_url}")

        # Group models by type.
        from collections import defaultdict
        by_type: dict[ModelType, list[ModelInfo]] = defaultdict(list)
        for m in self.models:
            by_type[m.model_type].append(m)

        for mtype in ModelType:
            models = by_type.get(mtype, [])
            if not models:
                continue
            lines.append(f"  Models ({mtype.value}): {len(models)}")
            for m in models:
                lines.append(f"    - {m.model_id}{_format_model_cost(m)}")

        return "\n".join(lines)


def _format_model_cost(m: ModelInfo) -> str:
    """Format the cost string for a model based on its type."""
    if isinstance(m, TextModelInfo):
        parts = []
        if m.context_window:
            parts.append(f"ctx: {m.context_window:,}")
        if m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None:
            parts.append(f"${m.input_cost_per_mtok:.2f}/${m.output_cost_per_mtok:.2f} per 1M tok")
        return f" | {' | '.join(parts)}" if parts else ""
    elif isinstance(m, ImageModelInfo):
        if m.cost_per_image is not None:
            return f" | ${m.cost_per_image:.3f}/image"
        return ""
    elif isinstance(m, AudioTTSModelInfo):
        if m.cost_per_mchars is not None:
            return f" | ${m.cost_per_mchars:.2f}/1M chars"
        if m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None:
            return f" | ${m.input_cost_per_mtok:.2f}/${m.output_cost_per_mtok:.2f} per 1M tok"
        return ""
    elif isinstance(m, AudioTranscriptionModelInfo):
        if m.cost_per_minute is not None:
            return f" | ${m.cost_per_minute:.3f}/min"
        return ""
    elif isinstance(m, EmbeddingModelInfo):
        parts = []
        if m.dimensions:
            parts.append(f"{m.dimensions}d")
        if m.input_cost_per_mtok is not None:
            parts.append(f"${m.input_cost_per_mtok:.2f}/1M tok")
        return f" | {' | '.join(parts)}" if parts else ""
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_discovery.py::test_summary_groups_by_model_type -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS (existing models still have `context_window` and pricing fields since `ModelInfo` still has them at this point — the base class hasn't been stripped yet)

- [ ] **Step 6: Commit**

```bash
git add llm_api_search/providers/base.py tests/test_discovery.py
git commit -m "feat: update ProviderInfo.summary() to group by model type"
```

### Task 3: Update exports in __init__.py files

**Files:**
- Modify: `llm_api_search/providers/__init__.py`
- Modify: `llm_api_search/__init__.py`

- [ ] **Step 1: Update providers/__init__.py**

Replace the imports and `__all__` in `llm_api_search/providers/__init__.py`:

```python
from llm_api_search.providers.base import (
    Provider, ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, ModelType,
    ProviderInfo, SUPPORTED_LANGUAGES,
)
from llm_api_search.providers.anthropic import AnthropicProvider
from llm_api_search.providers.google import GeminiProvider
from llm_api_search.providers.openai import OpenAIProvider
from llm_api_search.providers.inception import InceptionProvider

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "google": GeminiProvider,
    "openai": OpenAIProvider,
    "inception": InceptionProvider,
}

__all__ = [
    "Provider",
    "ModelInfo",
    "TextModelInfo",
    "ImageModelInfo",
    "AudioTTSModelInfo",
    "AudioTranscriptionModelInfo",
    "EmbeddingModelInfo",
    "ModelType",
    "ProviderInfo",
    "SUPPORTED_LANGUAGES",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "InceptionProvider",
    "PROVIDERS",
]
```

- [ ] **Step 2: Update llm_api_search/__init__.py**

Add re-exports and update `__all__`:

```python
from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.selector import select_provider
from llm_api_search.providers.base import (
    ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, ModelType,
)

__version__ = "0.1.0"
__all__ = [
    "discover", "discover_provider", "list_providers", "select_provider",
    "ModelInfo", "TextModelInfo", "ImageModelInfo", "AudioTTSModelInfo",
    "AudioTranscriptionModelInfo", "EmbeddingModelInfo", "ModelType",
]
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add llm_api_search/providers/__init__.py llm_api_search/__init__.py
git commit -m "feat: export new model type classes from package"
```

---

## Chunk 2: Migrate Existing Providers to TextModelInfo

### Task 4: Migrate Anthropic provider to TextModelInfo

**Files:**
- Modify: `llm_api_search/providers/anthropic.py`

- [ ] **Step 1: Update import and replace ModelInfo with TextModelInfo in _STATIC_MODELS**

In `llm_api_search/providers/anthropic.py`:
- Change import: `from llm_api_search.providers.base import ModelInfo, Provider, ProviderInfo` → add `TextModelInfo`
- Replace every `ModelInfo(` in `_STATIC_MODELS` with `TextModelInfo(`
- In `fetch_live_models()`, replace `ModelInfo(` with `TextModelInfo(`

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add llm_api_search/providers/anthropic.py
git commit -m "refactor: migrate Anthropic provider to TextModelInfo"
```

### Task 5: Migrate Google provider to TextModelInfo

**Files:**
- Modify: `llm_api_search/providers/google.py`

- [ ] **Step 1: Update import and replace ModelInfo with TextModelInfo**

Same pattern as Task 4. Additionally, for models with native image generation, add `supports_image_generation=True` to these models:
- `gemini-2.5-flash` (and any other models that support native image output)

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add llm_api_search/providers/google.py
git commit -m "refactor: migrate Google provider to TextModelInfo"
```

### Task 6: Migrate OpenAI provider to TextModelInfo

**Files:**
- Modify: `llm_api_search/providers/openai.py`

- [ ] **Step 1: Update import and replace ModelInfo with TextModelInfo**

Same pattern as Task 4. Replace all `ModelInfo(` with `TextModelInfo(` in `_STATIC_MODELS` and `fetch_live_models()`.

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add llm_api_search/providers/openai.py
git commit -m "refactor: migrate OpenAI provider to TextModelInfo"
```

### Task 7: Migrate Inception provider to TextModelInfo

**Files:**
- Modify: `llm_api_search/providers/inception.py`

- [ ] **Step 1: Update import and replace ModelInfo with TextModelInfo**

Same pattern as Task 4.

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add llm_api_search/providers/inception.py
git commit -m "refactor: migrate Inception provider to TextModelInfo"
```

### Task 8: Strip text-specific fields from base ModelInfo

Now that all providers use `TextModelInfo`, remove the text-specific fields from the base `ModelInfo` class.

**Files:**
- Modify: `llm_api_search/providers/base.py`
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Update test file imports first**

At the top of `tests/test_discovery.py`, replace the existing base import with:

```python
from llm_api_search.providers.base import (
    ProviderInfo, SUPPORTED_LANGUAGES, ModelType, ModelInfo,
    TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo,
)
```

- [ ] **Step 2: Remove text-specific fields from base ModelInfo**

The base `ModelInfo` in `base.py` should now be just:

```python
@dataclass
class ModelInfo:
    """Base class for all model types."""
    model_id: str
    display_name: str
    model_type: ModelType = ModelType.TEXT
    description: str = ""
```

Remove `context_window`, `max_output_tokens`, `supports_vision`, `supports_tool_use`, `input_cost_per_mtok`, `output_cost_per_mtok` from the base — they're now on `TextModelInfo`.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: Some tests FAIL — `test_model_pricing_fields` and `test_summary_includes_pricing` access text-specific fields on all models.

- [ ] **Step 4: Update test_model_pricing_fields to be type-aware**

Replace `test_model_pricing_fields` in `tests/test_discovery.py` with:

```python
def test_model_pricing_fields():
    """All static models should have type-appropriate pricing set."""
    results = discover(live=False)
    for key, info in results.items():
        for m in info.models:
            if isinstance(m, TextModelInfo):
                assert m.input_cost_per_mtok is not None, f"{key}/{m.model_id}: missing input_cost_per_mtok"
                assert m.output_cost_per_mtok is not None, f"{key}/{m.model_id}: missing output_cost_per_mtok"
                assert m.input_cost_per_mtok >= 0, f"{key}/{m.model_id}: negative input cost"
                assert m.output_cost_per_mtok >= 0, f"{key}/{m.model_id}: negative output cost"
                assert m.output_cost_per_mtok >= m.input_cost_per_mtok, f"{key}/{m.model_id}: output cost should be >= input cost"
            elif isinstance(m, ImageModelInfo):
                assert m.cost_per_image is not None, f"{key}/{m.model_id}: missing cost_per_image"
                assert m.cost_per_image >= 0, f"{key}/{m.model_id}: negative cost_per_image"
            elif isinstance(m, AudioTTSModelInfo):
                has_char_pricing = m.cost_per_mchars is not None
                has_token_pricing = m.input_cost_per_mtok is not None and m.output_cost_per_mtok is not None
                assert has_char_pricing or has_token_pricing, f"{key}/{m.model_id}: missing TTS pricing"
            elif isinstance(m, AudioTranscriptionModelInfo):
                assert m.cost_per_minute is not None, f"{key}/{m.model_id}: missing cost_per_minute"
                assert m.cost_per_minute >= 0, f"{key}/{m.model_id}: negative cost_per_minute"
            elif isinstance(m, EmbeddingModelInfo):
                assert m.input_cost_per_mtok is not None, f"{key}/{m.model_id}: missing input_cost_per_mtok"
                assert m.input_cost_per_mtok >= 0, f"{key}/{m.model_id}: negative input_cost_per_mtok"
```

(Imports were already updated in Step 1.)

- [ ] **Step 5: Update test_summary_includes_pricing to be type-aware**

Replace `test_summary_includes_pricing` with:

```python
def test_summary_includes_pricing():
    """Provider summary should show pricing for each model."""
    results = discover(live=False)
    for key, info in results.items():
        summary = info.summary()
        for m in info.models:
            if isinstance(m, TextModelInfo) and m.input_cost_per_mtok is not None:
                assert f"${m.input_cost_per_mtok:.2f}" in summary, (
                    f"{key}/{m.model_id}: pricing missing from summary"
                )
            elif isinstance(m, ImageModelInfo) and m.cost_per_image is not None:
                assert f"${m.cost_per_image:.3f}" in summary, (
                    f"{key}/{m.model_id}: pricing missing from summary"
                )
```

- [ ] **Step 6: Update test_connection_snippets_all_providers**

The current test asserts `"import" in sel.connection_snippet` which won't hold for all model types. Replace:

```python
def test_connection_snippets_all_providers():
    for key in list_providers():
        sel = select_provider(key, live=False)
        assert len(sel.connection_snippet) > 20
        assert sel.model.model_id in sel.connection_snippet
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add llm_api_search/providers/base.py tests/test_discovery.py
git commit -m "refactor: strip text-specific fields from base ModelInfo"
```

### Task 9: Update MCP tools for multi-type models

**Files:**
- Modify: `mcp_servers/llm_api_search.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Update llm_list_models with model_type filter and dataclasses.asdict()**

In `mcp_servers/llm_api_search.py`, replace the `llm_list_models` function:

```python
import dataclasses

@mcp.tool()
def llm_list_models(
    provider: str,
    live: bool = False,
    model_type: str | None = None,
) -> list[dict]:
    """List available models for a specific LLM provider.

    Args:
        provider: Provider key — one of "anthropic", "google", "openai", or "inception".
        live: If True, fetch live model lists (requires API key in environment).
        model_type: Optional filter — one of "text", "image", "audio_tts",
                    "audio_transcription", "embedding". Returns all types if omitted.

    Returns:
        A list of model details. Fields vary by model type.
    """
    info = discover_provider(provider, live=live)
    models = info.models
    if model_type:
        models = [m for m in models if m.model_type.value == model_type]
    return [dataclasses.asdict(m) for m in models]
```

- [ ] **Step 2: Update llm_compare_providers to handle multi-type models**

Replace the `llm_compare_providers` function to filter by `TextModelInfo` before accessing text-specific fields:

```python
from llm_api_search.providers.base import TextModelInfo

@mcp.tool()
def llm_compare_providers(live: bool = False) -> str:
    """Compare all LLM providers side-by-side: pricing tiers, model counts, context windows, and SDK info.

    Args:
        live: If True, fetch live model lists (requires API keys in environment).

    Returns:
        A formatted comparison table of all providers.
    """
    results = discover(live=live)
    lines = ["LLM Provider Comparison", "=" * 60, ""]

    for key, info in results.items():
        text_models = [m for m in info.models if isinstance(m, TextModelInfo)]
        max_ctx = max((m.context_window or 0) for m in text_models) if text_models else 0
        max_out = max((m.max_output_tokens or 0) for m in text_models) if text_models else 0
        priced = [m for m in text_models if m.input_cost_per_mtok is not None]
        model_types = sorted(set(m.model_type.value for m in info.models))

        lines.append(f"{info.name}")
        lines.append(f"  Models available:    {len(info.models)}")
        lines.append(f"  Model types:         {', '.join(model_types)}")
        if text_models:
            lines.append(f"  Max context window:  {max_ctx:,} tokens")
            lines.append(f"  Max output tokens:   {max_out:,} tokens")
        if priced:
            cheapest = min(priced, key=lambda m: m.input_cost_per_mtok)
            priciest = max(priced, key=lambda m: m.input_cost_per_mtok)
            lines.append(
                f"  Text price range:    "
                f"${cheapest.input_cost_per_mtok:.2f}/${cheapest.output_cost_per_mtok:.2f} "
                f"— ${priciest.input_cost_per_mtok:.2f}/${priciest.output_cost_per_mtok:.2f} per 1M tok"
            )
            lines.append(f"  Per-model pricing (text):")
            for m in priced:
                lines.append(
                    f"    {m.model_id:40s} ${m.input_cost_per_mtok:>7.2f} in / ${m.output_cost_per_mtok:>7.2f} out"
                )
        lines.append(f"  SDK:                 {info.sdk_install}")
        lines.append(f"  Auth env var:        {info.auth_env_var}")
        lines.append(f"  Docs:                {info.documentation_url}")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 3: Write test for MCP model_type filter**

Add to `tests/test_discovery.py`:

```python
def test_mcp_list_models_type_filter():
    """MCP llm_list_models tool should support model_type filtering."""
    from mcp_servers.llm_api_search import llm_list_models
    all_models = llm_list_models("openai")
    text_models = llm_list_models("openai", model_type="text")
    assert len(all_models) > len(text_models)
    assert all(m["model_type"] == "text" for m in text_models)
    assert all("model_type" in m for m in all_models)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS (the test will pass once non-text models are added to OpenAI in Chunk 3. If running before Chunk 3, this test will fail because OpenAI only has text models at this point. That's expected — it will pass after Tasks 11-14.)

- [ ] **Step 5: Commit**

```bash
git add mcp_servers/llm_api_search.py tests/test_discovery.py
git commit -m "feat: update MCP tools for multi-type models"
```

### Task 10: Update update_models.py for multi-type serialization

**Files:**
- Modify: `scripts/update_models.py`

- [ ] **Step 1: Update imports and _serialize_model()**

In `scripts/update_models.py`:

Update imports:
```python
from llm_api_search.providers.base import (
    ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, ModelType,
)
```

Replace `_serialize_model()`:

```python
# Pricing fields per model subclass — used for preserving pricing during merge.
_PRICING_FIELDS = {
    TextModelInfo: ("input_cost_per_mtok", "output_cost_per_mtok"),
    ImageModelInfo: ("cost_per_image",),
    AudioTTSModelInfo: ("cost_per_mchars", "input_cost_per_mtok", "output_cost_per_mtok"),
    AudioTranscriptionModelInfo: ("cost_per_minute",),
    EmbeddingModelInfo: ("input_cost_per_mtok",),
}


def _serialize_model(m: ModelInfo, indent: str = "    ") -> str:
    """Render a ModelInfo subclass as a Python constructor call."""
    cls_name = type(m).__name__
    lines = [f"{indent}{cls_name}("]
    for f in fields(type(m)):
        if f.name == "model_type":
            continue  # set automatically by subclass
        val = getattr(m, f.name)
        pricing_fields = set()
        for pfields in _PRICING_FIELDS.values():
            pricing_fields.update(pfields)
        if f.name in pricing_fields:
            if val is None:
                lines.append(f"{indent}    {f.name}=None,  # TODO: add pricing")
            elif isinstance(val, float):
                lines.append(f"{indent}    {f.name}={val},")
            else:
                lines.append(f"{indent}    {f.name}={val!r},")
        elif isinstance(val, str):
            lines.append(f"{indent}    {f.name}={val!r},")
        elif isinstance(val, bool):
            lines.append(f"{indent}    {f.name}={val},")
        elif isinstance(val, int):
            lines.append(f"{indent}    {f.name}={val:_},")
        elif isinstance(val, list):
            lines.append(f"{indent}    {f.name}={val!r},")
        else:
            lines.append(f"{indent}    {f.name}={val!r},")
    lines.append(f"{indent}),")
    return "\n".join(lines)
```

- [ ] **Step 2: Update merge logic in update_provider()**

Replace the merge section in `update_provider()` to handle any model subclass. The full updated `update_provider` function:

```python
def update_provider(key: str) -> tuple[int, int]:
    """Fetch live models for a provider and update its source file."""
    source_path = _PROVIDER_FILES[key]
    provider_cls = PROVIDERS[key]
    provider = provider_cls()

    # Load existing static models to preserve pricing.
    static_info = provider.get_static_info()
    # Build pricing map with type-specific pricing fields.
    pricing_map: dict[str, dict[str, float | None]] = {}
    for m in static_info.models:
        pfields = _PRICING_FIELDS.get(type(m), ())
        pricing_map[m.model_id] = {f: getattr(m, f) for f in pfields}
    # Preserve full model for enrichment and subclass detection.
    static_map: dict[str, ModelInfo] = {m.model_id: m for m in static_info.models}

    # Fetch live models.
    live_info = provider.fetch_live_models()
    live_map: dict[str, ModelInfo] = {m.model_id: m for m in live_info.models}

    # Merge: start with live models, enrich with static data.
    merged: list[ModelInfo] = []
    seen: set[str] = set()

    for model_id, live_m in live_map.items():
        seen.add(model_id)
        static_m = static_map.get(model_id)

        # Use the subclass from static data if available, otherwise from live.
        model_cls = type(static_m) if static_m else type(live_m)
        if model_cls is ModelInfo:
            model_cls = TextModelInfo  # Default unknown models to text

        # Build base kwargs shared by all types.
        kwargs: dict = {
            "model_id": model_id,
            "display_name": live_m.display_name or (static_m.display_name if static_m else model_id),
            "description": live_m.description or (static_m.description if static_m else ""),
        }

        # Add type-specific fields from static data (enrichment).
        if static_m:
            for f in fields(type(static_m)):
                if f.name in ("model_id", "display_name", "description", "model_type"):
                    continue
                if f.name not in kwargs:
                    kwargs[f.name] = getattr(static_m, f.name)

        # Overwrite pricing from static data (preserve hand-curated prices).
        if model_id in pricing_map:
            kwargs.update(pricing_map[model_id])

        merged.append(model_cls(**kwargs))

    # Keep any static-only models that didn't appear in live.
    for model_id, static_m in static_map.items():
        if model_id not in seen:
            merged.append(static_m)

    new_count = sum(1 for m in merged if m.model_id not in static_map)

    # Rewrite the source file.
    source = source_path.read_text()
    new_block = _serialize_models_block(merged)

    match = _STATIC_BLOCK_RE.search(source)
    if not match:
        print(f"  WARNING: Could not find _STATIC_MODELS block in {source_path.name}")
        return 0, len(merged)

    updated = source[:match.start()] + new_block + source[match.end():]
    source_path.write_text(updated)

    return new_count, len(merged)
```

- [ ] **Step 3: Run update script dry-check (no API keys needed)**

Run: `python -c "from scripts.update_models import _serialize_model; from llm_api_search.providers.base import TextModelInfo; print(_serialize_model(TextModelInfo(model_id='test', display_name='Test', input_cost_per_mtok=1.0, output_cost_per_mtok=5.0)))"`
Expected: Outputs `TextModelInfo(` constructor (not `ModelInfo(`)

- [ ] **Step 4: Commit**

```bash
git add scripts/update_models.py
git commit -m "refactor: update_models.py supports multi-type serialization"
```

---

## Chunk 3: Add Non-Text Models to OpenAI Provider

### Task 11: Add OpenAI image generation models

**Files:**
- Modify: `llm_api_search/providers/openai.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_openai_has_image_models():
    info = discover_provider("openai", live=False)
    image_models = [m for m in info.models if isinstance(m, ImageModelInfo)]
    assert len(image_models) >= 3
    ids = [m.model_id for m in image_models]
    assert "gpt-image-1.5" in ids
    assert "gpt-image-1" in ids
    assert "gpt-image-1-mini" in ids
    for m in image_models:
        assert m.cost_per_image is not None
        assert len(m.supported_sizes) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py::test_openai_has_image_models -v`
Expected: FAIL

- [ ] **Step 3: Add image models to OpenAI _STATIC_MODELS**

In `llm_api_search/providers/openai.py`, add import for `ImageModelInfo` and append after the text models:

```python
    # --- Image generation ---
    ImageModelInfo(
        model_id="gpt-image-1.5",
        display_name="GPT Image 1.5",
        description="Flagship image generation model with transparent backgrounds and streaming",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.034,
    ),
    ImageModelInfo(
        model_id="gpt-image-1",
        display_name="GPT Image 1",
        description="Standard image generation model",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.042,
    ),
    ImageModelInfo(
        model_id="gpt-image-1-mini",
        display_name="GPT Image 1 Mini",
        description="Cost-effective image generation model",
        supported_sizes=["1024x1024", "1024x1536", "1536x1024", "auto"],
        supported_qualities=["low", "medium", "high", "auto"],
        max_images_per_request=1,
        cost_per_image=0.011,
    ),
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_discovery.py::test_openai_has_image_models -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_api_search/providers/openai.py tests/test_discovery.py
git commit -m "feat: add OpenAI image generation models"
```

### Task 12: Add OpenAI audio TTS models

**Files:**
- Modify: `llm_api_search/providers/openai.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_openai_has_tts_models():
    info = discover_provider("openai", live=False)
    tts_models = [m for m in info.models if isinstance(m, AudioTTSModelInfo)]
    assert len(tts_models) >= 3
    ids = [m.model_id for m in tts_models]
    assert "gpt-4o-mini-tts" in ids
    assert "tts-1" in ids
    assert "tts-1-hd" in ids
    for m in tts_models:
        assert len(m.supported_voices) > 0
        assert len(m.supported_output_formats) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py::test_openai_has_tts_models -v`
Expected: FAIL

- [ ] **Step 3: Add TTS models to OpenAI _STATIC_MODELS**

```python
    # --- Audio TTS ---
    AudioTTSModelInfo(
        model_id="gpt-4o-mini-tts",
        display_name="GPT-4o Mini TTS",
        description="Instruction-steerable text-to-speech model",
        supported_voices=["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        input_cost_per_mtok=0.60,
        output_cost_per_mtok=12.00,
    ),
    AudioTTSModelInfo(
        model_id="tts-1",
        display_name="TTS-1",
        description="Low-latency text-to-speech model",
        supported_voices=["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        cost_per_mchars=15.00,
    ),
    AudioTTSModelInfo(
        model_id="tts-1-hd",
        display_name="TTS-1 HD",
        description="High-quality text-to-speech model",
        supported_voices=["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
        supported_output_formats=["mp3", "opus", "aac", "flac", "wav", "pcm"],
        cost_per_mchars=30.00,
    ),
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_discovery.py::test_openai_has_tts_models -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_api_search/providers/openai.py tests/test_discovery.py
git commit -m "feat: add OpenAI audio TTS models"
```

### Task 13: Add OpenAI audio transcription models

**Files:**
- Modify: `llm_api_search/providers/openai.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_openai_has_transcription_models():
    info = discover_provider("openai", live=False)
    trans_models = [m for m in info.models if isinstance(m, AudioTranscriptionModelInfo)]
    assert len(trans_models) >= 3
    ids = [m.model_id for m in trans_models]
    assert "gpt-4o-transcribe" in ids
    assert "whisper-1" in ids
    for m in trans_models:
        assert m.cost_per_minute is not None
        assert len(m.supported_input_formats) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py::test_openai_has_transcription_models -v`
Expected: FAIL

- [ ] **Step 3: Add transcription models to OpenAI _STATIC_MODELS**

```python
    # --- Audio transcription ---
    AudioTranscriptionModelInfo(
        model_id="gpt-4o-transcribe",
        display_name="GPT-4o Transcribe",
        description="High-accuracy speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
    AudioTranscriptionModelInfo(
        model_id="gpt-4o-mini-transcribe",
        display_name="GPT-4o Mini Transcribe",
        description="Cost-effective speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.003,
    ),
    AudioTranscriptionModelInfo(
        model_id="whisper-1",
        display_name="Whisper",
        description="Open-source speech-to-text model",
        supported_input_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"],
        max_file_size_mb=25,
        cost_per_minute=0.006,
    ),
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_discovery.py::test_openai_has_transcription_models -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_api_search/providers/openai.py tests/test_discovery.py
git commit -m "feat: add OpenAI audio transcription models"
```

### Task 14: Add OpenAI embedding models

**Files:**
- Modify: `llm_api_search/providers/openai.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_openai_has_embedding_models():
    info = discover_provider("openai", live=False)
    emb_models = [m for m in info.models if isinstance(m, EmbeddingModelInfo)]
    assert len(emb_models) >= 2
    ids = [m.model_id for m in emb_models]
    assert "text-embedding-3-large" in ids
    assert "text-embedding-3-small" in ids
    for m in emb_models:
        assert m.input_cost_per_mtok is not None
        assert m.dimensions is not None
```

- [ ] **Step 2: Run test, verify fail, add models**

```python
    # --- Embeddings ---
    EmbeddingModelInfo(
        model_id="text-embedding-3-large",
        display_name="Text Embedding 3 Large",
        description="Most capable embedding model",
        dimensions=3072,
        max_input_tokens=8192,
        input_cost_per_mtok=0.13,
    ),
    EmbeddingModelInfo(
        model_id="text-embedding-3-small",
        display_name="Text Embedding 3 Small",
        description="Cost-effective embedding model",
        dimensions=1536,
        max_input_tokens=8192,
        input_cost_per_mtok=0.02,
    ),
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_discovery.py::test_openai_has_embedding_models -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add llm_api_search/providers/openai.py tests/test_discovery.py
git commit -m "feat: add OpenAI embedding models"
```

### Task 15: Add OpenAI connection snippets for non-text model types

**Files:**
- Modify: `llm_api_search/providers/openai.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests for non-text snippets**

```python
def test_openai_image_snippet():
    sel = select_provider("openai", model_id="gpt-image-1", live=False)
    assert "images" in sel.connection_snippet.lower() or "generate" in sel.connection_snippet.lower()
    assert "gpt-image-1" in sel.connection_snippet

def test_openai_tts_snippet():
    sel = select_provider("openai", model_id="tts-1", live=False)
    assert "speech" in sel.connection_snippet.lower() or "audio" in sel.connection_snippet.lower()
    assert "tts-1" in sel.connection_snippet

def test_openai_transcription_snippet():
    sel = select_provider("openai", model_id="whisper-1", live=False)
    assert "transcri" in sel.connection_snippet.lower()
    assert "whisper-1" in sel.connection_snippet

def test_openai_embedding_snippet():
    sel = select_provider("openai", model_id="text-embedding-3-small", live=False)
    assert "embed" in sel.connection_snippet.lower()
    assert "text-embedding-3-small" in sel.connection_snippet
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_discovery.py -k "openai_image_snippet or openai_tts_snippet or openai_transcription_snippet or openai_embedding_snippet" -v`
Expected: FAIL — snippets still return chat completion code

- [ ] **Step 3: Implement _get_model_type() and type-specific snippet methods**

In `llm_api_search/providers/openai.py`, add a helper and refactor `get_connection_snippet()`:

```python
    def _get_model_type(self, model_id: str) -> ModelType:
        """Look up the model type from _STATIC_MODELS."""
        for m in _STATIC_MODELS:
            if m.model_id == model_id:
                return m.model_type
        return ModelType.TEXT

    def get_connection_snippet(
        self, model_id: str | None = None, language: str = "python"
    ) -> str:
        model = model_id or "gpt-5.4"
        mtype = self._get_model_type(model)

        if mtype == ModelType.IMAGE:
            return self._image_snippet(model, language)
        elif mtype == ModelType.AUDIO_TTS:
            return self._tts_snippet(model, language)
        elif mtype == ModelType.AUDIO_TRANSCRIPTION:
            return self._transcription_snippet(model, language)
        elif mtype == ModelType.EMBEDDING:
            return self._embedding_snippet(model, language)
        return self._text_snippet(model, language)
```

Rename existing snippet dict method to `_text_snippet(self, model, language)`.

Then add `_image_snippet`, `_tts_snippet`, `_transcription_snippet`, `_embedding_snippet` methods — each returning a dict of all 5 languages with the appropriate API call for that model type.

**Image snippet (Python example):**
```python
'from openai import OpenAI\n\n'
'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
'response = client.images.generate(\n'
f'    model="{model}",\n'
'    prompt="A white siamese cat",\n'
'    size="1024x1024",\n'
')\n'
'print(response.data[0].url)\n'
```

**TTS snippet (Python example):**
```python
'from openai import OpenAI\n\n'
'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
'response = client.audio.speech.create(\n'
f'    model="{model}",\n'
'    voice="alloy",\n'
'    input="Hello, world!",\n'
')\n'
'response.stream_to_file("output.mp3")\n'
```

**Transcription snippet (Python example):**
```python
'from openai import OpenAI\n\n'
'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
'with open("audio.mp3", "rb") as f:\n'
'    transcript = client.audio.transcriptions.create(\n'
f'        model="{model}",\n'
'        file=f,\n'
'    )\n'
'print(transcript.text)\n'
```

**Embedding snippet (Python example):**
```python
'from openai import OpenAI\n\n'
'client = OpenAI()  # uses OPENAI_API_KEY env var\n\n'
'response = client.embeddings.create(\n'
f'    model="{model}",\n'
'    input="Hello, world!",\n'
')\n'
'print(response.data[0].embedding[:5])\n'
```

Each of these needs TypeScript, JavaScript, Java, and C++ variants. TypeScript examples for each type:

**Image snippet (TypeScript):**
```typescript
import OpenAI from "openai";

const client = new OpenAI(); // uses OPENAI_API_KEY env var

const response = await client.images.generate({
  model: "{model}",
  prompt: "A white siamese cat",
  size: "1024x1024",
});
console.log(response.data[0].url);
```

**TTS snippet (TypeScript):**
```typescript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI(); // uses OPENAI_API_KEY env var

const response = await client.audio.speech.create({
  model: "{model}",
  voice: "alloy",
  input: "Hello, world!",
});
const buffer = Buffer.from(await response.arrayBuffer());
fs.writeFileSync("output.mp3", buffer);
```

**Transcription snippet (TypeScript):**
```typescript
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI(); // uses OPENAI_API_KEY env var

const transcript = await client.audio.transcriptions.create({
  model: "{model}",
  file: fs.createReadStream("audio.mp3"),
});
console.log(transcript.text);
```

**Embedding snippet (TypeScript):**
```typescript
import OpenAI from "openai";

const client = new OpenAI(); // uses OPENAI_API_KEY env var

const response = await client.embeddings.create({
  model: "{model}",
  input: "Hello, world!",
});
console.log(response.data[0].embedding.slice(0, 5));
```

JavaScript, Java, and C++ variants follow the same pattern translations as the existing text snippets (require for JS, HttpClient for Java, libcurl for C++).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_discovery.py -k "openai" -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add llm_api_search/providers/openai.py tests/test_discovery.py
git commit -m "feat: add OpenAI connection snippets for all model types"
```

### Task 16: Update OpenAI fetch_live_models for non-text prefixes

**Files:**
- Modify: `llm_api_search/providers/openai.py:200-231`

- [ ] **Step 1: Expand _PREFIXES to include non-text model prefixes**

Update `fetch_live_models()`:

```python
        _PREFIXES = ("gpt-5", "gpt-4", "o3", "o4", "gpt-image-", "tts-", "whisper-", "text-embedding-")
```

And update the model construction to use the appropriate subclass based on prefix:

```python
            for m in data.get("data", []):
                mid = m.get("id", "")
                if any(mid.startswith(p) for p in _PREFIXES):
                    model_cls = self._model_class_for_id(mid)
                    live_models.append(model_cls(model_id=mid, display_name=mid))
```

Add helper:
```python
    @staticmethod
    def _model_class_for_id(model_id: str):
        # Order matters: more specific prefixes before less specific ones.
        if model_id.startswith("gpt-image-"):
            return ImageModelInfo
        elif model_id.startswith("gpt-4o-mini-tts") or model_id.startswith("tts-"):
            return AudioTTSModelInfo
        elif model_id.startswith("whisper-") or model_id.startswith("gpt-4o-transcribe") or model_id.startswith("gpt-4o-mini-transcribe"):
            return AudioTranscriptionModelInfo
        elif model_id.startswith("text-embedding-"):
            return EmbeddingModelInfo
        return TextModelInfo
```

**Note:** `gpt-4o-mini-tts` must be checked before the default `return TextModelInfo` because it starts with `gpt-4`, which would otherwise match text model prefixes.

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add llm_api_search/providers/openai.py
git commit -m "feat: OpenAI live fetch includes non-text model types"
```

---

## Chunk 4: Add Non-Text Models to Google Provider

### Task 17: Add Google Imagen models

**Files:**
- Modify: `llm_api_search/providers/google.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
def test_google_has_image_models():
    info = discover_provider("google", live=False)
    image_models = [m for m in info.models if isinstance(m, ImageModelInfo)]
    assert len(image_models) >= 3
    ids = [m.model_id for m in image_models]
    assert "imagen-4.0-generate-001" in ids
    for m in image_models:
        assert m.cost_per_image is not None
```

- [ ] **Step 2: Run test, verify fail, add models**

In `llm_api_search/providers/google.py`, add imports for `ImageModelInfo, AudioTTSModelInfo, EmbeddingModelInfo, ModelType` and add to `_STATIC_MODELS`:

```python
    # --- Image generation (Imagen 4) ---
    ImageModelInfo(
        model_id="imagen-4.0-generate-001",
        display_name="Imagen 4",
        description="Standard image generation model",
        supported_sizes=["1024x1024", "2048x2048"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.04,
    ),
    ImageModelInfo(
        model_id="imagen-4.0-fast-generate-001",
        display_name="Imagen 4 Fast",
        description="Fast image generation model",
        supported_sizes=["1024x1024"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.02,
    ),
    ImageModelInfo(
        model_id="imagen-4.0-ultra-generate-001",
        display_name="Imagen 4 Ultra",
        description="Highest quality image generation model",
        supported_sizes=["1024x1024", "2048x2048"],
        supported_qualities=["standard"],
        max_images_per_request=4,
        cost_per_image=0.06,
    ),
```

- [ ] **Step 3: Run test, commit**

Run: `pytest tests/test_discovery.py::test_google_has_image_models -v`
Expected: PASS

```bash
git add llm_api_search/providers/google.py tests/test_discovery.py
git commit -m "feat: add Google Imagen 4 models"
```

### Task 18: Add Google TTS models

**Files:**
- Modify: `llm_api_search/providers/google.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test, add models, verify**

```python
def test_google_has_tts_models():
    info = discover_provider("google", live=False)
    tts_models = [m for m in info.models if isinstance(m, AudioTTSModelInfo)]
    assert len(tts_models) >= 2
    ids = [m.model_id for m in tts_models]
    assert "gemini-2.5-flash-preview-tts" in ids
    assert "gemini-2.5-pro-preview-tts" in ids
    for m in tts_models:
        assert m.input_cost_per_mtok is not None
        assert len(m.supported_voices) > 0
```

Add to `_STATIC_MODELS`:

```python
    # --- Audio TTS ---
    AudioTTSModelInfo(
        model_id="gemini-2.5-flash-preview-tts",
        display_name="Gemini 2.5 Flash TTS",
        description="Low-latency text-to-speech model",
        supported_voices=["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"],
        supported_output_formats=["wav"],
        input_cost_per_mtok=0.50,
        output_cost_per_mtok=10.00,
    ),
    AudioTTSModelInfo(
        model_id="gemini-2.5-pro-preview-tts",
        display_name="Gemini 2.5 Pro TTS",
        description="High-quality text-to-speech model",
        supported_voices=["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"],
        supported_output_formats=["wav"],
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=20.00,
    ),
```

- [ ] **Step 2: Run test, commit**

```bash
git add llm_api_search/providers/google.py tests/test_discovery.py
git commit -m "feat: add Google Gemini TTS models"
```

### Task 19: Add Google embedding models

**Files:**
- Modify: `llm_api_search/providers/google.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test, add models, verify**

```python
def test_google_has_embedding_models():
    info = discover_provider("google", live=False)
    emb_models = [m for m in info.models if isinstance(m, EmbeddingModelInfo)]
    assert len(emb_models) >= 2
    ids = [m.model_id for m in emb_models]
    assert "gemini-embedding-001" in ids
    assert "gemini-embedding-2-preview" in ids
    for m in emb_models:
        assert m.input_cost_per_mtok is not None
        assert m.dimensions is not None
```

Add to `_STATIC_MODELS`:

```python
    # --- Embeddings ---
    EmbeddingModelInfo(
        model_id="gemini-embedding-001",
        display_name="Gemini Embedding",
        description="Text embedding model",
        dimensions=3072,
        max_input_tokens=2048,
        input_cost_per_mtok=0.15,
    ),
    EmbeddingModelInfo(
        model_id="gemini-embedding-2-preview",
        display_name="Gemini Embedding 2",
        description="Multimodal embedding model (text, images, audio, video)",
        dimensions=3072,
        max_input_tokens=8192,
        supports_multimodal=True,
        input_cost_per_mtok=0.20,
    ),
```

- [ ] **Step 2: Run test, commit**

```bash
git add llm_api_search/providers/google.py tests/test_discovery.py
git commit -m "feat: add Google embedding models"
```

### Task 20: Add Google connection snippets for non-text model types

**Files:**
- Modify: `llm_api_search/providers/google.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write failing tests**

```python
def test_google_image_snippet():
    sel = select_provider("google", model_id="imagen-4.0-generate-001", live=False)
    assert "imagen" in sel.connection_snippet.lower() or "image" in sel.connection_snippet.lower()
    assert "imagen-4.0-generate-001" in sel.connection_snippet

def test_google_tts_snippet():
    sel = select_provider("google", model_id="gemini-2.5-flash-preview-tts", live=False)
    assert "audio" in sel.connection_snippet.lower() or "speech" in sel.connection_snippet.lower()
    assert "gemini-2.5-flash-preview-tts" in sel.connection_snippet

def test_google_embedding_snippet():
    sel = select_provider("google", model_id="gemini-embedding-001", live=False)
    assert "embed" in sel.connection_snippet.lower()
    assert "gemini-embedding-001" in sel.connection_snippet
```

- [ ] **Step 2: Implement _get_model_type() and type-specific snippets for Google**

Same pattern as OpenAI (Task 15). Add `_get_model_type`, refactor `get_connection_snippet`, add `_text_snippet`, `_image_snippet`, `_tts_snippet`, `_embedding_snippet` methods.

**Image snippet (Python example — Imagen uses predict API):**
```python
'from google import genai\n\n'
'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
'response = client.models.generate_images(\n'
f'    model="{model}",\n'
'    prompt="A white siamese cat",\n'
'    config=genai.types.GenerateImagesConfig(number_of_images=1),\n'
')\n'
'# Save the first generated image\n'
'response.generated_images[0].image.save("output.png")\n'
```

**TTS snippet (Python example):**
```python
'from google import genai\n'
'from google.genai import types\n\n'
'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
'response = client.models.generate_content(\n'
f'    model="{model}",\n'
'    contents="Hello, world!",\n'
'    config=types.GenerateContentConfig(\n'
'        response_modalities=["AUDIO"],\n'
'        speech_config=types.SpeechConfig(\n'
'            voice_config=types.VoiceConfig(\n'
'                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")\n'
'            )\n'
'        ),\n'
'    ),\n'
')\n'
'# Save audio from response\n'
'with open("output.wav", "wb") as f:\n'
'    f.write(response.candidates[0].content.parts[0].inline_data.data)\n'
```

**Embedding snippet (Python example):**
```python
'from google import genai\n\n'
'client = genai.Client()  # uses GEMINI_API_KEY env var\n\n'
'response = client.models.embed_content(\n'
f'    model="{model}",\n'
'    contents="Hello, world!",\n'
')\n'
'print(response.embeddings[0].values[:5])\n'
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add llm_api_search/providers/google.py tests/test_discovery.py
git commit -m "feat: add Google connection snippets for all model types"
```

### Task 21: Update Google fetch_live_models for non-text models

**Files:**
- Modify: `llm_api_search/providers/google.py:109-144`

- [ ] **Step 1: Update fetch_live_models to include embedding/TTS models**

The current code filters out embedding models (`if model_id.startswith("embedding"): continue`). Remove this filter and instead classify models into the correct subclass:

```python
            for m in data.get("models", []):
                model_id = m.get("name", "").removeprefix("models/")
                if not model_id:
                    continue
                model_cls = self._model_class_for_id(model_id)
                live_models.append(
                    model_cls(
                        model_id=model_id,
                        display_name=m.get("displayName", model_id),
                        description=m.get("description", ""),
                    )
                )
```

Add helper:
```python
    @staticmethod
    def _model_class_for_id(model_id: str):
        if model_id.startswith("imagen-"):
            return ImageModelInfo
        elif model_id.startswith("gemini-embedding") or model_id.startswith("text-embedding"):
            return EmbeddingModelInfo
        elif "tts" in model_id:
            return AudioTTSModelInfo
        return TextModelInfo
```

- [ ] **Step 2: Run all tests, commit**

```bash
git add llm_api_search/providers/google.py
git commit -m "feat: Google live fetch includes non-text model types"
```

---

## Chunk 5: Comprehensive Test Coverage

### Task 22: Add comprehensive multi-type tests

**Files:**
- Modify: `tests/test_discovery.py`

- [ ] **Step 1: Add model_type filter test**

```python
def test_list_models_filter_by_type():
    """MCP tool model_type filter should work correctly."""
    import dataclasses
    info = discover_provider("openai", live=False)
    all_models = [dataclasses.asdict(m) for m in info.models]
    text_only = [dataclasses.asdict(m) for m in info.models if m.model_type.value == "text"]
    image_only = [dataclasses.asdict(m) for m in info.models if m.model_type.value == "image"]
    assert len(all_models) > len(text_only)
    assert len(text_only) > 0
    assert len(image_only) > 0
    assert all(m["model_type"] == "text" for m in text_only)
    assert all(m["model_type"] == "image" for m in image_only)

def test_list_models_no_filter_returns_all():
    """Without model_type filter, all models should be returned."""
    info = discover_provider("openai", live=False)
    assert len(info.models) >= 20  # 14 text + 3 image + 3 tts + 3 transcription + 2 embedding
```

- [ ] **Step 2: Add snippet tests for all non-text types across all providers that support them**

```python
def test_connection_snippet_all_types_all_languages():
    """Every model type should produce a non-empty snippet in every language."""
    test_cases = [
        ("openai", "gpt-5.4"),             # text
        ("openai", "gpt-image-1"),         # image
        ("openai", "tts-1"),               # audio_tts
        ("openai", "whisper-1"),           # audio_transcription
        ("openai", "text-embedding-3-small"), # embedding
        ("google", "gemini-2.5-flash"),    # text
        ("google", "imagen-4.0-generate-001"), # image
        ("google", "gemini-2.5-flash-preview-tts"), # audio_tts
        ("google", "gemini-embedding-001"), # embedding
    ]
    for provider, model_id in test_cases:
        for lang in SUPPORTED_LANGUAGES:
            sel = select_provider(provider, model_id=model_id, live=False, language=lang)
            assert len(sel.connection_snippet) > 20, (
                f"{provider}/{model_id}/{lang}: snippet too short"
            )
            assert model_id in sel.connection_snippet, (
                f"{provider}/{model_id}/{lang}: model_id missing from snippet"
            )
```

- [ ] **Step 3: Add dataclasses.asdict test for all model types**

```python
def test_dataclasses_asdict_includes_type_specific_fields():
    """dataclasses.asdict should include all subclass-specific fields."""
    import dataclasses
    info = discover_provider("openai", live=False)
    for m in info.models:
        d = dataclasses.asdict(m)
        assert "model_type" in d
        assert "model_id" in d
        if isinstance(m, TextModelInfo):
            assert "context_window" in d
            assert "input_cost_per_mtok" in d
        elif isinstance(m, ImageModelInfo):
            assert "cost_per_image" in d
            assert "supported_sizes" in d
        elif isinstance(m, EmbeddingModelInfo):
            assert "dimensions" in d
            assert "input_cost_per_mtok" in d
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/test_discovery.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery.py
git commit -m "test: add comprehensive multi-model-type test coverage"
```

### Task 23: Final validation — run all tests and verify

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Quick smoke test of MCP tools**

Run: `python -c "from mcp_servers.llm_api_search import llm_list_providers, llm_discover_all; print(llm_list_providers()); print(llm_discover_all())"`
Expected: Shows all providers with grouped model types in summary

- [ ] **Step 3: Commit any final fixes if needed**

```bash
git add -A
git commit -m "fix: final adjustments from validation"
```
