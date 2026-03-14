# Multi-Model-Type Support Design

**Date:** 2026-03-14
**Status:** Draft

## Overview

Expand the LLM API Search tool to discover and surface all model types from each provider — not just text LLMs. This includes image generation, audio TTS, audio transcription, and embedding models.

## Data Model Changes

### New `ModelType` Enum

```python
class ModelType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO_TTS = "audio_tts"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    EMBEDDING = "embedding"
```

### Base `ModelInfo` (modified)

The base class retains only fields shared across all model types:

```python
@dataclass
class ModelInfo:
    model_id: str
    display_name: str
    model_type: ModelType
    description: str = ""
```

All type-specific fields (pricing, capabilities) move to subclasses.

### `TextModelInfo(ModelInfo)`

All current models become this. Adds `supports_image_generation` for Gemini models that can produce images natively.

```python
@dataclass
class TextModelInfo(ModelInfo):
    model_type: ModelType = field(default=ModelType.TEXT, init=False)
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_tool_use: bool = False
    supports_image_generation: bool = False
    input_cost_per_mtok: float | None = None
    output_cost_per_mtok: float | None = None
```

### `ImageModelInfo(ModelInfo)`

For standalone image generation models (OpenAI gpt-image-*, Google Imagen 4).

```python
@dataclass
class ImageModelInfo(ModelInfo):
    model_type: ModelType = field(default=ModelType.IMAGE, init=False)
    supported_sizes: list[str] = field(default_factory=list)
    supported_qualities: list[str] = field(default_factory=list)
    max_images_per_request: int | None = None
    cost_per_image: float | None = None   # representative price (medium quality, standard size)
```

`cost_per_image` stores a representative typical price. Pricing that varies by quality/size is noted in the description.

### `AudioTTSModelInfo(ModelInfo)`

For text-to-speech models.

```python
@dataclass
class AudioTTSModelInfo(ModelInfo):
    model_type: ModelType = field(default=ModelType.AUDIO_TTS, init=False)
    supported_voices: list[str] = field(default_factory=list)
    supported_output_formats: list[str] = field(default_factory=list)
    cost_per_mchars: float | None = None        # per 1M characters (OpenAI tts-1, tts-1-hd)
    input_cost_per_mtok: float | None = None     # per 1M input tokens (token-based models)
    output_cost_per_mtok: float | None = None    # per 1M output tokens (token-based models)
```

Two pricing modes: character-based (older OpenAI) and token-based (gpt-4o-mini-tts, Google TTS). Only the applicable fields are set.

### `AudioTranscriptionModelInfo(ModelInfo)`

For speech-to-text models.

```python
@dataclass
class AudioTranscriptionModelInfo(ModelInfo):
    model_type: ModelType = field(default=ModelType.AUDIO_TRANSCRIPTION, init=False)
    supported_input_formats: list[str] = field(default_factory=list)
    max_file_size_mb: int | None = None
    cost_per_minute: float | None = None
```

### `EmbeddingModelInfo(ModelInfo)`

For embedding/vector models.

```python
@dataclass
class EmbeddingModelInfo(ModelInfo):
    model_type: ModelType = field(default=ModelType.EMBEDDING, init=False)
    dimensions: int | None = None
    max_input_tokens: int | None = None
    supports_multimodal: bool = False
    input_cost_per_mtok: float | None = None
```

## Static Models Per Provider

### OpenAI

**Text (existing, unchanged):** gpt-5.4, gpt-5.4-pro, gpt-5.3-codex, gpt-5.2, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, o3, o3-pro, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano

**Image (new):**

| model_id | display_name | supported_sizes | supported_qualities | max_images | cost_per_image |
|---|---|---|---|---|---|
| gpt-image-1.5 | GPT Image 1.5 | 1024x1024, 1024x1536, 1536x1024, auto | low, medium, high, auto | 1 | $0.034 |
| gpt-image-1 | GPT Image 1 | 1024x1024, 1024x1536, 1536x1024, auto | low, medium, high, auto | 1 | $0.042 |
| gpt-image-1-mini | GPT Image 1 Mini | 1024x1024, 1024x1536, 1536x1024, auto | low, medium, high, auto | 1 | $0.011 |

Cost is representative (medium quality, 1024x1024).

**Audio TTS (new):**

| model_id | display_name | voices | formats | pricing |
|---|---|---|---|---|
| gpt-4o-mini-tts | GPT-4o Mini TTS | alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer, verse, marin, cedar | mp3, opus, aac, flac, wav, pcm | $0.60/1M input tok, $12.00/1M output tok |
| tts-1 | TTS-1 | alloy, ash, coral, echo, fable, nova, onyx, sage, shimmer | mp3, opus, aac, flac, wav, pcm | $15.00/1M chars |
| tts-1-hd | TTS-1 HD | alloy, ash, coral, echo, fable, nova, onyx, sage, shimmer | mp3, opus, aac, flac, wav, pcm | $30.00/1M chars |

**Audio Transcription (new):**

| model_id | display_name | input_formats | max_file_mb | cost_per_min |
|---|---|---|---|---|
| gpt-4o-transcribe | GPT-4o Transcribe | mp3, mp4, mpeg, mpga, m4a, wav, webm | 25 | $0.006 |
| gpt-4o-mini-transcribe | GPT-4o Mini Transcribe | mp3, mp4, mpeg, mpga, m4a, wav, webm | 25 | $0.003 |
| whisper-1 | Whisper | mp3, mp4, mpeg, mpga, m4a, wav, webm | 25 | $0.006 |

**Embedding (new):**

| model_id | display_name | dimensions | max_input_tokens | cost/1M tok |
|---|---|---|---|---|
| text-embedding-3-large | Text Embedding 3 Large | 3072 | 8192 | $0.13 |
| text-embedding-3-small | Text Embedding 3 Small | 1536 | 8192 | $0.02 |

### Google

**Text (existing, unchanged):** gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-3.1-flash-lite-preview, gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite

Note: `gemini-2.5-flash` and similar models with native image generation capability get `supports_image_generation=True` on their existing TextModelInfo.

**Image (new):**

| model_id | display_name | supported_sizes | supported_qualities | max_images | cost_per_image |
|---|---|---|---|---|---|
| imagen-4.0-generate-001 | Imagen 4 | 1024x1024, 2048x2048 | standard | 4 | $0.04 |
| imagen-4.0-fast-generate-001 | Imagen 4 Fast | 1024x1024 | standard | 4 | $0.02 |
| imagen-4.0-ultra-generate-001 | Imagen 4 Ultra | 1024x1024, 2048x2048 | standard | 4 | $0.06 |

Aspect ratios (1:1, 3:4, 4:3, 9:16, 16:9) are configured via a separate `aspectRatio` parameter, not size. Sizes listed are base resolutions; actual output dimensions vary by aspect ratio.

**Audio TTS (new):**

| model_id | display_name | voices | formats | pricing |
|---|---|---|---|---|
| gemini-2.5-flash-preview-tts | Gemini 2.5 Flash TTS | Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, + 22 more | wav (PCM 24kHz) | $0.50/1M in tok, $10.00/1M out tok |
| gemini-2.5-pro-preview-tts | Gemini 2.5 Pro TTS | (same 30 voices) | wav (PCM 24kHz) | $1.00/1M in tok, $20.00/1M out tok |

**Embedding (new):**

| model_id | display_name | dimensions | max_input_tokens | multimodal | cost/1M tok |
|---|---|---|---|---|---|
| gemini-embedding-001 | Gemini Embedding | 3072 | 2048 | false | $0.15 |
| gemini-embedding-2-preview | Gemini Embedding 2 | 3072 | 8192 | true | $0.20 |

### Anthropic

**Text (existing, unchanged):** claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5-20250929, claude-opus-4-5-20251124, claude-haiku-4-5-20251001

No non-text model types. Anthropic does not offer image generation, audio, or embeddings.

### Inception Labs

**Text (existing, unchanged):** mercury-2, mercury-edit

No non-text model types.

## Provider Changes

### `get_connection_snippet()` Updates

Each provider's snippet method looks up the model in `_STATIC_MODELS` to determine its `model_type`, then branches to return the appropriate API call:

- **TEXT:** existing chat/messages snippets (unchanged)
- **IMAGE:** `client.images.generate(...)` (OpenAI), Imagen predict endpoint (Google)
- **AUDIO_TTS:** `client.audio.speech.create(...)` (OpenAI), generateContent with responseModalities (Google)
- **AUDIO_TRANSCRIPTION:** `client.audio.transcriptions.create(...)` (OpenAI only)
- **EMBEDDING:** `client.embeddings.create(...)` (OpenAI), embedContent (Google)

Snippets are generated for all 5 languages (Python, TypeScript, JavaScript, Java, C++).

**Implementation:** Each provider adds a helper `_get_model_type(model_id: str) -> ModelType` that looks up the model in `_STATIC_MODELS` by ID and returns its `model_type`. If the model is not found, defaults to `ModelType.TEXT` (backwards compatible). The `get_connection_snippet()` method calls this helper and dispatches to type-specific snippet methods (e.g., `_text_snippet()`, `_image_snippet()`).

### `fetch_live_models()` Updates

- **OpenAI:** Expand prefix filter to include `tts-`, `whisper-`, `text-embedding-`, `gpt-image-` in addition to existing `gpt-5`, `gpt-4`, `o3`, `o4` prefixes. Map each to the correct ModelInfo subclass based on prefix.
- **Google:** Include embedding and TTS models (currently filtered out). Imagen models may need a separate API call or static-only treatment if not returned by the models list endpoint.
- **Anthropic/Inception:** No changes needed.

### `_STATIC_MODELS` ordering

Text models must always come first in each provider's `_STATIC_MODELS` list, followed by other types. This ensures `select_provider()` defaults to the first text model when no model_id is specified, maintaining backwards compatibility.

## Call Site Migration

Every location that accesses `ModelInfo` fields being moved to subclasses must be updated. Full audit:

### `ProviderInfo.summary()` (base.py ~line 73)

Currently accesses `m.context_window`, `m.input_cost_per_mtok`, `m.output_cost_per_mtok`. Update to group models by `m.model_type` and format each group with type-appropriate fields:

- **text:** `{context_window} ctx — ${input}/{output} per 1M tok`
- **image:** `${cost_per_image}/image`
- **audio_tts:** `${cost_per_mchars}/1M chars` or `${input}/{output} per 1M tok`
- **audio_transcription:** `${cost_per_minute}/min`
- **embedding:** `${input_cost_per_mtok}/1M tok, {dimensions}d`

### `llm_list_models()` (mcp_servers/llm_api_search.py ~line 105)

Currently constructs a dict with hardcoded keys. Replace with `dataclasses.asdict(m)` to automatically serialize whatever fields each subclass has. This ensures every subclass's fields appear without manual enumeration:

```python
models = info.models
if model_type:
    models = [m for m in models if m.model_type.value == model_type]
return [dataclasses.asdict(m) for m in models]
```

### `llm_compare_providers()` (mcp_servers/llm_api_search.py ~line 135)

Currently accesses `m.context_window`, `m.max_output_tokens`, `m.input_cost_per_mtok`, `m.output_cost_per_mtok` on all models. Rewrite to filter by type before accessing type-specific fields:

```python
text_models = [m for m in info.models if isinstance(m, TextModelInfo)]
max_ctx = max((m.context_window or 0) for m in text_models) if text_models else 0
# ... pricing from text_models only
# Add model type breakdown row
model_types = sorted(set(m.model_type.value for m in info.models))
```

### `select_provider()` (selector.py ~line 71)

Currently defaults to `info.models[0].model_id`. With text models first in `_STATIC_MODELS`, this continues to work. No code change needed — the ordering convention handles this.

### `_serialize_model()` (scripts/update_models.py ~line 46)

Currently hardcodes `ModelInfo(` and iterates `fields(ModelInfo)`. Update to use `type(model).__name__` for the constructor name and `fields(type(model))` for the field list:

```python
cls_name = type(model).__name__  # "TextModelInfo", "ImageModelInfo", etc.
lines = [f"{indent}{cls_name}("]
for f in fields(type(model)):
    ...
```

### Pricing merge in update_models.py (~line 99)

Currently hardcodes `input_cost_per_mtok` and `output_cost_per_mtok`. Update to detect the model subclass and preserve the correct pricing fields:

- **TextModelInfo:** preserve `input_cost_per_mtok`, `output_cost_per_mtok`
- **ImageModelInfo:** preserve `cost_per_image`
- **AudioTTSModelInfo:** preserve `cost_per_mchars`, `input_cost_per_mtok`, `output_cost_per_mtok`
- **AudioTranscriptionModelInfo:** preserve `cost_per_minute`
- **EmbeddingModelInfo:** preserve `input_cost_per_mtok`

Use a generic approach: for each pricing field in the model's dataclass fields, check if it exists in the old static data and preserve it.

### `test_model_pricing_fields` (tests/test_discovery.py ~line 201)

Replace the single pricing assertion with type-specific checks:

- **TextModelInfo:** `input_cost_per_mtok is not None and output_cost_per_mtok is not None; output >= input`
- **ImageModelInfo:** `cost_per_image is not None and cost_per_image >= 0`
- **AudioTTSModelInfo:** `cost_per_mchars is not None OR (input_cost_per_mtok is not None and output_cost_per_mtok is not None)`
- **AudioTranscriptionModelInfo:** `cost_per_minute is not None and cost_per_minute >= 0`
- **EmbeddingModelInfo:** `input_cost_per_mtok is not None and input_cost_per_mtok >= 0`

### `test_connection_snippets_all_providers` (tests/test_discovery.py ~line 77)

Currently asserts `"import" in sel.connection_snippet` for all providers. Update to only assert this for text model selections, or assert more generically that the snippet is non-empty and contains the model ID.

## Module Export Changes

### `providers/__init__.py`

Add exports for all new types:

```python
from .base import (
    ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, ModelType,
    ProviderInfo, SUPPORTED_LANGUAGES,
)
```

### `llm_api_search/__init__.py`

Re-export new types so they're accessible from the top-level package.

## MCP Tool Changes

### `llm_list_models`

Add optional `model_type` filter:

```python
def llm_list_models(
    provider: str,
    live: bool = False,
    model_type: str | None = None,  # NEW: "text", "image", "audio_tts", etc.
) -> list[dict]
```

- `model_type=None` returns all models (backwards compatible)
- Each returned dict includes `model_type` field
- Serialization uses `dataclasses.asdict()` — subclass-specific fields appear automatically

### `llm_discover_provider` / `llm_discover_all`

`ProviderInfo.summary()` groups models by type:

```
Provider: OpenAI
Models (text): 14
  gpt-5.4 — $2.50/$15.00 per 1M tokens
  ...
Models (image): 3
  gpt-image-1.5 — $0.034/image
  ...
Models (audio_tts): 3
  gpt-4o-mini-tts — $0.60/$12.00 per 1M tokens
  ...
```

### `llm_compare_providers`

Rewritten to iterate text models separately from other types. Adds:
- Model type breakdown row showing which types each provider offers
- Per-type pricing ranges (text: per-token, image: per-image, etc.)
- Text model comparison (context window, pricing) unchanged but only considers TextModelInfo instances

### `llm_get_connection_snippet`

No signature change. Looks up the model, detects its type, delegates to provider's type-aware snippet method.

## Discovery Module

`discovery.py` is unchanged. The `model_type` filtering happens at the MCP tool layer in `llm_list_models()`, not in the discovery module. `discover()` and `discover_provider()` return all model types; consumers filter as needed.

Google does not offer audio transcription through the Gemini API (Cloud Speech-to-Text is a separate product), so no `AudioTranscriptionModelInfo` models are added for Google. This is intentional.

## Update Script Changes

### `scripts/update_models.py`

- `_serialize_model()` uses `type(model).__name__` for the constructor and `fields(type(model))` for the field list
- `update_models.py` imports must be updated to include all subclass types (`TextModelInfo`, `ImageModelInfo`, `AudioTTSModelInfo`, `AudioTranscriptionModelInfo`, `EmbeddingModelInfo`, `ModelType`)
- Each provider source file must import the subclass types it uses (e.g., `openai.py` imports `TextModelInfo`, `ImageModelInfo`, `AudioTTSModelInfo`, `AudioTranscriptionModelInfo`, `EmbeddingModelInfo`) so the serialized `_STATIC_MODELS` block is valid Python
- Regex replacement pattern stays the same (replaces entire `_STATIC_MODELS` block)
- Pricing merge is type-aware: detects the model subclass and preserves the correct pricing fields (see Call Site Migration section above)
- New models of any type get `None` pricing and are flagged for review

## Test Changes

### New Tests

- `test_model_type_field` — every model has a valid `ModelType`
- `test_model_subclass_types` — TextModelInfo, ImageModelInfo, etc. are correct subclass instances
- `test_text_model_fields` — TextModelInfo instances have context_window, pricing
- `test_image_model_fields` — ImageModelInfo instances have cost_per_image, supported_sizes
- `test_embedding_model_fields` — EmbeddingModelInfo instances have dimensions, input_cost_per_mtok
- `test_audio_tts_model_fields` — AudioTTSModelInfo instances have voices, pricing
- `test_audio_transcription_model_fields` — AudioTranscriptionModelInfo instances have cost_per_minute
- `test_list_models_filter_by_type` — model_type filter returns correct subset
- `test_list_models_no_filter_returns_all` — backwards compatibility
- `test_connection_snippet_image_model` — image snippets show image generation API
- `test_connection_snippet_embedding_model` — embedding snippets show embedding API
- `test_connection_snippet_audio_model` — audio snippets show audio API
- `test_pricing_validation_per_type` — type-specific pricing assertions (see Call Site Migration)

### Updated Existing Tests

- `test_model_pricing_fields` — replaced by `test_pricing_validation_per_type` with per-type assertions
- `test_connection_snippets_all_providers` — relaxed assertion to check snippet is non-empty and contains model ID (not `"import"`)
- `test_summary_includes_pricing` — update to use type-appropriate pricing checks instead of accessing `m.input_cost_per_mtok` across all models. For text models check token pricing in summary; for image models check per-image pricing; etc.
- All other existing tests pass unchanged — current models are `TextModelInfo` with identical field values

## Backwards Compatibility

- `ModelInfo` base class still exists — `isinstance(model, ModelInfo)` works for all models
- `llm_list_models` without `model_type` returns all models as before
- `ProviderInfo.models` remains `list[ModelInfo]` (all subclasses are ModelInfo)
- Existing text model data is unchanged
- MCP tool signatures are additive only (new optional parameter)
- `select_provider()` defaults to first model, which is always text (ordering convention)
- `dataclasses.asdict()` serialization includes `model_type` discriminator in all dicts
