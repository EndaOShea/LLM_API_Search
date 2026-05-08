#!/usr/bin/env python3
"""Fetch live model data from each provider and update _STATIC_MODELS.

Preserves pricing, display_name, and description for models that already
exist in the static block — only new models pick these up from the live
API.  New models are added with pricing=None so a human can fill them in
via the resulting PR.

Usage:
    python scripts/update_models.py           # update all providers
    python scripts/update_models.py openai    # update one provider

Requires API keys in the environment:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
"""

from __future__ import annotations

import re
import sys
import textwrap
from dataclasses import fields
from pathlib import Path

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_api_search.providers import PROVIDERS
from llm_api_search.providers.base import (
    ModelInfo, TextModelInfo, ImageModelInfo, AudioTTSModelInfo,
    AudioTranscriptionModelInfo, EmbeddingModelInfo, MusicModelInfo, VideoModelInfo, ModelType,
)

_PRICING_FIELDS = {
    TextModelInfo: ("input_cost_per_mtok", "output_cost_per_mtok"),
    ImageModelInfo: ("cost_per_image",),
    AudioTTSModelInfo: ("cost_per_mchars", "input_cost_per_mtok", "output_cost_per_mtok"),
    AudioTranscriptionModelInfo: ("cost_per_minute",),
    EmbeddingModelInfo: ("input_cost_per_mtok",),
    MusicModelInfo: ("cost_per_second",),
    VideoModelInfo: ("cost_per_second",),
}

# Map provider key → source file path.
_PROVIDER_FILES = {
    "anthropic": PROJECT_ROOT / "llm_api_search" / "providers" / "anthropic.py",
    "google": PROJECT_ROOT / "llm_api_search" / "providers" / "google.py",
    "openai": PROJECT_ROOT / "llm_api_search" / "providers" / "openai.py",
    "inception": PROJECT_ROOT / "llm_api_search" / "providers" / "inception.py",
}

# Models to exclude from updates — superseded or deprecated model IDs per provider.
_EXCLUDED_MODELS: dict[str, set[str]] = {
    "inception": {"mercury", "mercury-coder"},
}

# Regex that matches the entire `_STATIC_MODELS = [...]` block.
_STATIC_BLOCK_RE = re.compile(
    r"^(_STATIC_MODELS\s*=\s*\[).*?(\n\])",
    re.MULTILINE | re.DOTALL,
)


def _serialize_model(m: ModelInfo, indent: str = "    ") -> str:
    """Render a ModelInfo subclass as a Python constructor call."""
    cls_name = type(m).__name__
    all_pricing = set()
    for pfields in _PRICING_FIELDS.values():
        all_pricing.update(pfields)

    lines = [f"{indent}{cls_name}("]
    for f in fields(type(m)):
        if f.name == "model_type":
            continue  # set automatically by subclass
        val = getattr(m, f.name)
        if f.name in all_pricing:
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


def _serialize_models_block(models: list[ModelInfo]) -> str:
    """Render the full _STATIC_MODELS = [...] block."""
    parts = ["_STATIC_MODELS = ["]
    for m in models:
        parts.append(_serialize_model(m))
    parts.append("]")
    return "\n".join(parts)


def update_provider(key: str) -> tuple[int, int]:
    """Fetch live models for a provider and update its source file."""
    source_path = _PROVIDER_FILES[key]
    provider_cls = PROVIDERS[key]
    provider = provider_cls()

    static_info = provider.get_static_info()
    pricing_map: dict[str, dict[str, float | None]] = {}
    for m in static_info.models:
        pfields = _PRICING_FIELDS.get(type(m), ())
        pricing_map[m.model_id] = {f: getattr(m, f) for f in pfields}
    static_map: dict[str, ModelInfo] = {m.model_id: m for m in static_info.models}

    live_info = provider.fetch_live_models()
    excluded = _EXCLUDED_MODELS.get(key, set())
    live_map: dict[str, ModelInfo] = {
        m.model_id: m for m in live_info.models if m.model_id not in excluded
    }

    merged: list[ModelInfo] = []
    seen: set[str] = set()

    for model_id, live_m in live_map.items():
        seen.add(model_id)
        static_m = static_map.get(model_id)

        model_cls = type(static_m) if static_m else type(live_m)
        if model_cls is ModelInfo:
            model_cls = TextModelInfo

        # Preserve existing display_name/description for known models so manual
        # curation (e.g. richer descriptions than the live API returns) is not
        # silently reverted on the weekly auto-update pass.  Live values are
        # only used when the model is new.
        if static_m:
            display_name = static_m.display_name or live_m.display_name or model_id
            description = static_m.description or live_m.description or ""
        else:
            display_name = live_m.display_name or model_id
            description = live_m.description or ""

        kwargs: dict = {
            "model_id": model_id,
            "display_name": display_name,
            "description": description,
        }

        if static_m:
            for f in fields(type(static_m)):
                if f.name in ("model_id", "display_name", "description", "model_type"):
                    continue
                if f.name not in kwargs:
                    kwargs[f.name] = getattr(static_m, f.name)

        if model_id in pricing_map:
            kwargs.update(pricing_map[model_id])

        merged.append(model_cls(**kwargs))

    for model_id, static_m in static_map.items():
        if model_id not in seen:
            merged.append(static_m)

    new_count = sum(1 for m in merged if m.model_id not in static_map)

    source = source_path.read_text()
    new_block = _serialize_models_block(merged)

    match = _STATIC_BLOCK_RE.search(source)
    if not match:
        print(f"  WARNING: Could not find _STATIC_MODELS block in {source_path.name}")
        return 0, len(merged)

    updated = source[:match.start()] + new_block + source[match.end():]
    source_path.write_text(updated)

    return new_count, len(merged)


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(_PROVIDER_FILES.keys())

    for key in targets:
        if key not in _PROVIDER_FILES:
            print(f"Unknown provider: {key}")
            continue

        print(f"Updating {key}...")
        new, total = update_provider(key)
        if new:
            print(f"  {total} models total ({new} NEW — check pricing!)")
        else:
            print(f"  {total} models total (no new models)")

    print("\nDone. Run `pytest tests/` to verify.")


if __name__ == "__main__":
    main()
