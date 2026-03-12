#!/usr/bin/env python3
"""Fetch live model data from each provider and update _STATIC_MODELS.

Preserves pricing for models that already have it set.  New models are
added with pricing=None so a human can fill them in via the resulting PR.

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
from llm_api_search.providers.base import ModelInfo

# Map provider key → source file path.
_PROVIDER_FILES = {
    "anthropic": PROJECT_ROOT / "llm_api_search" / "providers" / "anthropic.py",
    "google": PROJECT_ROOT / "llm_api_search" / "providers" / "google.py",
    "openai": PROJECT_ROOT / "llm_api_search" / "providers" / "openai.py",
}

# Regex that matches the entire `_STATIC_MODELS = [...]` block.
_STATIC_BLOCK_RE = re.compile(
    r"^(_STATIC_MODELS\s*=\s*\[).*?(\n\])",
    re.MULTILINE | re.DOTALL,
)


def _serialize_model(m: ModelInfo, indent: str = "    ") -> str:
    """Render a ModelInfo as a Python constructor call."""
    lines = [f"{indent}ModelInfo("]
    for f in fields(ModelInfo):
        val = getattr(m, f.name)
        if f.name in ("input_cost_per_mtok", "output_cost_per_mtok"):
            if val is None:
                lines.append(f"{indent}    {f.name}=None,  # TODO: add pricing")
            else:
                lines.append(f"{indent}    {f.name}={val:.2f},")
        elif isinstance(val, str):
            lines.append(f"{indent}    {f.name}={val!r},")
        elif isinstance(val, bool):
            lines.append(f"{indent}    {f.name}={val},")
        elif isinstance(val, int):
            lines.append(f"{indent}    {f.name}={val:_},")
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
    """Fetch live models for a provider and update its source file.

    Returns (new_count, total_count).
    """
    source_path = _PROVIDER_FILES[key]
    provider_cls = PROVIDERS[key]
    provider = provider_cls()

    # Load existing static models to preserve pricing.
    static_info = provider.get_static_info()
    pricing_map: dict[str, tuple[float | None, float | None]] = {
        m.model_id: (m.input_cost_per_mtok, m.output_cost_per_mtok)
        for m in static_info.models
    }
    # Also preserve full ModelInfo for models the live API might miss fields on.
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

        merged.append(ModelInfo(
            model_id=model_id,
            display_name=live_m.display_name or (static_m.display_name if static_m else model_id),
            description=live_m.description or (static_m.description if static_m else ""),
            context_window=live_m.context_window or (static_m.context_window if static_m else None),
            max_output_tokens=live_m.max_output_tokens or (static_m.max_output_tokens if static_m else None),
            supports_vision=live_m.supports_vision or (static_m.supports_vision if static_m else False),
            supports_tool_use=live_m.supports_tool_use or (static_m.supports_tool_use if static_m else False),
            # Preserve pricing from static data.
            input_cost_per_mtok=pricing_map.get(model_id, (None, None))[0],
            output_cost_per_mtok=pricing_map.get(model_id, (None, None))[1],
        ))

    # Keep any static-only models that didn't appear in live (could be
    # newly-added manually or the API filtered them out).
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
