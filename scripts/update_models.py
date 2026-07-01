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

import os
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
    "deepseek": PROJECT_ROOT / "llm_api_search" / "providers" / "deepseek.py",
    "zai": PROJECT_ROOT / "llm_api_search" / "providers" / "zai.py",
}

# Models to exclude from updates — superseded or deprecated model IDs per provider.
_EXCLUDED_MODELS: dict[str, set[str]] = {
    "inception": {"mercury", "mercury-coder"},
    "deepseek": {"deepseek-chat", "deepseek-reasoner"},
}

# Regex that matches the entire `_STATIC_MODELS = [...]` block.
_STATIC_BLOCK_RE = re.compile(
    r"^(_STATIC_MODELS\s*=\s*\[).*?(\n\])",
    re.MULTILINE | re.DOTALL,
)

# Pull the model_id off a field-assignment line at the start of a model entry.
_MODEL_ID_RE = re.compile(r"""^\s*model_id\s*=\s*['"]([^'"]+)['"]""")
# Pull a trailing inline comment off a `field=value,  # comment` line. Anchored
# to end-of-line so a `#` inside a string value (before the closing quote+comma)
# is not mistaken for a comment.
_TRAILING_COMMENT_RE = re.compile(r"^\s*(\w+)\s*=.*?,\s*(#.*?)\s*$")


def _extract_trailing_comments(block_text: str) -> dict[str, dict[str, str]]:
    """Map ``{model_id: {field_name: '# comment'}}`` from an existing block.

    Lets hand-authored inline notes (e.g. the Google pricing-tier breakdowns)
    survive the rewrite, which otherwise regenerates every line from dataclass
    values and drops comments. Keyed by (model_id, field) — not line position —
    so a comment follows its model even if the live API reorders the list.
    Auto-generated ``# TODO: add pricing`` notes are skipped; they're reproduced
    for any None-priced field anyway.
    """
    comments: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in block_text.splitlines():
        id_match = _MODEL_ID_RE.match(line)
        if id_match:
            current = id_match.group(1)
        if current is None:
            continue
        cm = _TRAILING_COMMENT_RE.match(line)
        if cm:
            field_name, comment = cm.group(1), cm.group(2).rstrip()
            if comment.startswith("# TODO: add pricing"):
                continue
            comments.setdefault(current, {})[field_name] = comment
    return comments


def _serialize_model(
    m: ModelInfo,
    comments: dict[str, str] | None = None,
    indent: str = "    ",
) -> str:
    """Render a ModelInfo subclass as a Python constructor call.

    ``comments`` maps field name → preserved trailing comment for this model,
    re-attached to the regenerated field line.
    """
    comments = comments or {}
    cls_name = type(m).__name__
    all_pricing = set()
    for pfields in _PRICING_FIELDS.values():
        all_pricing.update(pfields)

    lines = [f"{indent}{cls_name}("]
    for f in fields(type(m)):
        if f.name == "model_type":
            continue  # set automatically by subclass
        val = getattr(m, f.name)
        if f.name in all_pricing and val is None:
            # A video model priced per-video legitimately has cost_per_second
            # unset — that is not missing pricing, so don't stamp a TODO on it.
            priced_per_video = (
                isinstance(m, VideoModelInfo) and m.cost_per_video is not None
            )
            if not priced_per_video:
                # Generated TODO takes priority — never overlay a stale comment.
                lines.append(f"{indent}    {f.name}=None,  # TODO: add pricing")
                continue
            # else: fall through to emit a plain `cost_per_second=None,` line
        if f.name in all_pricing and isinstance(val, float):
            line = f"{indent}    {f.name}={val},"
        elif isinstance(val, bool):
            line = f"{indent}    {f.name}={val},"
        elif isinstance(val, int):
            line = f"{indent}    {f.name}={val:_},"
        else:
            # strings, lists, and any other value all round-trip via repr.
            line = f"{indent}    {f.name}={val!r},"
        comment = comments.get(f.name)
        if comment:
            line = f"{line}  {comment}"
        lines.append(line)
    lines.append(f"{indent}),")
    return "\n".join(lines)


def _serialize_models_block(
    models: list[ModelInfo],
    comments: dict[str, dict[str, str]] | None = None,
) -> str:
    """Render the full _STATIC_MODELS = [...] block."""
    comments = comments or {}
    parts = ["_STATIC_MODELS = ["]
    for m in models:
        parts.append(_serialize_model(m, comments.get(m.model_id)))
    parts.append("]")
    return "\n".join(parts)


def _merge_models(
    static_models: list[ModelInfo],
    live_models: list[ModelInfo],
    excluded: set[str],
) -> list[ModelInfo]:
    """Merge live model data into the curated static list, deterministically.

    Ordering is driven by the curated ``_STATIC_MODELS`` list, never by the
    live API's response order:

    * Existing curated models keep their position (so the selector default,
      ``models[0]``, and the "text models first" invariant are stable).
    * A curated model missing from the live response stays in its slot.
    * Genuinely-new live models are appended at the end (with ``pricing=None``
      so a human fills it in via the PR).

    Pricing, display_name, and description curation is preserved for known
    models — live values are only used for new ones.
    """
    pricing_map: dict[str, dict[str, float | None]] = {}
    for m in static_models:
        pfields = _PRICING_FIELDS.get(type(m), ())
        pricing_map[m.model_id] = {f: getattr(m, f) for f in pfields}
    static_map: dict[str, ModelInfo] = {m.model_id: m for m in static_models}
    live_map: dict[str, ModelInfo] = {
        m.model_id: m for m in live_models if m.model_id not in excluded
    }

    def _rebuild(model_id: str, live_m: ModelInfo) -> ModelInfo:
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

        return model_cls(**kwargs)

    merged: list[ModelInfo] = []

    # 1. Curated models in their existing order; refresh from live where present,
    #    otherwise keep the curated entry untouched.
    for model_id, static_m in static_map.items():
        live_m = live_map.get(model_id)
        merged.append(_rebuild(model_id, live_m) if live_m is not None else static_m)

    # 2. Genuinely-new live models, appended at the end in live order.
    for model_id, live_m in live_map.items():
        if model_id not in static_map:
            merged.append(_rebuild(model_id, live_m))

    return merged


def update_provider(key: str) -> tuple[int, int]:
    """Fetch live models for a provider and update its source file."""
    source_path = _PROVIDER_FILES[key]
    provider_cls = PROVIDERS[key]
    provider = provider_cls()

    static_info = provider.get_static_info()
    static_map: dict[str, ModelInfo] = {m.model_id: m for m in static_info.models}

    live_info = provider.fetch_live_models()
    excluded = _EXCLUDED_MODELS.get(key, set())

    merged = _merge_models(static_info.models, live_info.models, excluded)

    new_count = sum(1 for m in merged if m.model_id not in static_map)

    source = source_path.read_text()
    match = _STATIC_BLOCK_RE.search(source)
    if not match:
        print(f"  WARNING: Could not find _STATIC_MODELS block in {source_path.name}")
        return 0, len(merged)

    comments = _extract_trailing_comments(match.group(0))
    new_block = _serialize_models_block(merged, comments)
    updated = source[:match.start()] + new_block + source[match.end():]
    source_path.write_text(updated)

    return new_count, len(merged)


_NOTES_FILE = PROJECT_ROOT / "model_update_notes.md"


def _emit_discovery_notes(unrecognized: dict[str, set[str]]) -> None:
    """Surface live model IDs that aren't curated and aren't known aliases.

    These are providers (e.g. DeepSeek) whose live API lists a model the static
    data doesn't recognize and the update can't auto-add — a human must review
    and add it manually. Output goes to stdout, the GitHub Actions step summary
    (if running in CI), and ``model_update_notes.md`` for the PR body.
    """
    hits = {k: sorted(v) for k, v in unrecognized.items() if v}
    if not hits:
        # Clear any stale file so a later run's PR body doesn't show old hits.
        _NOTES_FILE.write_text("")
        return

    lines = ["### ⚠️ Unrecognized upstream models", ""]
    lines.append(
        "These IDs are served by the provider's live API but are neither "
        "curated in `_STATIC_MODELS` nor a known alias. They could not be "
        "auto-added — review and add them manually if they're real new models:"
    )
    lines.append("")
    for key, ids in hits.items():
        lines.append(f"- **{key}**: " + ", ".join(f"`{m}`" for m in ids))
    note = "\n".join(lines) + "\n"

    print("\n" + note)
    _NOTES_FILE.write_text(note)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(note)
    # A GitHub Actions annotation so the signal is visible on the run even when
    # there's no provider diff (hence no auto-update PR) to carry the note.
    if os.environ.get("GITHUB_ACTIONS"):
        flat = "; ".join(f"{k}: {', '.join(v)}" for k, v in hits.items())
        print(f"::warning title=Unrecognized upstream models::{flat}")


def main() -> None:
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(_PROVIDER_FILES.keys())

    unrecognized: dict[str, set[str]] = {}
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

        # Discovery signal for providers that keep fetch_live_models curated
        # (DeepSeek): genuinely-new live IDs never reach the merge above, so
        # flag them separately. unrecognized_live_model_ids never raises.
        unrecognized[key] = PROVIDERS[key]().unrecognized_live_model_ids()

    _emit_discovery_notes(unrecognized)

    print("\nDone. Run `pytest tests/` to verify.")


if __name__ == "__main__":
    main()
