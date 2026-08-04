"""Every ``_STATIC_MODELS`` block must already be in generator-canonical form.

The weekly auto-update regenerates each provider's ``_STATIC_MODELS`` block
from dataclass values via ``_serialize_models_block``. Strings are rendered
with ``repr()``, whose quote choice depends on the string's *content*:
``'...'`` normally, but ``"..."`` when the value contains an apostrophe and
no double-quote.

Hand-authored source, meanwhile, uses whatever quote style the curator typed.
When the two disagree the regenerated block is textually different but
semantically identical — and ``update-models.yml`` gates its PR on
``git diff --quiet llm_api_search/providers/``, so pure formatting churn is
reported as model drift and opens a PR titled "update model data from live
APIs" that changes no model data at all. That happened for real in PR #41:
two curated Qwen descriptions were authored with double quotes and contained
no apostrophe, so every update run rewrote them to single quotes.

This is the same class of defect as the no-op reorder PRs that
``test_update_models_ordering.py`` guards against — the generator leaking its
own formatting preferences into the file and dressing them up as drift.

The invariant enforced here: **regenerating a provider's block from its own
curated static data is a no-op.** A hand-edit in non-canonical style fails at
PR time, before the weekly run can misreport it as drift.

Fixing a failure: run ``python scripts/update_models.py <provider>`` (no API
key needed — it falls back to static data) and commit the reformatting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from llm_api_search.providers import PROVIDERS

# Load scripts/update_models.py as a module (it is not an importable package).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "update_models.py"
_spec = importlib.util.spec_from_file_location("update_models", _SCRIPT)
update_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_models)


def _regenerate_block(key: str) -> tuple[str, str]:
    """Return ``(on_disk_block, regenerated_block)`` for one provider.

    Mirrors ``update_provider``'s rewrite path exactly, minus the live fetch:
    read the file, locate the block, preserve trailing comments, re-serialize
    the provider's own curated models.
    """
    source_path = update_models._PROVIDER_FILES[key]
    source = source_path.read_text()
    match = update_models._STATIC_BLOCK_RE.search(source)
    assert match, f"{source_path.name}: no _STATIC_MODELS block found"

    on_disk = match.group(0)
    models = PROVIDERS[key]().get_static_info().models
    comments = update_models._extract_trailing_comments(on_disk)
    return on_disk, update_models._serialize_models_block(models, comments)


@pytest.mark.parametrize("key", sorted(PROVIDERS))
def test_static_block_is_generator_canonical(key: str) -> None:
    on_disk, regenerated = _regenerate_block(key)
    assert on_disk == regenerated, (
        f"{key}: _STATIC_MODELS is not in generator-canonical form, so the "
        f"next auto-update would rewrite it and report the reformatting as "
        f"model drift. Run `python scripts/update_models.py {key}` and commit "
        f"the result."
    )


def test_every_provider_file_is_covered() -> None:
    """_PROVIDER_FILES must track PROVIDERS, or a provider silently escapes.

    ``update_provider`` looks the path up by key, so a provider missing from
    ``_PROVIDER_FILES`` is skipped by the weekly update entirely — and by the
    canonical check above.
    """
    assert set(update_models._PROVIDER_FILES) == set(PROVIDERS), (
        "scripts/update_models.py `_PROVIDER_FILES` is out of sync with "
        "`PROVIDERS`: "
        f"missing={set(PROVIDERS) - set(update_models._PROVIDER_FILES)}, "
        f"extra={set(update_models._PROVIDER_FILES) - set(PROVIDERS)}"
    )
