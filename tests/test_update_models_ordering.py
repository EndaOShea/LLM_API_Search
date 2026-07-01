"""Tests for deterministic model ordering in scripts/update_models.py.

The weekly auto-update regenerates each provider's ``_STATIC_MODELS`` block.
Ordering must be driven by the *curated* static list, not by the live API's
response order — otherwise an upstream reshuffle produces no-op reorder PRs
and can silently change the default model (the selector uses ``models[0]``).

See issue #15.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_api_search.providers.base import TextModelInfo

# Load scripts/update_models.py as a module (it is not an importable package).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "update_models.py"
_spec = importlib.util.spec_from_file_location("update_models", _SCRIPT)
update_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(update_models)


def _model(model_id: str, **kwargs) -> TextModelInfo:
    return TextModelInfo(model_id=model_id, display_name=model_id, **kwargs)


def _ids(models) -> list[str]:
    return [m.model_id for m in models]


def test_reordered_live_preserves_curated_order():
    """Same IDs, different live order → output keeps the curated order."""
    static = [_model("a"), _model("b"), _model("c")]
    live = [_model("c"), _model("a"), _model("b")]  # API reshuffled

    merged = update_models._merge_models(static, live, excluded=set())

    assert _ids(merged) == ["a", "b", "c"]


def test_new_live_model_appended_at_end():
    """A genuinely-new live model lands at the end, regardless of live order."""
    static = [_model("a"), _model("b")]
    live = [_model("new"), _model("b"), _model("a")]  # new model first in API

    merged = update_models._merge_models(static, live, excluded=set())

    assert _ids(merged) == ["a", "b", "new"]


def test_curated_only_model_keeps_position():
    """A static model missing from the live list stays in its curated slot."""
    static = [_model("a"), _model("b"), _model("c")]
    live = [_model("a"), _model("c")]  # 'b' not returned this week

    merged = update_models._merge_models(static, live, excluded=set())

    assert _ids(merged) == ["a", "b", "c"]


def test_curated_pricing_preserved_when_live_reorders():
    """Reordering must not drop curated pricing for known models."""
    static = [
        _model("a", input_cost_per_mtok=0.25, output_cost_per_mtok=0.75),
        _model("b", input_cost_per_mtok=1.0, output_cost_per_mtok=2.0),
    ]
    live = [_model("b"), _model("a")]  # live carries no pricing

    merged = update_models._merge_models(static, live, excluded=set())
    by_id = {m.model_id: m for m in merged}

    assert by_id["a"].input_cost_per_mtok == 0.25
    assert by_id["a"].output_cost_per_mtok == 0.75
    assert by_id["b"].input_cost_per_mtok == 1.0


def test_zai_registered_in_update_files():
    from scripts.update_models import _PROVIDER_FILES
    assert "zai" in _PROVIDER_FILES
    assert _PROVIDER_FILES["zai"].name == "zai.py"


def test_zai_merge_preserves_order_and_types():
    # Re-merging the curated models against themselves must be a no-op that
    # keeps glm-5.2 first and preserves image/video subtypes + pricing.
    from scripts.update_models import _merge_models
    from llm_api_search.providers.zai import _STATIC_MODELS
    from llm_api_search.providers.base import ImageModelInfo, VideoModelInfo
    merged = _merge_models(list(_STATIC_MODELS), list(_STATIC_MODELS), set())
    assert [m.model_id for m in merged] == [m.model_id for m in _STATIC_MODELS]
    by_id = {m.model_id: m for m in merged}
    assert isinstance(by_id["cogview-4"], ImageModelInfo)
    assert by_id["cogview-4"].cost_per_image == 0.01
    assert isinstance(by_id["cogvideox-3"], VideoModelInfo)
    assert by_id["cogvideox-3"].cost_per_video == 0.20
