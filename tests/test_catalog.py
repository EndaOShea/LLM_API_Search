import re

from llm_api_search.catalog import build_catalog
from llm_api_search.providers import PROVIDERS


def test_catalog_top_level_shape():
    cat = build_catalog()
    assert cat["version"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cat["generated_at"])
    assert set(cat["providers"]) == set(PROVIDERS)


def test_catalog_models_are_text_only_with_required_fields():
    cat = build_catalog()
    for key, models in cat["providers"].items():
        assert len(models) > 0, f"{key} has no text models"
        for m in models:
            assert m["model_type"] == "text"
            for field in (
                "model_id", "display_name", "context_window",
                "input_cost_per_mtok", "output_cost_per_mtok",
            ):
                assert field in m, f"{key}:{m.get('model_id')} missing {field}"


def test_catalog_excludes_legacy_snapshots():
    # filter_models drops dated snapshots — no yyyy-mm-dd–suffixed ids
    # (except providers whose canonical id carries a date, e.g. claude-haiku-4-5-20251001).
    cat = build_catalog()
    openai_ids = [m["model_id"] for m in cat["providers"]["openai"]]
    assert not any(re.search(r"-\d{4}-\d{2}-\d{2}$", i) for i in openai_ids)


def test_catalog_route_registered():
    from mcp_server import build_app

    app = build_app("127.0.0.1", 0)
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/catalog.json" in paths
