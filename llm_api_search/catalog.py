"""Plain-JSON catalog export for downstream sync scripts.

Serves the same curated static model data the MCP tools expose, but as a
single dict suitable for a REST route — so consumers can sync with a bare
HTTP fetch instead of an MCP session handshake. Each text model carries a
"thinking" block (mode, control parameter, levels/budgets, can_disable,
and thinking-time sampling constraints) serialized from the same
ThinkingConfig registry the llm_get_thinking_config MCP tool serves.
"""
from __future__ import annotations

import dataclasses

from llm_api_search.data_version import DATA_UPDATED
from llm_api_search.providers import (
    PROVIDERS, filter_models, get_thinking_config, thinking_config_to_dict,
)
from llm_api_search.providers.base import ModelType

CATALOG_VERSION = 1


def build_catalog() -> dict:
    providers: dict[str, list[dict]] = {}
    for key, cls in PROVIDERS.items():
        info = cls().get_static_info()
        models = filter_models(info.models, key)
        entries: list[dict] = []
        for m in models:
            if m.model_type != ModelType.TEXT:
                continue
            d = dataclasses.asdict(m)
            # Coerce the str-Enum to its plain string value so the JSON export
            # is clean ("text") rather than a ModelType member repr.
            d["model_type"] = m.model_type.value
            # How to call the model: thinking mode/knob and thinking-time
            # sampling constraints. Non-capable models get an explicit
            # {"supported": false, "mode": "none"} rather than no key, so
            # "not capable" stays distinguishable from "not recorded".
            tc = get_thinking_config(key, m.model_id)[m.model_id]
            d["thinking"] = thinking_config_to_dict(tc)
            entries.append(d)
        providers[key] = entries
    return {
        "version": CATALOG_VERSION,
        "generated_at": DATA_UPDATED,
        "providers": providers,
    }
