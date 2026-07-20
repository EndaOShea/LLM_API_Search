"""Tests for the Qwen new-model discovery signal.

Qwen's ``fetch_live_models`` intentionally stays curated (DashScope's live
catalog spans hundreds of non-frontier model IDs — see provider docstring),
so the weekly update can't discover a brand-new Qwen model the normal way.
``unrecognized_live_model_ids`` is the separate, pattern-filtered signal:
only IDs shaped like a new frontier generation (qwen{N}.{M}-max/-plus) get
flagged, so the rest of DashScope's catalog doesn't generate weekly noise.
"""

import os
from unittest.mock import patch

from llm_api_search.providers.qwen import QwenProvider, _classify_unrecognized
from llm_api_search.providers.anthropic import AnthropicProvider


def test_classify_surfaces_new_frontier_generation():
    live = {"qwen3.7-max", "qwen3.7-plus", "qwen3.8-max"}
    static = {"qwen3.7-max", "qwen3.7-plus"}
    assert _classify_unrecognized(live, static) == {"qwen3.8-max"}


def test_classify_ignores_non_frontier_catalog_noise():
    live = {
        "qwen3.7-max", "qwen-vl-plus", "qwen2.5-72b-instruct",
        "qwen-audio-turbo", "qwen3.6-flash", "qwen-coder-plus",
    }
    static = {"qwen3.7-max", "qwen3.7-plus"}
    assert _classify_unrecognized(live, static) == set()


def test_classify_excludes_curated_static():
    live = {"qwen3.7-max", "qwen3.7-plus"}
    static = {"qwen3.7-max", "qwen3.7-plus"}
    assert _classify_unrecognized(live, static) == set()


def test_classify_empty_live_is_empty():
    assert _classify_unrecognized(set(), {"qwen3.7-max"}) == set()


def test_qwen_returns_empty_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        assert QwenProvider().unrecognized_live_model_ids() == set()


def test_base_default_returns_empty():
    with patch.dict(os.environ, {}, clear=True):
        assert AnthropicProvider().unrecognized_live_model_ids() == set()
