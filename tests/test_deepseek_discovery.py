"""Tests for the DeepSeek new-model discovery signal.

DeepSeek's ``fetch_live_models`` intentionally keeps runtime output curated
(see provider docstring), so the weekly update can't discover a brand-new
DeepSeek model the normal way. ``unrecognized_live_model_ids`` is the separate,
update-time signal that flags live IDs we neither curate nor treat as a known
generic alias. These tests cover the pure classification and the no-network
fallbacks (kept consistent with the repo convention of avoiding live calls).
"""

import os
from unittest.mock import patch

from llm_api_search.providers.deepseek import (
    DeepSeekProvider, _classify_unrecognized, _KNOWN_LIVE_ALIASES,
)
from llm_api_search.providers.anthropic import AnthropicProvider


def test_classify_surfaces_genuinely_new_id():
    live = {"deepseek-v4-pro", "deepseek-chat", "deepseek-v5"}
    static = {"deepseek-v4-pro", "deepseek-v4-flash"}
    aliases = {"deepseek-chat", "deepseek-reasoner"}
    assert _classify_unrecognized(live, static, aliases) == {"deepseek-v5"}


def test_classify_excludes_known_aliases():
    live = {"deepseek-chat", "deepseek-reasoner"}
    static = {"deepseek-v4-pro"}
    aliases = {"deepseek-chat", "deepseek-reasoner"}
    assert _classify_unrecognized(live, static, aliases) == set()


def test_classify_excludes_curated_static():
    live = {"deepseek-v4-pro", "deepseek-v4-flash"}
    static = {"deepseek-v4-pro", "deepseek-v4-flash"}
    assert _classify_unrecognized(live, static, set()) == set()


def test_classify_empty_live_is_empty():
    assert _classify_unrecognized(set(), {"deepseek-v4-pro"}, _KNOWN_LIVE_ALIASES) == set()


def test_deepseek_returns_empty_without_api_key():
    with patch.dict(os.environ, {}, clear=True):
        assert DeepSeekProvider().unrecognized_live_model_ids() == set()


def test_base_default_returns_empty():
    # Providers that don't curate against live data inherit the empty default.
    with patch.dict(os.environ, {}, clear=True):
        assert AnthropicProvider().unrecognized_live_model_ids() == set()
