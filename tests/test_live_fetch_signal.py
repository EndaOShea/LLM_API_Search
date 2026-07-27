"""Tests for the live-fetch failure signal and Anthropic /v1/models pagination.

A provider whose live API call fails silently reuses its curated static list.
That is the right runtime behaviour, but it means a broken API key and a week
with no new models produce byte-identical output — which is how a released
model can stay missing from the catalog indefinitely. ``live_fetch_error``
makes the fallback visible without changing what ``fetch_live_models`` returns.

These tests stub ``urllib.request.urlopen``; none of them touch the network.
"""

import json
import urllib.error
import urllib.request

from llm_api_search.providers import PROVIDERS
from llm_api_search.providers import anthropic as anthropic_mod


class _FakeResponse:
    """Minimal stand-in for the object ``urlopen`` yields as a context manager."""

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_live_fetch_error_defaults_to_none():
    """A freshly constructed provider has no recorded failure."""
    for key, cls in PROVIDERS.items():
        assert cls().live_fetch_error is None, key


def test_missing_api_key_is_not_reported_as_a_failure(monkeypatch):
    """No key is a skip, not a failure — CI without secrets must stay quiet."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = PROVIDERS["anthropic"]()
    provider.fetch_live_models()
    assert provider.live_fetch_error is None


def test_failed_fetch_is_recorded_and_falls_back_to_static(monkeypatch):
    """An HTTP error is recorded, and the returned models are still the curated ones."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    provider = PROVIDERS["anthropic"]()
    info = provider.fetch_live_models()

    assert provider.live_fetch_error is not None
    assert "401" in provider.live_fetch_error
    # Report-only: the fallback still hands back the full curated list.
    static_ids = {m.model_id for m in provider.get_static_info().models}
    assert {m.model_id for m in info.models} == static_ids


def test_empty_live_catalog_is_flagged(monkeypatch):
    """A 200 with zero models is not a normal outcome and must not pass silently."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse({"data": [], "has_more": False}),
    )

    provider = PROVIDERS["anthropic"]()
    info = provider.fetch_live_models()

    assert provider.live_fetch_error == "live /v1/models returned no models"
    assert info.models  # static list retained


def test_anthropic_fetch_follows_pagination(monkeypatch):
    """Models past the first page are collected, not silently truncated."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    pages = [
        {
            "data": [{"id": "claude-page1-a", "display_name": "P1A"}],
            "has_more": True,
            "last_id": "claude-page1-a",
        },
        {
            "data": [{"id": "claude-page2-b", "display_name": "P2B"}],
            "has_more": False,
        },
    ]
    seen_urls: list[str] = []

    def fake_urlopen(req, timeout=None):
        seen_urls.append(req.full_url)
        return _FakeResponse(pages[len(seen_urls) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    info = PROVIDERS["anthropic"]().fetch_live_models()

    assert [m.model_id for m in info.models] == ["claude-page1-a", "claude-page2-b"]
    assert len(seen_urls) == 2
    assert f"limit={anthropic_mod._MODELS_PAGE_SIZE}" in seen_urls[0]
    assert "after_id=claude-page1-a" in seen_urls[1]


def test_anthropic_fetch_stops_on_single_page(monkeypatch):
    """has_more=False means exactly one request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _FakeResponse(
            {"data": [{"id": "claude-only", "display_name": "Only"}], "has_more": False}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    info = PROVIDERS["anthropic"]().fetch_live_models()

    assert len(calls) == 1
    assert [m.model_id for m in info.models] == ["claude-only"]


def test_anthropic_pagination_is_bounded(monkeypatch):
    """A never-terminating has_more can't spin forever."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        mid = f"claude-m{len(calls)}"
        return _FakeResponse({"data": [{"id": mid}], "has_more": True, "last_id": mid})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    PROVIDERS["anthropic"]().fetch_live_models()

    assert len(calls) == anthropic_mod._MAX_MODEL_PAGES
