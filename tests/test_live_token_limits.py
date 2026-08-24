"""Tests that live-published token limits survive the trip onto disk.

Google's ``/v1beta/models`` publishes ``inputTokenLimit`` and
``outputTokenLimit`` for most endpoints, but that data used to be dropped
*twice* on the way into ``_STATIC_MODELS``:

1. ``GeminiProvider.fetch_live_models`` mapped only ``name``/``displayName``/
   ``description``, so the limits never left the JSON response, and
2. ``_merge_models._rebuild`` built brand-new models from those three fields
   alone, letting every other field fall to its dataclass default.

Either leak alone is enough to make an auto-added model land with
``context_window=None``, which is how the weekly update PR kept producing
Gemini entries that a human had to fill in by hand. Because the two layers
compose, fixing one without the other is invisible — hence a test per layer
plus one end-to-end.

These tests stub ``urllib.request.urlopen``; none of them touch the network.
"""

import json
import urllib.request

from llm_api_search.providers import PROVIDERS
from llm_api_search.providers.base import TextModelInfo
from scripts.update_models import _merge_models


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


def _stub_gemini_response(monkeypatch, models: list[dict]) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *a, **kw: _FakeResponse({"models": models}),
    )


def test_fetch_carries_token_limits_through(monkeypatch):
    """Layer 1: the fetcher must read the limits the API publishes."""
    _stub_gemini_response(monkeypatch, [
        {
            "name": "models/gemini-9-flash",
            "displayName": "Gemini 9 Flash",
            "description": "A model from the future.",
            "inputTokenLimit": 2_097_152,
            "outputTokenLimit": 65_536,
        },
    ])

    info = PROVIDERS["google"]().fetch_live_models()
    model = next(m for m in info.models if m.model_id == "gemini-9-flash")

    assert model.context_window == 2_097_152
    assert model.max_output_tokens == 65_536


def test_fetch_tolerates_endpoints_that_publish_no_limits(monkeypatch):
    """Live API and robotics endpoints omit the fields; that must not raise."""
    _stub_gemini_response(monkeypatch, [
        {
            "name": "models/gemini-9-flash-live-preview",
            "displayName": "Gemini 9 Flash Live Preview",
            "description": "Audio-to-audio, no published limits.",
        },
    ])

    info = PROVIDERS["google"]().fetch_live_models()
    model = next(
        m for m in info.models if m.model_id == "gemini-9-flash-live-preview"
    )

    assert model.context_window is None
    assert model.max_output_tokens is None


def test_merge_propagates_live_fields_for_new_models():
    """Layer 2: a genuinely-new model keeps what the live API supplied."""
    curated = [
        TextModelInfo(
            model_id="gemini-existing",
            display_name="Gemini Existing",
            description="Curated.",
            context_window=1_000_000,
            max_output_tokens=65_536,
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=2.0,
        ),
    ]
    live = curated + [
        TextModelInfo(
            model_id="gemini-brand-new",
            display_name="Gemini Brand New",
            description="From the live API.",
            context_window=2_097_152,
            max_output_tokens=32_768,
        ),
    ]

    merged = _merge_models(curated, live, excluded=set())
    new = next(m for m in merged if m.model_id == "gemini-brand-new")

    assert new.context_window == 2_097_152
    assert new.max_output_tokens == 32_768
    # Pricing is still the human's job — the live API does not serve it.
    assert new.input_cost_per_mtok is None
    assert new.output_cost_per_mtok is None


def test_merge_does_not_let_live_data_overwrite_curation():
    """Curated limits and pricing must win over whatever the live API says.

    The propagation above is scoped to *new* models only; without that scope
    a live response could silently revert hand-corrected catalog data.
    """
    curated = [
        TextModelInfo(
            model_id="gemini-existing",
            display_name="Gemini Existing",
            description="Hand-written description.",
            context_window=1_000_000,
            max_output_tokens=65_536,
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=2.0,
        ),
    ]
    live = [
        TextModelInfo(
            model_id="gemini-existing",
            display_name="Auto Name",
            description="Auto description.",
            context_window=123,
            max_output_tokens=456,
        ),
    ]

    merged = _merge_models(curated, live, excluded=set())
    model = next(m for m in merged if m.model_id == "gemini-existing")

    assert model.context_window == 1_000_000
    assert model.max_output_tokens == 65_536
    assert model.description == "Hand-written description."
    assert model.input_cost_per_mtok == 1.0
