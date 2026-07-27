"""Tests for Mistral's new-frontier-model discovery signal.

Mistral's live /v1/models keeps every legacy snapshot and a moving alias for
each family online indefinitely, so a family-prefix match is far too loose:
the 2026-07-27 auto-update flagged 14 IDs, none of which was a new model.

The hardened rule reports an ID only when it carries a YYMM release stamp
strictly newer than the newest curated stamp in the same family. These tests
pin both directions — the noise stays silent, and a real release still fires.

All pure functions; no network.
"""

from llm_api_search.providers.mistral import (
    MistralProvider,
    _classify_unrecognized,
    _family_of,
    _release_stamp,
)

# The curated frontier text IDs, as they appear in _STATIC_MODELS.
_CURATED = {
    "mistral-medium-3-5-26-04",  # stamp 2604
    "mistral-small-2603",        # stamp 2603
    "mistral-large-3-25-12",     # stamp 2512
    "codestral-2508",            # stamp 2508
}

# Verbatim: every ID the 2026-07-27 run flagged. All noise.
_JULY_27_FLAGGED = {
    "codestral-embed",
    "codestral-embed-2505",
    "codestral-latest",
    "mistral-large-2512",
    "mistral-large-latest",
    "mistral-medium-2505",
    "mistral-medium-2508",
    "mistral-medium-2604",
    "mistral-medium-3",
    "mistral-medium-3-5",
    "mistral-medium-3.5",
    "mistral-medium-latest",
    "mistral-small-2506",
    "mistral-small-latest",
}


def test_release_stamp_parses_both_id_shapes():
    assert _release_stamp("mistral-small-2603") == 2603
    assert _release_stamp("codestral-2508") == 2508
    assert _release_stamp("mistral-medium-3-5-26-04") == 2604
    assert _release_stamp("mistral-large-3-25-12") == 2512


def test_release_stamp_rejects_aliases_and_bare_versions():
    """An interior version fragment must never be read as a date."""
    assert _release_stamp("mistral-medium-latest") is None
    assert _release_stamp("mistral-medium-3") is None
    assert _release_stamp("mistral-medium-3-5") is None
    assert _release_stamp("mistral-medium-3.5") is None


def test_family_detection():
    assert _family_of("mistral-medium-2604") == "mistral-medium"
    assert _family_of("codestral-2508") == "codestral"
    assert _family_of("mistral-large-3-25-12") == "mistral-large"
    # Not frontier text families:
    assert _family_of("mistral-embed") is None
    assert _family_of("voxtral-tts-26-03") is None
    assert _family_of("ministral-3b-2410") is None


def test_july_27_noise_is_fully_silenced():
    """The regression this hardening exists for: 14 flagged IDs, 0 real."""
    assert _classify_unrecognized(_JULY_27_FLAGGED, _CURATED) == set()


def test_newer_release_is_still_flagged():
    """Hardening must not cost recall — a real new release still fires."""
    live = _CURATED | {"mistral-medium-2707"}
    assert _classify_unrecognized(live, _CURATED) == {"mistral-medium-2707"}


def test_newer_release_in_split_stamp_form_is_flagged():
    live = _CURATED | {"mistral-large-4-27-01"}
    assert _classify_unrecognized(live, _CURATED) == {"mistral-large-4-27-01"}


def test_stamp_comparison_is_per_family():
    """A large-family model newer than large's 2512 but older than medium's
    2604 must still be flagged — families are compared independently."""
    live = _CURATED | {"mistral-large-2601"}
    assert _classify_unrecognized(live, _CURATED) == {"mistral-large-2601"}


def test_year_boundary_compares_correctly():
    """2512 (Dec 2025) is older than 2603 (Mar 2026), not newer."""
    live = _CURATED | {"mistral-small-2512"}
    assert _classify_unrecognized(live, _CURATED) == set()


def test_non_text_classes_are_never_flagged():
    """codestral-embed is an embedding model; a new one is not a chat release."""
    live = _CURATED | {"codestral-embed-2707", "codestral-ocr-2707"}
    assert _classify_unrecognized(live, _CURATED) == set()


def test_family_with_nothing_curated_flags_its_first_dated_model():
    live = {"codestral-2601"}
    assert _classify_unrecognized(live, {"mistral-medium-3-5-26-04"}) == {"codestral-2601"}


def test_curated_ids_are_never_flagged():
    assert _classify_unrecognized(_CURATED, _CURATED) == set()


def test_empty_live_catalog_flags_nothing():
    assert _classify_unrecognized(set(), _CURATED) == set()


def test_every_curated_text_model_has_a_parseable_stamp():
    """The rule is only sound if each curated frontier ID yields a stamp —
    an unparseable one would floor its family at 0 and flag everything."""
    static_ids = {m.model_id for m in MistralProvider().get_static_info().models}
    for mid in static_ids:
        if _family_of(mid) is not None:
            assert _release_stamp(mid) is not None, mid


def test_provider_signal_is_quiet_without_credentials(monkeypatch):
    """unrecognized_live_model_ids must never raise (contract from base)."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert MistralProvider().unrecognized_live_model_ids() == set()
