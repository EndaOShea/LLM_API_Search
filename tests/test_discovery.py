"""Tests for the discovery and selector modules."""

import os
from unittest.mock import patch

from llm_api_search.discovery import discover, discover_provider, list_providers
from llm_api_search.selector import select_provider, Selection
from llm_api_search.providers.base import ProviderInfo, SUPPORTED_LANGUAGES


def test_list_providers():
    names = list_providers()
    assert "anthropic" in names
    assert "gemini" in names
    assert "openai" in names


def test_discover_provider_static():
    info = discover_provider("anthropic", live=False)
    assert isinstance(info, ProviderInfo)
    assert info.name == "Anthropic (Claude)"
    assert len(info.models) > 0
    assert info.sdk_package == "anthropic"


def test_discover_all_static():
    results = discover(live=False)
    assert set(results.keys()) == {"anthropic", "gemini", "openai"}
    for key, info in results.items():
        assert isinstance(info, ProviderInfo)
        assert len(info.models) > 0


def test_discover_subset():
    results = discover(live=False, providers=["anthropic", "openai"])
    assert set(results.keys()) == {"anthropic", "openai"}


def test_discover_provider_unknown():
    try:
        discover_provider("unknown_provider", live=False)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_select_provider_programmatic():
    sel = select_provider("anthropic", live=False)
    assert isinstance(sel, Selection)
    assert sel.provider_key == "anthropic"
    assert sel.model.model_id == sel.provider_info.models[0].model_id
    assert "anthropic" in sel.connection_snippet.lower() or "import" in sel.connection_snippet


def test_select_provider_with_model():
    sel = select_provider("openai", model_id="gpt-5.4", live=False)
    assert sel.model.model_id == "gpt-5.4"
    assert "gpt-5.4" in sel.connection_snippet


def test_select_provider_unknown_model():
    try:
        select_provider("anthropic", model_id="nonexistent-model", live=False)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_provider_summaries():
    results = discover(live=False)
    for info in results.values():
        summary = info.summary()
        assert info.name in summary
        assert info.api_base_url in summary


def test_connection_snippets_all_providers():
    for key in list_providers():
        sel = select_provider(key, live=False)
        assert "import" in sel.connection_snippet
        assert len(sel.connection_snippet) > 20


def test_provider_info_fields():
    """Verify all providers have required fields populated."""
    results = discover(live=False)
    for key, info in results.items():
        assert info.api_base_url.startswith("https://"), f"{key} missing https URL"
        assert info.auth_env_var, f"{key} missing auth_env_var"
        assert info.auth_header, f"{key} missing auth_header"
        assert info.sdk_package, f"{key} missing sdk_package"
        assert info.sdk_install.startswith("pip install"), f"{key} bad sdk_install"
        assert info.documentation_url.startswith("https://"), f"{key} missing docs URL"


def test_model_info_fields():
    """Verify static models have key fields set."""
    results = discover(live=False)
    for key, info in results.items():
        for m in info.models:
            assert m.model_id, f"{key}: model missing model_id"
            assert m.display_name, f"{key}: model missing display_name"


# --- Multi-language snippet tests ---


def test_connection_snippet_typescript():
    """Verify TypeScript snippets use TS import syntax."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="typescript")
        assert "import" in sel.connection_snippet
        # TS snippets should not use Python syntax
        assert "print(" not in sel.connection_snippet


def test_connection_snippet_javascript():
    """Verify JavaScript snippets use require syntax."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="javascript")
        assert "require(" in sel.connection_snippet
        assert "console.log" in sel.connection_snippet


def test_connection_snippet_java():
    """Verify Java snippets use Java import syntax."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="java")
        assert "import com." in sel.connection_snippet
        assert "System.out.println" in sel.connection_snippet


def test_connection_snippet_cpp():
    """Verify C++ snippets use libcurl and include directives."""
    for key in list_providers():
        sel = select_provider(key, live=False, language="cpp")
        assert "#include" in sel.connection_snippet
        assert "curl" in sel.connection_snippet.lower()


def test_connection_snippet_all_languages_all_providers():
    """Every provider should return a non-empty snippet for every supported language."""
    for key in list_providers():
        for lang in SUPPORTED_LANGUAGES:
            sel = select_provider(key, live=False, language=lang)
            assert len(sel.connection_snippet) > 20, (
                f"{key}/{lang}: snippet too short"
            )


def test_connection_snippet_model_id_in_all_languages():
    """Model ID should appear in snippets for all languages."""
    model = "claude-sonnet-4-6"
    for lang in SUPPORTED_LANGUAGES:
        sel = select_provider("anthropic", model_id=model, live=False, language=lang)
        assert model in sel.connection_snippet, (
            f"Model ID missing from {lang} snippet"
        )


def test_sdk_installs_per_language():
    """Verify sdk_installs contains entries for multiple languages."""
    results = discover(live=False)
    for key, info in results.items():
        assert "python" in info.sdk_installs, f"{key} missing python sdk_install"
        assert "typescript" in info.sdk_installs, f"{key} missing typescript sdk_install"
        assert "java" in info.sdk_installs, f"{key} missing java sdk_install"
        assert "cpp" in info.sdk_installs, f"{key} missing cpp sdk_install"


def test_sdk_install_for():
    """Verify sdk_install_for returns the correct install command."""
    info = discover_provider("anthropic", live=False)
    assert info.sdk_install_for("python") == "pip install anthropic"
    assert "npm" in info.sdk_install_for("typescript")
    assert info.sdk_install_for("cpp").lower().count("libcurl") or "REST" in info.sdk_install_for("cpp")


def test_unknown_language_falls_back_to_python():
    """Unknown language should fall back to Python snippet."""
    sel = select_provider("anthropic", live=False, language="rust")
    python_sel = select_provider("anthropic", live=False, language="python")
    assert sel.connection_snippet == python_sel.connection_snippet


def test_model_pricing_fields():
    """All static models should have pricing set."""
    results = discover(live=False)
    for key, info in results.items():
        for m in info.models:
            assert m.input_cost_per_mtok is not None, (
                f"{key}/{m.model_id}: missing input_cost_per_mtok"
            )
            assert m.output_cost_per_mtok is not None, (
                f"{key}/{m.model_id}: missing output_cost_per_mtok"
            )
            assert m.input_cost_per_mtok >= 0, (
                f"{key}/{m.model_id}: negative input cost"
            )
            assert m.output_cost_per_mtok >= 0, (
                f"{key}/{m.model_id}: negative output cost"
            )
            assert m.output_cost_per_mtok >= m.input_cost_per_mtok, (
                f"{key}/{m.model_id}: output cost should be >= input cost"
            )
