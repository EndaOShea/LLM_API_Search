"""Interactive and programmatic provider/model selector."""

from __future__ import annotations

from dataclasses import dataclass

from llm_api_search.providers.base import ModelInfo, ProviderInfo
from llm_api_search.discovery import discover


@dataclass
class Selection:
    """Result of a provider/model selection."""

    provider_key: str
    provider_info: ProviderInfo
    model: ModelInfo
    connection_snippet: str


def select_provider(
    provider_key: str | None = None,
    model_id: str | None = None,
    *,
    live: bool = True,
    interactive: bool = False,
    language: str = "python",
) -> Selection:
    """Select a provider and model, either programmatically or interactively.

    Args:
        provider_key: Provider key (e.g. "anthropic").  If None and
                      *interactive* is True, the user is prompted.
        model_id: Model ID to use.  If None and *interactive* is True,
                  the user is prompted.
        live: Fetch live model lists when True.
        interactive: When True, prompt the user via stdin for choices.
        language: Programming language for the connection snippet
                  (python, typescript, javascript, java, cpp).

    Returns:
        A ``Selection`` with provider info, model info, and a connection snippet.
    """
    all_info = discover(live=live)

    # --- resolve provider ---------------------------------------------------
    if provider_key is None and interactive:
        provider_key = _prompt_choice(
            "Select a provider:",
            list(all_info.keys()),
            labels=[info.name for info in all_info.values()],
        )
    elif provider_key is None:
        raise ValueError(
            "provider_key is required when interactive=False"
        )

    provider_key = provider_key.lower()
    if provider_key not in all_info:
        raise KeyError(f"Unknown provider: {provider_key!r}")
    info = all_info[provider_key]

    # --- resolve model -------------------------------------------------------
    if model_id is None and interactive:
        model_id = _prompt_choice(
            "Select a model:",
            [m.model_id for m in info.models],
        )
    elif model_id is None:
        # default to first model
        model_id = info.models[0].model_id

    model = next((m for m in info.models if m.model_id == model_id), None)
    if model is None:
        raise KeyError(
            f"Model {model_id!r} not found for {info.name}. "
            f"Available: {[m.model_id for m in info.models]}"
        )

    from llm_api_search.providers import PROVIDERS

    snippet = PROVIDERS[provider_key]().get_connection_snippet(model_id, language=language)

    return Selection(
        provider_key=provider_key,
        provider_info=info,
        model=model,
        connection_snippet=snippet,
    )


def _prompt_choice(
    prompt: str,
    keys: list[str],
    labels: list[str] | None = None,
) -> str:
    """Print a numbered menu and return the chosen key."""
    display = labels or keys
    print(f"\n{prompt}")
    for i, label in enumerate(display, 1):
        print(f"  [{i}] {label}")

    while True:
        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(keys):
                return keys[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"Please enter a number between 1 and {len(keys)}.")
