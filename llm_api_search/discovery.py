"""Discovery module — find the latest LLM API versions and models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_api_search.providers import PROVIDERS
from llm_api_search.providers.base import ProviderInfo


def list_providers() -> list[str]:
    """Return the names of all supported providers."""
    return list(PROVIDERS.keys())


def discover_provider(
    name: str,
    *,
    live: bool = True,
) -> ProviderInfo:
    """Discover API info for a single provider.

    Args:
        name: Provider key (e.g. "anthropic", "google", "openai").
        live: If True, attempt to fetch live model lists from the provider's
              API (requires the relevant API key in the environment).  Falls
              back to built-in static data on failure or when False.

    Returns:
        A ``ProviderInfo`` dataclass with models, URLs, and auth details.

    Raises:
        KeyError: If *name* is not a known provider.
    """
    name_lower = name.lower()
    if name_lower not in PROVIDERS:
        raise KeyError(
            f"Unknown provider {name!r}. "
            f"Available: {', '.join(PROVIDERS)}"
        )
    provider = PROVIDERS[name_lower]()
    if live:
        return provider.fetch_live_models()
    return provider.get_static_info()


def discover(
    *,
    live: bool = True,
    providers: list[str] | None = None,
) -> dict[str, ProviderInfo]:
    """Discover API info for all (or selected) providers in parallel.

    Args:
        live: Attempt live API calls when True.
        providers: Optional list of provider keys to query.  Defaults to all.

    Returns:
        A dict mapping provider key → ``ProviderInfo``.
    """
    targets = providers or list(PROVIDERS.keys())
    results: dict[str, ProviderInfo] = {}

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        futures = {
            pool.submit(discover_provider, name, live=live): name
            for name in targets
        }
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    return results
