from __future__ import annotations

from world_state.config import source_config
from world_state.ingest.base import DataSource
from world_state.ingest.providers import ECCCProvider, NWSProvider, SyntheticProvider

PROVIDERS: dict[str, type[DataSource]] = {
    "synthetic": SyntheticProvider,
    "eccc": ECCCProvider,
    "nws": NWSProvider,
}


def create_provider(name: str, config: dict) -> DataSource:
    try:
        provider_class = PROVIDERS[name]
    except KeyError as error:
        raise KeyError(f"Provider is not implemented: {name}") from error
    return provider_class(source_config(name, config), config.get("http", {}))


def implemented_providers() -> tuple[str, ...]:
    return tuple(PROVIDERS)
