from __future__ import annotations

from world_state.config import source_config
from world_state.ingest.base import DataSource
from world_state.ingest.providers import (
    AirNowProvider,
    ECCCProvider,
    EiaProvider,
    FirmsProvider,
    GoesProvider,
    HrrrProvider,
    MetarProvider,
    MrmsProvider,
    NWSProvider,
    ProbSevereProvider,
    RtmaProvider,
    SyntheticProvider,
    UsgsProvider,
)

PROVIDERS: dict[str, type[DataSource]] = {
    "synthetic": SyntheticProvider,
    "eccc": ECCCProvider,
    "nws": NWSProvider,
    "metar": MetarProvider,
    "rtma": RtmaProvider,
    "mrms": MrmsProvider,
    "probsevere": ProbSevereProvider,
    "hrrr": HrrrProvider,
    "goes": GoesProvider,
    "usgs": UsgsProvider,
    "firms": FirmsProvider,
    "airnow": AirNowProvider,
    "eia": EiaProvider,
}


def create_provider(name: str, config: dict) -> DataSource:
    try:
        provider_class = PROVIDERS[name]
    except KeyError as error:
        raise KeyError(f"Provider is not implemented: {name}") from error
    return provider_class(source_config(name, config), config.get("http", {}))


def implemented_providers() -> tuple[str, ...]:
    return tuple(PROVIDERS)
