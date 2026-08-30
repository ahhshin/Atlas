from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeAlias

import pandas as pd
import xarray as xr

from world_state.ingest.base import NormalizedPoint


@dataclass(frozen=True)
class ArtifactProvenance:
    source: str
    product: str
    data_class: str
    valid_time: datetime
    available_at: datetime
    ingested_at: datetime
    source_id: str
    source_url: str | None = None
    forecast_reference_time: datetime | None = None
    forecast_horizon_hours: int | None = None

    def identity(self, artifact_type: str) -> str:
        values = (
            artifact_type,
            self.source,
            self.product,
            self.source_id,
            self.valid_time.isoformat(),
            self.forecast_reference_time.isoformat() if self.forecast_reference_time else "",
            str(self.forecast_horizon_hours or ""),
        )
        return hashlib.sha256("|".join(values).encode()).hexdigest()


@dataclass(frozen=True)
class PointBatch:
    records: tuple[NormalizedPoint, ...]
    artifact_type: Literal["point_batch"] = field(default="point_batch", init=False)

    @property
    def valid_time(self) -> datetime:
        return max(record.valid_time for record in self.records)


@dataclass(frozen=True)
class GridField:
    dataset: xr.Dataset
    provenance: ArtifactProvenance
    variables: dict[str, str]
    bbox: tuple[float, float, float, float]
    native_resolution: str | None = None
    artifact_type: Literal["grid_field"] = field(default="grid_field", init=False)

    @property
    def asset_id(self) -> str:
        return self.provenance.identity(self.artifact_type)


@dataclass(frozen=True)
class ForecastField(GridField):
    artifact_type: Literal["forecast_field"] = field(default="forecast_field", init=False)

    def __post_init__(self) -> None:
        if self.provenance.forecast_reference_time is None:
            raise ValueError("ForecastField requires forecast_reference_time")
        if self.provenance.forecast_horizon_hours is None:
            raise ValueError("ForecastField requires forecast_horizon_hours")


@dataclass(frozen=True)
class EventCollection:
    events: pd.DataFrame
    provenance: ArtifactProvenance
    geometry_column: str = "geometry"
    crs: str = "OGC:CRS84"
    artifact_type: Literal["event_collection"] = field(default="event_collection", init=False)

    @property
    def asset_id(self) -> str:
        return self.provenance.identity(self.artifact_type)


NormalizedArtifact: TypeAlias = PointBatch | GridField | ForecastField | EventCollection
