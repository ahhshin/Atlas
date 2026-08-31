from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml


@dataclass(frozen=True)
class VariableSpec:
    source_name: str
    units: str
    temporal_kind: Literal["instant", "accumulation"] = "instant"
    scale: float = 1.0


VARIABLES: dict[str, VariableSpec] = {
    "temperature_2m": VariableSpec("t2m", "K"),
    "dewpoint_2m": VariableSpec("d2m", "K"),
    "surface_pressure": VariableSpec("sp", "Pa"),
    "u_wind_10m": VariableSpec("u10", "m s**-1"),
    "v_wind_10m": VariableSpec("v10", "m s**-1"),
    "total_precipitation": VariableSpec("tp", "mm", "accumulation", 1000.0),
    "total_cloud_cover": VariableSpec("tcc", "1"),
    "surface_solar_radiation_downwards": VariableSpec("ssrd", "J m**-2", "accumulation"),
    "soil_moisture_layer_1": VariableSpec("swvl1", "m**3 m**-3"),
    "skin_temperature": VariableSpec("skt", "K"),
}


@dataclass(frozen=True)
class BoundingBox:
    west: float
    south: float
    east: float
    north: float

    def shape(self, resolution: float) -> tuple[int, int]:
        latitude = round((self.north - self.south) / resolution) + 1
        longitude = round((self.east - self.west) / resolution) + 1
        return latitude, longitude


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> DateRange:
        return cls(_timestamp(value["start"]), _end_timestamp(value["end"]))

    def contains(self, timestamp: pd.Timestamp) -> bool:
        return self.start <= timestamp <= self.end


@dataclass(frozen=True)
class SplitConfig:
    train: DateRange
    validation: DateRange
    test: DateRange


@dataclass(frozen=True)
class ResearchConfig:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    bbox: BoundingBox
    resolution_degrees: float
    cadence_hours: int
    context_hours: int
    target_hours: int
    variables: tuple[str, ...]
    max_storage_gb: float
    storage_root: Path
    required_mount: Path | None
    source: dict[str, Any]
    splits: SplitConfig
    percentile: float = 0.95
    chunk_time: int = 8
    chunk_latitude: int = 32
    chunk_longitude: int = 32
    random_seed: int = 2020
    samples_per_time: int = 64
    max_train_samples: int = 300_000
    max_eval_samples: int = 150_000
    max_replay_times: int = 64
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResearchConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        split_raw = raw["splits"]
        config = cls(
            name=str(raw["name"]),
            start=_timestamp(raw["start"]),
            end=_end_timestamp(raw["end"]),
            bbox=BoundingBox(**raw["bbox"]),
            resolution_degrees=float(raw["resolution_degrees"]),
            cadence_hours=int(raw["cadence_hours"]),
            context_hours=int(raw.get("context_hours", 24)),
            target_hours=int(raw.get("target_hours", 6)),
            variables=tuple(raw["variables"]),
            max_storage_gb=float(raw["max_storage_gb"]),
            storage_root=Path(raw.get("storage_root", "/mnt/games/Atlas/data")),
            required_mount=(Path(raw["required_mount"]) if raw.get("required_mount") else None),
            source=dict(raw.get("source", {})),
            splits=SplitConfig(
                train=DateRange.from_mapping(split_raw["train"]),
                validation=DateRange.from_mapping(split_raw["validation"]),
                test=DateRange.from_mapping(split_raw["test"]),
            ),
            percentile=float(raw.get("percentile", 0.95)),
            chunk_time=int(raw.get("chunks", {}).get("time", 8)),
            chunk_latitude=int(raw.get("chunks", {}).get("latitude", 32)),
            chunk_longitude=int(raw.get("chunks", {}).get("longitude", 32)),
            random_seed=int(raw.get("training", {}).get("random_seed", 2020)),
            samples_per_time=int(raw.get("training", {}).get("samples_per_time", 64)),
            max_train_samples=int(raw.get("training", {}).get("max_train_samples", 300_000)),
            max_eval_samples=int(raw.get("training", {}).get("max_eval_samples", 150_000)),
            max_replay_times=int(raw.get("training", {}).get("max_replay_times", 64)),
            config_path=config_path,
        )
        config.validate()
        return config

    @property
    def context_steps(self) -> int:
        return self.context_hours // self.cadence_hours

    @property
    def target_steps(self) -> int:
        return self.target_hours // self.cadence_hours

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.start.year, self.end.year + 1))

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq=f"{self.cadence_hours}h")

    def validate(self) -> None:
        if self.start > self.end:
            raise ValueError("research start must not be after end")
        if self.resolution_degrees <= 0 or self.cadence_hours <= 0:
            raise ValueError("resolution and cadence must be positive")
        if self.context_hours % self.cadence_hours:
            raise ValueError("context_hours must be divisible by cadence_hours")
        if self.target_hours % self.cadence_hours:
            raise ValueError("target_hours must be divisible by cadence_hours")
        if not (0 < self.percentile < 1):
            raise ValueError("percentile must be between zero and one")
        unknown = sorted(set(self.variables) - VARIABLES.keys())
        if unknown:
            raise ValueError(f"unknown research variables: {', '.join(unknown)}")
        if "total_precipitation" not in self.variables:
            raise ValueError("total_precipitation is required for the target")
        if not (self.bbox.west < self.bbox.east and self.bbox.south < self.bbox.north):
            raise ValueError("invalid bounding box")
        ranges = (self.splits.train, self.splits.validation, self.splits.test)
        for earlier, later in pairwise(ranges):
            if earlier.end >= later.start:
                raise ValueError("chronological split ranges overlap or are out of order")


def _timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _end_timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    timestamp = _timestamp(value)
    raw = str(value)
    if len(raw) == 10:
        timestamp += pd.Timedelta(hours=23, minutes=59, seconds=59)
    return timestamp
