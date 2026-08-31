from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from world_state.research.config import VARIABLES, BoundingBox, ResearchConfig
from world_state.research.dataset import AtlasMiniDataset
from world_state.research.experiments import train_experiment
from world_state.research.metrics import score_probabilities
from world_state.research.models import build_tabular_samples
from world_state.research.pipeline import backfill, build_splits, validate_dataset
from world_state.research.source import EarthmoverERA5Source
from world_state.research.storage import (
    StorageGuardError,
    enforce_preflight,
    estimate_storage,
    inspect_storage,
)


class FixtureSource:
    def fetch_year(self, config: ResearchConfig, year: int) -> xr.Dataset:
        times = config.timestamps[config.timestamps.year == year]
        latitude = np.linspace(config.bbox.south, config.bbox.north, 3, dtype="float32")
        longitude = np.linspace(config.bbox.west, config.bbox.east, 3, dtype="float32")
        time_index = np.arange(len(times), dtype="float32")[:, None, None]
        lat_index = np.arange(3, dtype="float32")[None, :, None]
        lon_index = np.arange(3, dtype="float32")[None, None, :]
        output = {}
        for index, name in enumerate(config.variables):
            values = np.broadcast_to(
                index + time_index * 0.1 + lat_index * 0.01 + lon_index * 0.001,
                (len(times), 3, 3),
            ).copy()
            if name == "total_precipitation":
                values = (
                    (time_index % 11 == 0) * (lat_index + 1) * (2 + time_index / 30)
                    + (lon_index == 2) * 0.2
                ).astype("float32")
            if name == "temperature_2m":
                values += 270
                values[2, 0, 0] = np.nan
            output[name] = (("time", "latitude", "longitude"), values.astype("float32"))
            output[f"missing_{name}"] = (
                ("time", "latitude", "longitude"),
                ~np.isfinite(values),
            )
        dataset = xr.Dataset(
            output,
            coords={"time": times, "latitude": latitude, "longitude": longitude},
            attrs={
                "source": "fixture-era5",
                "data_class": "RETROSPECTIVE_REANALYSIS",
                "retrospective": True,
                "cadence_hours": config.cadence_hours,
                "resolution_degrees": config.resolution_degrees,
            },
        )
        for name in config.variables:
            dataset[name].attrs["units"] = VARIABLES[name].units
        return dataset.chunk({"time": 8, "latitude": 3, "longitude": 3})


@pytest.fixture
def research_config(tmp_path: Path) -> ResearchConfig:
    raw = {
        "name": "fixture-mini",
        "start": "2020-01-01",
        "end": "2020-01-10",
        "bbox": {"west": -1, "south": 0, "east": 1, "north": 2},
        "resolution_degrees": 1,
        "cadence_hours": 3,
        "context_hours": 24,
        "target_hours": 6,
        "percentile": 0.95,
        "max_storage_gb": 0.1,
        "storage_root": str(tmp_path / "atlas-data"),
        "variables": list(VARIABLES),
        "splits": {
            "train": {"start": "2020-01-01", "end": "2020-01-05"},
            "validation": {"start": "2020-01-06", "end": "2020-01-07"},
            "test": {"start": "2020-01-08", "end": "2020-01-10"},
        },
        "training": {
            "random_seed": 7,
            "samples_per_time": 4,
            "max_train_samples": 200,
            "max_eval_samples": 100,
            "max_replay_times": 2,
        },
    }
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path / "atlas-data").mkdir()
    return ResearchConfig.from_yaml(path)


def test_storage_estimate_and_cap_guard(research_config: ResearchConfig) -> None:
    estimate = estimate_storage(research_config)
    assert estimate.timestamps == 80
    assert (estimate.latitude, estimate.longitude) == (3, 3)
    status = inspect_storage(research_config)
    enforce_preflight(research_config, status)
    too_small = ResearchConfig(**{**research_config.__dict__, "max_storage_gb": 0.000001})
    with pytest.raises(StorageGuardError, match="exceeding"):
        enforce_preflight(too_small, status)


def test_chronological_splits_are_leakage_safe(research_config: ResearchConfig) -> None:
    splits = build_splits(research_config, research_config.timestamps)
    assert (splits.feature_end == splits.valid_time).all()
    assert (splits.feature_start <= splits.feature_end).all()
    assert (splits.target_start > splits.feature_end).all()
    assert (splits.target_end > splits.target_start).all()
    assert (
        splits.loc[splits.split == "train", "valid_time"].max()
        < splits.loc[splits.split == "validation", "valid_time"].min()
    )
    assert (
        splits.loc[splits.split == "validation", "valid_time"].max()
        < splits.loc[splits.split == "test", "valid_time"].min()
    )


def test_era5_hourly_slicing_units_and_accumulation(research_config: ResearchConfig) -> None:
    config = replace(
        research_config,
        start=pd.Timestamp("2020-01-01T00:00:00"),
        end=pd.Timestamp("2020-01-01T06:00:00"),
        bbox=BoundingBox(west=-125, south=24, east=-124.5, north=24.5),
        resolution_degrees=0.25,
    )
    times = pd.date_range("2019-12-31T22:00:00", "2020-01-01T06:00:00", freq="1h")
    latitude = np.array([24.75, 24.5, 24.25, 24.0])
    longitude = np.array([234.75, 235.0, 235.25, 235.5, 235.75])
    shape = (len(times), len(latitude), len(longitude))
    values = {}
    for index, spec in enumerate(VARIABLES.values()):
        data = np.full(shape, 0.001 if spec.source_name == "tp" else index + 1, dtype="float32")
        values[spec.source_name] = (("valid_time", "latitude", "longitude"), data)
    remote = xr.Dataset(
        values,
        coords={
            "valid_time": times,
            "latitude": latitude,
            "longitude": longitude,
            "lsm": (("latitude", "longitude"), np.ones((4, 5), dtype="float32")),
        },
    ).chunk({"valid_time": 9, "latitude": 2, "longitude": 2})
    source = EarthmoverERA5Source()
    source._dataset = remote
    normalized = source.fetch_year(config, 2020).compute()
    assert list(pd.to_datetime(normalized.time.values)) == list(
        pd.date_range("2020-01-01T00:00:00", "2020-01-01T06:00:00", freq="3h")
    )
    assert normalized.sizes["latitude"] == 3
    assert normalized.sizes["longitude"] == 3
    assert "lsm" not in normalized.coords
    np.testing.assert_allclose(normalized.total_precipitation.values, 3.0)
    assert normalized.total_precipitation.attrs["units"] == "mm"
    assert normalized.temperature_2m.attrs["units"] == "K"
    assert normalized.attrs["data_class"] == "RETROSPECTIVE_REANALYSIS"


def test_smoke_pipeline_targets_loader_and_determinism(research_config: ResearchConfig) -> None:
    diagnostics = backfill(research_config, source=FixtureSource())
    root = research_config.storage_root / "research" / research_config.name
    assert diagnostics["data_class"] == "RETROSPECTIVE_REANALYSIS"
    assert diagnostics["split_samples"]["test"] > 0
    assert (root / "state" / "2020.zarr").exists()
    assert (root / "targets" / "precipitation_targets.zarr").exists()
    targets = xr.open_zarr(root / "targets" / "precipitation_targets.zarr", consolidated=True)
    state = xr.open_zarr(root / "state" / "2020.zarr", consolidated=True)
    anchor = pd.Timestamp("2020-01-03T00:00:00")
    expected = state.total_precipitation.sel(
        time=anchor + pd.Timedelta(hours=3)
    ) + state.total_precipitation.sel(time=anchor + pd.Timedelta(hours=6))
    np.testing.assert_allclose(targets.precipitation_6h.sel(time=anchor).values, expected.values)
    splits = pd.read_parquet(root / "splits" / "splits.parquet")
    train_times = splits.loc[splits.split == "train", "valid_time"].to_numpy()
    expected_threshold = targets.precipitation_6h.sel(time=train_times).quantile(0.95, dim="time")
    np.testing.assert_allclose(targets.extreme_threshold.values, expected_threshold.values)
    dataset = AtlasMiniDataset(root, split="test")
    sample = dataset[0]
    assert sample["state"].shape == (8, 10, 3, 3)
    assert sample["target"].shape == (3, 3)
    assert sample["metadata"]["feature_end"] < sample["metadata"]["target_start"]
    first = build_tabular_samples(root, "test", max_samples=50, seed=42)
    second = build_tabular_samples(root, "test", max_samples=50, seed=42)
    np.testing.assert_array_equal(first.flat_indices, second.flat_indices)
    np.testing.assert_allclose(first.features, second.features, equal_nan=True)
    for model in ("climatology", "persistence", "simple", "neural"):
        experiment = train_experiment(research_config, model)
        assert experiment["model"] == model
        assert experiment["test_metrics"]["brier_score"] >= 0
        experiment_root = root / "experiments" / experiment["experiment_id"]
        assert (experiment_root / "replay.zarr").exists()
        assert (experiment_root / "spatial_skill.zarr").exists()
    validate_dataset(research_config)


def test_probabilistic_baseline_metrics() -> None:
    labels = np.array([0, 0, 1, 1], dtype="uint8")
    climatology = np.full(4, 0.5)
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    metrics = score_probabilities(labels, probabilities, climatology)
    assert metrics["brier_score"] < metrics["climatology_brier_score"]
    assert metrics["brier_skill_score"] > 0
    assert metrics["pr_auc"] == pytest.approx(1)
