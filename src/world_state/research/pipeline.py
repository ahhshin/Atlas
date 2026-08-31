from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import Blosc

from world_state.research.config import VARIABLES, ResearchConfig
from world_state.research.source import EarthmoverERA5Source, HistoricalSource
from world_state.research.storage import (
    GIB,
    StorageGuardError,
    enforce_cap,
    enforce_preflight,
    inspect_storage,
    tree_size,
)


def backfill(
    config: ResearchConfig,
    *,
    source: HistoricalSource | None = None,
    force: bool = False,
) -> dict[str, object]:
    status = inspect_storage(config, create=True)
    estimate = enforce_preflight(config, status)
    print(
        f"Research storage: {status.dataset_root} "
        f"({status.free_gb:.2f} GiB free on {status.mount_path})"
    )
    print(
        f"Estimated size: {estimate.expected_gb:.2f} GiB expected, "
        f"{estimate.upper_bound_gb:.2f} GiB conservative upper bound"
    )
    root = status.dataset_root
    _create_layout(root)
    _snapshot_config(config, root)
    historical_source = source or EarthmoverERA5Source(config.source)
    written: list[Path] = []
    for year in config.years:
        target = root / "state" / f"{year}.zarr"
        if target.exists() and not force:
            print(f"State {year}: already present, preserving immutable partition")
            continue
        if target.exists():
            shutil.rmtree(target)
        print(f"State {year}: fetching bounded ERA5 fields")
        dataset = historical_source.fetch_year(config, year)
        _validate_source_dataset(dataset, config)
        _write_zarr_atomic(dataset, target, config)
        written.append(target)
        size = enforce_cap(config, root)
        print(f"State {year}: complete; dataset uses {size / GIB:.3f} GiB")
    _build_research_artifacts(config, root, force=force or bool(written))
    size = enforce_cap(config, root)
    diagnostics = validate_dataset(config, root, deep_missing=True)
    diagnostics["disk_bytes"] = size
    diagnostics["disk_gb"] = size / GIB
    (root / "metadata" / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8"
    )
    return diagnostics


def _build_research_artifacts(config: ResearchConfig, root: Path, *, force: bool) -> None:
    targets_path = root / "targets" / "precipitation_targets.zarr"
    splits_path = root / "splits" / "splits.parquet"
    normalization_path = root / "climatology" / "normalization.parquet"
    climatology_path = root / "climatology" / "extreme_probability.zarr"
    if (
        all(
            path.exists()
            for path in (targets_path, splits_path, normalization_path, climatology_path)
        )
        and not force
    ):
        return
    states = _open_states(root)
    splits = build_splits(config, pd.DatetimeIndex(states.time.values))
    splits.to_parquet(splits_path, index=False)
    precipitation = states.total_precipitation
    future = sum(
        precipitation.shift(time=-step) for step in range(1, config.target_steps + 1)
    ).astype("float32")
    future.name = "precipitation_6h"
    train_times = splits.loc[splits.split == "train", "valid_time"].to_numpy()
    if len(train_times) == 0:
        raise ValueError("training split contains no model-ready timestamps")
    threshold = (
        future.sel(time=train_times)
        .chunk({"time": -1})
        .quantile(config.percentile, dim="time", skipna=True)
        .astype("float32")
        .drop_vars("quantile", errors="ignore")
    )
    threshold.name = "extreme_threshold"
    target_missing = future.isnull() | threshold.isnull()
    labels = (future > threshold).where(~target_missing, False).astype("uint8")
    labels.name = "extreme_precipitation_label"
    target_missing.name = "target_missing_mask"
    targets = xr.Dataset(
        {
            "precipitation_6h": future,
            "extreme_precipitation_label": labels,
            "target_missing_mask": target_missing,
            "extreme_threshold": threshold,
        }
    )
    targets.attrs.update(
        {
            "target_definition": (
                f"next {config.target_hours}h accumulated precipitation exceeds the "
                f"grid-cell training-only {config.percentile:.0%} percentile"
            ),
            "threshold_fit_split": "train",
            "data_class": "RETROSPECTIVE_REANALYSIS",
        }
    )
    _write_zarr_atomic(targets, targets_path, config)
    statistics = _normalization_statistics(states, config, train_times)
    statistics.to_parquet(normalization_path, index=False)
    train_labels = labels.sel(time=train_times)
    train_missing = target_missing.sel(time=train_times)
    probability = train_labels.where(~train_missing).mean("time", skipna=True).astype("float32")
    probability.name = "extreme_probability"
    probability.attrs.update(
        {
            "fit_split": "train",
            "description": "grid-cell extreme-event frequency in the training period",
        }
    )
    _write_zarr_atomic(probability.to_dataset(), climatology_path, config)


def build_splits(config: ResearchConfig, times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cadence = pd.Timedelta(hours=config.cadence_hours)
    context_delta = pd.Timedelta(hours=config.cadence_hours * (config.context_steps - 1))
    target_delta = pd.Timedelta(hours=config.target_hours)
    first_time = pd.Timestamp(times.min())
    last_time = pd.Timestamp(times.max())
    named_ranges = (
        ("train", config.splits.train),
        ("validation", config.splits.validation),
        ("test", config.splits.test),
    )
    for value in times:
        anchor = pd.Timestamp(value)
        feature_start = anchor - context_delta
        target_start = anchor + cadence
        target_end = anchor + target_delta
        if feature_start < first_time or target_end > last_time:
            continue
        for name, period in named_ranges:
            if period.contains(anchor) and target_end <= period.end:
                rows.append(
                    {
                        "valid_time": anchor,
                        "split": name,
                        "feature_start": feature_start,
                        "feature_end": anchor,
                        "target_start": target_start,
                        "target_end": target_end,
                    }
                )
                break
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("split configuration produced no model-ready timestamps")
    return frame.sort_values("valid_time").reset_index(drop=True)


def validate_dataset(
    config: ResearchConfig,
    root: Path | None = None,
    *,
    deep_missing: bool = False,
) -> dict[str, object]:
    dataset_root = root or inspect_storage(config).dataset_root
    states = _open_states(dataset_root)
    targets = xr.open_zarr(
        dataset_root / "targets" / "precipitation_targets.zarr",
        consolidated=True,
        chunks={},
    )
    splits = pd.read_parquet(dataset_root / "splits" / "splits.parquet")
    times = pd.DatetimeIndex(states.time.values)
    deltas = pd.Series(times[1:] - times[:-1]).drop_duplicates()
    expected_delta = pd.Timedelta(hours=config.cadence_hours)
    if len(deltas) != 1 or deltas.iloc[0] != expected_delta:
        raise ValueError(f"state cadence is not uniformly {config.cadence_hours} hours")
    expected_shape = config.bbox.shape(config.resolution_degrees)
    actual_shape = (states.sizes["latitude"], states.sizes["longitude"])
    if actual_shape != expected_shape:
        raise ValueError(f"grid shape {actual_shape} does not match expected {expected_shape}")
    for variable in config.variables:
        if states[variable].dtype != np.dtype("float32"):
            raise ValueError(f"{variable} is not float32")
        if states[variable].attrs.get("units") != VARIABLES[variable].units:
            raise ValueError(f"unexpected units for {variable}")
        if f"missing_{variable}" not in states:
            raise ValueError(f"missing-data mask absent for {variable}")
    if states.attrs.get("data_class") != "RETROSPECTIVE_REANALYSIS":
        raise ValueError("historical state is not marked as retrospective reanalysis")
    _validate_splits(splits)
    if not np.array_equal(targets.time.values, states.time.values):
        raise ValueError("state and target timestamps are not aligned")
    normalization = pd.read_parquet(dataset_root / "climatology" / "normalization.parquet")
    if set(normalization.fit_split) != {"train"}:
        raise ValueError("normalization statistics are not training-only")
    if targets.attrs.get("threshold_fit_split") != "train":
        raise ValueError("target threshold is not marked training-only")
    disk = tree_size(dataset_root)
    if disk > config.max_storage_gb * GIB:
        raise StorageGuardError("validated dataset exceeds its configured storage cap")
    diagnostics_path = dataset_root / "metadata" / "diagnostics.json"
    cached_diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists() and not deep_missing
        else None
    )
    use_cached_diagnostics = bool(
        cached_diagnostics
        and cached_diagnostics.get("dataset") == config.name
        and cached_diagnostics.get("timestamps") == len(times)
        and cached_diagnostics.get("variables") == list(config.variables)
        and set(cached_diagnostics.get("missing_percent", {})) == set(config.variables)
        and "test_event_prevalence" in cached_diagnostics
    )
    if use_cached_diagnostics:
        missing = {
            name: float(cached_diagnostics["missing_percent"][name])
            for name in config.variables
        }
    else:
        missing = {
            name: float(states[f"missing_{name}"].mean().compute()) * 100
            for name in config.variables
        }
    label = targets.extreme_precipitation_label
    target_mask = targets.target_missing_mask
    split_counts = splits.groupby("split").size().to_dict()
    test_times = splits.loc[splits.split == "test", "valid_time"].to_numpy()
    if use_cached_diagnostics:
        prevalence = float(cached_diagnostics["test_event_prevalence"])
    else:
        prevalence = (
            float(
                label.sel(time=test_times)
                .where(~target_mask.sel(time=test_times))
                .mean()
                .compute()
            )
            if len(test_times)
            else float("nan")
        )
    return {
        "dataset": config.name,
        "years": list(config.years),
        "timestamps": len(times),
        "latitude": actual_shape[0],
        "longitude": actual_shape[1],
        "variables": list(config.variables),
        "missing_percent": missing,
        "split_samples": {str(key): int(value) for key, value in split_counts.items()},
        "test_event_prevalence": prevalence,
        "threshold_percentile": config.percentile,
        "threshold_fit_split": "train",
        "data_class": states.attrs["data_class"],
        "disk_bytes": disk,
        "disk_gb": disk / GIB,
    }


def _validate_splits(splits: pd.DataFrame) -> None:
    for row in splits.itertuples(index=False):
        if not (row.feature_start <= row.feature_end < row.target_start <= row.target_end):
            raise ValueError(f"leakage detected in sample at {row.valid_time}")
        if row.feature_end != row.valid_time:
            raise ValueError("feature window extends after anchor")
    grouped = splits.groupby("split").valid_time.agg(["min", "max"])
    order = [name for name in ("train", "validation", "test") if name in grouped.index]
    for earlier, later in pairwise(order):
        if grouped.loc[earlier, "max"] >= grouped.loc[later, "min"]:
            raise ValueError("chronological splits overlap")


def _normalization_statistics(
    states: xr.Dataset, config: ResearchConfig, train_times: np.ndarray
) -> pd.DataFrame:
    rows = []
    for variable in config.variables:
        values = states[variable].sel(time=train_times)
        mean = values.mean(skipna=True).compute()
        standard_deviation = values.std(skipna=True).compute()
        rows.append(
            {
                "variable": variable,
                "mean": float(mean),
                "std": float(standard_deviation),
                "fit_split": "train",
                "fit_start": pd.Timestamp(train_times.min()),
                "fit_end": pd.Timestamp(train_times.max()),
            }
        )
    return pd.DataFrame(rows)


def _open_states(root: Path) -> xr.Dataset:
    stores = sorted((root / "state").glob("*.zarr"))
    if not stores:
        raise FileNotFoundError(f"no state stores found beneath {root}")
    return xr.concat(
        [xr.open_zarr(path, consolidated=True, chunks={}) for path in stores], dim="time"
    ).sortby("time")


def _validate_source_dataset(dataset: xr.Dataset, config: ResearchConfig) -> None:
    for variable in config.variables:
        if variable not in dataset or f"missing_{variable}" not in dataset:
            raise ValueError(f"source omitted {variable} or its missing mask")
    if dataset.attrs.get("data_class") != "RETROSPECTIVE_REANALYSIS":
        raise ValueError("source must identify historical ERA5 as retrospective reanalysis")


def _write_zarr_atomic(dataset: xr.Dataset, target: Path, config: ResearchConfig) -> None:
    partial = target.with_name(f".{target.name}.partial-{uuid4().hex}")
    if partial.exists():
        shutil.rmtree(partial)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding: dict[str, dict[str, object]] = {}
    for name, values in dataset.data_vars.items():
        encoding[name] = {"compressor": compressor}
        if np.issubdtype(values.dtype, np.floating):
            encoding[name]["dtype"] = "float32"
    try:
        dataset.to_zarr(
            partial,
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding,
        )
        if target.exists():
            shutil.rmtree(target)
        partial.replace(target)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise


def _create_layout(root: Path) -> None:
    for name in ("state", "targets", "climatology", "splits", "experiments", "metadata"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _snapshot_config(config: ResearchConfig, root: Path) -> None:
    if config.config_path is None:
        raise ValueError("backfill requires a configuration loaded from YAML")
    destination = root / "metadata" / "config.yaml"
    shutil.copy2(config.config_path, destination)
    manifest = {
        "dataset": config.name,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "Earthmover public ERA5 Icechunk on AWS",
        "data_class": "RETROSPECTIVE_REANALYSIS",
        "config": asdict(config),
    }
    (root / "metadata" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
