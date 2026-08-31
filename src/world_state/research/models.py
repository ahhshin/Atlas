from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from world_state.research.dataset import AtlasMiniDataset


@dataclass
class TabularSamples:
    features: np.ndarray
    labels: np.ndarray
    precipitation: np.ndarray
    valid_times: np.ndarray
    flat_indices: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    feature_names: tuple[str, ...]


def build_tabular_samples(
    dataset_root: str | Path,
    split: str,
    *,
    max_samples: int,
    seed: int,
) -> TabularSamples:
    dataset = AtlasMiniDataset(dataset_root, split=split)
    cache_path = _sample_cache_path(dataset, split, max_samples, seed)
    if cache_path.exists():
        return _load_samples(cache_path)
    assignments = dataset.assignments
    per_time = min(
        dataset.config.samples_per_time,
        dataset.state.sizes["latitude"] * dataset.state.sizes["longitude"],
    )
    max_times = max(1, max_samples // per_time)
    rng = np.random.default_rng(seed)
    if len(assignments) > max_times:
        chosen = np.sort(rng.choice(len(assignments), size=max_times, replace=False))
        assignments = assignments.iloc[chosen].reset_index(drop=True)
    time_index = pd.Index(pd.to_datetime(dataset.state.time.values))
    positions = np.array(
        [time_index.get_loc(pd.Timestamp(value)) for value in assignments.valid_time]
    )
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    precipitation_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    latitude_parts: list[np.ndarray] = []
    longitude_parts: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    height = dataset.state.sizes["latitude"]
    width = dataset.state.sizes["longitude"]
    total_cells = height * width
    for batch_start in range(0, len(assignments), 64):
        batch = assignments.iloc[batch_start : batch_start + 64]
        batch_positions = positions[batch_start : batch_start + 64]
        load_start = int(batch_positions.min()) - dataset.config.context_steps + 1
        load_stop = int(batch_positions.max()) + 1
        loaded = (
            dataset.state[list(dataset.config.variables)]
            .isel(time=slice(load_start, load_stop))
            .to_array("channel")
            .transpose("time", "channel", "latitude", "longitude")
            .compute()
            .values.astype("float32", copy=False)
        )
        targets = dataset.targets.sel(time=batch.valid_time.to_numpy()).compute()
        for local_row, (row, position) in enumerate(
            zip(batch.itertuples(), batch_positions, strict=True)
        ):
            offset = int(position) - load_start
            window = loaded[offset - dataset.config.context_steps + 1 : offset + 1]
            selected_cells = np.sort(rng.choice(total_cells, size=per_time, replace=False))
            matrix, names = engineer_features(window, dataset.config.variables, selected_cells)
            target_values = targets.extreme_precipitation_label.isel(time=local_row).values.ravel()
            target_missing = targets.target_missing_mask.isel(time=local_row).values.ravel()
            precipitation = targets.precipitation_6h.isel(time=local_row).values.ravel()
            keep = ~target_missing[selected_cells]
            selected_cells = selected_cells[keep]
            feature_parts.append(matrix[keep])
            label_parts.append(target_values[selected_cells].astype("uint8"))
            precipitation_parts.append(precipitation[selected_cells].astype("float32"))
            time_parts.append(np.repeat(np.datetime64(row.valid_time), len(selected_cells)))
            index_parts.append(selected_cells.astype("int32"))
            latitude_parts.append(dataset.state.latitude.values[selected_cells // width])
            longitude_parts.append(dataset.state.longitude.values[selected_cells % width])
            feature_names = names
    if not feature_parts or feature_names is None:
        raise ValueError(f"no usable {split} samples")
    samples = TabularSamples(
        features=np.concatenate(feature_parts)[:max_samples],
        labels=np.concatenate(label_parts)[:max_samples],
        precipitation=np.concatenate(precipitation_parts)[:max_samples],
        valid_times=np.concatenate(time_parts)[:max_samples],
        flat_indices=np.concatenate(index_parts)[:max_samples],
        latitudes=np.concatenate(latitude_parts)[:max_samples],
        longitudes=np.concatenate(longitude_parts)[:max_samples],
        feature_names=feature_names,
    )
    _save_samples(cache_path, samples)
    return samples


def _sample_cache_path(dataset: AtlasMiniDataset, split: str, max_samples: int, seed: int) -> Path:
    signature = json.dumps(
        {
            "dataset": dataset.config.name,
            "variables": dataset.config.variables,
            "context_steps": dataset.config.context_steps,
            "split": split,
            "assignments": len(dataset.assignments),
            "max_samples": max_samples,
            "samples_per_time": dataset.config.samples_per_time,
            "seed": seed,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
    root = dataset.root / "metadata" / "sample_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{split}-{digest}.npz"


def _save_samples(path: Path, samples: TabularSamples) -> None:
    partial = path.with_name(f".{path.stem}.partial.npz")
    np.savez_compressed(
        partial,
        features=samples.features,
        labels=samples.labels,
        precipitation=samples.precipitation,
        valid_times=samples.valid_times,
        flat_indices=samples.flat_indices,
        latitudes=samples.latitudes,
        longitudes=samples.longitudes,
        feature_names=np.asarray(samples.feature_names, dtype="U"),
    )
    partial.replace(path)


def _load_samples(path: Path) -> TabularSamples:
    with np.load(path, allow_pickle=False) as values:
        return TabularSamples(
            features=values["features"],
            labels=values["labels"],
            precipitation=values["precipitation"],
            valid_times=values["valid_times"],
            flat_indices=values["flat_indices"],
            latitudes=values["latitudes"],
            longitudes=values["longitudes"],
            feature_names=tuple(values["feature_names"].tolist()),
        )


def engineer_features(
    window: np.ndarray,
    variables: tuple[str, ...],
    flat_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    _, channels, height, width = window.shape
    if channels != len(variables):
        raise ValueError("state channel count does not match configuration")
    if flat_indices is None:
        flat_indices = np.arange(height * width)
    latest = window[-1]
    temporal_mean = np.nanmean(window, axis=0)
    temporal_std = np.nanstd(window, axis=0)
    spatial_mean = np.stack([uniform_filter(values, size=3, mode="nearest") for values in latest])
    missing_fraction = np.mean(~np.isfinite(window), axis=0)
    blocks = [latest, temporal_mean, temporal_std, spatial_mean, missing_fraction]
    names: list[str] = []
    for prefix in ("latest", "mean_24h", "std_24h", "neighbor_mean", "missing_fraction"):
        names.extend(f"{prefix}_{name}" for name in variables)
    precipitation_index = variables.index("total_precipitation")
    precipitation = window[:, precipitation_index]
    recent = [
        precipitation[-1],
        np.nansum(precipitation[-2:], axis=0),
        np.nansum(precipitation[-4:], axis=0),
        np.nansum(precipitation, axis=0),
    ]
    names.extend(
        ("precipitation_3h", "precipitation_6h_recent", "precipitation_12h", "precipitation_24h")
    )
    temperature = latest[variables.index("temperature_2m")]
    dewpoint = latest[variables.index("dewpoint_2m")]
    temperature_c = temperature - 273.15
    dewpoint_c = dewpoint - 273.15
    humidity_exponent = (17.625 * dewpoint_c) / (243.04 + dewpoint_c) - (17.625 * temperature_c) / (
        243.04 + temperature_c
    )
    relative_humidity = np.exp(np.clip(humidity_exponent, -20, 20))
    relative_humidity = np.clip(relative_humidity, 0, 1)
    u_wind = latest[variables.index("u_wind_10m")]
    v_wind = latest[variables.index("v_wind_10m")]
    wind_speed = np.hypot(u_wind, v_wind)
    surface_pressure = window[:, variables.index("surface_pressure")]
    pressure_tendency = surface_pressure[-1] - surface_pressure[0]
    blocks.extend(recent)
    blocks.extend((relative_humidity, wind_speed, pressure_tendency))
    names.extend(("relative_humidity", "wind_speed", "pressure_tendency_24h"))
    flattened = [
        block.reshape((-1, height * width))
        if block.ndim == 3
        else block.reshape((1, height * width))
        for block in blocks
    ]
    matrix = np.concatenate(flattened, axis=0)[:, flat_indices].T.astype("float32")
    return matrix, tuple(names)


def make_model(name: str, seed: int) -> Pipeline:
    common = [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if name == "simple":
        estimator = LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
            tol=1e-3,
        )
    elif name == "neural":
        estimator = MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            solver="adam",
            batch_size="auto",
            learning_rate_init=5e-4,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=6,
            random_state=seed,
        )
    elif name == "persistence":
        estimator = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"unsupported learned model: {name}")
    return Pipeline([*common, ("model", estimator)])


def model_parameter_count(model: Pipeline, input_features: int) -> int:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coefs_"):
        return int(
            sum(values.size for values in estimator.coefs_)
            + sum(values.size for values in estimator.intercepts_)
        )
    if hasattr(estimator, "coef_"):
        return int(estimator.coef_.size + estimator.intercept_.size)
    return input_features
