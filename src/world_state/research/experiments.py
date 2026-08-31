from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from world_state.research.config import ResearchConfig
from world_state.research.dataset import AtlasMiniDataset
from world_state.research.metrics import calibration_table, score_probabilities, spatial_brier_skill
from world_state.research.models import (
    TabularSamples,
    build_tabular_samples,
    engineer_features,
    make_model,
    model_parameter_count,
)
from world_state.research.pipeline import _write_zarr_atomic
from world_state.research.storage import enforce_cap, inspect_storage


def train_experiment(config: ResearchConfig, model_name: str) -> dict[str, object]:
    if model_name not in {"climatology", "persistence", "simple", "neural"}:
        raise ValueError("model must be climatology, persistence, simple, or neural")
    status = inspect_storage(config)
    root = status.dataset_root
    experiment_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{model_name}-{uuid4().hex[:8]}"
    experiment_root = root / "experiments" / experiment_id
    experiment_root.mkdir(parents=True, exist_ok=False)
    train = build_tabular_samples(
        root,
        "train",
        max_samples=config.max_train_samples,
        seed=config.random_seed,
    )
    validation = build_tabular_samples(
        root,
        "validation",
        max_samples=config.max_eval_samples,
        seed=config.random_seed + 1,
    )
    test = build_tabular_samples(
        root,
        "test",
        max_samples=config.max_eval_samples,
        seed=config.random_seed + 2,
    )
    climatology_grid = xr.open_zarr(
        root / "climatology" / "extreme_probability.zarr", consolidated=True
    ).extreme_probability.values.ravel()
    estimator = None
    selected_features = np.arange(train.features.shape[1])
    if model_name == "climatology":
        parameter_count = 0
    else:
        if model_name == "persistence":
            selected_features = np.array(
                [
                    index
                    for index, name in enumerate(train.feature_names)
                    if name.startswith("precipitation_")
                ]
            )
        estimator = make_model(model_name, config.random_seed)
        fit_kwargs = {}
        if model_name == "neural":
            positives = max(1, int(train.labels.sum()))
            negatives = max(1, len(train.labels) - positives)
            weights = np.where(
                train.labels == 1,
                len(train.labels) / (2 * positives),
                len(train.labels) / (2 * negatives),
            )
            fit_kwargs["model__sample_weight"] = weights
        estimator.fit(train.features[:, selected_features], train.labels, **fit_kwargs)
        parameter_count = model_parameter_count(estimator, len(selected_features))
        joblib.dump(
            {
                "model": estimator,
                "feature_indices": selected_features,
                "feature_names": train.feature_names,
            },
            experiment_root / "model.joblib",
        )
    split_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for split_name, samples in (("train", train), ("validation", validation), ("test", test)):
        climate = climatology_grid[samples.flat_indices]
        probability = (
            climate
            if estimator is None
            else estimator.predict_proba(samples.features[:, selected_features])[:, 1]
        )
        predictions[split_name] = probability
        split_metrics[split_name] = score_probabilities(samples.labels, probability, climate)
    calibration_table(test.labels, predictions["test"]).to_parquet(
        experiment_root / "calibration.parquet", index=False
    )
    _write_evaluation_artifacts(
        config,
        root,
        experiment_root,
        model_name,
        estimator,
        selected_features,
        climatology_grid,
        test,
        predictions["test"],
    )
    checkpoint = experiment_root / "model.joblib" if estimator is not None else None
    metadata: dict[str, object] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": config.name,
        "variables": list(config.variables),
        "context_length_hours": config.context_hours,
        "target_definition": (
            f"next-{config.target_hours}h precipitation above grid-cell "
            f"training {config.percentile:.0%} percentile"
        ),
        "model": model_name,
        "parameter_count": parameter_count,
        "training_configuration": {
            "random_seed": config.random_seed,
            "train_samples": len(train.labels),
            "evaluation_samples": len(test.labels),
            "feature_count": len(selected_features),
        },
        "train_metrics": split_metrics["train"],
        "validation_metrics": split_metrics["validation"],
        "test_metrics": split_metrics["test"],
        "checkpoint_path": str(checkpoint) if checkpoint else None,
    }
    (experiment_root / "metrics.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
    )
    _append_experiment_index(root, metadata)
    size = enforce_cap(config, root)
    _refresh_diagnostics_size(root, size)
    return metadata


def evaluate_experiment(experiment_id: str, data_root: Path) -> dict[str, object]:
    matches = list((data_root / "research").glob(f"*/experiments/{experiment_id}/metrics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely locate experiment {experiment_id}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _write_evaluation_artifacts(
    config: ResearchConfig,
    dataset_root: Path,
    experiment_root: Path,
    model_name: str,
    estimator: object | None,
    feature_indices: np.ndarray,
    climatology_grid: np.ndarray,
    test: TabularSamples,
    test_probability: np.ndarray,
) -> None:
    dataset = AtlasMiniDataset(dataset_root, split="test")
    shape = (dataset.state.sizes["latitude"], dataset.state.sizes["longitude"])
    test_climate = climatology_grid[test.flat_indices]
    brier, skill, counts = spatial_brier_skill(
        test.labels,
        test_probability,
        test_climate,
        test.flat_indices,
        shape,
    )
    spatial = xr.Dataset(
        {
            "brier_score": (("latitude", "longitude"), brier.astype("float32")),
            "brier_skill_score": (("latitude", "longitude"), skill.astype("float32")),
            "samples": (("latitude", "longitude"), counts.astype("int32")),
        },
        coords={"latitude": dataset.state.latitude, "longitude": dataset.state.longitude},
    )
    _write_zarr_atomic(spatial, experiment_root / "spatial_skill.zarr", config)
    timeline = pd.DataFrame(
        {
            "valid_time": test.valid_times,
            "label": test.labels,
            "probability": test_probability,
            "climatology": test_climate,
        }
    )
    timeline["squared_error"] = (timeline.probability - timeline.label) ** 2
    timeline.groupby("valid_time", as_index=False).agg(
        brier_score=("squared_error", "mean"),
        event_prevalence=("label", "mean"),
        mean_probability=("probability", "mean"),
        samples=("label", "size"),
    ).to_parquet(experiment_root / "timeline.parquet", index=False)
    _write_regional_metrics(experiment_root, test, test_probability, test_climate)
    replay_count = min(config.max_replay_times, len(dataset))
    replay_indices = np.linspace(0, len(dataset) - 1, replay_count, dtype=int)
    replay_fields: list[xr.Dataset] = []
    for index in replay_indices:
        sample = dataset[index]
        height, width = sample["target"].shape
        all_cells = np.arange(height * width)
        features, _ = engineer_features(sample["state"], config.variables, all_cells)
        climate = climatology_grid.reshape((height, width))
        probability = (
            climate
            if estimator is None
            else estimator.predict_proba(features[:, feature_indices])[:, 1].reshape(
                (height, width)
            )
        )
        replay_fields.append(
            xr.Dataset(
                {
                    "probability": (("latitude", "longitude"), probability.astype("float32")),
                    "precipitation_6h": (("latitude", "longitude"), sample["precipitation_6h"]),
                    "extreme_label": (("latitude", "longitude"), sample["target"]),
                    "probability_error": (
                        ("latitude", "longitude"),
                        (probability - sample["target"]).astype("float32"),
                    ),
                },
                coords={
                    "time": np.datetime64(sample["valid_time"]),
                    "latitude": dataset.state.latitude.values,
                    "longitude": dataset.state.longitude.values,
                },
            ).expand_dims("time")
        )
    replay = xr.concat(replay_fields, dim="time")
    replay.attrs.update({"model": model_name, "dataset": config.name})
    _write_zarr_atomic(replay, experiment_root / "replay.zarr", config)


def _write_regional_metrics(
    root: Path,
    samples: TabularSamples,
    probability: np.ndarray,
    climatology: np.ndarray,
) -> None:
    boundaries = np.quantile(samples.longitudes, [1 / 3, 2 / 3])
    region = np.where(
        samples.longitudes <= boundaries[0],
        "west",
        np.where(samples.longitudes <= boundaries[1], "central", "east"),
    )
    rows = []
    for name in ("west", "central", "east"):
        selected = region == name
        if not np.any(selected):
            rows.append({"region": name, "samples": 0})
            continue
        rows.append(
            {
                "region": name,
                **score_probabilities(
                    samples.labels[selected], probability[selected], climatology[selected]
                ),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "regional_metrics.parquet", index=False)


def _append_experiment_index(root: Path, metadata: dict[str, object]) -> None:
    path = root / "experiments" / "experiments.parquet"
    row = pd.DataFrame(
        [
            {
                "experiment_id": metadata["experiment_id"],
                "timestamp": metadata["timestamp"],
                "git_commit": metadata["git_commit"],
                "dataset_version": metadata["dataset_version"],
                "model": metadata["model"],
                "parameter_count": metadata["parameter_count"],
                "brier_score": metadata["test_metrics"]["brier_score"],
                "brier_skill_score": metadata["test_metrics"]["brier_skill_score"],
                "pr_auc": metadata["test_metrics"]["pr_auc"],
                "roc_auc": metadata["test_metrics"]["roc_auc"],
                "test_event_prevalence": metadata["test_metrics"]["positive_event_prevalence"],
                "experiment_path": str(root / "experiments" / str(metadata["experiment_id"])),
            }
        ]
    )
    if path.exists():
        row = pd.concat([pd.read_parquet(path), row], ignore_index=True)
    row.to_parquet(path, index=False)


def _refresh_diagnostics_size(root: Path, size: int) -> None:
    path = root / "metadata" / "diagnostics.json"
    if not path.exists():
        return
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    diagnostics.update({"disk_bytes": size, "disk_gb": size / (1024**3)})
    partial = path.with_suffix(".partial.json")
    partial.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    partial.replace(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
