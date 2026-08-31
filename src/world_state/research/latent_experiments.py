from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from world_state.research.config import ResearchConfig
from world_state.research.latent_config import LatentWorldConfig
from world_state.research.latent_data import LatentSequenceDataset
from world_state.research.latent_metrics import (
    compare_with_persistence,
    compression_ratio,
    latent_mse,
    physical_state_metrics,
)
from world_state.research.latent_models import (
    FrozenLatentProbe,
    LatentDynamics,
    RegionalDecoder,
    RegionalEncoder,
    masked_loss,
    parameter_count,
)
from world_state.research.metrics import calibration_table, score_probabilities
from world_state.research.pipeline import _write_zarr_atomic
from world_state.research.storage import GIB, enforce_cap, inspect_storage, tree_size

STAGES = ("autoencoder", "dynamics", "probe")


def run_latent_smoke(config: LatentWorldConfig) -> list[dict[str, Any]]:
    summaries = []
    for dimensions in config.ablation_dimensions:
        variant = config.with_latent_dimensions(dimensions)
        summaries.append(run_latent_stages(variant, STAGES))
    return summaries


def train_latent_stage(
    config: LatentWorldConfig,
    stage: str,
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown latent stage: {stage}")
    return run_latent_stages(config, (stage,), experiment_id=experiment_id)


def run_latent_stages(
    config: LatentWorldConfig,
    stages: Iterable[str],
    *,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    stages = tuple(stages)
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError("latent stages must be autoencoder, dynamics, and/or probe")
    _set_determinism(config)
    device = _device(config)
    data_root, dataset_root, dataset_config = _storage_context(config)
    experiment_root, metadata = _resolve_experiment(
        config,
        dataset_root,
        stages,
        experiment_id,
    )
    datasets = _datasets(data_root, config)
    loaders = _loaders(datasets, config)
    encoder, decoder, dynamics, probe = _models(config, len(datasets["train"].channels))
    modules = (encoder, decoder, dynamics, probe)
    total_parameters = parameter_count(*modules)
    if total_parameters > config.max_parameters:
        raise ValueError(
            f"latent model has {total_parameters:,} parameters, exceeding "
            f"configured maximum {config.max_parameters:,}"
        )
    for module in modules:
        module.to(device)
    metadata.update(
        _model_metadata(config, datasets["train"], encoder, decoder, dynamics, probe, device)
    )
    print(
        f"Latent artifacts: {_estimated_artifact_bytes(config, datasets['train']) / GIB:.3f} "
        f"GiB estimated; {config.additional_artifact_cap_gb:.2f} GiB additional cap"
    )
    print(f"Training device: {device}; parameters: {total_parameters:,}")
    for stage in stages:
        if stage == "autoencoder":
            history = _train_autoencoder(config, encoder, decoder, loaders, device)
            checkpoint = experiment_root / "autoencoder.pt"
            _save_checkpoint(
                checkpoint,
                {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                config,
                stage,
            )
            reconstruction = _evaluate_reconstruction(
                config, encoder, decoder, loaders["test"], datasets["test"], device
            )
            _save_reconstruction_maps(
                config,
                experiment_root,
                encoder,
                decoder,
                loaders["test"],
                datasets["test"],
                dataset_config,
                device,
            )
            metadata["autoencoder"] = {
                "training_history": history,
                "selected_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
                "test_metrics": reconstruction,
                "checkpoint_path": str(checkpoint),
            }
        elif stage == "dynamics":
            _load_autoencoder(experiment_root, encoder, decoder, device)
            history = _train_dynamics(
                config, encoder, decoder, dynamics, loaders, datasets, device
            )
            checkpoint = experiment_root / "dynamics.pt"
            _save_checkpoint(checkpoint, {"dynamics": dynamics.state_dict()}, config, stage)
            evaluation = _evaluate_dynamics(
                config,
                encoder,
                decoder,
                dynamics,
                loaders["test"],
                datasets["test"],
                device,
            )
            _save_world_diagnostics(
                config,
                experiment_root,
                encoder,
                decoder,
                dynamics,
                loaders["test"],
                datasets["test"],
                dataset_config,
                device,
            )
            metadata["dynamics"] = {
                "training_history": history,
                "selected_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
                "test_metrics": evaluation,
                "checkpoint_path": str(checkpoint),
            }
        elif stage == "probe":
            _load_autoencoder(experiment_root, encoder, decoder, device)
            history = _train_probe(config, encoder, probe, loaders, device)
            checkpoint = experiment_root / "probe.pt"
            _save_checkpoint(checkpoint, {"probe": probe.state_dict()}, config, stage)
            evaluation, calibration = _evaluate_probe(
                config,
                encoder,
                probe,
                loaders["test"],
                datasets["test"],
                device,
            )
            calibration.to_parquet(experiment_root / "probe_calibration.parquet", index=False)
            metadata["probe"] = {
                "training_history": history,
                "selected_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
                "test_metrics": evaluation,
                "checkpoint_path": str(checkpoint),
                "encoder_frozen": True,
            }
        metadata["completed_stages"] = sorted(
            set(metadata.get("completed_stages", [])) | {stage}, key=STAGES.index
        )
        metadata["updated_at"] = datetime.now(UTC).isoformat()
        _write_metadata(experiment_root, metadata)
        _upsert_index(dataset_root, metadata, experiment_root)
        _enforce_latent_storage(config, dataset_config, dataset_root)
    return metadata


def evaluate_latent_experiment(experiment_id: str, data_root: Path) -> dict[str, Any]:
    matches = list(
        (data_root / "research").glob(
            f"*/latent_world/experiments/{experiment_id}/metrics.json"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely locate latent experiment {experiment_id}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def load_latent_models(
    experiment_root: str | Path,
    config: LatentWorldConfig,
    physical_channels: int,
    device: torch.device | str = "cpu",
) -> tuple[RegionalEncoder, RegionalDecoder, LatentDynamics, FrozenLatentProbe]:
    root = Path(experiment_root)
    encoder, decoder, dynamics, probe = _models(config, physical_channels)
    _load_autoencoder(root, encoder, decoder, torch.device(device))
    dynamics_path = root / "dynamics.pt"
    if dynamics_path.exists():
        dynamics.load_state_dict(torch.load(dynamics_path, map_location=device)["state"]["dynamics"])
    probe_path = root / "probe.pt"
    if probe_path.exists():
        probe.load_state_dict(torch.load(probe_path, map_location=device)["state"]["probe"])
    return encoder, decoder, dynamics, probe


def _train_autoencoder(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    loaders: dict[str, DataLoader[dict[str, Any]]],
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=config.training.learning_rate
    )
    history = []
    best_validation = float("inf")
    best_encoder: dict[str, Tensor] | None = None
    best_decoder: dict[str, Tensor] | None = None
    for epoch in range(config.training.autoencoder_epochs):
        encoder.train()
        decoder.train()
        train_loss = _autoencoder_epoch(
            config, encoder, decoder, loaders["train"], device, optimizer
        )
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            validation_loss = _autoencoder_epoch(
                config, encoder, decoder, loaders["validation"], device, None
            )
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_encoder = deepcopy(encoder.state_dict())
            best_decoder = deepcopy(decoder.state_dict())
        print(
            f"autoencoder epoch {epoch + 1}/{config.training.autoencoder_epochs}: "
            f"train={train_loss:.5f} validation={validation_loss:.5f}"
        )
    if best_encoder is not None and best_decoder is not None:
        encoder.load_state_dict(best_encoder)
        decoder.load_state_dict(best_decoder)
    return history


def _autoencoder_epoch(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total = 0.0
    batches = 0
    for batch in loader:
        sequence = batch["history"].to(device)
        sequence_mask = batch["history_mask"].to(device)
        values = sequence.reshape(-1, *sequence.shape[2:])
        mask = sequence_mask.reshape(-1, *sequence_mask.shape[2:])
        latent = encoder(values, mask)
        reconstruction = decoder(latent, values.shape[-2:])
        loss = masked_loss(
            reconstruction, values, mask, kind=config.loss.reconstruction
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def _train_dynamics(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    dynamics: LatentDynamics,
    loaders: dict[str, DataLoader[dict[str, Any]]],
    datasets: dict[str, LatentSequenceDataset],
    device: torch.device,
) -> list[dict[str, float]]:
    _freeze(encoder)
    _freeze(decoder)
    dynamics.train()
    optimizer = torch.optim.Adam(dynamics.parameters(), lr=config.training.learning_rate)
    history = []
    best_validation = float("inf")
    best_dynamics: dict[str, Tensor] | None = None
    for epoch in range(config.training.dynamics_epochs):
        dynamics.train()
        train = _dynamics_epoch(
            config, encoder, decoder, dynamics, loaders["train"], device, optimizer
        )
        dynamics.eval()
        with torch.no_grad():
            validation = _dynamics_epoch(
                config,
                encoder,
                decoder,
                dynamics,
                loaders["validation"],
                device,
                None,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train["combined"],
                "train_latent_loss": train["latent"],
                "train_physical_loss": train["physical"],
                "validation_loss": validation["combined"],
                "validation_latent_loss": validation["latent"],
                "validation_physical_loss": validation["physical"],
            }
        )
        if validation["combined"] < best_validation:
            best_validation = validation["combined"]
            best_dynamics = deepcopy(dynamics.state_dict())
        print(
            f"dynamics epoch {epoch + 1}/{config.training.dynamics_epochs}: "
            f"train={train['combined']:.5f} validation={validation['combined']:.5f}"
        )
    if best_dynamics is not None:
        dynamics.load_state_dict(best_dynamics)
    _ = datasets
    return history


def _dynamics_epoch(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    dynamics: LatentDynamics,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    totals = {"combined": 0.0, "latent": 0.0, "physical": 0.0}
    batches = 0
    for batch in loader:
        history = batch["history"].to(device)
        history_mask = batch["history_mask"].to(device)
        future = batch["future"].to(device)
        future_mask = batch["future_mask"].to(device)
        with torch.no_grad():
            latent_history = encoder.encode_sequence(history, history_mask)
            latent_actual = encoder(future, future_mask)
        latent_prediction = dynamics(latent_history)
        physical_prediction = decoder(latent_prediction, future.shape[-2:])
        latent_loss = F.mse_loss(latent_prediction, latent_actual)
        physical_loss = masked_loss(
            physical_prediction, future, future_mask, kind=config.loss.reconstruction
        )
        combined = (
            config.loss.latent_weight * latent_loss
            + config.loss.physical_weight * physical_loss
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            combined.backward()
            optimizer.step()
        for name, value in (
            ("combined", combined),
            ("latent", latent_loss),
            ("physical", physical_loss),
        ):
            totals[name] += float(value.detach())
        batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


def _train_probe(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    probe: FrozenLatentProbe,
    loaders: dict[str, DataLoader[dict[str, Any]]],
    device: torch.device,
) -> list[dict[str, float]]:
    _freeze(encoder)
    positive_weight = _probe_positive_weight(loaders["train"])
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.training.learning_rate)
    history = []
    best_validation = float("inf")
    best_probe: dict[str, Tensor] | None = None
    for epoch in range(config.training.probe_epochs):
        probe.train()
        train_loss = _probe_epoch(
            encoder, probe, loaders["train"], device, positive_weight, optimizer
        )
        probe.eval()
        with torch.no_grad():
            validation_loss = _probe_epoch(
                encoder,
                probe,
                loaders["validation"],
                device,
                positive_weight,
                None,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "positive_weight": positive_weight,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_probe = deepcopy(probe.state_dict())
        print(
            f"probe epoch {epoch + 1}/{config.training.probe_epochs}: "
            f"train={train_loss:.5f} validation={validation_loss:.5f}"
        )
    if best_probe is not None:
        probe.load_state_dict(best_probe)
    return history


def _probe_epoch(
    encoder: RegionalEncoder,
    probe: FrozenLatentProbe,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    positive_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    total = 0.0
    batches = 0
    weight = torch.tensor(positive_weight, dtype=torch.float32, device=device)
    for batch in loader:
        values = batch["history"][:, -1].to(device)
        mask = batch["history_mask"][:, -1].to(device)
        target = batch["extreme_target"].to(device)
        target_mask = batch["extreme_mask"].to(device)
        with torch.no_grad():
            latent = encoder(values, mask)
        logits = probe(latent, target.shape[-2:])
        losses = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=weight
        )
        valid = 1 - target_mask
        loss = (losses * valid).sum() / valid.sum().clamp_min(1)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def _evaluate_reconstruction(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    loader: DataLoader[dict[str, Any]],
    dataset: LatentSequenceDataset,
    device: torch.device,
) -> dict[str, Any]:
    encoder.eval()
    decoder.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            values = batch["history"][:, -1].to(device)
            mask = batch["history_mask"][:, -1].to(device)
            prediction = decoder(encoder(values, mask), values.shape[-2:])
            predictions.append(prediction.cpu().numpy())
            targets.append(values.cpu().numpy())
            masks.append(mask.cpu().numpy().astype(bool))
    metrics = physical_state_metrics(
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(masks),
        dataset.standard_deviations,
        dataset.channels,
    )
    latent_height, latent_width = dataset.latent_shape
    metrics["compression_ratio"] = compression_ratio(
        len(dataset.channels),
        *dataset.spatial_shape,
        config.latent_dimensions,
        latent_height,
        latent_width,
    )
    return metrics


def _evaluate_dynamics(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    dynamics: LatentDynamics,
    loader: DataLoader[dict[str, Any]],
    dataset: LatentSequenceDataset,
    device: torch.device,
) -> dict[str, Any]:
    encoder.eval()
    decoder.eval()
    dynamics.eval()
    model_fields, latent_fields, persistence_fields, targets, masks = [], [], [], [], []
    latent_predictions, latent_actuals, latent_current = [], [], []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            history_mask = batch["history_mask"].to(device)
            future = batch["future"].to(device)
            future_mask = batch["future_mask"].to(device)
            latent_history = encoder.encode_sequence(history, history_mask)
            actual = encoder(future, future_mask)
            predicted = dynamics(latent_history)
            model_fields.append(decoder(predicted, future.shape[-2:]).cpu().numpy())
            latent_fields.append(decoder(latent_history[:, -1], future.shape[-2:]).cpu().numpy())
            persistence_fields.append(history[:, -1].cpu().numpy())
            targets.append(future.cpu().numpy())
            masks.append(future_mask.cpu().numpy().astype(bool))
            latent_predictions.append(predicted.cpu().numpy())
            latent_actuals.append(actual.cpu().numpy())
            latent_current.append(latent_history[:, -1].cpu().numpy())
    target = np.concatenate(targets)
    missing = np.concatenate(masks)
    physical_persistence = physical_state_metrics(
        np.concatenate(persistence_fields),
        target,
        missing,
        dataset.standard_deviations,
        dataset.channels,
    )
    latent_persistence = physical_state_metrics(
        np.concatenate(latent_fields),
        target,
        missing,
        dataset.standard_deviations,
        dataset.channels,
    )
    model = physical_state_metrics(
        np.concatenate(model_fields),
        target,
        missing,
        dataset.standard_deviations,
        dataset.channels,
    )
    predicted_latent = np.concatenate(latent_predictions)
    actual_latent = np.concatenate(latent_actuals)
    current_latent = np.concatenate(latent_current)
    return {
        "physical_persistence": physical_persistence,
        "latent_persistence": latent_persistence,
        "latent_dynamics": model,
        "improvement_over_physical_persistence": compare_with_persistence(
            model, physical_persistence
        ),
        "latent_prediction_mse": latent_mse(predicted_latent, actual_latent),
        "latent_persistence_mse": latent_mse(current_latent, actual_latent),
    }


def _evaluate_probe(
    config: LatentWorldConfig,
    encoder: RegionalEncoder,
    probe: FrozenLatentProbe,
    loader: DataLoader[dict[str, Any]],
    dataset: LatentSequenceDataset,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    encoder.eval()
    probe.eval()
    labels, probabilities, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            values = batch["history"][:, -1].to(device)
            missing = batch["history_mask"][:, -1].to(device)
            target = batch["extreme_target"].numpy()
            target_mask = batch["extreme_mask"].numpy().astype(bool)
            probability = torch.sigmoid(
                probe(encoder(values, missing), target.shape[-2:])
            ).cpu().numpy()
            labels.append(target)
            masks.append(target_mask)
            probabilities.append(probability)
    label = np.concatenate(labels)
    missing = np.concatenate(masks)
    probability = np.concatenate(probabilities)
    climate = xr.open_zarr(
        dataset.root / "climatology" / "extreme_probability.zarr", consolidated=True
    ).extreme_probability.isel(
        latitude=dataset.latitude_slice, longitude=dataset.longitude_slice
    ).values
    climate = np.broadcast_to(climate, label.shape)
    valid = ~missing
    metrics = score_probabilities(label[valid], probability[valid], climate[valid])
    reference_path = dataset.root / "experiments" / "experiments.parquet"
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        reference = reference.loc[reference.pr_auc.notna()]
        if not reference.empty:
            best = reference.loc[reference.pr_auc.idxmax()]
            metrics["engineered_feature_reference"] = {
                "experiment_id": best.experiment_id,
                "model": best.model,
                "pr_auc": float(best.pr_auc),
                "roc_auc": float(best.roc_auc),
                "brier_score": float(best.brier_score),
                "brier_skill_score": float(best.brier_skill_score),
            }
    return metrics, calibration_table(label[valid], probability[valid])


def _save_reconstruction_maps(
    config: LatentWorldConfig,
    experiment_root: Path,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    loader: DataLoader[dict[str, Any]],
    dataset: LatentSequenceDataset,
    dataset_config: ResearchConfig,
    device: torch.device,
) -> None:
    records = _diagnostic_batches(loader, config.training.diagnostic_samples)
    actuals, reconstructions, times = [], [], []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch in records:
            values = batch["history"][:, -1].to(device)
            mask = batch["history_mask"][:, -1].to(device)
            reconstructed = decoder(encoder(values, mask), values.shape[-2:])
            actuals.append(_inverse_batch(values.cpu().numpy(), dataset))
            reconstructions.append(_inverse_batch(reconstructed.cpu().numpy(), dataset))
            times.extend(batch["valid_time"])
    output = xr.Dataset(
        {
            "actual": (
                ("time", "channel", "latitude", "longitude"),
                np.concatenate(actuals).astype("float32"),
            ),
            "reconstructed": (
                ("time", "channel", "latitude", "longitude"),
                np.concatenate(reconstructions).astype("float32"),
            ),
        },
        coords={
            "time": pd.to_datetime(times),
            "channel": list(dataset.channels),
            "latitude": dataset.latitudes,
            "longitude": dataset.longitudes,
        },
        attrs={"data_class": "RETROSPECTIVE_REANALYSIS", "latent_dimensions": config.latent_dimensions},
    )
    _write_zarr_atomic(output, experiment_root / "reconstruction.zarr", dataset_config)


def _save_world_diagnostics(
    config: LatentWorldConfig,
    experiment_root: Path,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    dynamics: LatentDynamics,
    loader: DataLoader[dict[str, Any]],
    dataset: LatentSequenceDataset,
    dataset_config: ResearchConfig,
    device: torch.device,
) -> None:
    records = _diagnostic_batches(loader, config.training.diagnostic_samples)
    physical: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "current",
            "reconstructed_current",
            "actual_future",
            "physical_persistence",
            "latent_persistence",
            "latent_prediction",
        )
    }
    latent_maps: dict[str, list[np.ndarray]] = {
        "latent_magnitude": [],
        "latent_change_magnitude": [],
        "latent_prediction_error": [],
    }
    embeddings, embedding_times = [], []
    times, future_times = [], []
    encoder.eval()
    decoder.eval()
    dynamics.eval()
    with torch.no_grad():
        for batch in records:
            history = batch["history"].to(device)
            history_mask = batch["history_mask"].to(device)
            future = batch["future"].to(device)
            future_mask = batch["future_mask"].to(device)
            latent_history = encoder.encode_sequence(history, history_mask)
            latent_actual = encoder(future, future_mask)
            latent_prediction = dynamics(latent_history)
            current = history[:, -1]
            reconstructed = decoder(latent_history[:, -1], current.shape[-2:])
            decoded_persistence = decoder(latent_history[:, -1], future.shape[-2:])
            decoded_prediction = decoder(latent_prediction, future.shape[-2:])
            for name, values in (
                ("current", current),
                ("reconstructed_current", reconstructed),
                ("actual_future", future),
                ("physical_persistence", current),
                ("latent_persistence", decoded_persistence),
                ("latent_prediction", decoded_prediction),
            ):
                physical[name].append(_inverse_batch(values.cpu().numpy(), dataset))
            latent_maps["latent_magnitude"].append(
                torch.linalg.vector_norm(latent_history[:, -1], dim=1).cpu().numpy()
            )
            latent_maps["latent_change_magnitude"].append(
                torch.linalg.vector_norm(
                    latent_history[:, -1] - latent_history[:, -2], dim=1
                ).cpu().numpy()
            )
            latent_maps["latent_prediction_error"].append(
                torch.linalg.vector_norm(latent_prediction - latent_actual, dim=1).cpu().numpy()
            )
            embeddings.append(latent_history[:, -1].cpu().numpy())
            embedding_times.extend(batch["valid_time"])
            times.extend(batch["valid_time"])
            future_times.extend(batch["future_time"])
    physical_values = {name: np.concatenate(values) for name, values in physical.items()}
    variables: dict[str, Any] = {
        name: (("time", "channel", "latitude", "longitude"), values.astype("float32"))
        for name, values in physical_values.items()
    }
    variables["model_error"] = (
        ("time", "channel", "latitude", "longitude"),
        (physical_values["latent_prediction"] - physical_values["actual_future"]).astype(
            "float32"
        ),
    )
    variables["persistence_error"] = (
        ("time", "channel", "latitude", "longitude"),
        (physical_values["physical_persistence"] - physical_values["actual_future"]).astype(
            "float32"
        ),
    )
    for name, values in latent_maps.items():
        variables[name] = (
            ("time", "latent_latitude", "latent_longitude"),
            np.concatenate(values).astype("float32"),
        )
    diagnostics = xr.Dataset(
        variables,
        coords={
            "time": pd.to_datetime(times),
            "future_time": ("time", pd.to_datetime(future_times)),
            "channel": list(dataset.channels),
            "latitude": dataset.latitudes,
            "longitude": dataset.longitudes,
            "latent_latitude": np.arange(dataset.latent_shape[0]),
            "latent_longitude": np.arange(dataset.latent_shape[1]),
        },
        attrs={
            "model": "Atlas Latent World Model V0",
            "data_class": "RETROSPECTIVE_REANALYSIS",
            "patch_size": config.patch_size,
            "latent_dimensions": config.latent_dimensions,
        },
    )
    _write_zarr_atomic(diagnostics, experiment_root / "world_diagnostics.zarr", dataset_config)
    _save_embedding_diagnostics(
        experiment_root, np.concatenate(embeddings), embedding_times, dataset.latent_shape
    )


def _save_embedding_diagnostics(
    experiment_root: Path,
    latent: np.ndarray,
    times: list[str],
    latent_shape: tuple[int, int],
) -> None:
    samples, dimensions, height, width = latent.shape
    matrix = latent.transpose(0, 2, 3, 1).reshape(-1, dimensions)
    components = min(3, dimensions, len(matrix))
    projection = PCA(n_components=components, random_state=0).fit_transform(matrix)
    time_values = np.repeat(np.asarray(times), height * width)
    latitude_index = np.tile(np.repeat(np.arange(height), width), samples)
    longitude_index = np.tile(np.tile(np.arange(width), height), samples)
    frame = pd.DataFrame(
        {
            "time": time_values,
            "latent_latitude": latitude_index,
            "latent_longitude": longitude_index,
            "pca_1": projection[:, 0],
            "pca_2": projection[:, 1] if components > 1 else 0.0,
            "pca_3": projection[:, 2] if components > 2 else 0.0,
            "latent_magnitude": np.linalg.norm(matrix, axis=1),
        }
    )
    frame.to_parquet(experiment_root / "latent_pca.parquet", index=False)
    rows = []
    center = (latent_shape[0] // 2, latent_shape[1] // 2)
    for sample_index, query_time in enumerate(times):
        query_flat = sample_index * height * width + center[0] * width + center[1]
        distance = np.linalg.norm(matrix - matrix[query_flat], axis=1)
        nearest = np.argsort(distance)[1:9]
        for rank, flat_index in enumerate(nearest, start=1):
            neighbor_sample, regional_index = divmod(int(flat_index), height * width)
            regional_latitude, regional_longitude = divmod(regional_index, width)
            rows.append(
                {
                    "query_time": query_time,
                    "query_latent_latitude": center[0],
                    "query_latent_longitude": center[1],
                    "rank": rank,
                    "neighbor_time": times[neighbor_sample],
                    "neighbor_latent_latitude": regional_latitude,
                    "neighbor_latent_longitude": regional_longitude,
                    "distance": float(distance[flat_index]),
                }
            )
    pd.DataFrame(rows).to_parquet(experiment_root / "latent_neighbors.parquet", index=False)


def _models(
    config: LatentWorldConfig, physical_channels: int
) -> tuple[RegionalEncoder, RegionalDecoder, LatentDynamics, FrozenLatentProbe]:
    return (
        RegionalEncoder(
            physical_channels,
            config.latent_dimensions,
            config.patch_size,
            config.hidden_channels,
        ),
        RegionalDecoder(
            physical_channels,
            config.latent_dimensions,
            config.patch_size,
            config.hidden_channels,
        ),
        LatentDynamics(config.latent_dimensions),
        FrozenLatentProbe(config.latent_dimensions),
    )


def _datasets(
    data_root: Path, config: LatentWorldConfig
) -> dict[str, LatentSequenceDataset]:
    limits = {
        "train": config.training.max_train_samples,
        "validation": config.training.max_validation_samples,
        "test": config.training.max_test_samples,
    }
    datasets = {
        split: LatentSequenceDataset(data_root, config, split, max_samples=limit)
        for split, limit in limits.items()
    }
    if config.training.cache_samples_in_memory:
        memory_cap = int(config.training.max_memory_cache_gb * GIB)
        estimated = sum(dataset.preload(memory_cap) for dataset in datasets.values())
        if estimated > memory_cap:
            raise MemoryError(
                f"combined latent sample cache estimate {estimated / GIB:.2f} GiB exceeds "
                f"configured {config.training.max_memory_cache_gb:.2f} GiB cap"
            )
        print(f"Preloaded bounded latent sample cache: {estimated / 1024**2:.1f} MiB")
    return datasets


def _loaders(
    datasets: dict[str, LatentSequenceDataset], config: LatentWorldConfig
) -> dict[str, DataLoader[dict[str, Any]]]:
    generator = torch.Generator().manual_seed(config.training.random_seed)
    return {
        split: DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=config.training.num_workers,
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }


def _storage_context(
    config: LatentWorldConfig,
) -> tuple[Path, Path, ResearchConfig]:
    configured_root = Path(os.environ.get("ATLAS_DATA_ROOT", config.storage_root)).resolve()
    source_root = configured_root / "research" / config.dataset
    dataset_config = ResearchConfig.from_yaml(source_root / "metadata" / "config.yaml")
    status = inspect_storage(dataset_config, create=True)
    if status.data_root != configured_root:
        raise ValueError("latent config and source dataset resolve to different data roots")
    if config.required_mount and status.mount_path != config.required_mount.resolve():
        raise ValueError("latent artifacts are not resolving to the required SSD mount")
    return configured_root, source_root, dataset_config


def _resolve_experiment(
    config: LatentWorldConfig,
    dataset_root: Path,
    stages: tuple[str, ...],
    experiment_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    experiments_root = dataset_root / "latent_world" / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    if experiment_id is None and stages[0] != "autoencoder":
        experiment_id = _latest_compatible_experiment(experiments_root, config, stages[0])
    if experiment_id is None:
        experiment_id = (
            f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-latent-d{config.latent_dimensions}-"
            f"{uuid4().hex[:8]}"
        )
        root = experiments_root / experiment_id
        root.mkdir(parents=False, exist_ok=False)
        if config.config_path:
            shutil.copy2(config.config_path, root / "config.yaml")
        metadata = {
            "experiment_id": experiment_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            "model_version": "atlas-latent-world-v0",
            "config_name": config.name,
            "dataset_version": config.dataset,
            "completed_stages": [],
        }
        return root, metadata
    root = experiments_root / experiment_id
    if not root.exists():
        raise FileNotFoundError(f"latent experiment does not exist: {experiment_id}")
    metadata = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if metadata.get("latent_dimensions") != config.latent_dimensions:
        raise ValueError("experiment latent dimensions do not match the requested config")
    return root, metadata


def _latest_compatible_experiment(
    experiments_root: Path, config: LatentWorldConfig, stage: str
) -> str:
    required = "autoencoder" if stage in {"dynamics", "probe"} else None
    candidates = []
    for path in experiments_root.glob("*/metrics.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata.get("dataset_version") == config.dataset
            and metadata.get("latent_dimensions") == config.latent_dimensions
            and (required is None or required in metadata.get("completed_stages", []))
        ):
            candidates.append((metadata.get("updated_at", ""), metadata["experiment_id"]))
    if not candidates:
        raise FileNotFoundError(f"no compatible latent experiment is ready for {stage}")
    return max(candidates)[1]


def _model_metadata(
    config: LatentWorldConfig,
    dataset: LatentSequenceDataset,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    dynamics: LatentDynamics,
    probe: FrozenLatentProbe,
    device: torch.device,
) -> dict[str, Any]:
    latent_height, latent_width = dataset.latent_shape
    return {
        "patch_size": config.patch_size,
        "latent_dimensions": config.latent_dimensions,
        "latent_grid_dimensions": [latent_height, latent_width],
        "physical_grid_dimensions": list(dataset.spatial_shape),
        "channels": list(dataset.channels),
        "context_hours": config.context_hours,
        "forecast_hours": config.forecast_hours,
        "training_period": [
            str(dataset.dataset_config.splits.train.start),
            str(dataset.dataset_config.splits.train.end),
        ],
        "architectures": {
            "encoder": "two 3x3 convolutions plus patch-strided regional projection",
            "decoder": "patch transposed convolution plus two 3x3 convolutions",
            "dynamics": "3x3 ConvGRU with spatial residual prediction head",
            "probe": "frozen encoder plus bilinearly upsampled 1x1 logistic head",
        },
        "parameter_counts": {
            "encoder": parameter_count(encoder),
            "decoder": parameter_count(decoder),
            "dynamics": parameter_count(dynamics),
            "probe": parameter_count(probe),
            "total": parameter_count(encoder, decoder, dynamics, probe),
        },
        "loss_weights": asdict(config.loss),
        "optimizer": "Adam",
        "learning_rate": config.training.learning_rate,
        "epochs": {
            "autoencoder": config.training.autoencoder_epochs,
            "dynamics": config.training.dynamics_epochs,
            "probe": config.training.probe_epochs,
        },
        "sample_limits": {
            "train": config.training.max_train_samples,
            "validation": config.training.max_validation_samples,
            "test": config.training.max_test_samples,
        },
        "device": str(device),
        "compression_ratio": compression_ratio(
            len(dataset.channels),
            *dataset.spatial_shape,
            config.latent_dimensions,
            latent_height,
            latent_width,
        ),
        "missing_value_handling": (
            "normalized missing values are zero-filled only alongside one explicit mask channel "
            "per physical variable; losses exclude missing targets"
        ),
    }


def _save_checkpoint(
    path: Path,
    state: dict[str, Any],
    config: LatentWorldConfig,
    stage: str,
) -> None:
    partial = path.with_suffix(".partial.pt")
    torch.save(
        {
            "stage": stage,
            "latent_dimensions": config.latent_dimensions,
            "patch_size": config.patch_size,
            "state": state,
        },
        partial,
    )
    partial.replace(path)


def _load_autoencoder(
    experiment_root: Path,
    encoder: RegionalEncoder,
    decoder: RegionalDecoder,
    device: torch.device,
) -> None:
    path = experiment_root / "autoencoder.pt"
    if not path.exists():
        raise FileNotFoundError("autoencoder checkpoint is required before this stage")
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["state"]["encoder"])
    decoder.load_state_dict(checkpoint["state"]["decoder"])


def _write_metadata(root: Path, metadata: dict[str, Any]) -> None:
    path = root / "metrics.json"
    partial = root / ".metrics.partial.json"
    partial.write_text(json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8")
    partial.replace(path)


def _upsert_index(dataset_root: Path, metadata: dict[str, Any], root: Path) -> None:
    path = dataset_root / "latent_world" / "experiments.parquet"
    dynamics = metadata.get("dynamics", {}).get("test_metrics", {})
    probe = metadata.get("probe", {}).get("test_metrics", {})
    row = pd.DataFrame(
        [
            {
                "experiment_id": metadata["experiment_id"],
                "updated_at": metadata["updated_at"],
                "dataset_version": metadata["dataset_version"],
                "latent_dimensions": metadata["latent_dimensions"],
                "patch_size": metadata["patch_size"],
                "latent_grid": "×".join(map(str, metadata["latent_grid_dimensions"])),
                "parameter_count": metadata["parameter_counts"]["total"],
                "completed_stages": ", ".join(metadata["completed_stages"]),
                "model_normalized_rmse": dynamics.get("latent_dynamics", {}).get(
                    "overall_normalized_rmse", np.nan
                ),
                "persistence_normalized_rmse": dynamics.get(
                    "physical_persistence", {}
                ).get("overall_normalized_rmse", np.nan),
                "probe_pr_auc": probe.get("pr_auc", np.nan),
                "probe_roc_auc": probe.get("roc_auc", np.nan),
                "experiment_path": str(root),
            }
        ]
    )
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing.loc[existing.experiment_id != metadata["experiment_id"]]
        row = pd.concat([existing, row], ignore_index=True)
    row.to_parquet(path, index=False)


def _probe_positive_weight(loader: DataLoader[dict[str, Any]]) -> float:
    positives = 0.0
    negatives = 0.0
    for batch in loader:
        target = batch["extreme_target"]
        valid = 1 - batch["extreme_mask"]
        positives += float((target * valid).sum())
        negatives += float(((1 - target) * valid).sum())
    return float(np.clip(negatives / max(1.0, positives), 1.0, 20.0))


def _diagnostic_batches(
    loader: DataLoader[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    records = []
    count = 0
    for batch in loader:
        remaining = max_samples - count
        if remaining <= 0:
            break
        if len(batch["valid_time"]) > remaining:
            sliced: dict[str, Any] = {}
            for key, values in batch.items():
                sliced[key] = values[:remaining]
            batch = sliced
        records.append(batch)
        count += len(batch["valid_time"])
    if not records:
        raise ValueError("no test samples available for latent diagnostics")
    return records


def _inverse_batch(values: np.ndarray, dataset: LatentSequenceDataset) -> np.ndarray:
    return (
        values * dataset.standard_deviations[None, :, None, None]
        + dataset.means[None, :, None, None]
    )


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _device(config: LatentWorldConfig) -> torch.device:
    requested = config.training.device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _set_determinism(config: LatentWorldConfig) -> None:
    seed = config.training.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, config.training.torch_threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _estimated_artifact_bytes(
    config: LatentWorldConfig, dataset: LatentSequenceDataset
) -> int:
    encoder, decoder, dynamics, probe = _models(config, len(dataset.channels))
    checkpoints = parameter_count(encoder, decoder, dynamics, probe) * 16
    height, width = dataset.spatial_shape
    latent_height, latent_width = dataset.latent_shape
    samples = config.training.diagnostic_samples
    physical_maps = samples * len(dataset.channels) * height * width * 4 * 9
    latent_maps = samples * latent_height * latent_width * 4 * 4
    metadata = 10_000_000
    return checkpoints + physical_maps + latent_maps + metadata


def _enforce_latent_storage(
    config: LatentWorldConfig,
    dataset_config: ResearchConfig,
    dataset_root: Path,
) -> None:
    latent_size = tree_size(dataset_root / "latent_world")
    if latent_size > config.additional_artifact_cap_gb * GIB:
        raise RuntimeError(
            f"latent artifacts reached {latent_size / GIB:.2f} GiB and exceeded the "
            f"configured {config.additional_artifact_cap_gb:.2f} GiB additional cap"
        )
    total = enforce_cap(dataset_config, dataset_root)
    if total > config.max_storage_gb * GIB:
        raise RuntimeError("latent experiment exceeded its total dataset storage cap")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
