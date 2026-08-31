from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class LatentLossConfig:
    latent_weight: float = 1.0
    physical_weight: float = 1.0
    reconstruction: Literal["mse", "huber"] = "huber"


@dataclass(frozen=True)
class SpatialCrop:
    latitude_start: int = 0
    longitude_start: int = 0
    height: int | None = None
    width: int | None = None


@dataclass(frozen=True)
class LatentTrainingConfig:
    random_seed: int = 2020
    device: str = "auto"
    batch_size: int = 2
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-4
    autoencoder_epochs: int = 5
    dynamics_epochs: int = 5
    probe_epochs: int = 5
    max_train_samples: int = 512
    max_validation_samples: int = 128
    max_test_samples: int = 128
    diagnostic_samples: int = 8
    num_workers: int = 0
    torch_threads: int = 4
    cache_samples_in_memory: bool = False
    max_memory_cache_gb: float = 0.5


@dataclass(frozen=True)
class LatentWorldConfig:
    name: str
    dataset: str
    patch_size: int
    latent_dimensions: int
    ablation_dimensions: tuple[int, ...]
    context_hours: int
    forecast_hours: int
    cadence_hours: int
    max_parameters: int
    max_storage_gb: float
    additional_artifact_cap_gb: float
    storage_root: Path
    required_mount: Path | None
    hidden_channels: int
    loss: LatentLossConfig
    training: LatentTrainingConfig
    crop: SpatialCrop
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> LatentWorldConfig:
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        latent = raw.get("latent", {})
        training = raw.get("training", {})
        loss = raw.get("loss", {})
        crop = raw.get("crop", {})
        config = cls(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            patch_size=int(raw.get("patch_size", latent.get("patch_size", 4))),
            latent_dimensions=int(
                raw.get("latent_dimensions", latent.get("dimensions", 64))
            ),
            ablation_dimensions=tuple(
                int(value)
                for value in raw.get(
                    "ablation_dimensions",
                    [raw.get("latent_dimensions", latent.get("dimensions", 64))],
                )
            ),
            context_hours=int(raw.get("context_hours", 24)),
            forecast_hours=int(raw.get("forecast_hours", 3)),
            cadence_hours=int(raw.get("cadence_hours", 3)),
            max_parameters=int(raw.get("max_parameters", 10_000_000)),
            max_storage_gb=float(raw.get("max_storage_gb", 15)),
            additional_artifact_cap_gb=float(raw.get("additional_artifact_cap_gb", 2)),
            storage_root=Path(raw.get("storage_root", "/mnt/games/Atlas/data")),
            required_mount=(Path(raw["required_mount"]) if raw.get("required_mount") else None),
            hidden_channels=int(raw.get("hidden_channels", 48)),
            loss=LatentLossConfig(
                latent_weight=float(loss.get("latent_weight", 1.0)),
                physical_weight=float(loss.get("physical_weight", 1.0)),
                reconstruction=str(loss.get("reconstruction", "huber")),
            ),
            training=LatentTrainingConfig(
                random_seed=int(training.get("random_seed", 2020)),
                device=str(training.get("device", "auto")),
                batch_size=int(training.get("batch_size", 2)),
                learning_rate=float(training.get("learning_rate", 1e-3)),
                fine_tune_learning_rate=float(
                    training.get("fine_tune_learning_rate", 1e-4)
                ),
                autoencoder_epochs=int(training.get("autoencoder_epochs", 5)),
                dynamics_epochs=int(training.get("dynamics_epochs", 5)),
                probe_epochs=int(training.get("probe_epochs", 5)),
                max_train_samples=int(training.get("max_train_samples", 512)),
                max_validation_samples=int(training.get("max_validation_samples", 128)),
                max_test_samples=int(training.get("max_test_samples", 128)),
                diagnostic_samples=int(training.get("diagnostic_samples", 8)),
                num_workers=int(training.get("num_workers", 0)),
                torch_threads=int(training.get("torch_threads", 4)),
                cache_samples_in_memory=bool(
                    training.get("cache_samples_in_memory", False)
                ),
                max_memory_cache_gb=float(training.get("max_memory_cache_gb", 0.5)),
            ),
            crop=SpatialCrop(
                latitude_start=int(crop.get("latitude_start", 0)),
                longitude_start=int(crop.get("longitude_start", 0)),
                height=(int(crop["height"]) if crop.get("height") is not None else None),
                width=(int(crop["width"]) if crop.get("width") is not None else None),
            ),
            config_path=config_path,
        )
        config.validate()
        return config

    @property
    def context_steps(self) -> int:
        return self.context_hours // self.cadence_hours

    @property
    def forecast_steps(self) -> int:
        return self.forecast_hours // self.cadence_hours

    def with_latent_dimensions(self, dimensions: int) -> LatentWorldConfig:
        return replace(self, latent_dimensions=dimensions, name=f"{self.name}-d{dimensions}")

    def validate(self) -> None:
        if self.patch_size <= 0 or self.latent_dimensions <= 0:
            raise ValueError("patch size and latent dimensions must be positive")
        if self.context_hours % self.cadence_hours:
            raise ValueError("context_hours must be divisible by cadence_hours")
        if self.forecast_hours % self.cadence_hours:
            raise ValueError("forecast_hours must be divisible by cadence_hours")
        if self.forecast_steps != 1:
            raise ValueError("latent world V0 currently predicts exactly one cadence step")
        if self.loss.latent_weight < 0 or self.loss.physical_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.loss.latent_weight + self.loss.physical_weight == 0:
            raise ValueError("at least one training loss must be enabled")
        if self.additional_artifact_cap_gb <= 0:
            raise ValueError("additional artifact cap must be positive")
        if not self.ablation_dimensions:
            raise ValueError("at least one ablation dimension is required")
        if any(value <= 0 for value in self.ablation_dimensions):
            raise ValueError("ablation dimensions must be positive")
