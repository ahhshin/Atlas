from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

torch = pytest.importorskip("torch", reason="latent-world tests require the optional torch extra")

from world_state.research.config import VARIABLES, ResearchConfig
from world_state.research.latent_config import LatentWorldConfig
from world_state.research.latent_data import LatentSequenceDataset
from world_state.research.latent_experiments import load_latent_models, run_latent_stages
from world_state.research.latent_models import (
    FrozenLatentProbe,
    LatentDynamics,
    RegionalDecoder,
    RegionalEncoder,
    masked_loss,
    padded_shape,
    parameter_count,
)
from world_state.research.pipeline import backfill


class LatentFixtureSource:
    def fetch_year(self, config: ResearchConfig, year: int) -> xr.Dataset:
        times = config.timestamps[config.timestamps.year == year]
        latitude = np.linspace(0, 2, 3, dtype="float32")
        longitude = np.linspace(-1, 1, 3, dtype="float32")
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
def latent_fixture(tmp_path: Path) -> tuple[LatentWorldConfig, Path]:
    data_root = tmp_path / "atlas-data"
    data_root.mkdir()
    research_raw = {
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
        "storage_root": str(data_root),
        "variables": list(VARIABLES),
        "splits": {
            "train": {"start": "2020-01-01", "end": "2020-01-05"},
            "validation": {"start": "2020-01-06", "end": "2020-01-07"},
            "test": {"start": "2020-01-08", "end": "2020-01-10"},
        },
    }
    research_path = tmp_path / "research.yaml"
    research_path.write_text(yaml.safe_dump(research_raw), encoding="utf-8")
    research_config = ResearchConfig.from_yaml(research_path)
    backfill(research_config, source=LatentFixtureSource())
    latent_raw = {
        "name": "fixture-latent",
        "dataset": research_config.name,
        "latent": {"patch_size": 2, "dimensions": 4},
        "ablation_dimensions": [4],
        "context_hours": 24,
        "forecast_hours": 3,
        "cadence_hours": 3,
        "hidden_channels": 4,
        "max_parameters": 100_000,
        "storage_root": str(data_root),
        "max_storage_gb": 0.1,
        "additional_artifact_cap_gb": 0.05,
        "loss": {"latent_weight": 1, "physical_weight": 1, "reconstruction": "mse"},
        "training": {
            "random_seed": 17,
            "device": "cpu",
            "torch_threads": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "autoencoder_epochs": 1,
            "dynamics_epochs": 1,
            "probe_epochs": 1,
            "max_train_samples": 2,
            "max_validation_samples": 1,
            "max_test_samples": 1,
            "diagnostic_samples": 1,
        },
    }
    latent_path = tmp_path / "latent.yaml"
    latent_path.write_text(yaml.safe_dump(latent_raw), encoding="utf-8")
    return LatentWorldConfig.from_yaml(latent_path), data_root


def test_regional_latent_shapes_padding_masking_and_determinism() -> None:
    torch.manual_seed(11)
    values = torch.randn(2, 10, 5, 7)
    missing = torch.zeros_like(values)
    missing[:, :, 0, 0] = 1
    encoder = RegionalEncoder(10, 8, patch_size=4, hidden_channels=6)
    decoder = RegionalDecoder(10, 8, patch_size=4, hidden_channels=6)
    dynamics = LatentDynamics(8)
    latent = encoder(values, missing)
    assert padded_shape(5, 7, 4) == (8, 8)
    assert latent.shape == (2, 8, 2, 2)
    assert decoder(latent, (5, 7)).shape == values.shape
    assert dynamics(latent[:, None].repeat(1, 8, 1, 1, 1)).shape == latent.shape
    assert parameter_count(encoder, decoder, dynamics, FrozenLatentProbe(8)) < 100_000
    prediction = values.clone()
    prediction[:, :, 0, 0] = 1_000_000
    assert masked_loss(prediction, values, missing, kind="mse") == pytest.approx(0)
    torch.manual_seed(11)
    repeated = RegionalEncoder(10, 8, patch_size=4, hidden_channels=6)
    torch.manual_seed(11)
    expected = RegionalEncoder(10, 8, patch_size=4, hidden_channels=6)
    np.testing.assert_allclose(
        repeated(values, missing).detach().numpy(), expected(values, missing).detach().numpy()
    )


def test_latent_dataset_alignment_masks_and_frozen_probe(
    latent_fixture: tuple[LatentWorldConfig, Path],
) -> None:
    config, data_root = latent_fixture
    first = LatentSequenceDataset(data_root, config, "test", max_samples=1)
    second = LatentSequenceDataset(data_root, config, "test", max_samples=1)
    sample = first[0]
    assert sample["history"].shape == (8, 10, 3, 3)
    assert sample["future"].shape == (10, 3, 3)
    assert first.latent_shape == (2, 2)
    assert pd.Timestamp(sample["feature_start"]) < pd.Timestamp(sample["valid_time"])
    assert pd.Timestamp(sample["future_time"]) == pd.Timestamp(sample["valid_time"]) + pd.Timedelta(
        hours=3
    )
    assert np.all(sample["history"][sample["history_mask"].astype(bool)] == 0)
    np.testing.assert_array_equal(first[0]["history"], second[0]["history"])
    with pytest.raises(MemoryError, match="memory cap"):
        first.preload(1)
    assert first.preload(10 * 1024**2) > 0
    encoder = RegionalEncoder(10, 4, 2, 4)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    probe = FrozenLatentProbe(4)
    latent = encoder(
        torch.from_numpy(sample["history"][-1:]),
        torch.from_numpy(sample["history_mask"][-1:]),
    )
    probe(latent, sample["extreme_target"].shape).mean().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert any(parameter.grad is not None for parameter in probe.parameters())


def test_cpu_latent_smoke_checkpoint_metadata_and_storage(
    latent_fixture: tuple[LatentWorldConfig, Path],
) -> None:
    config, data_root = latent_fixture
    metadata = run_latent_stages(config, ("autoencoder", "dynamics", "probe"))
    assert metadata["completed_stages"] == ["autoencoder", "dynamics", "probe"]
    assert metadata["latent_grid_dimensions"] == [2, 2]
    assert metadata["parameter_counts"]["total"] < config.max_parameters
    assert metadata["probe"]["encoder_frozen"] is True
    assert "physical_persistence" in metadata["dynamics"]["test_metrics"]
    assert "latent_persistence_mse" in metadata["dynamics"]["test_metrics"]
    root = (
        data_root
        / "research"
        / config.dataset
        / "latent_world"
        / "experiments"
        / metadata["experiment_id"]
    )
    for name in (
        "autoencoder.pt",
        "dynamics.pt",
        "probe.pt",
        "metrics.json",
        "reconstruction.zarr",
        "world_diagnostics.zarr",
        "latent_pca.parquet",
        "latent_neighbors.parquet",
        "probe_calibration.parquet",
    ):
        assert (root / name).exists()
    encoder, decoder, dynamics, probe = load_latent_models(root, config, 10)
    assert parameter_count(encoder, decoder, dynamics, probe) == metadata["parameter_counts"][
        "total"
    ]
    assert sum(item.stat().st_size for item in root.rglob("*") if item.is_file()) < 0.05 * 1024**3
