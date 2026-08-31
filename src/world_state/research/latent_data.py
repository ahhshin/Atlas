from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from world_state.research.config import ResearchConfig
from world_state.research.latent_config import LatentWorldConfig


class LatentSequenceDataset(Dataset[dict[str, Any]]):
    """Existing Atlas state exposed as normalized histories and one-step targets."""

    def __init__(
        self,
        data_root: str | Path,
        config: LatentWorldConfig,
        split: str,
        *,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split}")
        self.data_root = Path(data_root)
        self.root = self.data_root / "research" / config.dataset
        self.config = config
        self.dataset_config = ResearchConfig.from_yaml(self.root / "metadata" / "config.yaml")
        if config.cadence_hours != self.dataset_config.cadence_hours:
            raise ValueError("latent cadence must match the source dataset")
        if config.context_hours > self.dataset_config.context_hours:
            raise ValueError("latent context exceeds the existing sequence support")
        stores = sorted((self.root / "state").glob("*.zarr"))
        if not stores:
            raise FileNotFoundError(f"no state stores found beneath {self.root}")
        self.state = xr.concat(
            [xr.open_zarr(path, consolidated=True, chunks={}) for path in stores], dim="time"
        ).sortby("time")
        self.targets = xr.open_zarr(
            self.root / "targets" / "precipitation_targets.zarr",
            consolidated=True,
            chunks={},
        )
        assignments = pd.read_parquet(self.root / "splits" / "splits.parquet")
        assignments = assignments.loc[assignments.split == split].reset_index(drop=True)
        if max_samples is not None and len(assignments) > max_samples:
            selected = np.linspace(0, len(assignments) - 1, max_samples, dtype=int)
            assignments = assignments.iloc[selected].reset_index(drop=True)
        self.assignments = assignments
        self.split = split
        self._time_index = pd.Index(pd.to_datetime(self.state.time.values))
        normalization = pd.read_parquet(
            self.root / "climatology" / "normalization.parquet"
        ).set_index("variable")
        normalization = normalization.loc[list(self.dataset_config.variables)]
        self.means = normalization["mean"].to_numpy(dtype="float32")
        self.standard_deviations = normalization["std"].to_numpy(dtype="float32")
        self.standard_deviations[
            ~np.isfinite(self.standard_deviations) | (self.standard_deviations == 0)
        ] = 1
        self.latitude_slice, self.longitude_slice = self._crop_slices()
        self.latitudes = self.state.latitude.isel(latitude=self.latitude_slice).values
        self.longitudes = self.state.longitude.isel(longitude=self.longitude_slice).values
        self._cache: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.assignments)

    @property
    def channels(self) -> tuple[str, ...]:
        return self.dataset_config.variables

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return len(self.latitudes), len(self.longitudes)

    @property
    def latent_shape(self) -> tuple[int, int]:
        height, width = self.spatial_shape
        patch = self.config.patch_size
        return (height + patch - 1) // patch, (width + patch - 1) // patch

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_item(index)

    def preload(self, max_bytes: int) -> int:
        channels = len(self.channels)
        height, width = self.spatial_shape
        estimated = len(self) * (self.config.context_steps + 1) * channels * height * width * 8
        if estimated > max_bytes:
            raise MemoryError(
                f"latent sample cache estimate {estimated / 1024**3:.2f} GiB exceeds "
                f"configured {max_bytes / 1024**3:.2f} GiB memory cap"
            )
        self._cache = [self._load_item(index) for index in range(len(self))]
        return estimated

    def _load_item(self, index: int) -> dict[str, Any]:
        row = self.assignments.iloc[index]
        valid_time = pd.Timestamp(row.valid_time)
        position = int(self._time_index.get_loc(valid_time))
        history_start = position - self.config.context_steps + 1
        future_position = position + self.config.forecast_steps
        if history_start < 0 or future_position >= len(self._time_index):
            raise IndexError(f"sample at {valid_time} lacks required context or target")
        positions = list(range(history_start, position + 1)) + [future_position]
        selected = self.state.isel(
            time=positions,
            latitude=self.latitude_slice,
            longitude=self.longitude_slice,
        )
        values = (
            selected[list(self.channels)]
            .to_array("channel")
            .transpose("time", "channel", "latitude", "longitude")
            .compute()
            .values.astype("float32", copy=False)
        )
        masks = (
            selected[[f"missing_{name}" for name in self.channels]]
            .to_array("channel")
            .transpose("time", "channel", "latitude", "longitude")
            .compute()
            .values.astype(bool, copy=False)
        )
        masks |= ~np.isfinite(values)
        normalized = (values - self.means[None, :, None, None]) / self.standard_deviations[
            None, :, None, None
        ]
        normalized = np.where(masks, 0.0, normalized).astype("float32", copy=False)
        history = normalized[:-1]
        history_mask = masks[:-1]
        future = normalized[-1]
        future_mask = masks[-1]
        target = self.targets.sel(time=np.datetime64(valid_time)).isel(
            latitude=self.latitude_slice,
            longitude=self.longitude_slice,
        )
        return {
            "history": history,
            "history_mask": history_mask.astype("float32"),
            "future": future,
            "future_mask": future_mask.astype("float32"),
            "extreme_target": target.extreme_precipitation_label.values.astype(
                "float32", copy=False
            ),
            "extreme_mask": target.target_missing_mask.values.astype("float32", copy=False),
            "valid_time": valid_time.isoformat(),
            "future_time": pd.Timestamp(self._time_index[future_position]).isoformat(),
            "feature_start": pd.Timestamp(self._time_index[history_start]).isoformat(),
        }

    def inverse_normalize(self, values: np.ndarray) -> np.ndarray:
        return (
            values * self.standard_deviations[:, None, None]
            + self.means[:, None, None]
        )

    def _crop_slices(self) -> tuple[slice, slice]:
        total_height = self.state.sizes["latitude"]
        total_width = self.state.sizes["longitude"]
        latitude_start = self.config.crop.latitude_start
        longitude_start = self.config.crop.longitude_start
        height = self.config.crop.height or (total_height - latitude_start)
        width = self.config.crop.width or (total_width - longitude_start)
        if latitude_start < 0 or longitude_start < 0 or height <= 0 or width <= 0:
            raise ValueError("latent crop must have non-negative offsets and positive size")
        if latitude_start + height > total_height or longitude_start + width > total_width:
            raise ValueError("latent crop exceeds the physical source grid")
        return (
            slice(latitude_start, latitude_start + height),
            slice(longitude_start, longitude_start + width),
        )
