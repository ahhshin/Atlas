from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from world_state.research.config import ResearchConfig


class AtlasMiniDataset:
    """Model-ready sequence API that preserves NaNs and returns explicit masks."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str | None = None,
        normalize: bool = False,
    ) -> None:
        self.root = Path(dataset_root)
        self.config = ResearchConfig.from_yaml(self.root / "metadata" / "config.yaml")
        stores = sorted((self.root / "state").glob("*.zarr"))
        if not stores:
            raise FileNotFoundError(f"no yearly state stores found beneath {self.root}")
        self.state = xr.concat(
            [xr.open_zarr(path, consolidated=True, chunks={}) for path in stores],
            dim="time",
        ).sortby("time")
        self.targets = xr.open_zarr(
            self.root / "targets" / "precipitation_targets.zarr",
            consolidated=True,
            chunks={},
        )
        assignments = pd.read_parquet(self.root / "splits" / "splits.parquet")
        if split is not None:
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"unknown split: {split}")
            assignments = assignments.loc[assignments.split == split]
        self.assignments = assignments.reset_index(drop=True)
        self.normalize = normalize
        self._normalization = self._read_normalization() if normalize else None
        self._time_index = pd.Index(pd.to_datetime(self.state.time.values))

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.assignments.iloc[index]
        valid_time = pd.Timestamp(row.valid_time)
        position = self._time_index.get_loc(valid_time)
        start = position - self.config.context_steps + 1
        if start < 0:
            raise IndexError(f"insufficient context before {valid_time}")
        window = self.state.isel(time=slice(start, position + 1))
        values = (
            window[list(self.config.variables)]
            .to_array("channel")
            .transpose("time", "channel", "latitude", "longitude")
            .compute()
            .values.astype("float32", copy=False)
        )
        masks = (
            window[[f"missing_{name}" for name in self.config.variables]]
            .to_array("channel")
            .transpose("time", "channel", "latitude", "longitude")
            .compute()
            .values.astype(bool, copy=False)
        )
        if self._normalization is not None:
            means, standard_deviations = self._normalization
            values = (values - means[None, :, None, None]) / standard_deviations[
                None, :, None, None
            ]
        target = self.targets.sel(time=np.datetime64(valid_time)).compute()
        return {
            "state": values,
            "missing_mask": masks,
            "target": target.extreme_precipitation_label.values.astype("uint8", copy=False),
            "target_missing_mask": target.target_missing_mask.values.astype(bool, copy=False),
            "precipitation_6h": target.precipitation_6h.values.astype("float32", copy=False),
            "extreme_threshold": target.extreme_threshold.values.astype("float32", copy=False),
            "valid_time": valid_time,
            "metadata": {
                "dataset": self.config.name,
                "split": row.split,
                "feature_start": pd.Timestamp(row.feature_start),
                "feature_end": pd.Timestamp(row.feature_end),
                "target_start": pd.Timestamp(row.target_start),
                "target_end": pd.Timestamp(row.target_end),
                "channels": list(self.config.variables),
                "data_class": "RETROSPECTIVE_REANALYSIS",
            },
        }

    def _read_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / "climatology" / "normalization.parquet"
        frame = pd.read_parquet(path).set_index("variable").loc[list(self.config.variables)]
        mean = frame["mean"].to_numpy(dtype="float32")
        std = frame["std"].to_numpy(dtype="float32")
        std[~np.isfinite(std) | (std == 0)] = 1
        return mean, std
