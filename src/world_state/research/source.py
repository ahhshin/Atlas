from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
import xarray as xr

from world_state.research.config import VARIABLES, ResearchConfig


class HistoricalSource(Protocol):
    def fetch_year(self, config: ResearchConfig, year: int) -> xr.Dataset: ...


class EarthmoverERA5Source:
    """Anonymous, analysis-ready ERA5 access through the public AWS Icechunk archive."""

    def __init__(self, source_config: dict[str, object] | None = None) -> None:
        values = source_config or {}
        self.bucket = str(values.get("bucket", "earthmover-icechunk-era5"))
        self.prefix = str(values.get("prefix", "icechunkV2"))
        self.region = str(values.get("region", "us-east-1"))
        self.branch = str(values.get("branch", "main"))
        self.group = str(values.get("group", "single/temporal"))
        self._dataset: xr.Dataset | None = None

    def open(self) -> xr.Dataset:
        if self._dataset is None:
            import icechunk
            import pcodec  # noqa: F401 -- registers the remote PCodec decoder

            storage = icechunk.s3_storage(
                bucket=self.bucket,
                prefix=self.prefix,
                region=self.region,
                anonymous=True,
            )
            repository = icechunk.Repository.open(storage)
            session = repository.readonly_session(self.branch)
            self._dataset = xr.open_zarr(
                session.store,
                group=self.group,
                consolidated=False,
                chunks={},
            )
        return self._dataset

    def fetch_year(self, config: ResearchConfig, year: int) -> xr.Dataset:
        period_start = max(config.start, pd.Timestamp(year=year, month=1, day=1))
        period_end = min(config.end, pd.Timestamp(year=year, month=12, day=31, hour=23))
        anchors = pd.date_range(
            period_start.ceil(f"{config.cadence_hours}h"),
            period_end,
            freq=f"{config.cadence_hours}h",
        )
        if anchors.empty:
            raise ValueError(f"year {year} has no configured timestamps")
        expanded_start = anchors[0] - pd.Timedelta(hours=config.cadence_hours - 1)
        remote = self.open()
        source_names = [VARIABLES[name].source_name for name in config.variables]
        west = config.bbox.west % 360
        east = config.bbox.east % 360
        selected = remote[source_names].sel(
            valid_time=slice(expanded_start, anchors[-1]),
            latitude=slice(config.bbox.north, config.bbox.south),
            longitude=slice(west, east),
        )
        output: dict[str, xr.DataArray] = {}
        for name in config.variables:
            spec = VARIABLES[name]
            values = selected[spec.source_name]
            if spec.temporal_kind == "accumulation":
                values = values.coarsen(valid_time=config.cadence_hours, boundary="exact").sum()
                values = values.assign_coords(valid_time=anchors)
            else:
                values = values.sel(valid_time=anchors)
            values = (values * np.float32(spec.scale)).astype("float32")
            values.attrs = {
                "units": spec.units,
                "source_variable": spec.source_name,
                "temporal_semantics": (
                    f"{config.cadence_hours}-hour accumulation ending at valid time"
                    if spec.temporal_kind == "accumulation"
                    else "instantaneous at valid time"
                ),
            }
            output[name] = values
            output[f"missing_{name}"] = values.isnull()
        dataset = xr.Dataset(output).rename(valid_time="time").reset_coords(drop=True)
        dataset = dataset.assign_coords(longitude=((dataset.longitude + 180) % 360) - 180)
        dataset = dataset.sortby("latitude").sortby("longitude")
        for variable in dataset.variables.values():
            variable.encoding = {}
        dataset.attrs.update(
            {
                "source": "earthmover-era5",
                "source_product": "ERA5 single-level temporal Icechunk",
                "data_class": "RETROSPECTIVE_REANALYSIS",
                "retrospective": True,
                "cadence_hours": config.cadence_hours,
                "resolution_degrees": config.resolution_degrees,
                "provenance": (
                    "Copernicus ERA5 via the public Earthmover AWS archive; "
                    "not an operational point-in-time feed"
                ),
            }
        )
        return dataset.chunk(
            {
                "time": config.chunk_time,
                "latitude": config.chunk_latitude,
                "longitude": config.chunk_longitude,
            }
        )
