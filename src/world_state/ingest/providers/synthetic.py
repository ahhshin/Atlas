from __future__ import annotations

import json
from datetime import datetime

import httpx

from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.synthetic import build_synthetic_dataset
from world_state.variables import VARIABLES


class SyntheticProvider(DataSource):
    name = "synthetic"
    product = "deterministic-conus-v1"
    data_class = DataClass.SYNTHETIC

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del client, now
        content = json.dumps({"seed": 42, "periods": 80, "generator": self.product}).encode()
        return [RawPayload("synthetic-manifest", content, "local://world_state.synthetic")]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[NormalizedPoint]:
        del payloads
        ds = build_synthetic_dataset()
        latest = ds.isel(time=-1)
        valid_time = utc_datetime(str(ds.time.values[-1]))
        records: list[NormalizedPoint] = []
        for variable in ds.data_vars:
            for latitude in ds.latitude.values:
                for longitude in ds.longitude.values:
                    records.append(
                        NormalizedPoint(
                            source=self.name,
                            source_product=self.product,
                            data_class=self.data_class,
                            valid_time=valid_time,
                            available_at=valid_time,
                            ingested_at=ingested_at,
                            latitude=float(latitude),
                            longitude=float(longitude),
                            variable=variable,
                            value=float(
                                latest[variable].sel(latitude=latitude, longitude=longitude).values
                            ),
                            unit=VARIABLES[variable]["unit"],
                            source_id=f"synthetic-{valid_time.isoformat()}-{latitude}-{longitude}",
                            station_id=f"grid-{latitude}-{longitude}",
                            station_name="Synthetic grid cell",
                            quality_flag="synthetic",
                        )
                    )
        return records
