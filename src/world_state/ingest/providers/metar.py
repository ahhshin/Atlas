from __future__ import annotations

import gzip
import io
import math
from datetime import datetime

import httpx
import pandas as pd

from world_state.ingest.artifacts import PointBatch
from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_bytes


class MetarProvider(DataSource):
    name = "metar"
    product = "aviation-weather-bulk-cache"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        content, request_url = get_bytes(
            client,
            self.config["endpoint"],
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [
            RawPayload(
                "metars.cache.csv.gz",
                content,
                request_url,
                "application/gzip",
            )
        ]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[PointBatch]:
        if not payloads:
            return []
        frame = pd.read_csv(io.BytesIO(gzip.decompress(payloads[0].content)), low_memory=False)
        west, south, east, north = self.config.get("bbox", [-170, 15, -50, 75])
        numeric_columns = [
            "latitude",
            "longitude",
            "temp_c",
            "dewpoint_c",
            "wind_dir_degrees",
            "wind_speed_kt",
            "wind_gust_kt",
            "visibility_statute_mi",
            "altim_in_hg",
            "sea_level_pressure_mb",
            "precip_in",
        ]
        for column in numeric_columns:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[
            frame.latitude.between(south, north) & frame.longitude.between(west, east)
        ]

        records: list[NormalizedPoint] = []
        for row in frame.itertuples(index=False):
            try:
                valid_time = utc_datetime(str(row.observation_time))
                station_id = str(row.station_id)
                latitude = float(row.latitude)
                longitude = float(row.longitude)
            except (AttributeError, TypeError, ValueError):
                continue
            values: dict[str, tuple[float, str]] = {}
            candidates = {
                "temperature": (getattr(row, "temp_c", None), "°C"),
                "dewpoint": (getattr(row, "dewpoint_c", None), "°C"),
                "wind_direction": (getattr(row, "wind_dir_degrees", None), "°"),
                "wind_speed": (getattr(row, "wind_speed_kt", None), "kt"),
                "wind_gust": (getattr(row, "wind_gust_kt", None), "kt"),
                "visibility": (getattr(row, "visibility_statute_mi", None), "mi"),
                "precipitation": (getattr(row, "precip_in", None), "in"),
            }
            for variable, (value, unit) in candidates.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    if unit == "kt":
                        value, unit = float(value) * 0.514444, "m/s"
                    elif unit == "in" and variable == "precipitation":
                        value, unit = float(value) * 25.4, "mm"
                    values[variable] = (float(value), unit)
            pressure = getattr(row, "sea_level_pressure_mb", None)
            if not isinstance(pressure, (int, float)) or not math.isfinite(pressure):
                altimeter = getattr(row, "altim_in_hg", None)
                pressure = (
                    float(altimeter) * 33.8638866667
                    if isinstance(altimeter, (int, float)) and math.isfinite(altimeter)
                    else None
                )
            if pressure is not None:
                values["pressure"] = (float(pressure), "hPa")

            source_id = f"{station_id}-{valid_time.isoformat()}"
            for variable, (value, unit) in values.items():
                records.append(
                    NormalizedPoint(
                        source=self.name,
                        source_product=self.product,
                        data_class=self.data_class,
                        valid_time=valid_time,
                        available_at=ingested_at,
                        ingested_at=ingested_at,
                        latitude=latitude,
                        longitude=longitude,
                        variable=variable,
                        value=value,
                        unit=unit,
                        source_id=source_id,
                        station_id=station_id,
                        station_name=station_id,
                        quality_flag=getattr(row, "flight_category", None),
                    )
                )
        return [PointBatch(tuple(records))] if records else []
