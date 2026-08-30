from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import httpx

from world_state.ingest.artifacts import PointBatch
from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes

VARIABLES = {
    "PM25": ("pm2_5", "µg/m³"),
    "PM10": ("pm10", "µg/m³"),
    "OZONE": ("ozone", "ppm"),
    "CO": ("carbon_monoxide", "ppm"),
    "NO2": ("nitrogen_dioxide", "ppb"),
    "SO2": ("sulfur_dioxide", "ppb"),
}


class AirNowProvider(DataSource):
    name = "airnow"
    product = "airnow-hourly-observations"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        environment_name = self.config.get("api_key_env", "AIRNOW_API_KEY")
        key = os.environ.get(environment_name)
        if not key:
            raise RuntimeError(f"Set {environment_name} to enable AirNow ingestion")
        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=int(self.config.get("lookback_hours", 2)))
        params = {
            "startDate": start.strftime("%Y-%m-%dT%H"),
            "endDate": end.strftime("%Y-%m-%dT%H"),
            "parameters": ",".join(self.config.get("parameters", VARIABLES)),
            "BBOX": ",".join(str(value) for value in self.config.get("bbox", [-125, 24, -66, 50])),
            "dataType": "A",
            "format": "application/json",
            "verbose": "1",
            "monitorType": "0",
            "includerawconcentrations": "0",
            "API_KEY": key,
        }
        content, request_url = get_json_bytes(
            client,
            self.config["endpoint"],
            params=params,
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [RawPayload("airnow-current", content, request_url.replace(key, "REDACTED"))]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[PointBatch]:
        if not payloads:
            return []
        records: list[NormalizedPoint] = []
        for row in json.loads(payloads[0].content):
            parameter = str(row.get("Parameter", "")).upper()
            if parameter not in VARIABLES:
                continue
            variable, default_unit = VARIABLES[parameter]
            try:
                value = float(row["Value"])
                latitude = float(row["Latitude"])
                longitude = float(row["Longitude"])
                valid_time = utc_datetime(row["UTC"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            site_id = str(row.get("FullAQSCode") or row.get("SiteName") or f"{latitude}-{longitude}")
            records.append(
                NormalizedPoint(
                    self.name,
                    self.product,
                    self.data_class,
                    valid_time,
                    ingested_at,
                    ingested_at,
                    latitude,
                    longitude,
                    variable,
                    value,
                    str(row.get("Unit") or default_unit),
                    f"{site_id}-{parameter}-{valid_time.isoformat()}",
                    station_id=site_id,
                    station_name=row.get("SiteName") or row.get("ReportingArea"),
                    quality_flag=str(row.get("Category") or row.get("AQI") or ""),
                )
            )
        return [PointBatch(tuple(records))] if records else []
