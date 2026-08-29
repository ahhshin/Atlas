from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import httpx

from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes

FIELD_MAP = {
    "temperature": ("temperature", "°C"),
    "dew_point": ("dewpoint", "°C"),
    "humidity": ("relativeHumidity", "%"),
    "pressure": ("barometricPressure", "hPa"),
    "wind_speed": ("windSpeed", "m/s"),
    "wind_direction": ("windDirection", "°"),
    "precipitation": ("precipitationLastHour", "mm"),
}


def _convert(value: float, unit_code: str | None, canonical_unit: str) -> float:
    unit = (unit_code or "").split(":")[-1]
    if canonical_unit == "hPa" and unit == "Pa":
        return value / 100
    if canonical_unit == "m/s" and unit in {"km_h-1", "km/h"}:
        return value / 3.6
    if canonical_unit == "°C" and unit == "K":
        return value - 273.15
    return value


class NWSProvider(DataSource):
    name = "nws"
    product = "station-observations-latest"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        self.fetch_errors = []
        payloads: list[RawPayload] = []
        headers = {
            "Accept": "application/geo+json",
            "User-Agent": self.http_config.get("user_agent", "world-state-personal-research/0.1"),
        }
        for station in self.config.get("stations", []):
            url = self.config["endpoint"].format(station=station)
            try:
                content, request_url = get_json_bytes(
                    client,
                    url,
                    headers=headers,
                    retries=int(self.http_config.get("retries", 3)),
                    backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
                )
                payloads.append(RawPayload(station, content, request_url))
            except (httpx.HTTPError, ValueError) as error:
                self.fetch_errors.append(f"{station}: {error}")
        if not payloads and self.fetch_errors:
            raise RuntimeError("; ".join(self.fetch_errors))
        return payloads

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[NormalizedPoint]:
        records: list[NormalizedPoint] = []
        for payload in payloads:
            feature: dict[str, Any] = json.loads(payload.content)
            properties = feature.get("properties", {})
            coordinates = (feature.get("geometry") or {}).get("coordinates", [])
            timestamp = properties.get("timestamp")
            if len(coordinates) < 2 or not timestamp:
                continue
            longitude, latitude = map(float, coordinates[:2])
            if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                continue
            valid_time = utc_datetime(timestamp)
            source_id = str(properties.get("@id") or feature.get("id") or payload.identifier)
            for variable, (field, canonical_unit) in FIELD_MAP.items():
                quantity = properties.get(field) or {}
                value = quantity.get("value")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
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
                        value=_convert(float(value), quantity.get("unitCode"), canonical_unit),
                        unit=canonical_unit,
                        quality_flag=quantity.get("qualityControl"),
                        source_id=source_id,
                        station_id=properties.get("stationId") or payload.identifier,
                        station_name=properties.get("stationName"),
                    )
                )
        return records
