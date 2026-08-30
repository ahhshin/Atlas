from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

import httpx

from world_state.ingest.artifacts import PointBatch
from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes

FIELD_MAP: dict[str, tuple[tuple[str, ...], str]] = {
    "temperature": (("air_temp",), "°C"),
    "dew_point": (("dwpt_temp", "avg_dwpt_temp_pst1hr"), "°C"),
    "humidity": (("rel_hum", "avg_rel_hum_pst1hr"), "%"),
    "pressure": (("stn_pres", "mslp"), "hPa"),
    "wind_speed": (
        (
            "avg_wnd_spd_10m_pst10mts",
            "avg_wnd_spd_10m_pst2mts",
            "avg_wnd_spd_10m_pst1mt",
        ),
        "m/s",
    ),
    "wind_direction": (
        (
            "avg_wnd_dir_10m_pst10mts",
            "avg_wnd_dir_10m_pst2mts",
            "avg_wnd_dir_10m_pst1mt",
        ),
        "°",
    ),
    "precipitation": (("pcpn_amt_pst1hr", "rnfl_amt_pst1hr"), "mm"),
}


def _convert(value: float, source_unit: str | None, canonical_unit: str) -> float:
    if canonical_unit == "m/s" and source_unit in {"km/h", "km h-1", "km_h-1"}:
        return value / 3.6
    if canonical_unit == "hPa" and source_unit == "Pa":
        return value / 100
    return value


def _first_value(properties: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, float] | None:
    for field in fields:
        value = properties.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return field, float(value)
    return None


class ECCCProvider(DataSource):
    name = "eccc"
    product = "swob-realtime"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        self.fetch_errors = []
        payloads: list[RawPayload] = []
        endpoint = self.config["endpoint"]
        for station in self.config.get("stations", []):
            try:
                content, request_url = get_json_bytes(
                    client,
                    endpoint,
                    params={
                        "f": "json",
                        "lang": "en",
                        "limit": 1,
                        "sortby": "-date_tm-value",
                        "url": station,
                    },
                    retries=int(self.http_config.get("retries", 3)),
                    backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
                )
                payloads.append(RawPayload(station, content, request_url))
            except (httpx.HTTPError, ValueError) as error:
                self.fetch_errors.append(f"{station}: {error}")
        if not payloads and self.fetch_errors:
            raise RuntimeError("; ".join(self.fetch_errors))
        return payloads

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[PointBatch]:
        records: list[NormalizedPoint] = []
        for payload in payloads:
            document = json.loads(payload.content)
            for feature in document.get("features", []):
                coordinates = (feature.get("geometry") or {}).get("coordinates", [])
                properties = feature.get("properties", {})
                if len(coordinates) < 2:
                    continue
                longitude, latitude = map(float, coordinates[:2])
                if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
                    continue
                valid_value = properties.get("date_tm-value") or properties.get("obs_date_tm")
                if not valid_value:
                    continue
                valid_time = utc_datetime(valid_value)
                available_at = utc_datetime(properties.get("processed_date_tm") or ingested_at)
                source_id = str(properties.get("id") or feature.get("id") or payload.identifier)
                station_id = str(
                    properties.get("tc_id-value")
                    or properties.get("wmo_synop_id-value")
                    or payload.identifier
                )
                station_name = properties.get("stn_nam-value")
                for variable, (fields, canonical_unit) in FIELD_MAP.items():
                    selected = _first_value(properties, fields)
                    if selected is None:
                        continue
                    field, value = selected
                    source_unit = properties.get(f"{field}-uom")
                    quality = properties.get(f"{field}-qa")
                    if quality is None:
                        quality = properties.get(f"{field}-data_flag-value")
                    records.append(
                        NormalizedPoint(
                            source=self.name,
                            source_product=self.product,
                            data_class=self.data_class,
                            valid_time=valid_time,
                            available_at=available_at,
                            ingested_at=ingested_at,
                            latitude=latitude,
                            longitude=longitude,
                            variable=variable,
                            value=_convert(value, source_unit, canonical_unit),
                            unit=canonical_unit,
                            quality_flag=None if quality is None else str(quality),
                            source_id=source_id,
                            station_id=station_id,
                            station_name=station_name,
                        )
                    )
        return [PointBatch(tuple(records))] if records else []
