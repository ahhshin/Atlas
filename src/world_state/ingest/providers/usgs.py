from __future__ import annotations

import json
import math
from datetime import datetime

import httpx

from world_state.ingest.artifacts import PointBatch
from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes

PARAMETERS = {
    "00060": ("streamflow", "ft³/s"),
    "00065": ("gauge_height", "ft"),
}


class UsgsProvider(DataSource):
    name = "usgs"
    product = "nwis-instantaneous-values"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        params = {
            "format": "json",
            "sites": ",".join(self.config["sites"]),
            "parameterCd": ",".join(self.config.get("parameters", PARAMETERS)),
            "period": self.config.get("period", "P1D"),
            "siteStatus": "active",
        }
        content, request_url = get_json_bytes(
            client,
            self.config["endpoint"],
            params=params,
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [RawPayload("usgs-nwis-current", content, request_url)]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[PointBatch]:
        if not payloads:
            return []
        document = json.loads(payloads[0].content)
        records: list[NormalizedPoint] = []
        for series in document.get("value", {}).get("timeSeries", []):
            source = series.get("sourceInfo") or {}
            location = (source.get("geoLocation") or {}).get("geogLocation") or {}
            site_codes = source.get("siteCode") or []
            variable_info = series.get("variable") or {}
            variable_codes = variable_info.get("variableCode") or []
            if not site_codes or not variable_codes:
                continue
            site_id = str(site_codes[0].get("value"))
            parameter = str(variable_codes[0].get("value"))
            if parameter not in PARAMETERS:
                continue
            variable, default_unit = PARAMETERS[parameter]
            values = [
                value
                for block in series.get("values", [])
                for value in block.get("value", [])
                if value.get("value") not in {None, ""}
            ]
            if not values:
                continue
            latest = max(values, key=lambda value: utc_datetime(value["dateTime"]))
            try:
                numeric = float(latest["value"])
                latitude = float(location["latitude"])
                longitude = float(location["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                continue
            valid_time = utc_datetime(latest["dateTime"])
            unit = (variable_info.get("unit") or {}).get("unitCode") or default_unit
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
                    numeric,
                    unit,
                    f"{site_id}-{parameter}-{valid_time.isoformat()}",
                    station_id=site_id,
                    station_name=source.get("siteName"),
                    quality_flag=",".join(latest.get("qualifiers") or []),
                )
            )
        return [PointBatch(tuple(records))] if records else []
