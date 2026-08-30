from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta

import httpx

from world_state.ingest.artifacts import PointBatch
from world_state.ingest.base import DataClass, DataSource, NormalizedPoint, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes

TYPES = {"D": "electricity_demand", "NG": "net_generation"}


class EiaProvider(DataSource):
    name = "eia"
    product = "electricity-rto-region-data"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        environment_name = self.config.get("api_key_env", "EIA_API_KEY")
        key = os.environ.get(environment_name)
        if not key:
            raise RuntimeError(f"Set {environment_name} to enable EIA ingestion")
        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=int(self.config.get("lookback_hours", 6)))
        params = {
            "api_key": key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[type][]": list(self.config.get("types", TYPES)),
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "offset": 0,
            "length": int(self.config.get("length", 5000)),
        }
        content, request_url = get_json_bytes(
            client,
            self.config["endpoint"],
            params=params,
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [RawPayload("eia-rto-current", content, request_url.replace(key, "REDACTED"))]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[PointBatch]:
        if not payloads:
            return []
        rows = json.loads(payloads[0].content).get("response", {}).get("data", [])
        records: list[NormalizedPoint] = []
        for row in rows:
            kind = str(row.get("type"))
            if kind not in TYPES:
                continue
            try:
                value = float(row["value"])
                valid_time = utc_datetime(row["period"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            respondent = str(row.get("respondent") or "unknown")
            records.append(
                NormalizedPoint(
                    self.name,
                    self.product,
                    self.data_class,
                    valid_time,
                    ingested_at,
                    ingested_at,
                    None,
                    None,
                    TYPES[kind],
                    value,
                    str(row.get("value-units") or "MWh"),
                    f"{respondent}-{kind}-{valid_time.isoformat()}",
                    station_id=respondent,
                    station_name=row.get("respondent-name"),
                )
            )
        return [PointBatch(tuple(records))] if records else []
