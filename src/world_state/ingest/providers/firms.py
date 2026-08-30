from __future__ import annotations

import io
import os
from datetime import UTC, datetime

import httpx
import pandas as pd

from world_state.ingest.artifacts import ArtifactProvenance, EventCollection
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.http import get_bytes


class FirmsProvider(DataSource):
    name = "firms"
    product = "viirs-snpp-nrt"
    data_class = DataClass.OBSERVED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        environment_name = self.config.get("api_key_env", "FIRMS_MAP_KEY")
        key = os.environ.get(environment_name)
        if not key:
            raise RuntimeError(f"Set {environment_name} to enable NASA FIRMS ingestion")
        west, south, east, north = self.config.get("bbox", [-170, 15, -50, 75])
        area = f"{west},{south},{east},{north}"
        url = self.config["endpoint"].format(
            map_key=key,
            source=self.config.get("source", "VIIRS_SNPP_NRT"),
            area=area,
            days=int(self.config.get("days", 1)),
        )
        content, request_url = get_bytes(
            client,
            url,
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [
            RawPayload(
                "firms-viirs-current",
                content,
                request_url.replace(key, "REDACTED"),
                "text/csv",
            )
        ]

    def normalize(
        self, payloads: list[RawPayload], ingested_at: datetime
    ) -> list[EventCollection]:
        if not payloads:
            return []
        frame = pd.read_csv(io.BytesIO(payloads[0].content))
        rows: list[dict] = []
        for row in frame.itertuples(index=False):
            time_text = str(int(getattr(row, "acq_time", 0))).zfill(4)
            valid_time = datetime.strptime(
                f"{row.acq_date} {time_text}", "%Y-%m-%d %H%M"
            ).replace(tzinfo=UTC)
            event_id = (
                f"{row.latitude}-{row.longitude}-{valid_time.isoformat()}-"
                f"{getattr(row, 'satellite', '')}"
            )
            rows.append(
                {
                    "event_id": event_id,
                    "valid_time": valid_time,
                    "available_at": ingested_at,
                    "brightness_temperature_k": getattr(row, "bright_ti4", None),
                    "fire_radiative_power_mw": getattr(row, "frp", None),
                    "confidence": str(getattr(row, "confidence", "")),
                    "satellite": getattr(row, "satellite", None),
                    "daynight": getattr(row, "daynight", None),
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row.longitude), float(row.latitude)],
                    },
                }
            )
        if not rows:
            return []
        valid_time = max(value["valid_time"] for value in rows)
        provenance = ArtifactProvenance(
            self.name,
            self.product,
            self.data_class,
            valid_time,
            ingested_at,
            ingested_at,
            f"firms-{valid_time.isoformat()}",
            payloads[0].request_url,
        )
        return [EventCollection(pd.DataFrame(rows), provenance)]
