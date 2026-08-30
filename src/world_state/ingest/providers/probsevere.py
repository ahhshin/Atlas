from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import httpx
import pandas as pd
from shapely.geometry import box, shape

from world_state.ingest.artifacts import ArtifactProvenance, EventCollection
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.http import get_bytes, get_json_bytes

FILE_PATTERN = re.compile(r"MRMS_PROBSEVERE_(\d{8}_\d{6})\.json")


def _timestamp(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    cleaned = value.replace(" UTC", "")
    try:
        return datetime.strptime(cleaned, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        parsed = pd.Timestamp(value).to_pydatetime()
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _number(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


class ProbSevereProvider(DataSource):
    name = "probsevere"
    product = "mrms-probsevere-v3"
    data_class = DataClass.ANALYZED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        listing, _ = get_bytes(
            client,
            self.config["directory"],
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        matches = sorted(set(FILE_PATTERN.findall(listing.decode(errors="ignore"))))
        if not matches:
            raise RuntimeError("No current ProbSevere JSON file found")
        filename = f"MRMS_PROBSEVERE_{matches[-1]}.json"
        url = f"{self.config['directory'].rstrip('/')}/{filename}"
        content, request_url = get_json_bytes(
            client,
            url,
            retries=int(self.http_config.get("retries", 3)),
            backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
        )
        return [RawPayload(filename, content, request_url, "application/geo+json")]

    def normalize(
        self, payloads: list[RawPayload], ingested_at: datetime
    ) -> list[EventCollection]:
        if not payloads:
            return []
        payload = payloads[0]
        document = json.loads(payload.content)
        valid_time = _timestamp(document.get("validTime"), ingested_at)
        production_time = _timestamp(document.get("productionTime"), ingested_at)
        available_at = production_time
        west, south, east, north = self.config.get("bbox", [-125, 24, -66, 50])
        region = box(west, south, east, north)
        rows: list[dict] = []
        for feature in document.get("features", []):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            parsed_geometry = shape(geometry)
            if parsed_geometry.is_empty or not parsed_geometry.intersects(region):
                continue
            properties = feature.get("properties") or {}
            event_id = str(properties.get("ID") or feature.get("id") or len(rows))
            rows.append(
                {
                    "event_id": event_id,
                    "valid_time": valid_time,
                    "available_at": available_at,
                    "probability_severe": _number(properties.get("ProbSevere")),
                    "probability_hail": _number(properties.get("ProbHail")),
                    "probability_wind": _number(properties.get("ProbWind")),
                    "probability_tornado": _number(properties.get("ProbTor")),
                    "mesh_mm": _number(properties.get("MESH")),
                    "motion_east_mps": _number(properties.get("EastMotion")),
                    "motion_south_mps": _number(properties.get("SouthMotion")),
                    "models_json": json.dumps(feature.get("models") or {}, sort_keys=True),
                    "geometry": geometry,
                }
            )
        if not rows:
            return []
        provenance = ArtifactProvenance(
            source=self.name,
            product=self.product,
            data_class=self.data_class,
            valid_time=valid_time,
            available_at=available_at,
            ingested_at=ingested_at,
            source_id=payload.identifier,
            source_url=payload.request_url,
        )
        return [EventCollection(pd.DataFrame(rows), provenance)]
