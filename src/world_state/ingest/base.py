from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import httpx


class DataClass(StrEnum):
    OBSERVED = "observed"
    ANALYZED = "analyzed"
    FORECAST = "forecast"
    DERIVED = "derived"
    SYNTHETIC = "synthetic"


def utc_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class RawPayload:
    identifier: str
    content: bytes
    request_url: str
    media_type: str = "application/json"


@dataclass(frozen=True)
class NormalizedPoint:
    source: str
    source_product: str
    data_class: str
    valid_time: datetime
    available_at: datetime
    ingested_at: datetime
    latitude: float
    longitude: float
    variable: str
    value: float
    unit: str
    source_id: str
    station_id: str | None = None
    station_name: str | None = None
    quality_flag: str | None = None
    forecast_reference_time: datetime | None = None
    forecast_horizon_hours: int | None = None

    @property
    def record_id(self) -> str:
        identity = (
            f"{self.source}|{self.source_product}|{self.source_id}|{self.variable}|"
            f"{self.valid_time.isoformat()}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["record_id"] = self.record_id
        return record


class DataSource(ABC):
    name: str
    product: str
    data_class: DataClass

    def __init__(self, config: dict[str, Any], http_config: dict[str, Any] | None = None):
        self.config = config
        self.http_config = http_config or {}
        self.fetch_errors: list[str] = []

    @abstractmethod
    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        """Fetch source-native payloads without altering their content."""

    @abstractmethod
    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[NormalizedPoint]:
        """Convert raw payloads to canonical point observations."""
