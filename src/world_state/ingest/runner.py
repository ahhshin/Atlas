from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from world_state.ingest.artifacts import EventCollection, ForecastField, GridField, PointBatch
from world_state.ingest.base import DataSource
from world_state.ingest.ledger import IngestionLedger
from world_state.ingest.storage import RawArchive, StorageRouter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionOutcome:
    run_id: str
    source: str
    status: str
    records_written: int
    payloads_archived: int
    latest_source_timestamp: datetime | None
    error_message: str | None = None


def run_provider(
    provider: DataSource,
    *,
    client: httpx.Client,
    ledger: IngestionLedger,
    raw_archive: RawArchive,
    storage_router: StorageRouter,
    now: datetime | None = None,
) -> IngestionOutcome:
    started_at = now or datetime.now(UTC)
    run_id = ledger.start(provider.name, provider.product, started_at)
    payloads = []
    try:
        payloads = provider.fetch(client, started_at)
        raw_archive.write(provider.name, run_id, payloads, started_at)
        artifacts = provider.normalize(payloads, started_at)
        if not artifacts:
            raise ValueError("Provider returned no usable normalized artifacts")
        result = storage_router.write(artifacts, run_id)
        valid_times = []
        source_times = []
        available_times = []
        normalized_objects = 0
        for artifact in artifacts:
            if isinstance(artifact, PointBatch):
                valid_times.extend(record.valid_time for record in artifact.records)
                source_times.extend(record.valid_time for record in artifact.records)
                available_times.extend(record.available_at for record in artifact.records)
                normalized_objects += len(artifact.records)
            elif isinstance(artifact, (GridField, EventCollection)):
                valid_times.append(artifact.provenance.valid_time)
                source_times.append(
                    artifact.provenance.forecast_reference_time
                    if isinstance(artifact, ForecastField)
                    else artifact.provenance.valid_time
                )
                available_times.append(artifact.provenance.available_at)
                normalized_objects += 1
        latest_valid = max(valid_times)
        latest_source = max(source_times)
        latest_available = max(available_times)
        status = "partial" if provider.fetch_errors else "success"
        error_message = "; ".join(provider.fetch_errors) or None
        completed_at = datetime.now(UTC)
        ledger.finish(
            run_id,
            status=status,
            completed_at=completed_at,
            latest_source_timestamp=latest_source,
            records_or_objects=normalized_objects,
            bytes_downloaded=sum(len(payload.content) for payload in payloads),
            error_message=error_message,
        )
        storage_router.catalog.update_freshness(
            provider.name,
            provider.product,
            completed_at,
            status,
            latest_valid,
            latest_available,
            error_message,
        )
        LOGGER.info(
            "source=%s product=%s status=%s records=%d payloads=%d bytes=%d",
            provider.name,
            provider.product,
            status,
            result.records_written or result.artifacts_written,
            len(payloads),
            sum(len(payload.content) for payload in payloads),
        )
        return IngestionOutcome(
            run_id,
            provider.name,
            status,
            result.records_written or result.artifacts_written,
            len(payloads),
            latest_source,
            error_message,
        )
    # A worker must isolate arbitrary provider/parser/storage failures from other sources.
    except Exception as error:  # noqa: BLE001
        completed_at = datetime.now(UTC)
        ledger.finish(
            run_id,
            status="failed",
            completed_at=completed_at,
            records_or_objects=0,
            bytes_downloaded=sum(len(payload.content) for payload in payloads),
            error_message=str(error),
        )
        storage_router.catalog.update_freshness(
            provider.name,
            provider.product,
            completed_at,
            "failed",
            None,
            None,
            str(error),
        )
        LOGGER.error(
            "source=%s product=%s status=failed error=%s", provider.name, provider.product, error
        )
        return IngestionOutcome(run_id, provider.name, "failed", 0, len(payloads), None, str(error))
