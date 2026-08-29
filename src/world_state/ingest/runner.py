from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from world_state.ingest.base import DataSource
from world_state.ingest.ledger import IngestionLedger
from world_state.ingest.storage import PointObservationStore, RawArchive

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
    point_store: PointObservationStore,
    now: datetime | None = None,
) -> IngestionOutcome:
    started_at = now or datetime.now(UTC)
    run_id = ledger.start(provider.name, provider.product, started_at)
    payloads = []
    try:
        payloads = provider.fetch(client, started_at)
        raw_archive.write(provider.name, run_id, payloads, started_at)
        records = provider.normalize(payloads, started_at)
        if not records:
            raise ValueError("Provider returned no usable normalized observations")
        written = point_store.append(records, run_id)
        latest = max(record.valid_time for record in records)
        status = "partial" if provider.fetch_errors else "success"
        error_message = "; ".join(provider.fetch_errors) or None
        completed_at = datetime.now(UTC)
        ledger.finish(
            run_id,
            status=status,
            completed_at=completed_at,
            latest_source_timestamp=latest,
            records_or_objects=len(records),
            bytes_downloaded=sum(len(payload.content) for payload in payloads),
            error_message=error_message,
        )
        LOGGER.info(
            "source=%s product=%s status=%s records=%d payloads=%d bytes=%d",
            provider.name,
            provider.product,
            status,
            written,
            len(payloads),
            sum(len(payload.content) for payload in payloads),
        )
        return IngestionOutcome(
            run_id,
            provider.name,
            status,
            written,
            len(payloads),
            latest,
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
        LOGGER.error(
            "source=%s product=%s status=failed error=%s", provider.name, provider.product, error
        )
        return IngestionOutcome(run_id, provider.name, "failed", 0, len(payloads), None, str(error))
