from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from world_state.paths import INGESTION_LEDGER_PATH


class IngestionLedger:
    def __init__(self, path: Path = INGESTION_LEDGER_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(str(self.path))
        connection.execute("SET TimeZone='UTC'")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    product VARCHAR NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    latest_source_timestamp TIMESTAMPTZ,
                    records_or_objects BIGINT DEFAULT 0,
                    bytes_downloaded BIGINT DEFAULT 0,
                    status VARCHAR NOT NULL,
                    error_message VARCHAR
                )
                """
            )

    def start(self, source: str, product: str, started_at: datetime | None = None) -> str:
        run_id = uuid4().hex
        started_at = started_at or datetime.now(UTC)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_runs VALUES (?, ?, ?, ?, NULL, NULL, 0, 0, 'running', NULL)",
                [run_id, source, product, started_at],
            )
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        completed_at: datetime,
        latest_source_timestamp: datetime | None = None,
        records_or_objects: int = 0,
        bytes_downloaded: int = 0,
        error_message: str | None = None,
    ) -> None:
        if status not in {"success", "partial", "failed"}:
            raise ValueError(f"Invalid ingestion status: {status}")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs
                SET completed_at = ?, latest_source_timestamp = ?, records_or_objects = ?,
                    bytes_downloaded = ?, status = ?, error_message = ?
                WHERE run_id = ?
                """,
                [
                    completed_at,
                    latest_source_timestamp,
                    records_or_objects,
                    bytes_downloaded,
                    status,
                    error_message,
                    run_id,
                ],
            )

    def runs(self) -> pd.DataFrame:
        with self._connect() as connection:
            return connection.sql("SELECT * FROM ingestion_runs ORDER BY started_at DESC").df()

    def source_health(self, config: dict) -> pd.DataFrame:
        runs = self.runs()
        rows: list[dict] = []
        now = pd.Timestamp.now(tz="UTC")
        for source, source_settings in config.get("sources", {}).items():
            source_runs = runs.loc[runs.source == source] if not runs.empty else pd.DataFrame()
            latest = source_runs.iloc[0] if not source_runs.empty else None
            successes = (
                source_runs.loc[source_runs.status == "success"]
                if not source_runs.empty
                else pd.DataFrame()
            )
            latest_success = successes.iloc[0] if not successes.empty else None
            if not source_settings.get("enabled", False):
                display_status = "disabled"
            elif latest is None:
                display_status = "unavailable"
            elif latest["status"] == "failed":
                display_status = "failed"
            elif latest["status"] == "partial":
                display_status = "partial"
            elif latest_success is None or pd.isna(latest_success["latest_source_timestamp"]):
                display_status = "unavailable"
            else:
                cadence = max(int(source_settings.get("cadence_minutes", 60)), 1)
                age_minutes = (
                    now - pd.Timestamp(latest_success["latest_source_timestamp"])
                ).total_seconds() / 60
                display_status = "healthy" if age_minutes <= max(cadence * 3, 60) else "stale"
            rows.append(
                {
                    "source": source,
                    "status": display_status,
                    "last_attempt": None if latest is None else latest["started_at"],
                    "last_success": None
                    if latest_success is None
                    else latest_success["completed_at"],
                    "data_current_through": (
                        None
                        if latest_success is None
                        else latest_success["latest_source_timestamp"]
                    ),
                    "records_or_objects": (
                        0 if latest is None else int(latest["records_or_objects"])
                    ),
                    "error": None if latest is None else latest["error_message"],
                }
            )
        return pd.DataFrame(rows)
