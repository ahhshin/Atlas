from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from world_state.ingest.artifacts import EventCollection, GridField
from world_state.paths import INGESTION_LEDGER_PATH


class AtlasCatalog:
    """Small DuckDB catalog for immutable assets and fast current-state lookups."""

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
                CREATE TABLE IF NOT EXISTS point_record_ids (
                    record_id VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    parquet_path VARCHAR NOT NULL
                );
                CREATE TABLE IF NOT EXISTS latest_point_state (
                    source VARCHAR NOT NULL,
                    source_product VARCHAR NOT NULL,
                    data_class VARCHAR NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL,
                    latitude DOUBLE NOT NULL,
                    longitude DOUBLE NOT NULL,
                    variable VARCHAR NOT NULL,
                    value DOUBLE NOT NULL,
                    unit VARCHAR NOT NULL,
                    source_id VARCHAR NOT NULL,
                    station_id VARCHAR NOT NULL,
                    station_name VARCHAR,
                    quality_flag VARCHAR,
                    forecast_reference_time TIMESTAMPTZ,
                    forecast_horizon_hours INTEGER,
                    record_id VARCHAR NOT NULL,
                    PRIMARY KEY (source, station_id, variable)
                );
                CREATE TABLE IF NOT EXISTS point_partitions (
                    parquet_path VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    valid_date DATE NOT NULL,
                    minimum_valid_time TIMESTAMPTZ NOT NULL,
                    maximum_valid_time TIMESTAMPTZ NOT NULL,
                    minimum_available_at TIMESTAMPTZ NOT NULL,
                    maximum_available_at TIMESTAMPTZ NOT NULL,
                    record_count BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS grid_assets (
                    asset_id VARCHAR PRIMARY KEY,
                    artifact_type VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    product VARCHAR NOT NULL,
                    data_class VARCHAR NOT NULL,
                    variables_json VARCHAR NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL,
                    forecast_reference_time TIMESTAMPTZ,
                    forecast_horizon_hours INTEGER,
                    bbox_json VARCHAR NOT NULL,
                    native_resolution VARCHAR,
                    zarr_path VARCHAR NOT NULL,
                    source_id VARCHAR NOT NULL,
                    source_url VARCHAR
                );
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    asset_id VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    product VARCHAR NOT NULL,
                    forecast_reference_time TIMESTAMPTZ NOT NULL,
                    forecast_horizon_hours INTEGER NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    zarr_path VARCHAR NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_assets (
                    asset_id VARCHAR PRIMARY KEY,
                    source VARCHAR NOT NULL,
                    product VARCHAR NOT NULL,
                    data_class VARCHAR NOT NULL,
                    valid_time TIMESTAMPTZ NOT NULL,
                    available_at TIMESTAMPTZ NOT NULL,
                    ingested_at TIMESTAMPTZ NOT NULL,
                    event_count BIGINT NOT NULL,
                    bbox_json VARCHAR,
                    geoparquet_path VARCHAR NOT NULL,
                    source_id VARCHAR NOT NULL,
                    source_url VARCHAR
                );
                CREATE TABLE IF NOT EXISTS source_freshness (
                    source VARCHAR PRIMARY KEY,
                    product VARCHAR NOT NULL,
                    latest_valid_time TIMESTAMPTZ,
                    latest_available_at TIMESTAMPTZ,
                    last_ingested_at TIMESTAMPTZ NOT NULL,
                    status VARCHAR NOT NULL,
                    detail VARCHAR
                );
                CREATE TABLE IF NOT EXISTS catalog_migrations (
                    migration_name VARCHAR PRIMARY KEY,
                    completed_at TIMESTAMPTZ NOT NULL
                );
                CREATE INDEX IF NOT EXISTS point_record_source_time
                    ON point_record_ids(source, valid_time);
                CREATE INDEX IF NOT EXISTS grid_source_time
                    ON grid_assets(source, valid_time);
                CREATE INDEX IF NOT EXISTS forecast_cycle_horizon
                    ON forecast_runs(source, forecast_reference_time, forecast_horizon_hours);
                CREATE INDEX IF NOT EXISTS event_source_time
                    ON event_assets(source, valid_time);
                """
            )
            connection.execute("ALTER TABLE latest_point_state ALTER latitude DROP NOT NULL")
            connection.execute("ALTER TABLE latest_point_state ALTER longitude DROP NOT NULL")

    def migration_applied(self, name: str) -> bool:
        with self._connect() as connection:
            return bool(
                connection.sql(
                    "SELECT COUNT(*) FROM catalog_migrations WHERE migration_name = ?",
                    params=[name],
                ).fetchone()[0]
            )

    def mark_migration(self, name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO catalog_migrations VALUES (?, now()) ON CONFLICT DO NOTHING",
                [name],
            )

    def unseen_record_ids(self, record_ids: pd.Series) -> set[str]:
        unique = pd.DataFrame({"record_id": record_ids.drop_duplicates().astype(str)})
        if unique.empty:
            return set()
        with self._connect() as connection:
            connection.register("candidate_record_ids", unique)
            existing = connection.sql(
                """
                SELECT candidate.record_id
                FROM candidate_record_ids candidate
                JOIN point_record_ids known USING (record_id)
                """
            ).df()
        return set(unique.record_id) - set(existing.record_id)

    def register_point_partition(self, frame: pd.DataFrame, path: Path) -> None:
        registered = frame.copy()
        registered["parquet_path"] = str(path)
        registered["station_id"] = registered.station_id.fillna(registered.source_id)
        with self._connect() as connection:
            connection.register("new_point_records", registered)
            connection.execute("BEGIN")
            connection.execute(
                """
                INSERT INTO point_record_ids
                SELECT record_id, source, valid_time, available_at, parquet_path
                FROM new_point_records
                ON CONFLICT DO NOTHING
                """
            )
            connection.execute(
                """
                INSERT INTO latest_point_state BY NAME
                SELECT * EXCLUDE (parquet_path) FROM new_point_records
                ON CONFLICT (source, station_id, variable) DO UPDATE SET
                    source_product = excluded.source_product,
                    data_class = excluded.data_class,
                    valid_time = excluded.valid_time,
                    available_at = excluded.available_at,
                    ingested_at = excluded.ingested_at,
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    value = excluded.value,
                    unit = excluded.unit,
                    source_id = excluded.source_id,
                    station_name = excluded.station_name,
                    quality_flag = excluded.quality_flag,
                    forecast_reference_time = excluded.forecast_reference_time,
                    forecast_horizon_hours = excluded.forecast_horizon_hours,
                    record_id = excluded.record_id
                WHERE excluded.valid_time >= latest_point_state.valid_time
                """
            )
            connection.execute(
                """
                INSERT INTO point_partitions
                SELECT ?, source, CAST(valid_time AS DATE), MIN(valid_time), MAX(valid_time),
                       MIN(available_at), MAX(available_at), COUNT(*)
                FROM new_point_records GROUP BY source, CAST(valid_time AS DATE)
                ON CONFLICT DO NOTHING
                """,
                [str(path)],
            )
            connection.execute("COMMIT")

    def register_grid(self, artifact: GridField, path: Path) -> None:
        p = artifact.provenance
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO grid_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    artifact.asset_id,
                    artifact.artifact_type,
                    p.source,
                    p.product,
                    p.data_class,
                    json.dumps(artifact.variables, sort_keys=True),
                    p.valid_time,
                    p.available_at,
                    p.ingested_at,
                    p.forecast_reference_time,
                    p.forecast_horizon_hours,
                    json.dumps(artifact.bbox),
                    artifact.native_resolution,
                    str(path),
                    p.source_id,
                    p.source_url,
                ],
            )
            if artifact.artifact_type == "forecast_field":
                connection.execute(
                    """
                    INSERT INTO forecast_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        artifact.asset_id,
                        p.source,
                        p.product,
                        p.forecast_reference_time,
                        p.forecast_horizon_hours,
                        p.valid_time,
                        p.available_at,
                        str(path),
                    ],
                )

    def register_events(
        self, artifact: EventCollection, path: Path, bbox: tuple[float, ...] | None
    ) -> None:
        p = artifact.provenance
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO event_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                [
                    artifact.asset_id,
                    p.source,
                    p.product,
                    p.data_class,
                    p.valid_time,
                    p.available_at,
                    p.ingested_at,
                    len(artifact.events),
                    json.dumps(bbox) if bbox else None,
                    str(path),
                    p.source_id,
                    p.source_url,
                ],
            )

    def update_freshness(
        self,
        source: str,
        product: str,
        ingested_at: datetime,
        status: str,
        latest_valid_time: datetime | None,
        latest_available_at: datetime | None,
        detail: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_freshness VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source) DO UPDATE SET
                    product = excluded.product,
                    latest_valid_time = COALESCE(excluded.latest_valid_time,
                                                 source_freshness.latest_valid_time),
                    latest_available_at = COALESCE(excluded.latest_available_at,
                                                   source_freshness.latest_available_at),
                    last_ingested_at = excluded.last_ingested_at,
                    status = excluded.status,
                    detail = excluded.detail
                """,
                [
                    source,
                    product,
                    latest_valid_time,
                    latest_available_at,
                    ingested_at,
                    status,
                    detail,
                ],
            )

    def table(self, name: str) -> pd.DataFrame:
        allowed = {
            "latest_point_state",
            "point_partitions",
            "grid_assets",
            "forecast_runs",
            "event_assets",
            "source_freshness",
        }
        if name not in allowed:
            raise ValueError(f"Unknown catalog table: {name}")
        with self._connect() as connection:
            return connection.sql(f"SELECT * FROM {name}").df()

    def assets_available_at(self, timestamp: datetime) -> dict[str, pd.DataFrame]:
        with self._connect() as connection:
            partitions = connection.sql(
                """
                SELECT parquet_path FROM point_partitions
                WHERE minimum_available_at <= ? AND minimum_valid_time <= ?
                """,
                params=[timestamp, timestamp],
            ).df()
            if partitions.empty:
                points = pd.DataFrame()
            else:
                points = connection.sql(
                    """
                    SELECT * FROM read_parquet(?, union_by_name=true)
                    WHERE available_at <= ? AND ingested_at <= ? AND valid_time <= ?
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY source, COALESCE(station_id, source_id), variable
                        ORDER BY valid_time DESC, available_at DESC
                    ) = 1
                    """,
                    params=[partitions.parquet_path.tolist(), timestamp, timestamp, timestamp],
                ).df()
            return {
                "points": points,
                "grids": connection.sql(
                    """
                    SELECT * FROM grid_assets
                    WHERE artifact_type = 'grid_field'
                      AND available_at <= ? AND ingested_at <= ? ORDER BY valid_time DESC
                    """,
                    params=[timestamp, timestamp],
                ).df(),
                "events": connection.sql(
                    """
                    SELECT * FROM event_assets
                    WHERE available_at <= ? AND ingested_at <= ? ORDER BY valid_time DESC
                    """,
                    params=[timestamp, timestamp],
                ).df(),
                "forecasts": connection.sql(
                    """
                    SELECT f.* FROM forecast_runs f
                    JOIN grid_assets g USING (asset_id)
                    WHERE f.available_at <= ? AND g.ingested_at <= ?
                    ORDER BY f.forecast_reference_time DESC, f.forecast_horizon_hours
                    """,
                    params=[timestamp, timestamp],
                ).df(),
            }
