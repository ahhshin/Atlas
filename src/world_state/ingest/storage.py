from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import shape

from world_state.ingest.artifacts import (
    EventCollection,
    ForecastField,
    GridField,
    NormalizedArtifact,
    PointBatch,
)
from world_state.ingest.base import NormalizedPoint, RawPayload
from world_state.ingest.catalog import AtlasCatalog
from world_state.paths import (
    EVENT_COLLECTIONS_DIR,
    FORECAST_FIELDS_DIR,
    GRID_FIELDS_DIR,
    POINT_OBSERVATIONS_DIR,
    RAW_DIR,
)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "payload"


class RawArchive:
    def __init__(self, root: Path = RAW_DIR):
        self.root = root

    def write(
        self, source: str, run_id: str, payloads: list[RawPayload], now: datetime
    ) -> list[Path]:
        target = self.root / source / now.strftime("%Y/%m/%d")
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for index, payload in enumerate(payloads):
            if "json" in payload.media_type:
                suffix = ".json"
            elif "gzip" in payload.media_type:
                suffix = ".gz"
            elif "grib" in payload.media_type:
                suffix = ".grib2"
            else:
                suffix = ".bin"
            filename = f"{run_id}-{index:03d}-{_safe_filename(payload.identifier)}{suffix}"
            path = target / filename
            with path.open("xb") as handle:
                handle.write(payload.content)
            metadata_path = path.with_suffix(f"{path.suffix}.meta.json")
            with metadata_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "identifier": payload.identifier,
                        "request_url": payload.request_url,
                        "media_type": payload.media_type,
                    },
                    handle,
                    indent=2,
                )
            written.append(path)
        return written


class PointObservationStore:
    def __init__(
        self,
        root: Path = POINT_OBSERVATIONS_DIR,
        catalog: AtlasCatalog | None = None,
    ):
        self.root = root
        self.catalog = catalog or (
            AtlasCatalog(root.parent / "atlas.duckdb")
            if root != POINT_OBSERVATIONS_DIR
            else AtlasCatalog()
        )
        self._bootstrap_existing_partitions()

    def _bootstrap_existing_partitions(self) -> None:
        migration = f"point-catalog-v1:{self.root.resolve()}"
        if self.catalog.migration_applied(migration):
            return
        for path in self.parquet_files():
            frame = pd.read_parquet(path)
            if not frame.empty and "record_id" in frame:
                self.catalog.register_point_partition(frame, path)
        self.catalog.mark_migration(migration)

    def parquet_files(self) -> list[Path]:
        return sorted(self.root.glob("**/*.parquet")) if self.root.exists() else []

    def append(self, records: list[NormalizedPoint], run_id: str) -> int:
        if not records:
            return 0
        frame = pd.DataFrame(record.as_record() for record in records)
        frame = frame.drop_duplicates("record_id")
        unseen = self.catalog.unseen_record_ids(frame.record_id)
        frame = frame.loc[frame.record_id.isin(unseen)]
        if frame.empty:
            return 0

        written = 0
        for source, source_frame in frame.groupby("source", sort=True):
            for date, date_frame in source_frame.groupby(
                source_frame.valid_time.dt.date, sort=True
            ):
                target = self.root / f"source={source}" / f"date={date.isoformat()}"
                target.mkdir(parents=True, exist_ok=True)
                path = target / f"part-{run_id}-{uuid4().hex[:8]}.parquet"
                date_frame.to_parquet(path, index=False)
                self.catalog.register_point_partition(date_frame, path)
                written += len(date_frame)
        return written

    def read(self) -> pd.DataFrame:
        files = self.parquet_files()
        if not files:
            return pd.DataFrame()
        return duckdb.sql(
            "SELECT * FROM read_parquet(?, union_by_name=true)",
            params=[[str(path) for path in files]],
        ).df()

    def latest(self) -> pd.DataFrame:
        return self.catalog.table("latest_point_state")


class GridStore:
    def __init__(
        self,
        grid_root: Path = GRID_FIELDS_DIR,
        forecast_root: Path = FORECAST_FIELDS_DIR,
        catalog: AtlasCatalog | None = None,
    ):
        self.grid_root = grid_root
        self.forecast_root = forecast_root
        self.catalog = catalog or (
            AtlasCatalog(grid_root.parent / "atlas.duckdb")
            if grid_root != GRID_FIELDS_DIR
            else AtlasCatalog()
        )

    def write(self, artifact: GridField | ForecastField) -> int:
        p = artifact.provenance
        root = self.forecast_root if isinstance(artifact, ForecastField) else self.grid_root
        target = (
            root
            / f"source={_safe_filename(p.source)}"
            / f"product={_safe_filename(p.product)}"
            / f"date={p.valid_time.date().isoformat()}"
            / f"{artifact.asset_id}.zarr"
        )
        created = not target.exists()
        if created:
            target.parent.mkdir(parents=True, exist_ok=True)
            dataset = artifact.dataset.copy()
            dataset.attrs.update(
                {
                    "source": p.source,
                    "product": p.product,
                    "data_class": p.data_class,
                    "valid_time": p.valid_time.isoformat(),
                    "available_at": p.available_at.isoformat(),
                    "ingested_at": p.ingested_at.isoformat(),
                    "source_id": p.source_id,
                    "source_url": p.source_url or "",
                    "forecast_reference_time": (
                        p.forecast_reference_time.isoformat()
                        if p.forecast_reference_time
                        else ""
                    ),
                    "forecast_horizon_hours": (
                        p.forecast_horizon_hours
                        if p.forecast_horizon_hours is not None
                        else -1
                    ),
                }
            )
            dataset.to_zarr(target, mode="w", consolidated=True, zarr_format=2)
        self.catalog.register_grid(artifact, target)
        return int(created)


class EventStore:
    def __init__(
        self,
        root: Path = EVENT_COLLECTIONS_DIR,
        catalog: AtlasCatalog | None = None,
    ):
        self.root = root
        self.catalog = catalog or (
            AtlasCatalog(root.parent / "atlas.duckdb")
            if root != EVENT_COLLECTIONS_DIR
            else AtlasCatalog()
        )

    def write(self, artifact: EventCollection) -> int:
        p = artifact.provenance
        target = (
            self.root
            / f"source={_safe_filename(p.source)}"
            / f"date={p.valid_time.date().isoformat()}"
            / f"{artifact.asset_id}.parquet"
        )
        bbox: tuple[float, ...] | None = None
        created = not target.exists()
        if created:
            target.parent.mkdir(parents=True, exist_ok=True)
            frame = artifact.events.copy()
            geometries = [shape(value) for value in frame[artifact.geometry_column]]
            if geometries:
                bounds = [geometry.bounds for geometry in geometries]
                bbox = (
                    min(value[0] for value in bounds),
                    min(value[1] for value in bounds),
                    max(value[2] for value in bounds),
                    max(value[3] for value in bounds),
                )
            frame[artifact.geometry_column] = [geometry.wkb for geometry in geometries]
            table = pa.Table.from_pandas(frame, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata[b"geo"] = json.dumps(
                {
                    "version": "1.1.0",
                    "primary_column": artifact.geometry_column,
                    "columns": {
                        artifact.geometry_column: {
                            "encoding": "WKB",
                            "geometry_types": sorted({geometry.geom_type for geometry in geometries}),
                            "crs": artifact.crs,
                            "bbox": list(bbox) if bbox else None,
                        }
                    },
                }
            ).encode()
            pq.write_table(table.replace_schema_metadata(metadata), target)
        self.catalog.register_events(artifact, target, bbox)
        return len(artifact.events) if created else 0


@dataclass(frozen=True)
class StorageResult:
    artifacts_written: int = 0
    records_written: int = 0


class StorageRouter:
    def __init__(
        self,
        catalog: AtlasCatalog | None = None,
        point_store: PointObservationStore | None = None,
        grid_store: GridStore | None = None,
        event_store: EventStore | None = None,
    ):
        self.catalog = catalog or AtlasCatalog()
        self.point_store = point_store or PointObservationStore(catalog=self.catalog)
        self.grid_store = grid_store or GridStore(catalog=self.catalog)
        self.event_store = event_store or EventStore(catalog=self.catalog)

    def write(self, artifacts: list[NormalizedArtifact], run_id: str) -> StorageResult:
        artifact_count = 0
        record_count = 0
        for artifact in artifacts:
            if isinstance(artifact, PointBatch):
                count = self.point_store.append(list(artifact.records), run_id)
                artifact_count += int(count > 0)
                record_count += count
            elif isinstance(artifact, (GridField, ForecastField)):
                artifact_count += self.grid_store.write(artifact)
            elif isinstance(artifact, EventCollection):
                count = self.event_store.write(artifact)
                artifact_count += int(count > 0)
                record_count += count
            else:
                raise TypeError(f"Unsupported normalized artifact: {type(artifact)!r}")
        return StorageResult(artifact_count, record_count)
