from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from world_state.ingest.base import NormalizedPoint, RawPayload
from world_state.paths import POINT_OBSERVATIONS_DIR, RAW_DIR


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
            suffix = ".json" if "json" in payload.media_type else ".bin"
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
    def __init__(self, root: Path = POINT_OBSERVATIONS_DIR):
        self.root = root

    def parquet_files(self) -> list[Path]:
        return sorted(self.root.glob("**/*.parquet")) if self.root.exists() else []

    def append(self, records: list[NormalizedPoint], run_id: str) -> int:
        if not records:
            return 0
        frame = pd.DataFrame(record.as_record() for record in records)
        frame = frame.drop_duplicates("record_id")
        existing_files = self.parquet_files()
        if existing_files:
            existing = duckdb.sql(
                "SELECT record_id FROM read_parquet(?, union_by_name=true)",
                params=[[str(path) for path in existing_files]],
            ).df()
            frame = frame.loc[~frame.record_id.isin(existing.record_id)]
        if frame.empty:
            return 0

        for source, source_frame in frame.groupby("source", sort=True):
            for date, date_frame in source_frame.groupby(
                source_frame.valid_time.dt.date, sort=True
            ):
                target = self.root / f"source={source}" / f"date={date.isoformat()}"
                target.mkdir(parents=True, exist_ok=True)
                path = target / f"part-{run_id}-{uuid4().hex[:8]}.parquet"
                date_frame.to_parquet(path, index=False)
        return len(frame)

    def read(self) -> pd.DataFrame:
        files = self.parquet_files()
        if not files:
            return pd.DataFrame()
        return duckdb.sql(
            "SELECT * FROM read_parquet(?, union_by_name=true)",
            params=[[str(path) for path in files]],
        ).df()
