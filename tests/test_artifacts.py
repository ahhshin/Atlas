from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from world_state.ingest.artifacts import (
    ArtifactProvenance,
    EventCollection,
    ForecastField,
    GridField,
    PointBatch,
)
from world_state.ingest.base import NormalizedPoint, RawPayload
from world_state.ingest.catalog import AtlasCatalog
from world_state.ingest.providers.hrrr import HrrrProvider
from world_state.ingest.providers.metar import MetarProvider
from world_state.ingest.providers.probsevere import ProbSevereProvider
from world_state.ingest.storage import EventStore, GridStore, PointObservationStore, StorageRouter

NOW = datetime(2026, 8, 30, 18, tzinfo=UTC)


def router(tmp_path: Path) -> tuple[StorageRouter, AtlasCatalog]:
    catalog = AtlasCatalog(tmp_path / "atlas.duckdb")
    return (
        StorageRouter(
            catalog=catalog,
            point_store=PointObservationStore(tmp_path / "points", catalog),
            grid_store=GridStore(tmp_path / "grids", tmp_path / "forecasts", catalog),
            event_store=EventStore(tmp_path / "events", catalog),
        ),
        catalog,
    )


def test_storage_router_routes_all_artifact_types_and_catalogs_availability(tmp_path: Path):
    storage, catalog = router(tmp_path)
    provenance = ArtifactProvenance(
        "rtma", "analysis", "analyzed", NOW, NOW, NOW + timedelta(minutes=1), "run"
    )
    dataset = xr.Dataset(
        {"temperature": (("latitude", "longitude"), np.ones((2, 2), dtype=np.float32))},
        coords={"latitude": [40.0, 41.0], "longitude": [-80.0, -79.0]},
    )
    point = NormalizedPoint(
        "metar",
        "bulk",
        "observed",
        NOW,
        NOW,
        NOW,
        40,
        -80,
        "temperature",
        20,
        "°C",
        "KAAA-time",
        station_id="KAAA",
    )
    event = EventCollection(
        pd.DataFrame(
            [{"event_id": "storm-1", "geometry": {"type": "Point", "coordinates": [-80, 40]}}]
        ),
        ArtifactProvenance("probsevere", "v3", "analyzed", NOW, NOW, NOW, "storm-run"),
    )
    forecast_provenance = ArtifactProvenance(
        "hrrr",
        "surface",
        "forecast",
        NOW + timedelta(hours=3),
        NOW,
        NOW,
        "hrrr-run-f03",
        forecast_reference_time=NOW,
        forecast_horizon_hours=3,
    )
    result = storage.write(
        [
            PointBatch((point,)),
            GridField(dataset, provenance, {"temperature": "°C"}, (-80, 40, -79, 41)),
            ForecastField(
                dataset,
                forecast_provenance,
                {"temperature": "°C"},
                (-80, 40, -79, 41),
            ),
            event,
        ],
        "run-1",
    )

    assert result.artifacts_written == 4
    assert len(catalog.table("latest_point_state")) == 1
    assert len(catalog.table("grid_assets")) == 2
    assert len(catalog.table("forecast_runs")) == 1
    event_path = Path(catalog.table("event_assets").iloc[0].geoparquet_path)
    assert b"geo" in (pq.read_metadata(event_path).metadata or {})
    assert len(catalog.assets_available_at(NOW)["points"]) == 1
    assert catalog.assets_available_at(NOW)["grids"].empty
    assert len(catalog.assets_available_at(NOW)["forecasts"]) == 1
    assert len(catalog.assets_available_at(NOW + timedelta(minutes=2))["grids"]) == 1


def test_point_dedup_uses_catalog_without_reading_historical_partitions(
    tmp_path: Path, monkeypatch
):
    catalog = AtlasCatalog(tmp_path / "atlas.duckdb")
    store = PointObservationStore(tmp_path / "points", catalog)
    point = NormalizedPoint(
        "metar", "bulk", "observed", NOW, NOW, NOW, 40, -80, "temperature", 20, "°C", "id"
    )
    assert store.append([point], "first") == 1
    monkeypatch.setattr(store, "parquet_files", lambda: (_ for _ in ()).throw(AssertionError()))
    assert store.append([point], "second") == 0


def test_metar_bulk_cache_is_bounded_and_normalized():
    csv = (
        b"station_id,observation_time,latitude,longitude,temp_c,dewpoint_c,"
        b"wind_dir_degrees,wind_speed_kt,wind_gust_kt,visibility_statute_mi,"
        b"altim_in_hg,sea_level_pressure_mb,precip_in,flight_category\n"
        b"KAAA,2026-08-30T17:55:00Z,40,-80,20,10,180,10,15,8,30.00,,0.10,VFR\n"
        b"ZZZZ,2026-08-30T17:55:00Z,0,0,30,20,90,5,,10,29.90,,,VFR\n"
    )
    artifacts = MetarProvider({"bbox": [-125, 24, -66, 50]}, {}).normalize(
        [RawPayload("metar", gzip.compress(csv), "https://example.test", "application/gzip")],
        NOW,
    )
    records = artifacts[0].records
    assert {record.station_id for record in records} == {"KAAA"}
    by_variable = {record.variable: record for record in records}
    assert by_variable["wind_speed"].value == 5.14444
    assert by_variable["precipitation"].value == 2.54
    assert round(by_variable["pressure"].value, 2) == 1015.92


def test_probsevere_preserves_valid_production_and_ingestion_times():
    document = {
        "type": "FeatureCollection",
        "validTime": "20260830_175500 UTC",
        "productionTime": "20260830_175700 UTC",
        "features": [
            {
                "type": "Feature",
                "properties": {"ID": "1", "ProbSevere": "42", "ProbTor": "3"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-81, 39], [-79, 39], [-79, 41], [-81, 39]]],
                },
            }
        ],
    }
    artifacts = ProbSevereProvider({"bbox": [-125, 24, -66, 50]}, {}).normalize(
        [RawPayload("prob.json", json.dumps(document).encode(), "https://example.test")], NOW
    )
    artifact = artifacts[0]
    assert artifact.provenance.valid_time.minute == 55
    assert artifact.provenance.available_at.minute == 57
    assert artifact.provenance.ingested_at == NOW
    assert artifact.events.iloc[0].probability_severe == 42


def test_hrrr_selects_latest_short_cycle_and_latest_complete_extended_cycle():
    listing = "\n".join(
        [f'hrrr.t21z.wrfsfcf{horizon:02d}.grib2' for horizon in [1, 3, 6, 12]]
        + [f'hrrr.t18z.wrfsfcf{horizon:02d}.grib2' for horizon in [1, 3, 6, 12, 24, 48]]
    )

    def handler(request):
        return httpx.Response(200, text=listing, request=request)

    provider = HrrrProvider(
        {
            "directory": "https://example.test/hrrr.{date}/",
            "horizons": [1, 3, 6, 12, 24, 48],
            "extended_cycles": [0, 6, 12, 18],
        },
        {"retries": 1},
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runs = provider._choose_runs(client, NOW)

    assert runs[0] == ("20260830", "21", [1, 3, 6, 12])
    assert runs[1] == ("20260830", "18", [24, 48])
