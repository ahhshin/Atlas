from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from world_state.ingest.base import DataClass, DataSource, RawPayload, utc_datetime
from world_state.ingest.http import get_json_bytes
from world_state.ingest.ledger import IngestionLedger
from world_state.ingest.providers.eccc import ECCCProvider
from world_state.ingest.providers.nws import NWSProvider
from world_state.ingest.runner import run_provider
from world_state.ingest.storage import PointObservationStore, RawArchive

FIXTURES = Path(__file__).parent / "fixtures"
INGESTED_AT = datetime(2026, 8, 29, 18, 5, tzinfo=UTC)


def payload(name: str) -> RawPayload:
    return RawPayload(name, (FIXTURES / name).read_bytes(), f"https://example.test/{name}")


def test_utc_normalization_handles_offsets_and_naive_values():
    assert utc_datetime("2026-08-29T14:00:00-04:00").hour == 18
    naive = datetime(2026, 8, 29, 18)  # noqa: DTZ001 - intentionally exercises naive input
    assert utc_datetime(naive).tzinfo == UTC


def test_eccc_normalizes_available_values_and_converts_wind_speed():
    provider = ECCCProvider({}, {})
    records = provider.normalize([payload("eccc_swob.json")], INGESTED_AT)
    by_variable = {record.variable: record for record in records}

    assert set(by_variable) == {
        "temperature",
        "dew_point",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
    }
    assert by_variable["wind_speed"].value == pytest.approx(2.0)
    assert by_variable["wind_speed"].unit == "m/s"
    assert by_variable["temperature"].data_class == "observed"
    assert by_variable["temperature"].available_at.minute == 2
    assert by_variable["temperature"].longitude == -79.63


def test_nws_normalizes_units_and_skips_null_measurements():
    provider = NWSProvider({}, {})
    records = provider.normalize([payload("nws_observation.json")], INGESTED_AT)
    by_variable = {record.variable: record for record in records}

    assert "precipitation" not in by_variable
    assert by_variable["pressure"].value == pytest.approx(1023.7)
    assert by_variable["wind_speed"].value == pytest.approx(2.6)
    assert by_variable["temperature"].quality_flag == "V"
    assert by_variable["temperature"].station_id == "KJFK"


def test_malformed_feature_is_ignored_without_inventing_values():
    provider = ECCCProvider({}, {})
    malformed = RawPayload("bad", b'{"features":[{"geometry":null}]}', "https://example.test")

    assert provider.normalize([malformed], INGESTED_AT) == []


def test_http_retry_recovers_from_timeout():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        content, _ = get_json_bytes(client, "https://example.test", retries=2, backoff_seconds=0)

    assert content == b'{"ok":true}'
    assert calls == 2


def test_duplicate_observations_are_not_appended_twice(tmp_path: Path):
    provider = NWSProvider({}, {})
    records = provider.normalize([payload("nws_observation.json")], INGESTED_AT)
    store = PointObservationStore(tmp_path / "points")

    assert store.append(records, "first") == len(records)
    assert store.append(records, "second") == 0
    assert len(store.read()) == len(records)


def test_raw_archive_retains_request_provenance(tmp_path: Path):
    archive = RawArchive(tmp_path / "raw")
    paths = archive.write("nws", "run-1", [payload("nws_observation.json")], INGESTED_AT)

    assert paths[0].read_bytes() == payload("nws_observation.json").content
    metadata = paths[0].with_suffix(".json.meta.json").read_text()
    assert "https://example.test/nws_observation.json" in metadata


def test_provider_failure_is_recorded_and_does_not_escape_runner(tmp_path: Path):
    class BrokenProvider(DataSource):
        name = "broken"
        product = "test"
        data_class = DataClass.OBSERVED

        def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
            del client, now
            request = httpx.Request("GET", "https://example.test")
            raise httpx.ConnectTimeout("offline", request=request)

        def normalize(self, payloads, ingested_at):
            del payloads, ingested_at
            return []

    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as client:
        ledger = IngestionLedger(tmp_path / "ledger.duckdb")
        outcome = run_provider(
            BrokenProvider({}, {}),
            client=client,
            ledger=ledger,
            raw_archive=RawArchive(tmp_path / "raw"),
            point_store=PointObservationStore(tmp_path / "points"),
            now=INGESTED_AT,
        )

    assert outcome.status == "failed"
    assert ledger.runs().iloc[0].status == "failed"
    assert "offline" in ledger.runs().iloc[0].error_message
