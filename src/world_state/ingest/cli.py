from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import httpx

from world_state.config import load_config
from world_state.ingest.base import utc_datetime
from world_state.ingest.catalog import AtlasCatalog
from world_state.ingest.ledger import IngestionLedger
from world_state.ingest.registry import create_provider, implemented_providers
from world_state.ingest.runner import run_provider
from world_state.ingest.storage import RawArchive, StorageRouter


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def fetch_sources(names: list[str]) -> int:
    config = load_config()
    timeout = float(config.get("http", {}).get("timeout_seconds", 20))
    ledger = IngestionLedger()
    archive = RawArchive()
    catalog = AtlasCatalog()
    router = StorageRouter(catalog=catalog)
    outcomes = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for name in names:
            provider = create_provider(name, config)
            outcome = run_provider(
                provider,
                client=client,
                ledger=ledger,
                raw_archive=archive,
                storage_router=router,
                now=datetime.now(UTC),
            )
            outcomes.append(outcome)
            print(
                f"source={outcome.source} status={outcome.status} "
                f"records={outcome.records_written} payloads={outcome.payloads_archived}"
            )
            if outcome.error_message:
                print(f"error={outcome.error_message}")
    return 1 if outcomes and all(outcome.status == "failed" for outcome in outcomes) else 0


def show_health() -> int:
    health = IngestionLedger().source_health(load_config())
    if health.empty:
        print("No sources configured")
    else:
        print(health.to_string(index=False))
    return 0


def show_catalog(table: str) -> int:
    frame = AtlasCatalog().table(table)
    print("No catalog entries" if frame.empty else frame.to_string(index=False))
    return 0


def show_state_at(timestamp: str) -> int:
    moment = utc_datetime(timestamp)
    state = AtlasCatalog().assets_available_at(moment)
    print(f"state_at={moment.isoformat()}")
    for artifact_type, frame in state.items():
        print(f"{artifact_type}={len(frame)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas public-data ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch one provider or all enabled providers"
    )
    fetch_parser.add_argument("source", choices=[*implemented_providers(), "all"])
    subparsers.add_parser("health", help="Show ingestion-ledger health")
    catalog_parser = subparsers.add_parser("catalog", help="Inspect an artifact catalog table")
    catalog_parser.add_argument(
        "table",
        choices=[
            "latest_point_state",
            "point_partitions",
            "grid_assets",
            "forecast_runs",
            "event_assets",
            "source_freshness",
        ],
    )
    state_parser = subparsers.add_parser(
        "state-at", help="Count artifacts that were actually available at an ISO-8601 time"
    )
    state_parser.add_argument("timestamp")
    args = parser.parse_args()
    configure_logging()
    if args.command == "health":
        return show_health()
    if args.command == "catalog":
        return show_catalog(args.table)
    if args.command == "state-at":
        return show_state_at(args.timestamp)
    config = load_config()
    names = (
        [
            name
            for name in implemented_providers()
            if config["sources"].get(name, {}).get("enabled", False)
        ]
        if args.source == "all"
        else [args.source]
    )
    return fetch_sources(names)


if __name__ == "__main__":
    raise SystemExit(main())
