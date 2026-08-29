from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import httpx

from world_state.config import load_config
from world_state.ingest.ledger import IngestionLedger
from world_state.ingest.registry import create_provider, implemented_providers
from world_state.ingest.runner import run_provider
from world_state.ingest.storage import PointObservationStore, RawArchive


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def fetch_sources(names: list[str]) -> int:
    config = load_config()
    timeout = float(config.get("http", {}).get("timeout_seconds", 20))
    ledger = IngestionLedger()
    archive = RawArchive()
    store = PointObservationStore()
    outcomes = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for name in names:
            provider = create_provider(name, config)
            outcome = run_provider(
                provider,
                client=client,
                ledger=ledger,
                raw_archive=archive,
                point_store=store,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="World State public-data ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser(
        "fetch", help="Fetch one provider or all enabled providers"
    )
    fetch_parser.add_argument("source", choices=[*implemented_providers(), "all"])
    subparsers.add_parser("health", help="Show ingestion-ledger health")
    args = parser.parse_args()
    configure_logging()
    if args.command == "health":
        return show_health()
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
