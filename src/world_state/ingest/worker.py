from __future__ import annotations

import argparse
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from world_state.config import load_config
from world_state.ingest.cli import configure_logging, fetch_sources
from world_state.ingest.registry import implemented_providers

LOGGER = logging.getLogger(__name__)


def enabled_scheduled_sources(config: dict) -> list[str]:
    return [
        name
        for name in implemented_providers()
        if config["sources"].get(name, {}).get("enabled", False)
        and config["sources"][name].get("scheduled", False)
    ]


def run_scheduler() -> None:
    config = load_config()
    sources = enabled_scheduled_sources(config)
    if not sources:
        raise RuntimeError("No implemented scheduled sources are enabled")
    scheduler = BlockingScheduler(timezone="UTC")
    for source in sources:
        cadence = int(config["sources"][source]["cadence_minutes"])
        scheduler.add_job(
            fetch_sources,
            "interval",
            minutes=cadence,
            args=[[source]],
            id=f"ingest-{source}",
            max_instances=1,
            coalesce=True,
        )
        LOGGER.info("source=%s status=scheduled cadence_minutes=%d", source, cadence)
    fetch_sources(sources)
    scheduler.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="World State scheduled ingestion worker")
    parser.add_argument("--once", action="store_true", help="Fetch all enabled live sources once")
    args = parser.parse_args()
    configure_logging()
    config = load_config()
    sources = enabled_scheduled_sources(config)
    if args.once:
        raise SystemExit(fetch_sources(sources))
    run_scheduler()


if __name__ == "__main__":
    main()
