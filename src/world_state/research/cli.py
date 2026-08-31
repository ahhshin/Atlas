from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_state.research.config import ResearchConfig
from world_state.research.experiments import evaluate_experiment, train_experiment
from world_state.research.pipeline import backfill, validate_dataset
from world_state.research.storage import estimate_storage, inspect_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-research")
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("estimate", help="estimate bounded research storage")
    estimate.add_argument("config", type=Path)
    build = commands.add_parser("backfill", help="build yearly state and leakage-safe targets")
    build.add_argument("config", type=Path)
    build.add_argument("--force", action="store_true")
    validate = commands.add_parser("validate", help="validate a built research dataset")
    validate.add_argument("config", type=Path)
    validate.add_argument(
        "--deep",
        action="store_true",
        help="rescan every immutable missing-data mask instead of using saved diagnostics",
    )
    train = commands.add_parser("train", help="train and evaluate a research baseline")
    train.add_argument("config", type=Path)
    train.add_argument(
        "--model",
        choices=("climatology", "persistence", "simple", "neural"),
        required=True,
    )
    evaluate = commands.add_parser("evaluate", help="show saved experiment metrics")
    evaluate.add_argument("experiment_id")
    evaluate.add_argument("--data-root", type=Path)
    latent_smoke = commands.add_parser(
        "latent-smoke", help="run all latent-world stages on a bounded real-data subset"
    )
    latent_smoke.add_argument("config", type=Path)
    latent_train = commands.add_parser(
        "latent-train", help="train one staged latent-world component"
    )
    latent_train.add_argument("config", type=Path)
    latent_train.add_argument(
        "--stage", choices=("autoencoder", "dynamics", "probe"), required=True
    )
    latent_train.add_argument("--experiment-id")
    latent_evaluate = commands.add_parser(
        "latent-evaluate", help="show saved latent-world experiment metrics"
    )
    latent_evaluate.add_argument("experiment_id")
    latent_evaluate.add_argument("--data-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        data_root = args.data_root or Path("/mnt/games/Atlas/data")
        print(json.dumps(evaluate_experiment(args.experiment_id, data_root), indent=2))
        return
    if args.command in {"latent-smoke", "latent-train", "latent-evaluate"}:
        from world_state.research.latent_config import LatentWorldConfig
        from world_state.research.latent_experiments import (
            evaluate_latent_experiment,
            run_latent_smoke,
            train_latent_stage,
        )

        if args.command == "latent-evaluate":
            data_root = args.data_root or Path("/mnt/games/Atlas/data")
            print(
                json.dumps(
                    evaluate_latent_experiment(args.experiment_id, data_root),
                    indent=2,
                )
            )
            return
        latent_config = LatentWorldConfig.from_yaml(args.config)
        result = (
            run_latent_smoke(latent_config)
            if args.command == "latent-smoke"
            else train_latent_stage(
                latent_config,
                args.stage,
                experiment_id=args.experiment_id,
            )
        )
        print(json.dumps(result, indent=2, default=str, allow_nan=True))
        return
    config = ResearchConfig.from_yaml(args.config)
    if args.command == "estimate":
        estimate = estimate_storage(config)
        status = inspect_storage(config)
        print(f"Resolved storage path: {status.dataset_root}")
        print(f"Mounted filesystem: {status.mount_path}")
        print(f"Available space: {status.free_gb:.2f} GiB")
        print(f"Grid: {estimate.latitude} x {estimate.longitude}")
        print(f"Timestamps: {estimate.timestamps:,}")
        print(f"Variables: {estimate.variables}")
        print(f"Expected compressed size: {estimate.expected_gb:.2f} GiB")
        print(f"Conservative upper bound: {estimate.upper_bound_gb:.2f} GiB")
        print(f"Configured cap: {config.max_storage_gb:.2f} GiB")
    elif args.command == "backfill":
        print(json.dumps(backfill(config, force=args.force), indent=2, default=str))
    elif args.command == "validate":
        print(
            json.dumps(
                validate_dataset(config, deep_missing=args.deep),
                indent=2,
                default=str,
            )
        )
    elif args.command == "train":
        print(json.dumps(train_experiment(config, args.model), indent=2, default=str))


if __name__ == "__main__":
    main()
