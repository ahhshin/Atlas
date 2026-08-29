from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from world_state.paths import SOURCES_CONFIG_PATH


def load_config(path: Path = SOURCES_CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    contact = os.getenv("WORLD_STATE_CONTACT")
    if contact:
        config.setdefault("http", {})["user_agent"] = (
            f"world-state-personal-research/0.1 ({contact})"
        )
    return config


def source_config(name: str, config: dict[str, Any]) -> dict[str, Any]:
    try:
        return config["sources"][name]
    except KeyError as error:
        raise KeyError(f"Unknown source: {name}") from error
