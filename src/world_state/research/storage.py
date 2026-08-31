from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from world_state.research.config import ResearchConfig

GIB = 1024**3


class StorageGuardError(RuntimeError):
    """Raised before research output could spill onto an unsafe filesystem."""


@dataclass(frozen=True)
class StorageEstimate:
    timestamps: int
    latitude: int
    longitude: int
    variables: int
    raw_bytes: int
    expected_bytes: int
    upper_bound_bytes: int

    @property
    def expected_gb(self) -> float:
        return self.expected_bytes / GIB

    @property
    def upper_bound_gb(self) -> float:
        return self.upper_bound_bytes / GIB


@dataclass(frozen=True)
class StorageStatus:
    data_root: Path
    dataset_root: Path
    mount_path: Path
    free_bytes: int
    current_dataset_bytes: int

    @property
    def free_gb(self) -> float:
        return self.free_bytes / GIB


def estimate_storage(config: ResearchConfig) -> StorageEstimate:
    timestamps = len(config.timestamps)
    latitude, longitude = config.bbox.shape(config.resolution_degrees)
    cells = timestamps * latitude * longitude
    state_values = cells * len(config.variables) * 4
    missing_masks = cells * len(config.variables)
    targets = cells * (4 + 1 + 1)
    static = latitude * longitude * 12
    coordinate_and_metadata = timestamps * 32 + 8 * (latitude + longitude) + 5_000_000
    raw = state_values + missing_masks + targets + static + coordinate_and_metadata
    expected = int(raw * 0.62)
    upper = int(raw * 1.08)
    return StorageEstimate(
        timestamps=timestamps,
        latitude=latitude,
        longitude=longitude,
        variables=len(config.variables),
        raw_bytes=raw,
        expected_bytes=expected,
        upper_bound_bytes=upper,
    )


def resolve_data_root(config: ResearchConfig) -> Path:
    configured = os.environ.get("ATLAS_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else config.storage_root.resolve()


def inspect_storage(config: ResearchConfig, *, create: bool = False) -> StorageStatus:
    data_root = resolve_data_root(config)
    mount_path = _mount_for(data_root)
    if config.required_mount:
        required = config.required_mount.resolve()
        try:
            data_root.relative_to(required)
        except ValueError as error:
            raise StorageGuardError(
                f"ATLAS_DATA_ROOT resolves to {data_root}, outside required SSD mount {required}"
            ) from error
        if not required.is_mount():
            raise StorageGuardError(f"required SSD path is not mounted: {required}")
        mount_path = required
    if mount_path == Path("/") and config.required_mount:
        raise StorageGuardError("research storage would fall back to the system filesystem")
    if create:
        data_root.mkdir(parents=True, exist_ok=True)
        probe = data_root / ".atlas-research-write-test"
        try:
            probe.write_text("atlas", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            raise StorageGuardError(
                f"research storage is not writable: {data_root}: {error}"
            ) from error
    elif not data_root.exists():
        raise StorageGuardError(
            f"configured research storage does not exist: {data_root}; refusing fallback"
        )
    usage = shutil.disk_usage(mount_path)
    dataset_root = data_root / "research" / config.name
    return StorageStatus(
        data_root=data_root,
        dataset_root=dataset_root,
        mount_path=mount_path,
        free_bytes=usage.free,
        current_dataset_bytes=tree_size(dataset_root),
    )


def enforce_preflight(config: ResearchConfig, status: StorageStatus) -> StorageEstimate:
    estimate = estimate_storage(config)
    cap = int(config.max_storage_gb * GIB)
    if estimate.expected_bytes > cap or estimate.upper_bound_bytes > cap:
        raise StorageGuardError(
            f"estimated research dataset upper bound is {estimate.upper_bound_gb:.2f} GiB, "
            f"exceeding the configured {config.max_storage_gb:.2f} GiB cap"
        )
    required = max(0, estimate.upper_bound_bytes - status.current_dataset_bytes)
    reserve = min(2 * GIB, int(cap * 0.1))
    if status.free_bytes < required + reserve:
        raise StorageGuardError(
            f"insufficient space on {status.mount_path}: {status.free_gb:.2f} GiB free; "
            f"need about {(required + reserve) / GIB:.2f} GiB"
        )
    return estimate


def enforce_cap(config: ResearchConfig, dataset_root: Path) -> int:
    size = tree_size(dataset_root)
    cap = int(config.max_storage_gb * GIB)
    if size > cap:
        raise StorageGuardError(
            f"dataset reached {size / GIB:.2f} GiB and exceeded the "
            f"{config.max_storage_gb:.2f} GiB cap; backfill stopped"
        )
    return size


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _mount_for(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    while candidate != candidate.parent and not candidate.is_mount():
        candidate = candidate.parent
    return candidate
