from __future__ import annotations

from typing import Any

import numpy as np


def physical_state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    missing_mask: np.ndarray,
    standard_deviations: np.ndarray,
    channels: tuple[str, ...],
) -> dict[str, Any]:
    if prediction.shape != target.shape or target.shape != missing_mask.shape:
        raise ValueError("metric arrays must have identical shapes")
    error = prediction - target
    valid = ~missing_mask.astype(bool)
    rows: dict[str, dict[str, float]] = {}
    for channel_index, name in enumerate(channels):
        selected = valid[:, channel_index]
        channel_error = error[:, channel_index][selected]
        normalized_rmse = _rmse(channel_error)
        normalized_mae = _mae(channel_error)
        scale = float(standard_deviations[channel_index])
        rows[name] = {
            "normalized_rmse": normalized_rmse,
            "normalized_mae": normalized_mae,
            "physical_rmse": normalized_rmse * scale,
            "physical_mae": normalized_mae * scale,
            "valid_cells": int(selected.sum()),
        }
    all_error = error[valid]
    return {
        "overall_normalized_rmse": _rmse(all_error),
        "overall_normalized_mae": _mae(all_error),
        "per_variable": rows,
    }


def compare_with_persistence(
    model: dict[str, Any], persistence: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "overall_normalized_rmse_improvement_percent": _improvement(
            persistence["overall_normalized_rmse"], model["overall_normalized_rmse"]
        ),
        "overall_normalized_mae_improvement_percent": _improvement(
            persistence["overall_normalized_mae"], model["overall_normalized_mae"]
        ),
        "per_variable": {},
    }
    for name, values in model["per_variable"].items():
        baseline = persistence["per_variable"][name]
        comparison["per_variable"][name] = {
            "physical_rmse_improvement_percent": _improvement(
                baseline["physical_rmse"], values["physical_rmse"]
            ),
            "physical_mae_improvement_percent": _improvement(
                baseline["physical_mae"], values["physical_mae"]
            ),
        }
    return comparison


def compression_ratio(
    physical_channels: int,
    height: int,
    width: int,
    latent_dimensions: int,
    latent_height: int,
    latent_width: int,
) -> float:
    physical = physical_channels * height * width
    latent = latent_dimensions * latent_height * latent_width
    return float(physical / latent)


def latent_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction.astype("float64") - target.astype("float64"))))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values.astype("float64"))))) if values.size else float("nan")


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values.astype("float64")))) if values.size else float("nan")


def _improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return float((baseline - candidate) / baseline * 100)
