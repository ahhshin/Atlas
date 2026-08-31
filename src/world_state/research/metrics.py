from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    climatology: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype="uint8")
    probabilities = np.clip(np.asarray(probabilities, dtype="float64"), 1e-7, 1 - 1e-7)
    climatology = np.clip(np.asarray(climatology, dtype="float64"), 1e-7, 1 - 1e-7)
    brier = float(np.mean((probabilities - labels) ** 2))
    climatology_brier = float(np.mean((climatology - labels) ** 2))
    positive = labels == 1
    metrics = {
        "brier_score": brier,
        "brier_skill_score": (
            1 - brier / climatology_brier if climatology_brier > 0 else float("nan")
        ),
        "climatology_brier_score": climatology_brier,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "precision": float(precision_score(labels, probabilities >= 0.5, zero_division=0)),
        "recall": float(recall_score(labels, probabilities >= 0.5, zero_division=0)),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if np.any(positive)
            else float("nan")
        ),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if np.any(positive) and np.any(~positive)
            else float("nan")
        ),
        "positive_event_prevalence": float(np.mean(positive)),
        "positive_event_brier": (
            float(np.mean((probabilities[positive] - 1) ** 2)) if np.any(positive) else float("nan")
        ),
        "positive_event_log_loss": (
            float(np.mean(-np.log(probabilities[positive]))) if np.any(positive) else float("nan")
        ),
        "samples": len(labels),
        "positive_events": int(np.sum(positive)),
    }
    return metrics


def calibration_table(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    frame = pd.DataFrame({"label": labels, "probability": probabilities})
    frame["bin"] = pd.cut(
        frame.probability, edges, include_lowest=True, labels=False, duplicates="drop"
    )
    grouped = frame.groupby("bin", observed=False)
    result = grouped.agg(
        mean_probability=("probability", "mean"),
        observed_frequency=("label", "mean"),
        samples=("label", "size"),
    ).reset_index()
    result["bin_lower"] = result.bin.map(lambda value: edges[int(value)])
    result["bin_upper"] = result.bin.map(lambda value: edges[int(value) + 1])
    return result


def spatial_brier_skill(
    labels: np.ndarray,
    probabilities: np.ndarray,
    climatology: np.ndarray,
    flat_indices: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.bincount(flat_indices, minlength=np.prod(shape)).astype("float64")
    model_sum = np.bincount(
        flat_indices, weights=(probabilities - labels) ** 2, minlength=np.prod(shape)
    )
    climate_sum = np.bincount(
        flat_indices, weights=(climatology - labels) ** 2, minlength=np.prod(shape)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        model_brier = model_sum / count
        climate_brier = climate_sum / count
        skill = 1 - model_brier / climate_brier
    model_brier[count == 0] = np.nan
    skill[(count == 0) | (climate_brier == 0)] = np.nan
    return model_brier.reshape(shape), skill.reshape(shape), count.reshape(shape)
