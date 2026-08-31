from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.store import load_metrics
from world_state.synthetic import SYNTHETIC_VARIABLES

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("Experiments")
mode = st.radio(
    "Experiment family",
    ("Historical · Atlas Mini", "Synthetic lab"),
    horizontal=True,
    label_visibility="collapsed",
)


def render_synthetic() -> None:
    st.caption("Synthetic baseline evaluation · lower RMSE is better")
    metrics = load_metrics()
    target = st.selectbox(
        "Target",
        list(SYNTHETIC_VARIABLES),
        format_func=lambda name: SYNTHETIC_VARIABLES[name]["label"],
    )
    filtered = metrics.loc[metrics.target == target].copy()
    figure = px.line(
        filtered,
        x="forecast_horizon_hours",
        y="rmse",
        color="model",
        markers=True,
        labels={
            "forecast_horizon_hours": "Forecast horizon (hours)",
            "rmse": f"RMSE ({SYNTHETIC_VARIABLES[target]['unit']})",
        },
    )
    figure.update_layout(height=430)
    st.plotly_chart(figure, width="stretch")
    st.dataframe(
        filtered[["model", "forecast_horizon_hours", "mae", "rmse", "bias", "samples"]],
        hide_index=True,
        width="stretch",
    )


@st.cache_data(show_spinner=False)
def research_datasets(data_root: str) -> list[str]:
    research = Path(data_root) / "research"
    if not research.exists():
        return []
    return sorted(
        [
            path.name
            for path in research.iterdir()
            if path.is_dir() and (path / "metadata" / "diagnostics.json").exists()
        ],
        reverse=True,
    )


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def open_zarr(path: str) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def render_historical() -> None:
    data_root = Path(os.environ.get("ATLAS_DATA_ROOT", "/mnt/games/Atlas/data"))
    datasets = research_datasets(str(data_root))
    st.caption(
        "Atlas Mini · Extreme Precipitation · retrospective ERA5 reanalysis, not an operational feed"
    )
    if not datasets:
        st.info(
            "No historical research dataset is available. Run `atlas-research backfill "
            "configs/research/precip_smoke.yaml` first."
        )
        return
    selected = st.selectbox("Dataset", datasets)
    root = data_root / "research" / selected
    diagnostics = read_json(str(root / "metadata" / "diagnostics.json"))
    render_diagnostics(diagnostics)
    index_path = root / "experiments" / "experiments.parquet"
    if not index_path.exists():
        st.info("The dataset is ready; train a baseline to populate historical experiments.")
        return
    experiments = read_parquet(str(index_path)).sort_values("timestamp")
    st.subheader("Model comparison")
    comparison = experiments[
        [
            "model",
            "brier_score",
            "brier_skill_score",
            "pr_auc",
            "roc_auc",
            "test_event_prevalence",
            "experiment_id",
        ]
    ]
    st.dataframe(comparison, hide_index=True, width="stretch")
    metric = st.selectbox(
        "Comparison metric", ("brier_score", "brier_skill_score", "pr_auc", "roc_auc")
    )
    chart = px.bar(
        experiments,
        x="model",
        y=metric,
        color="model",
        hover_data=["experiment_id"],
    )
    chart.update_layout(height=350, showlegend=False)
    st.plotly_chart(chart, width="stretch")
    labels = {
        row.experiment_id: f"{row.model} · {row.experiment_id}" for row in experiments.itertuples()
    }
    experiment_id = st.selectbox(
        "Experiment",
        experiments.experiment_id.tolist(),
        index=len(experiments) - 1,
        format_func=lambda value: labels[value],
    )
    experiment_root = root / "experiments" / experiment_id
    left, right = st.columns(2)
    with left:
        render_calibration(experiment_root)
    with right:
        render_timeline(experiment_root)
    render_spatial_skill(experiment_root)
    render_replay(experiment_root)


def render_diagnostics(diagnostics: dict[str, object]) -> None:
    st.subheader("Dataset health")
    columns = st.columns(5)
    columns[0].metric("Years", len(diagnostics["years"]))
    columns[1].metric("Timestamps", f"{diagnostics['timestamps']:,}")
    columns[2].metric("Grid", f"{diagnostics['latitude']} × {diagnostics['longitude']}")
    columns[3].metric("Disk", f"{diagnostics['disk_gb']:.2f} GiB")
    columns[4].metric("Test prevalence", f"{diagnostics['test_event_prevalence']:.2%}")
    with st.expander("Variables, missingness, and leakage controls"):
        missing = pd.DataFrame(
            {
                "variable": diagnostics["variables"],
                "missing_percent": [
                    diagnostics["missing_percent"][name] for name in diagnostics["variables"]
                ],
            }
        )
        first, second = st.columns(2)
        first.dataframe(missing, hide_index=True, width="stretch")
        second.json(
            {
                "split_samples": diagnostics["split_samples"],
                "threshold_percentile": diagnostics["threshold_percentile"],
                "threshold_fit_split": diagnostics["threshold_fit_split"],
                "data_class": diagnostics["data_class"],
            }
        )


def render_calibration(experiment_root: Path) -> None:
    st.markdown("#### Calibration")
    calibration = read_parquet(str(experiment_root / "calibration.parquet"))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=calibration.mean_probability,
            y=calibration.observed_frequency,
            mode="lines+markers",
            name="Model",
        )
    )
    figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dot"})
    )
    figure.update_layout(
        height=340,
        xaxis_title="Predicted probability",
        yaxis_title="Observed frequency",
    )
    st.plotly_chart(figure, width="stretch")


def render_timeline(experiment_root: Path) -> None:
    st.markdown("#### Test-period timeline")
    timeline = read_parquet(str(experiment_root / "timeline.parquet"))
    figure = px.line(
        timeline,
        x="valid_time",
        y=["brier_score", "event_prevalence", "mean_probability"],
    )
    figure.update_layout(height=340, legend_title=None, yaxis_title=None)
    st.plotly_chart(figure, width="stretch")


def render_spatial_skill(experiment_root: Path) -> None:
    st.markdown("#### Spatial Brier skill")
    spatial = open_zarr(str(experiment_root / "spatial_skill.zarr"))
    figure = px.imshow(
        spatial.brier_skill_score,
        x=spatial.longitude,
        y=spatial.latitude,
        origin="lower",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto",
        labels={"color": "BSS"},
    )
    figure.update_layout(height=420)
    st.plotly_chart(figure, width="stretch")


def render_replay(experiment_root: Path) -> None:
    st.subheader("Historical replay")
    replay = open_zarr(str(experiment_root / "replay.zarr"))
    timestamp = st.select_slider(
        "Anchor time T",
        options=list(pd.to_datetime(replay.time.values)),
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M UTC"),
    )
    selected = replay.sel(time=timestamp)
    fields = (
        ("probability", "Predicted extreme-rain probability", "Viridis", (0, 1)),
        ("precipitation_6h", "Actual next-6h precipitation (mm)", "Blues", None),
        ("extreme_label", "Actual extreme label", "Greys", (0, 1)),
        ("probability_error", "Probability error", "RdBu", (-1, 1)),
    )
    for start in (0, 2):
        columns = st.columns(2)
        for column, (name, title, colorscale, limits) in zip(
            columns, fields[start : start + 2], strict=True
        ):
            with column:
                kwargs = {"zmin": limits[0], "zmax": limits[1]} if limits else {}
                figure = go.Figure(
                    go.Heatmap(
                        z=selected[name].values,
                        x=selected.longitude.values,
                        y=selected.latitude.values,
                        colorscale=colorscale,
                        colorbar={"title": title},
                        **kwargs,
                    )
                )
                figure.update_layout(height=350, title=title, margin={"t": 45, "b": 20})
                st.plotly_chart(figure, width="stretch")


if mode == "Historical · Atlas Mini":
    render_historical()
else:
    render_synthetic()
