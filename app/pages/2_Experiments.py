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

from world_state.research.config import VARIABLES
from world_state.store import load_metrics
from world_state.synthetic import SYNTHETIC_VARIABLES

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("Experiments")
mode = st.radio(
    "Experiment family",
    ("Historical · Atlas Mini", "Historical · Latent World Model", "Synthetic lab"),
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


@st.cache_data(show_spinner=False)
def directory_size(path: str) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


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


def render_latent_world() -> None:
    data_root = Path(os.environ.get("ATLAS_DATA_ROOT", "/mnt/games/Atlas/data"))
    datasets = [
        name
        for name in research_datasets(str(data_root))
        if (data_root / "research" / name / "latent_world" / "experiments.parquet").exists()
    ]
    st.caption(
        "Atlas Latent World Model V0 · learned regional environmental state representation"
    )
    if not datasets:
        st.info(
            "No latent experiment is available. Run `atlas-research latent-smoke "
            "configs/research/latent_world_smoke.yaml` first."
        )
        return
    dataset_name = st.selectbox("Source dataset", datasets)
    latent_root = data_root / "research" / dataset_name / "latent_world"
    index = read_parquet(str(latent_root / "experiments.parquet")).sort_values("updated_at")
    st.subheader("Representation-size comparison")
    st.dataframe(
        index[
            [
                "latent_dimensions",
                "latent_grid",
                "parameter_count",
                "model_normalized_rmse",
                "persistence_normalized_rmse",
                "probe_pr_auc",
                "probe_roc_auc",
                "completed_stages",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    complete = index.loc[index.completed_stages.str.contains("probe")]
    if len(complete) > 1:
        comparison = px.bar(
            complete,
            x="latent_dimensions",
            y=["model_normalized_rmse", "persistence_normalized_rmse"],
            barmode="group",
            labels={"value": "Normalized RMSE", "latent_dimensions": "Latent dimensions"},
        )
        comparison.update_layout(height=330, legend_title=None)
        st.plotly_chart(comparison, width="stretch")
    labels = {
        row.experiment_id: (
            f"D={row.latent_dimensions} · {row.completed_stages} · {row.experiment_id}"
        )
        for row in index.itertuples()
    }
    experiment_id = st.selectbox(
        "Latent experiment",
        index.experiment_id.tolist(),
        index=len(index) - 1,
        format_func=lambda value: labels[value],
    )
    experiment_root = latent_root / "experiments" / experiment_id
    metadata = read_json(str(experiment_root / "metrics.json"))
    render_latent_overview(metadata, experiment_root)
    reconstruction_path = experiment_root / "reconstruction.zarr"
    if reconstruction_path.exists():
        render_latent_reconstruction(reconstruction_path)
    diagnostics_path = experiment_root / "world_diagnostics.zarr"
    if diagnostics_path.exists():
        render_latent_future(diagnostics_path)
        render_latent_state(diagnostics_path, experiment_root)
    if "dynamics" in metadata:
        render_latent_metrics(metadata)
    if "probe" in metadata:
        render_latent_probe(metadata, experiment_root)


def render_latent_overview(metadata: dict[str, object], experiment_root: Path) -> None:
    st.subheader("Overview")
    columns = st.columns(6)
    columns[0].metric("Version", "V0")
    columns[1].metric("Latent D", metadata["latent_dimensions"])
    columns[2].metric("Patch", f"{metadata['patch_size']} × {metadata['patch_size']}")
    columns[3].metric("Latent grid", " × ".join(map(str, metadata["latent_grid_dimensions"])))
    columns[4].metric("Parameters", f"{metadata['parameter_counts']['total']:,}")
    columns[5].metric("Artifacts", f"{directory_size(str(experiment_root)) / 1024**2:.1f} MiB")
    with st.expander("Architecture and training provenance"):
        st.json(
            {
                "dataset": metadata["dataset_version"],
                "git_commit": metadata["git_commit"],
                "training_period": metadata["training_period"],
                "physical_grid": metadata["physical_grid_dimensions"],
                "compression_ratio": metadata["compression_ratio"],
                "architectures": metadata["architectures"],
                "loss_weights": metadata["loss_weights"],
                "optimizer": metadata["optimizer"],
                "learning_rate": metadata["learning_rate"],
                "epochs": metadata["epochs"],
                "completed_stages": metadata["completed_stages"],
            }
        )


def render_latent_reconstruction(path: Path) -> None:
    st.subheader("Reconstruction")
    reconstruction = open_zarr(str(path))
    controls = st.columns(2)
    variable = controls[0].selectbox(
        "Reconstruction variable", reconstruction.channel.values.tolist()
    )
    timestamp = controls[1].selectbox(
        "Reconstruction time",
        list(pd.to_datetime(reconstruction.time.values)),
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M UTC"),
    )
    selected = reconstruction.sel(time=timestamp, channel=variable)
    unit = VARIABLES[str(variable)].units
    columns = st.columns(2)
    for column, field, title in zip(
        columns,
        ("actual", "reconstructed"),
        ("Actual X(t)", "Reconstructed X̂(t)"),
        strict=True,
    ):
        with column:
            _field_map(selected[field], f"{title} · {variable} ({unit})", "Viridis")


def render_latent_future(path: Path) -> None:
    st.subheader("Future prediction · T + 3h")
    diagnostics = open_zarr(str(path))
    controls = st.columns(2)
    variable = controls[0].selectbox("Future variable", diagnostics.channel.values.tolist())
    timestamp = controls[1].selectbox(
        "Prediction anchor T",
        list(pd.to_datetime(diagnostics.time.values)),
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M UTC"),
    )
    selected = diagnostics.sel(time=timestamp, channel=variable)
    unit = VARIABLES[str(variable)].units
    fields = (
        ("current", "X(t)"),
        ("physical_persistence", "Physical persistence"),
        ("latent_prediction", "Atlas latent prediction"),
        ("actual_future", "Actual X(t+3h)"),
    )
    columns = st.columns(4)
    for column, (field, title) in zip(columns, fields, strict=True):
        with column:
            _field_map(selected[field], f"{title} · {unit}", "Viridis", height=290)
    error_columns = st.columns(2)
    for column, field, title in (
        (error_columns[0], "persistence_error", "Persistence error"),
        (error_columns[1], "model_error", "Latent-model error"),
    ):
        with column:
            _field_map(selected[field], f"{title} · {unit}", "RdBu", midpoint=0)


def render_latent_state(path: Path, experiment_root: Path) -> None:
    st.subheader("Latent state diagnostics")
    diagnostics = open_zarr(str(path))
    timestamp = st.selectbox(
        "Latent diagnostic time",
        list(pd.to_datetime(diagnostics.time.values)),
        format_func=lambda value: value.strftime("%Y-%m-%d %H:%M UTC"),
    )
    selected = diagnostics.sel(time=timestamp)
    columns = st.columns(3)
    for column, field, title in zip(
        columns,
        ("latent_magnitude", "latent_change_magnitude", "latent_prediction_error"),
        ("‖Z(t)‖", "‖Z(t) − Z(t−3h)‖", "‖Ẑ(t+3h) − Z(t+3h)‖"),
        strict=True,
    ):
        with column:
            figure = px.imshow(
                selected[field],
                origin="lower",
                aspect="auto",
                color_continuous_scale="Magma",
                labels={"color": title},
            )
            figure.update_layout(height=290, title=title, margin={"t": 45, "b": 20})
            st.plotly_chart(figure, width="stretch")
    pca_path = experiment_root / "latent_pca.parquet"
    if pca_path.exists():
        pca = read_parquet(str(pca_path))
        figure = px.scatter(
            pca,
            x="pca_1",
            y="pca_2",
            color="time",
            size="latent_magnitude",
            hover_data=["latent_latitude", "latent_longitude"],
            title="Regional embedding PCA projection",
        )
        figure.update_layout(height=430)
        st.plotly_chart(figure, width="stretch")
    neighbors_path = experiment_root / "latent_neighbors.parquet"
    if neighbors_path.exists():
        with st.expander("Nearest regional states to the selected central token"):
            neighbors = read_parquet(str(neighbors_path))
            query_options = neighbors.query_time.unique().tolist()
            query = st.selectbox("Query timestamp", query_options)
            st.dataframe(
                neighbors.loc[neighbors.query_time == query], hide_index=True, width="stretch"
            )


def render_latent_metrics(metadata: dict[str, object]) -> None:
    st.subheader("Persistence comparison by variable")
    metrics = metadata["dynamics"]["test_metrics"]
    rows = []
    for variable in metadata["channels"]:
        persistence = metrics["physical_persistence"]["per_variable"][variable]
        latent_persistence = metrics["latent_persistence"]["per_variable"][variable]
        model = metrics["latent_dynamics"]["per_variable"][variable]
        improvement = metrics["improvement_over_physical_persistence"]["per_variable"][variable]
        rows.append(
            {
                "variable": variable,
                "unit": VARIABLES[variable].units,
                "physical_persistence_rmse": persistence["physical_rmse"],
                "latent_persistence_rmse": latent_persistence["physical_rmse"],
                "latent_model_rmse": model["physical_rmse"],
                "latent_model_mae": model["physical_mae"],
                "rmse_improvement_percent": improvement[
                    "physical_rmse_improvement_percent"
                ],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    columns = st.columns(3)
    columns[0].metric(
        "Latent prediction MSE", f"{metrics['latent_prediction_mse']:.4f}"
    )
    columns[1].metric(
        "Latent persistence MSE", f"{metrics['latent_persistence_mse']:.4f}"
    )
    columns[2].metric(
        "RMSE improvement",
        f"{metrics['improvement_over_physical_persistence']['overall_normalized_rmse_improvement_percent']:.1f}%",
    )


def render_latent_probe(metadata: dict[str, object], experiment_root: Path) -> None:
    st.subheader("Frozen-encoder precipitation probe")
    metrics = metadata["probe"]["test_metrics"]
    columns = st.columns(5)
    for column, key, label in zip(
        columns,
        ("pr_auc", "roc_auc", "brier_score", "brier_skill_score", "positive_event_prevalence"),
        ("PR-AUC", "ROC-AUC", "Brier", "Brier skill", "Prevalence"),
        strict=True,
    ):
        column.metric(label, f"{metrics[key]:.3f}")
    reference = metrics.get("engineered_feature_reference")
    if reference:
        st.caption(
            "Engineered-feature reference · "
            f"{reference['model']} · PR-AUC {reference['pr_auc']:.3f} · "
            f"ROC-AUC {reference['roc_auc']:.3f} · Brier {reference['brier_score']:.3f}"
        )
    calibration_path = experiment_root / "probe_calibration.parquet"
    if calibration_path.exists():
        calibration = read_parquet(str(calibration_path))
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=calibration.mean_probability,
                y=calibration.observed_frequency,
                mode="lines+markers",
                name="Frozen Z(t) probe",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dot"}
            )
        )
        figure.update_layout(
            height=350,
            xaxis_title="Predicted probability",
            yaxis_title="Observed frequency",
        )
        st.plotly_chart(figure, width="stretch")


def _field_map(
    values: xr.DataArray,
    title: str,
    colorscale: str,
    *,
    midpoint: float | None = None,
    height: int = 340,
) -> None:
    figure = px.imshow(
        values,
        x=values.longitude,
        y=values.latitude,
        origin="lower",
        aspect="auto",
        color_continuous_scale=colorscale,
        color_continuous_midpoint=midpoint,
    )
    figure.update_layout(height=height, title=title, margin={"t": 45, "b": 20})
    st.plotly_chart(figure, width="stretch")


if mode == "Historical · Atlas Mini":
    render_historical()
elif mode == "Historical · Latent World Model":
    render_latent_world()
else:
    render_synthetic()
