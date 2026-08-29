from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.store import (
    available_live_sources,
    forecast_slice,
    load_latest_point_observations,
    load_observations,
    load_station_snapshot,
)
from world_state.synthetic import SYNTHETIC_VARIABLES
from world_state.variables import LIVE_VARIABLES

st.set_page_config(page_title="World State map", page_icon="🗺️", layout="wide")
st.title("World State")


@st.cache_resource
def synthetic_observations():
    return load_observations()


@st.cache_data(ttl=60)
def latest_points(source: str, variable: str):
    return load_latest_point_observations(source=source, variable=variable)


@st.cache_data(ttl=60)
def station_snapshot(source: str, station_id: str):
    return load_station_snapshot(source, station_id)


@st.cache_data
def forecasts(model: str, target: str, horizon: int, valid_at: pd.Timestamp):
    return forecast_slice(model=model, target=target, horizon=horizon, valid_at=valid_at)


def _utc(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def render_live() -> None:
    sources = available_live_sources()
    st.caption("Live public station observations · OBSERVED")
    if not sources:
        st.warning(
            "No real observations have been ingested. Run `world-state-ingest fetch eccc` or "
            "`world-state-ingest fetch nws`, then refresh this page."
        )
        return

    controls = st.columns([1, 2])
    source = controls[0].selectbox("Source", sources, format_func=str.upper)
    available = load_latest_point_observations(source=source)
    variable_names = [name for name in LIVE_VARIABLES if name in set(available.variable)]
    variable = controls[1].selectbox(
        "Variable", variable_names, format_func=lambda name: LIVE_VARIABLES[name]["label"]
    )
    frame = latest_points(source, variable)
    if frame.empty:
        st.info("This source has no current values for the selected variable.")
        return

    unit = LIVE_VARIABLES[variable]["unit"]
    fig = px.scatter_geo(
        frame,
        lat="latitude",
        lon="longitude",
        color="value",
        hover_name="station_name",
        color_continuous_scale="Turbo",
        custom_data=["source", "station_id", "valid_time", "ingested_at", "value"],
        labels={"value": f"{LIVE_VARIABLES[variable]['label']} ({unit})"},
    )
    fig.update_traces(marker={"size": 13, "opacity": 0.92, "line": {"width": 0.5}})
    fig.update_geos(
        projection_type="natural earth",
        center={"lat": 47, "lon": -98},
        lataxis_range=[23, 72],
        lonaxis_range=[-145, -50],
        showland=True,
        landcolor="#111827",
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
        showcountries=True,
    )
    fig.update_layout(height=590, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points", key="live-map"
    )
    points = event.selection.points if event and event.selection else []
    if points:
        custom = points[0].get("customdata", [])
        if len(custom) >= 2:
            st.session_state["live_selected_station"] = (custom[0], custom[1])

    selected_source, selected_station = st.session_state.get(
        "live_selected_station", (source, str(frame.iloc[0].station_id))
    )
    if selected_source != source:
        selected_station = str(frame.iloc[0].station_id)
    snapshot = station_snapshot(source, selected_station)
    if snapshot.empty:
        return
    selected_rows = snapshot.loc[snapshot.variable == variable]
    selected = selected_rows.iloc[0] if not selected_rows.empty else snapshot.iloc[0]

    st.subheader(selected.station_name or selected.station_id)
    provenance, readings = st.columns([1, 2])
    provenance.markdown(
        f"""
**Source**  
{str(selected.source).upper()}

**Type**  
{str(selected.data_class).upper()}

**Observed**  
{_utc(selected.valid_time)}

**Available to system**  
{_utc(selected.available_at)}

**Ingested**  
{_utc(selected.ingested_at)}
"""
    )
    display = snapshot[["variable", "value", "unit", "quality_flag"]].copy()
    display["variable"] = display.variable.map(
        lambda name: LIVE_VARIABLES.get(name, {"label": name})["label"]
    )
    display["value"] = display.value.round(2)
    readings.dataframe(
        display.rename(
            columns={
                "variable": "Variable",
                "value": "Value",
                "unit": "Unit",
                "quality_flag": "Source quality",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def render_synthetic() -> None:
    st.caption("Synthetic CONUS forecast demonstration · SYNTHETIC · not observed data")
    ds = synthetic_observations()
    controls = st.columns([2, 1, 1, 1, 2])
    variable = controls[0].selectbox(
        "Variable",
        list(SYNTHETIC_VARIABLES),
        format_func=lambda name: SYNTHETIC_VARIABLES[name]["label"],
    )
    model = controls[1].selectbox("Model", ["persistence", "climatology"])
    horizon = controls[2].selectbox(
        "Horizon", [6, 12, 24, 48], format_func=lambda value: f"+{value} h"
    )
    layer = controls[3].selectbox("Layer", ["prediction", "actual", "error"])
    offset = horizon // 6
    available_times = pd.to_datetime(ds.time.values[offset:])
    valid_at = controls[4].select_slider(
        "Valid time",
        options=list(available_times),
        value=available_times[-1],
        format_func=lambda value: value.strftime("%b %d %H:%M"),
    )
    frame = forecasts(model, variable, horizon, pd.Timestamp(valid_at))
    unit = SYNTHETIC_VARIABLES[variable]["unit"]
    scale = "RdBu_r" if layer == "error" else ("Blues" if variable == "precipitation" else "Turbo")
    range_color = None
    if layer == "error":
        bound = max(float(frame[layer].abs().quantile(0.98)), 0.01)
        range_color = [-bound, bound]
    fig = px.scatter_geo(
        frame,
        lat="latitude",
        lon="longitude",
        color=layer,
        color_continuous_scale=scale,
        range_color=range_color,
        custom_data=["latitude", "longitude", "prediction", "actual", "error"],
        labels={layer: f"{layer.title()} ({unit})"},
    )
    fig.update_traces(marker={"size": 16, "opacity": 0.92})
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#111827",
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
    )
    fig.update_layout(height=590, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    event = st.plotly_chart(
        fig, width="stretch", on_select="rerun", selection_mode="points", key="synthetic-map"
    )
    points = event.selection.points if event and event.selection else []
    if points:
        st.session_state["synthetic_selected_cell"] = (points[0]["lat"], points[0]["lon"])
    selected_lat, selected_lon = st.session_state.get("synthetic_selected_cell", (39.0, -97.0))
    nearest_lat = float(ds.latitude.sel(latitude=selected_lat, method="nearest"))
    nearest_lon = float(ds.longitude.sel(longitude=selected_lon, method="nearest"))
    cell = frame.loc[(frame.latitude == nearest_lat) & (frame.longitude == nearest_lon)].iloc[0]
    st.subheader(f"Synthetic grid cell · {nearest_lat:.1f}°, {nearest_lon:.1f}°")
    metrics = st.columns(3)
    metrics[0].metric("Prediction", f"{cell.prediction:.2f} {unit}")
    metrics[1].metric("Synthetic actual", f"{cell.actual:.2f} {unit}")
    metrics[2].metric("Error", f"{cell.error:+.2f} {unit}")
    history = (
        ds[variable].sel(latitude=nearest_lat, longitude=nearest_lon).to_dataframe().reset_index()
    )
    history_fig = px.line(
        history,
        x="time",
        y=variable,
        labels={variable: f"{SYNTHETIC_VARIABLES[variable]['label']} ({unit})"},
    )
    history_fig.add_vline(
        x=pd.Timestamp(valid_at).timestamp() * 1000, line_dash="dash", line_color="#ef4444"
    )
    history_fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(history_fig, width="stretch")


mode = st.segmented_control(
    "Data mode",
    ["Live observations", "Synthetic forecast"],
    default="Live observations",
    selection_mode="single",
)
if mode == "Synthetic forecast":
    render_synthetic()
else:
    render_live()
