from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.store import (
    available_live_sources,
    forecast_slice,
    load_catalog_table,
    load_event_asset,
    load_grid_asset,
    load_latest_point_observations,
    load_observations,
    load_station_snapshot,
)
from world_state.synthetic import SYNTHETIC_VARIABLES

st.set_page_config(page_title="Atlas State Explorer", page_icon="🗺️", layout="wide")
st.title("Atlas State Explorer")


@st.cache_resource
def synthetic_observations():
    return load_observations()


@st.cache_data(ttl=60)
def latest_points(source: str, variable: str):
    return load_latest_point_observations(source=source, variable=variable)


@st.cache_data(ttl=60)
def station_snapshot(source: str, station_id: str):
    return load_station_snapshot(source, station_id)


@st.cache_data(ttl=30)
def catalog_table(name: str):
    return load_catalog_table(name)


@st.cache_resource
def grid_asset(path: str):
    return load_grid_asset(path)


@st.cache_data(ttl=60)
def event_asset(path: str):
    return load_event_asset(path)


@st.cache_data
def forecasts(model: str, target: str, horizon: int, valid_at: pd.Timestamp):
    return forecast_slice(model=model, target=target, horizon=horizon, valid_at=valid_at)


def _utc(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def _variables(row: pd.Series) -> dict[str, str]:
    return json.loads(row.variables_json)


def _polygon_traces(fig: go.Figure, events: pd.DataFrame) -> None:
    for row in events.head(100).itertuples():
        geometry = row.geometry
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            ring = polygon[0]
            probability = getattr(row, "probability_severe", None)
            label = "ProbSevere" if pd.isna(probability) else f"ProbSevere {probability:.0f}%"
            fig.add_trace(
                go.Scatter(
                    x=[value[0] for value in ring],
                    y=[value[1] for value in ring],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(239,68,68,0.16)",
                    line={"color": "#ef4444", "width": 1.2},
                    name=label,
                    hovertext=label,
                    showlegend=False,
                )
            )


def _transform(values: np.ndarray, view: str) -> tuple[np.ndarray, str]:
    if view == "Spatial anomaly":
        return values - np.nanmedian(values), "spatial anomaly"
    if view == "Spatial percentile":
        flat = pd.Series(values.ravel())
        return flat.rank(pct=True).to_numpy().reshape(values.shape) * 100, "percentile"
    return values, "value"


def _render_station_map(allowed_sources: set[str] | None = None) -> None:
    sources = available_live_sources()
    if allowed_sources is not None:
        sources = [source for source in sources if source in allowed_sources]
    if not sources:
        st.info("No point observations are stored yet. Run `world-state-ingest fetch metar`.")
        return
    controls = st.columns(2)
    source = controls[0].selectbox("Point source", sources, format_func=str.upper)
    available = load_latest_point_observations(source=source)
    variable = controls[1].selectbox("Variable", sorted(available.variable.unique()))
    frame = latest_points(source, variable)
    frame = frame.dropna(subset=["latitude", "longitude"])
    if frame.empty:
        st.info("This time-series source has no map coordinates.")
        return
    fig = px.scatter_geo(
        frame,
        lat="latitude",
        lon="longitude",
        color="value",
        hover_name="station_name",
        custom_data=["source", "station_id", "valid_time", "available_at", "ingested_at"],
        color_continuous_scale="Turbo",
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.85})
    fig.update_geos(
        projection_type="natural earth",
        center={"lat": 42, "lon": -98},
        lataxis_range=[15, 75],
        lonaxis_range=[-170, -50],
        showland=True,
        landcolor="#111827",
        showcountries=True,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=610, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"OBSERVED · {source.upper()} · {len(frame):,} current station values")


def _render_event_points(source: str, events: pd.DataFrame) -> None:
    scoped = events.loc[events.source == source]
    if scoped.empty:
        st.info(f"No {source.upper()} event collection has been ingested yet.")
        return
    row = scoped.sort_values("valid_time").iloc[-1]
    frame = event_asset(row.geoparquet_path)
    frame["longitude"] = [value["coordinates"][0] for value in frame.geometry]
    frame["latitude"] = [value["coordinates"][1] for value in frame.geometry]
    color = "fire_radiative_power_mw" if "fire_radiative_power_mw" in frame else None
    fig = px.scatter_geo(
        frame,
        lat="latitude",
        lon="longitude",
        color=color,
        hover_name="event_id",
        color_continuous_scale="Inferno",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    fig.update_geos(
        projection_type="natural earth",
        center={"lat": 42, "lon": -98},
        lataxis_range=[15, 75],
        lonaxis_range=[-170, -50],
        showland=True,
        landcolor="#111827",
        showcountries=True,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(height=610, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, width="stretch")


def _render_grid_explorer(section: str, assets: pd.DataFrame, events: pd.DataFrame) -> None:
    forecast_mode = section == "Atmosphere" and "hrrr" in set(assets.source)
    mode_options = ["Current analysis"] + (["HRRR forecast"] if forecast_mode else [])
    mode = st.radio("State class", mode_options, horizontal=True)
    scoped = assets.loc[assets.source == "hrrr"] if mode == "HRRR forecast" else assets
    if mode != "HRRR forecast":
        scoped = scoped.loc[scoped.source != "hrrr"]
    if scoped.empty:
        st.info(f"No {section.lower()} grid assets have been ingested yet.")
        return

    if mode == "HRRR forecast":
        runs = sorted(pd.to_datetime(scoped.forecast_reference_time.dropna().unique()))
        run = st.selectbox("Model run", runs, index=len(runs) - 1, format_func=_utc)
        scoped = scoped.loc[pd.to_datetime(scoped.forecast_reference_time) == run]
        horizons = sorted(scoped.forecast_horizon_hours.dropna().astype(int).unique())
        horizon = st.selectbox("Forecast horizon", horizons, format_func=lambda value: f"+{value} h")
        scoped = scoped.loc[scoped.forecast_horizon_hours == horizon]

    product_options = sorted(scoped["product"].unique())
    product = st.selectbox("Product", product_options)
    scoped = scoped.loc[scoped["product"] == product].copy()
    variable_options = sorted({key for _, row in scoped.iterrows() for key in _variables(row)})
    controls = st.columns([2, 1])
    variable = controls[0].selectbox("Layer", variable_options)
    view = controls[1].selectbox("View", ["Value", "Spatial anomaly", "Spatial percentile"])
    scoped = scoped.loc[scoped.apply(lambda row: variable in _variables(row), axis=1)]
    scoped = scoped.sort_values("valid_time")
    time_options = list(pd.to_datetime(scoped.valid_time))
    playback_key = f"frame-{section}-{mode}-{product}-{variable}"
    valid_time = st.select_slider(
        "Valid time / playback frame",
        options=time_options,
        value=time_options[-1],
        format_func=_utc,
        key=playback_key,
    )
    playback = st.columns([1, 1, 6])
    frame_index = time_options.index(valid_time)
    if playback[0].button("◀", help="Previous frame", key=f"previous-{playback_key}"):
        st.session_state[playback_key] = time_options[max(0, frame_index - 1)]
        st.rerun()
    if playback[1].button("▶", help="Next frame", key=f"next-{playback_key}"):
        st.session_state[playback_key] = time_options[min(len(time_options) - 1, frame_index + 1)]
        st.rerun()
    row = scoped.iloc[int(np.argmin(np.abs(pd.to_datetime(scoped.valid_time) - valid_time)))]
    dataset = grid_asset(row.zarr_path)
    field = dataset[variable]
    values, color_title = _transform(np.asarray(field.values), view)
    unit = _variables(row).get(variable, "")
    colorscale = "RdBu_r" if view == "Spatial anomaly" else "Turbo"
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=np.asarray(dataset.longitude),
            y=np.asarray(dataset.latitude),
            colorscale=colorscale,
            colorbar={"title": f"{color_title}<br>{unit}"},
            hovertemplate="%{y:.2f}°, %{x:.2f}°<br>%{z:.2f}<extra></extra>",
        )
    )

    if section == "Atmosphere" and st.toggle("METAR station overlay", value=True):
        stations = latest_points("metar", variable)
        if not stations.empty:
            fig.add_trace(
                go.Scatter(
                    x=stations.longitude,
                    y=stations.latitude,
                    mode="markers",
                    marker={"size": 4, "color": "white", "opacity": 0.7},
                    text=stations.station_id,
                    name="METAR",
                )
            )
    if section == "Radar" and not events.empty and st.toggle("ProbSevere polygons", value=True):
        severe = events.loc[events.source == "probsevere"]
        if not severe.empty:
            latest_event = severe.sort_values("valid_time").iloc[-1]
            _polygon_traces(fig, event_asset(latest_event.geoparquet_path))
    if section == "Satellite" and not events.empty and st.toggle("GLM lightning", value=True):
        lightning_assets = events.loc[events.source == "goes"]
        if not lightning_assets.empty:
            lightning_row = lightning_assets.sort_values("valid_time").iloc[-1]
            lightning = event_asset(lightning_row.geoparquet_path)
            longitude = [value["coordinates"][0] for value in lightning.geometry]
            latitude = [value["coordinates"][1] for value in lightning.geometry]
            fig.add_trace(
                go.Scatter(
                    x=longitude,
                    y=latitude,
                    mode="markers",
                    marker={"size": 5, "color": "#f8fafc", "symbol": "x"},
                    name="GLM flash",
                )
            )

    fig.update_layout(
        height=610,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis={"title": "Longitude", "range": [-125, -66]},
        yaxis={"title": "Latitude", "range": [24, 50]},
        legend={"orientation": "h"},
    )
    st.plotly_chart(fig, width="stretch")

    if mode == "HRRR forecast":
        observed_variable = {
            "radar_reflectivity_composite": "radar_reflectivity",
        }.get(variable, variable)
        observed = assets.loc[
            (assets.source != "hrrr")
            & assets.apply(lambda value: observed_variable in _variables(value), axis=1)
        ].copy()
        if not observed.empty:
            observed["distance"] = abs(
                pd.to_datetime(observed.valid_time) - pd.Timestamp(row.valid_time)
            )
            outcome = observed.sort_values("distance").iloc[0]
            if outcome.distance <= pd.Timedelta("90min"):
                actual = grid_asset(outcome.zarr_path)[observed_variable].interp_like(field)
                mae = float(np.nanmean(np.abs(np.asarray(field) - np.asarray(actual))))
                st.metric("HRRR vs eventual analyzed outcome · MAE", f"{mae:.2f} {unit}")

    with st.expander("Source and provenance"):
        st.json(
            {
                "source": row.source,
                "product": row["product"],
                "class": row.data_class,
                "valid_time": _utc(row.valid_time),
                "available_time": _utc(row.available_at),
                "ingestion_time": _utc(row.ingested_at),
                "forecast_reference_time": _utc(row.forecast_reference_time),
                "forecast_horizon_hours": row.forecast_horizon_hours,
                "source_id": row.source_id,
                "source_url": row.source_url,
            }
        )


def render_live() -> None:
    st.caption("Observed, analyzed, and forecast environmental state · provenance preserved")
    section = st.segmented_control(
        "Section",
        ["Atmosphere", "Radar", "Satellite", "Hydrology", "Fire", "Air", "Energy"],
        default="Atmosphere",
    )
    assets = catalog_table("grid_assets")
    events = catalog_table("event_assets")
    section_sources = {
        "Atmosphere": {"rtma", "hrrr"},
        "Radar": {"mrms"},
        "Satellite": {"goes"},
        "Hydrology": {"usgs", "nwm"},
        "Fire": {"firms"},
        "Air": {"openaq", "airnow"},
        "Energy": {"eia"},
    }
    scoped = assets.loc[assets.source.isin(section_sources[section])] if not assets.empty else assets
    point_sources = {
        "Atmosphere": {"metar", "eccc", "nws"},
        "Hydrology": {"usgs"},
        "Air": {"airnow", "openaq"},
        "Energy": {"eia"},
    }
    available_points = load_latest_point_observations()
    section_points = (
        available_points.loc[available_points.source.isin(point_sources.get(section, set()))]
        if not available_points.empty
        else available_points
    )
    if scoped.empty and section == "Fire" and not events.empty:
        _render_event_points("firms", events)
    elif scoped.empty and section == "Energy" and not section_points.empty:
        st.dataframe(
            section_points[
                ["station_name", "variable", "value", "unit", "valid_time", "available_at"]
            ].sort_values(["variable", "value"], ascending=[True, False]),
            hide_index=True,
            width="stretch",
        )
    elif scoped.empty and section in point_sources and not section_points.empty:
        _render_station_map(point_sources[section])
    elif scoped.empty:
        st.info(
            f"{section} is scaffolded in the shared artifact/catalog model; its provider is in the "
            "next secondary-modality increment."
        )
    else:
        _render_grid_explorer(section, scoped, events)


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
