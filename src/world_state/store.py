from __future__ import annotations

import duckdb
import pandas as pd
import xarray as xr
from shapely import from_wkb
from shapely.geometry import mapping

from world_state.ingest.catalog import AtlasCatalog
from world_state.paths import FORECASTS_PATH, METRICS_PATH, OBSERVATIONS_PATH


def load_observations() -> xr.Dataset:
    if not OBSERVATIONS_PATH.exists():
        raise FileNotFoundError("Demo data is missing. Run `world-state-generate` first.")
    return xr.open_zarr(OBSERVATIONS_PATH)


def forecast_slice(
    *, model: str, target: str, horizon: int, valid_at: pd.Timestamp
) -> pd.DataFrame:
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError("Demo data is missing. Run `world-state-generate` first.")
    return duckdb.sql(
        """
        SELECT latitude, longitude, prediction, actual, error
        FROM read_parquet(?)
        WHERE model = ? AND target = ? AND forecast_horizon_hours = ? AND valid_at = ?
        ORDER BY latitude, longitude
        """,
        params=[str(FORECASTS_PATH), model, target, horizon, valid_at.to_pydatetime()],
    ).df()


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Demo data is missing. Run `world-state-generate` first.")
    return duckdb.sql(
        "SELECT * FROM read_parquet(?) ORDER BY target, forecast_horizon_hours, model",
        params=[str(METRICS_PATH)],
    ).df()


def load_latest_point_observations(
    *, source: str | None = None, variable: str | None = None
) -> pd.DataFrame:
    frame = AtlasCatalog().table("latest_point_state")
    if frame.empty:
        return pd.DataFrame()
    if source:
        frame = frame.loc[frame.source == source]
    if variable:
        frame = frame.loc[frame.variable == variable]
    return frame.sort_values(["source", "station_name", "variable"], na_position="last")


def load_station_snapshot(source: str, station_id: str) -> pd.DataFrame:
    observations = load_latest_point_observations(source=source)
    if observations.empty:
        return observations
    return observations.loc[observations.station_id == station_id].sort_values("variable")


def available_live_sources() -> list[str]:
    observations = load_latest_point_observations()
    if observations.empty:
        return []
    return sorted(observations.loc[observations.data_class != "synthetic", "source"].unique())


def load_catalog_table(name: str) -> pd.DataFrame:
    return AtlasCatalog().table(name)


def load_grid_asset(path: str) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=True)


def load_event_asset(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "geometry" in frame:
        frame["geometry"] = [mapping(from_wkb(value)) for value in frame.geometry]
    return frame
