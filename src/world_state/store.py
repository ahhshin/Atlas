from __future__ import annotations

import duckdb
import pandas as pd
import xarray as xr

from world_state.ingest.storage import PointObservationStore
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
    files = PointObservationStore().parquet_files()
    if not files:
        return pd.DataFrame()
    conditions: list[str] = []
    params: list[object] = [[str(path) for path in files]]
    if source:
        conditions.append("source = ?")
        params.append(source)
    if variable:
        conditions.append("variable = ?")
        params.append(variable)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return duckdb.sql(
        f"""
        SELECT * FROM read_parquet(?, union_by_name=true)
        {where}
        QUALIFY row_number() OVER (
            PARTITION BY source, station_id, variable ORDER BY valid_time DESC, ingested_at DESC
        ) = 1
        ORDER BY source, station_name, variable
        """,
        params=params,
    ).df()


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
