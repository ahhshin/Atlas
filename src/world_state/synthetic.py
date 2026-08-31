from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from world_state.paths import FORECASTS_PATH, METRICS_PATH, OBSERVATIONS_PATH
from world_state.variables import VARIABLES

SYNTHETIC_VARIABLES = {
    name: VARIABLES[name]
    for name in ("temperature", "pressure", "humidity", "precipitation", "u_wind", "v_wind")
}


def build_synthetic_dataset(seed: int = 42, periods: int = 80) -> xr.Dataset:
    """Create deterministic, spatially coherent weather-like fields on a coarse CONUS grid."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-05-01", periods=periods, freq="6h")
    latitude = np.arange(25.0, 49.1, 2.0)
    longitude = np.arange(-124.0, -66.9, 3.0)

    t = np.arange(periods)[:, None, None]
    lat = latitude[None, :, None]
    lon = longitude[None, None, :]
    shape = (periods, latitude.size, longitude.size)

    diurnal = np.sin(2 * np.pi * t / 4)
    slow_wave = np.sin(2 * np.pi * t / 28 + (lon + 100) / 8)
    front = np.tanh((lon + 112 - 0.65 * t + 0.35 * (lat - 37)) / 4)
    storm = np.exp(-((lon + 101 - 0.40 * t) ** 2) / 55 - ((lat - 36) ** 2) / 20)

    temperature = 27 - 0.62 * (lat - 25) + 4.5 * diurnal + 3.2 * slow_wave + 2 * front
    temperature = temperature + rng.normal(0, 0.35, shape)
    pressure = 1014 + 6 * front - 13 * storm + rng.normal(0, 0.45, shape)
    humidity = np.clip(55 - 18 * front + 36 * storm - 6 * diurnal + rng.normal(0, 2, shape), 5, 100)
    precipitation = np.maximum(0, 12 * storm * (0.65 + 0.35 * np.sin(t / 2) ** 2) - 0.7)
    precipitation = precipitation + rng.gamma(0.7, 0.18, shape)
    u_wind = 4 + 9 * storm + 1.6 * slow_wave + rng.normal(0, 0.4, shape)
    v_wind = -1 + 5 * storm - 1.2 * front + rng.normal(0, 0.4, shape)

    ds = xr.Dataset(
        data_vars={
            "temperature": (("time", "latitude", "longitude"), temperature.astype("float32")),
            "pressure": (("time", "latitude", "longitude"), pressure.astype("float32")),
            "humidity": (("time", "latitude", "longitude"), humidity.astype("float32")),
            "precipitation": (
                ("time", "latitude", "longitude"),
                precipitation.astype("float32"),
            ),
            "u_wind": (("time", "latitude", "longitude"), u_wind.astype("float32")),
            "v_wind": (("time", "latitude", "longitude"), v_wind.astype("float32")),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        attrs={"title": "Deterministic synthetic CONUS environmental state", "seed": seed},
    )
    for variable, metadata in SYNTHETIC_VARIABLES.items():
        ds[variable].attrs.update(metadata)
    return ds


def build_baseline_forecasts(
    ds: xr.Dataset, horizons: tuple[int, ...] = (6, 12, 24, 48)
) -> pd.DataFrame:
    """Create persistence and per-cell climatology forecasts under one tidy schema."""
    step_hours = int((ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "h"))
    records: list[pd.DataFrame] = []
    climatology = ds.mean("time")

    for horizon in horizons:
        offset = horizon // step_hours
        if horizon % step_hours or offset >= ds.sizes["time"]:
            raise ValueError(f"Horizon {horizon}h is incompatible with the dataset frequency")
        issued = ds.isel(time=slice(None, -offset))
        actual = ds.isel(time=slice(offset, None))

        for target in ds.data_vars:
            actual_values = actual[target].values
            issued_at = pd.to_datetime(issued.time.values)
            valid_at = pd.to_datetime(actual.time.values)
            for model, prediction in (
                ("persistence", issued[target].values),
                ("climatology", np.broadcast_to(climatology[target].values, actual_values.shape)),
            ):
                frame = pd.DataFrame(
                    {
                        "issued_at": np.repeat(
                            issued_at, ds.sizes["latitude"] * ds.sizes["longitude"]
                        ),
                        "valid_at": np.repeat(
                            valid_at, ds.sizes["latitude"] * ds.sizes["longitude"]
                        ),
                        "latitude": np.tile(
                            np.repeat(ds.latitude.values, ds.sizes["longitude"]), len(valid_at)
                        ),
                        "longitude": np.tile(
                            ds.longitude.values, len(valid_at) * ds.sizes["latitude"]
                        ),
                        "forecast_horizon_hours": horizon,
                        "target": target,
                        "model": model,
                        "prediction": prediction.reshape(-1),
                        "actual": actual_values.reshape(-1),
                    }
                )
                frame["error"] = frame["prediction"] - frame["actual"]
                records.append(frame)
    return pd.concat(records, ignore_index=True)


def aggregate_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    grouped = forecasts.groupby(["model", "target", "forecast_horizon_hours"], sort=True)
    return grouped.apply(
        lambda frame: pd.Series(
            {
                "mae": frame["error"].abs().mean(),
                "rmse": np.sqrt(np.square(frame["error"]).mean()),
                "bias": frame["error"].mean(),
                "samples": len(frame),
            }
        ),
        include_groups=False,
    ).reset_index()


def write_demo_data(output_root: Path | None = None) -> tuple[Path, Path, Path]:
    observations_path = OBSERVATIONS_PATH
    forecasts_path = FORECASTS_PATH
    metrics_path = METRICS_PATH
    if output_root is not None:
        observations_path = output_root / "processed" / "environment.zarr"
        forecasts_path = output_root / "predictions" / "forecasts.parquet"
        metrics_path = output_root / "predictions" / "metrics.parquet"

    observations_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts_path.parent.mkdir(parents=True, exist_ok=True)
    ds = build_synthetic_dataset()
    forecasts = build_baseline_forecasts(ds)
    metrics = aggregate_metrics(forecasts)
    chunk_shape = (16, ds.sizes["latitude"], ds.sizes["longitude"])
    encoding = {variable: {"chunks": chunk_shape} for variable in ds.data_vars}
    ds.to_zarr(observations_path, mode="w", encoding=encoding, zarr_format=2)
    forecasts.to_parquet(forecasts_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    return observations_path, forecasts_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the World State demo dataset")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    paths = write_demo_data(args.output_root)
    print("Generated demo data:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
