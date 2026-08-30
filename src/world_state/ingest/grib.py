from __future__ import annotations

import gzip
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


@contextmanager
def grib_datasets(content: bytes, *, gzip_encoded: bool = False) -> Iterator[list[xr.Dataset]]:
    body = gzip.decompress(content) if gzip_encoded else content
    with tempfile.NamedTemporaryFile(suffix=".grib2") as handle:
        handle.write(body)
        handle.flush()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="In a future version of xarray the default value for compat will change",
                category=FutureWarning,
            )
            datasets = cfgrib.open_datasets(Path(handle.name), backend_kwargs={"indexpath": ""})
        yield datasets


def target_coordinates(
    bbox: tuple[float, float, float, float], resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = bbox
    latitude = np.arange(south, north + resolution / 2, resolution, dtype=np.float32)
    longitude = np.arange(west, east + resolution / 2, resolution, dtype=np.float32)
    return latitude, longitude


def sample_dataset(
    dataset: xr.Dataset,
    variables: dict[str, str],
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> xr.Dataset:
    """Nearest-neighbour sample regular or projected GRIB grids onto a small lat/lon grid."""
    target_latitude, target_longitude = target_coordinates(bbox, resolution)
    latitude = dataset.latitude
    longitude = dataset.longitude
    output: dict[str, tuple[tuple[str, str], np.ndarray]] = {}

    if latitude.ndim == 1 and longitude.ndim == 1:
        source_longitude = longitude.values
        requested_longitude = (
            np.mod(target_longitude, 360)
            if np.nanmax(source_longitude) > 180
            else target_longitude
        )
        selected = dataset[list(variables)].sel(
            latitude=xr.DataArray(target_latitude, dims="latitude"),
            longitude=xr.DataArray(requested_longitude, dims="longitude"),
            method="nearest",
        )
        for source_name, canonical_name in variables.items():
            output[canonical_name] = (
                ("latitude", "longitude"),
                np.asarray(selected[source_name].values, dtype=np.float32),
            )
    else:
        flat_latitude = np.asarray(latitude.values).ravel()
        flat_longitude = np.asarray(longitude.values).ravel()
        flat_longitude = np.where(flat_longitude > 180, flat_longitude - 360, flat_longitude)
        valid = np.isfinite(flat_latitude) & np.isfinite(flat_longitude)
        tree = cKDTree(np.column_stack((flat_latitude[valid], flat_longitude[valid])))
        target_lon_mesh, target_lat_mesh = np.meshgrid(target_longitude, target_latitude)
        _, valid_indices = tree.query(
            np.column_stack((target_lat_mesh.ravel(), target_lon_mesh.ravel()))
        )
        source_indices = np.flatnonzero(valid)[valid_indices]
        for source_name, canonical_name in variables.items():
            values = np.asarray(dataset[source_name].values).ravel()[source_indices]
            output[canonical_name] = (
                ("latitude", "longitude"),
                values.reshape(target_lat_mesh.shape).astype(np.float32),
            )

    return xr.Dataset(
        output,
        coords={"latitude": target_latitude, "longitude": target_longitude},
    )
