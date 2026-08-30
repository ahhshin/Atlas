from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from world_state.ingest.artifacts import ArtifactProvenance, EventCollection, GridField
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.grib import target_coordinates
from world_state.ingest.http import get_bytes


def _timestamp(value, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    parsed = pd.Timestamp(value).to_pydatetime()
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _keys(document: bytes) -> list[str]:
    root = ET.fromstring(document)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [value.text for value in root.findall("s3:Contents/s3:Key", namespace) if value.text]


def _nearest_indices(coordinates: np.ndarray, requested: np.ndarray) -> np.ndarray:
    descending = coordinates[0] > coordinates[-1]
    ordered = coordinates[::-1] if descending else coordinates
    indices = np.searchsorted(ordered, requested)
    indices = np.clip(indices, 1, len(ordered) - 1)
    left = ordered[indices - 1]
    right = ordered[indices]
    indices -= (np.abs(requested - left) <= np.abs(right - requested)).astype(int)
    return len(coordinates) - 1 - indices if descending else indices


class GoesProvider(DataSource):
    name = "goes"
    product = "goes-east-abi-glm"
    data_class = DataClass.OBSERVED

    def _latest_keys(
        self, client: httpx.Client, now: datetime, product: str, stem: str, count: int
    ) -> list[str]:
        for offset in range(3):
            moment = now - timedelta(hours=offset)
            prefix = f"{product}/{moment.year}/{moment.strftime('%j')}/{moment:%H}/{stem}"
            listing, _ = get_bytes(
                client,
                self.config["bucket_url"],
                params={"list-type": "2", "prefix": prefix, "max-keys": 1000},
                retries=int(self.http_config.get("retries", 3)),
                backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
            )
            matches = sorted(_keys(listing))
            if matches:
                return matches[-count:]
        return []

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        requests: list[str] = []
        for channel in self.config.get("abi_channels", [8, 13]):
            matches = self._latest_keys(
                client,
                now,
                "ABI-L2-CMIPC",
                f"OR_ABI-L2-CMIPC-M6C{int(channel):02d}",
                1,
            )
            requests.extend(matches)
        requests.extend(
            self._latest_keys(
                client,
                now,
                "GLM-L2-LCFA",
                "OR_GLM-L2-LCFA",
                int(self.config.get("glm_files", 15)),
            )
        )
        if not requests:
            raise RuntimeError("No current GOES ABI or GLM objects found")
        payloads: list[RawPayload] = []
        self.fetch_errors = []
        for key in requests:
            url = f"{self.config['bucket_url'].rstrip('/')}/{key}"
            try:
                content, request_url = get_bytes(
                    client,
                    url,
                    retries=int(self.http_config.get("retries", 3)),
                    backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
                )
                payloads.append(
                    RawPayload(Path(key).name, content, request_url, "application/x-netcdf")
                )
            except httpx.HTTPError as error:
                self.fetch_errors.append(f"{Path(key).name}: {error}")
        if not payloads:
            raise RuntimeError("; ".join(self.fetch_errors))
        return payloads

    def _abi_field(self, payload: RawPayload, ingested_at: datetime) -> GridField:
        bbox = tuple(self.config.get("bbox", [-125, 24, -66, 50]))
        resolution = float(self.config.get("storage_resolution_degrees", 0.25))
        target_latitude, target_longitude = target_coordinates(bbox, resolution)
        target_lon_mesh, target_lat_mesh = np.meshgrid(target_longitude, target_latitude)
        with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
            handle.write(payload.content)
            handle.flush()
            with xr.open_dataset(handle.name, engine="h5netcdf") as source:
                projection = source.goes_imager_projection.attrs
                height = float(projection["perspective_point_height"])
                geos = CRS.from_proj4(
                    "+proj=geos "
                    f"+h={height} +lon_0={projection['longitude_of_projection_origin']} "
                    f"+sweep={projection['sweep_angle_axis']} "
                    f"+a={projection['semi_major_axis']} +b={projection['semi_minor_axis']}"
                )
                transformer = Transformer.from_crs("EPSG:4326", geos, always_xy=True)
                projected_x, projected_y = transformer.transform(target_lon_mesh, target_lat_mesh)
                x_requested = projected_x / height
                y_requested = projected_y / height
                valid = np.isfinite(x_requested) & np.isfinite(y_requested)
                x_index = _nearest_indices(np.asarray(source.x), np.nan_to_num(x_requested))
                y_index = _nearest_indices(np.asarray(source.y), np.nan_to_num(y_requested))
                values = np.asarray(source.CMI.values)[y_index, x_index].astype(np.float32)
                if "DQF" in source:
                    quality = np.asarray(source.DQF.values)[y_index, x_index]
                    values = np.where((quality <= 1) & valid, values, np.nan)
                valid_time = _timestamp(source.attrs.get("time_coverage_start"), ingested_at)
                available_at = _timestamp(source.attrs.get("date_created"), ingested_at)

        channel = 8 if "C08_" in payload.identifier else 13
        variable = (
            "water_vapor_brightness_temperature"
            if channel == 8
            else "infrared_cloud_top_temperature"
        )
        values -= 273.15
        dataset = xr.Dataset(
            {variable: (("latitude", "longitude"), values)},
            coords={"latitude": target_latitude, "longitude": target_longitude},
        )
        dataset[variable].attrs.update({"units": "°C", "abi_channel": channel})
        provenance = ArtifactProvenance(
            self.name,
            f"ABI-L2-CMIPC-C{channel:02d}",
            self.data_class,
            valid_time,
            available_at,
            ingested_at,
            payload.identifier,
            payload.request_url,
        )
        return GridField(dataset, provenance, {variable: "°C"}, bbox, "2 km native; resampled")

    def _glm_events(
        self, payloads: list[RawPayload], ingested_at: datetime
    ) -> EventCollection | None:
        west, south, east, north = self.config.get("bbox", [-125, 24, -66, 50])
        rows: list[dict] = []
        valid_times: list[datetime] = []
        available_times: list[datetime] = []
        for payload in payloads:
            with tempfile.NamedTemporaryFile(suffix=".nc") as handle:
                handle.write(payload.content)
                handle.flush()
                with xr.open_dataset(handle.name, engine="h5netcdf") as source:
                    if "flash_lon" not in source or "flash_lat" not in source:
                        continue
                    longitude = np.asarray(source.flash_lon.values)
                    latitude = np.asarray(source.flash_lat.values)
                    energy = (
                        np.asarray(source.flash_energy.values)
                        if "flash_energy" in source
                        else np.full(longitude.shape, np.nan)
                    )
                    identifiers = (
                        np.asarray(source.flash_id.values)
                        if "flash_id" in source
                        else np.arange(len(longitude))
                    )
                    valid_time = _timestamp(source.attrs.get("time_coverage_start"), ingested_at)
                    available_at = _timestamp(source.attrs.get("date_created"), ingested_at)
                    valid_times.append(valid_time)
                    available_times.append(available_at)
                    mask = (
                        (longitude >= west)
                        & (longitude <= east)
                        & (latitude >= south)
                        & (latitude <= north)
                    )
                    for lon, lat, value, identifier in zip(
                        longitude[mask], latitude[mask], energy[mask], identifiers[mask], strict=True
                    ):
                        rows.append(
                            {
                                "event_id": f"{payload.identifier}-{identifier}",
                                "valid_time": valid_time,
                                "available_at": available_at,
                                "flash_energy_j": float(value),
                                "source_file": payload.identifier,
                                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                            }
                        )
        if not rows:
            return None
        provenance = ArtifactProvenance(
            self.name,
            "GLM-L2-LCFA",
            self.data_class,
            max(valid_times),
            max(available_times),
            ingested_at,
            f"glm-{min(valid_times).isoformat()}-{max(valid_times).isoformat()}",
            payloads[-1].request_url,
        )
        return EventCollection(pd.DataFrame(rows), provenance)

    def normalize(
        self, payloads: list[RawPayload], ingested_at: datetime
    ) -> list[GridField | EventCollection]:
        artifacts: list[GridField | EventCollection] = []
        abi = [payload for payload in payloads if "ABI-L2-CMIPC" in payload.identifier]
        glm = [payload for payload in payloads if "GLM-L2-LCFA" in payload.identifier]
        artifacts.extend(self._abi_field(payload, ingested_at) for payload in abi)
        lightning = self._glm_events(glm, ingested_at)
        if lightning is not None:
            artifacts.append(lightning)
        return artifacts
