from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import xarray as xr

from world_state.ingest.artifacts import ArtifactProvenance, ForecastField
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.grib import grib_datasets, sample_dataset
from world_state.ingest.http import get_bytes

FILE_PATTERN = re.compile(r"hrrr\.t(\d{2})z\.wrfsfcf(\d{2})\.grib2")


class HrrrProvider(DataSource):
    name = "hrrr"
    product = "hrrr-conus-surface"
    data_class = DataClass.FORECAST

    def _choose_runs(
        self, client: httpx.Client, now: datetime
    ) -> list[tuple[str, str, list[int]]]:
        requested = [int(value) for value in self.config.get("horizons", [1, 3, 6, 12, 24, 48])]
        extended_cycles = set(self.config.get("extended_cycles", [0, 6, 12, 18]))
        retries = int(self.http_config.get("retries", 3))
        backoff = float(self.http_config.get("backoff_seconds", 0.5))
        candidates: list[tuple[datetime, str, int, set[int]]] = []
        for offset in (0, 1):
            date = (now - timedelta(days=offset)).strftime("%Y%m%d")
            directory = self.config["directory"].format(date=date)
            try:
                listing, _ = get_bytes(
                    client, directory, retries=retries, backoff_seconds=backoff
                )
            except httpx.HTTPError:
                continue
            available: dict[int, set[int]] = {}
            for cycle, horizon in set(FILE_PATTERN.findall(listing.decode(errors="ignore"))):
                available.setdefault(int(cycle), set()).add(int(horizon))
            for cycle, horizons in available.items():
                reference = datetime.strptime(f"{date}{cycle:02d}", "%Y%m%d%H").replace(tzinfo=UTC)
                candidates.append((reference, date, cycle, horizons))
        short_horizons = [value for value in requested if value <= 12]
        long_horizons = [value for value in requested if value > 12]
        chosen: list[tuple[str, str, list[int]]] = []
        short = [value for value in candidates if set(short_horizons).issubset(value[3])]
        if short:
            _, date, cycle, _ = max(short)
            chosen.append((date, f"{cycle:02d}", short_horizons))
        extended = [
            value
            for value in candidates
            if value[2] in extended_cycles and set(long_horizons).issubset(value[3])
        ]
        if long_horizons and extended:
            _, date, cycle, _ = max(extended)
            chosen.append((date, f"{cycle:02d}", long_horizons))
        if not chosen:
            raise RuntimeError("No complete current HRRR cycle found for configured horizons")
        return chosen

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        runs = self._choose_runs(client, now)
        west, south, east, north = self.config.get("bbox", [-125, 24, -66, 50])
        payloads: list[RawPayload] = []
        self.fetch_errors = []
        for date, cycle, horizons in runs:
            for horizon in horizons:
                filename = f"hrrr.t{cycle}z.wrfsfcf{horizon:02d}.grib2"
                params = {
                    "file": filename,
                    "lev_2_m_above_ground": "on",
                    "lev_10_m_above_ground": "on",
                    "lev_surface": "on",
                    "lev_entire_atmosphere": "on",
                    "var_TMP": "on",
                    "var_DPT": "on",
                    "var_UGRD": "on",
                    "var_VGRD": "on",
                    "var_APCP": "on",
                    "var_REFC": "on",
                    "subregion": "",
                    "leftlon": west,
                    "rightlon": east,
                    "toplat": north,
                    "bottomlat": south,
                    "dir": f"/hrrr.{date}/conus",
                }
                try:
                    content, request_url = get_bytes(
                        client,
                        self.config["filter_endpoint"],
                        params=params,
                        retries=int(self.http_config.get("retries", 3)),
                        backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
                    )
                    if not content.startswith(b"GRIB"):
                        raise ValueError(f"HRRR f{horizon:02d} response was not GRIB2")
                    payloads.append(
                        RawPayload(
                            f"hrrr-{date}-t{cycle}z-f{horizon:02d}",
                            content,
                            request_url,
                            "application/x-grib2",
                        )
                    )
                except (httpx.HTTPError, ValueError) as error:
                    self.fetch_errors.append(f"t{cycle}z f{horizon:02d}: {error}")
        if not payloads:
            raise RuntimeError("; ".join(self.fetch_errors) or "No HRRR horizons fetched")
        return payloads

    def normalize(
        self, payloads: list[RawPayload], ingested_at: datetime
    ) -> list[ForecastField]:
        bbox = tuple(self.config.get("bbox", [-125, 24, -66, 50]))
        resolution = float(self.config.get("storage_resolution_degrees", 0.25))
        artifacts: list[ForecastField] = []
        for payload in payloads:
            match = re.search(r"hrrr-(\d{8})-t(\d{2})z-f(\d{2})", payload.identifier)
            if match is None:
                continue
            date, cycle, horizon_text = match.groups()
            horizon = int(horizon_text)
            reference_time = datetime.strptime(f"{date}{cycle}", "%Y%m%d%H").replace(
                tzinfo=UTC
            )
            valid_time = reference_time + timedelta(hours=horizon)
            parts: list[xr.Dataset] = []
            with grib_datasets(payload.content) as datasets:
                for dataset in datasets:
                    mapping = {
                        source: canonical
                        for source, canonical in {
                            "t2m": "temperature",
                            "d2m": "dewpoint",
                            "u10": "wind_u",
                            "v10": "wind_v",
                            "tp": "precipitation_accumulated",
                            "refc": "radar_reflectivity_composite",
                        }.items()
                        if source in dataset
                    }
                    if mapping:
                        parts.append(sample_dataset(dataset, mapping, bbox, resolution).load())
            if not parts:
                continue
            output = xr.merge(parts, compat="override")
            for variable in ("temperature", "dewpoint"):
                if variable in output:
                    output[variable] = output[variable] - 273.15
                    output[variable].attrs["units"] = "°C"
            if "wind_u" in output and "wind_v" in output:
                output["wind_speed"] = np.hypot(output.wind_u, output.wind_v)
                for variable in ("wind_u", "wind_v", "wind_speed"):
                    output[variable].attrs["units"] = "m/s"
            if "precipitation_accumulated" in output:
                output.precipitation_accumulated.attrs["units"] = "mm"
            if "radar_reflectivity_composite" in output:
                output.radar_reflectivity_composite.attrs["units"] = "dBZ"
            units = {
                name: str(array.attrs.get("units", ""))
                for name, array in output.data_vars.items()
            }
            provenance = ArtifactProvenance(
                source=self.name,
                product=self.product,
                data_class=self.data_class,
                valid_time=valid_time,
                available_at=ingested_at,
                ingested_at=ingested_at,
                source_id=payload.identifier,
                source_url=payload.request_url,
                forecast_reference_time=reference_time,
                forecast_horizon_hours=horizon,
            )
            artifacts.append(
                ForecastField(
                    output,
                    provenance,
                    units,
                    bbox,
                    "3 km native; resampled",
                )
            )
        return artifacts
