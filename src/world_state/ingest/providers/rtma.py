from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import xarray as xr

from world_state.ingest.artifacts import ArtifactProvenance, GridField
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.grib import grib_datasets, sample_dataset
from world_state.ingest.http import get_bytes

FILE_PATTERN = re.compile(r"rtma2p5\.t(\d{2})z\.2dvaranl_ndfd\.grb2(?:_wexp)?")


class RtmaProvider(DataSource):
    name = "rtma"
    product = "rtma2p5-analysis"
    data_class = DataClass.ANALYZED

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        retries = int(self.http_config.get("retries", 3))
        backoff = float(self.http_config.get("backoff_seconds", 0.5))
        chosen: tuple[str, str, str] | None = None
        for offset in (0, 1):
            date = (now - timedelta(days=offset)).strftime("%Y%m%d")
            directory = self.config["directory"].format(date=date)
            try:
                listing, _ = get_bytes(
                    client, directory, retries=retries, backoff_seconds=backoff
                )
            except httpx.HTTPError:
                continue
            filenames = sorted(set(FILE_PATTERN.findall(listing.decode(errors="ignore"))))
            if filenames:
                cycle = filenames[-1]
                filename = f"rtma2p5.t{cycle}z.2dvaranl_ndfd.grb2_wexp"
                chosen = date, cycle, filename
                break
        if chosen is None:
            raise RuntimeError("No current RTMA analysis file found")

        date, cycle, filename = chosen
        west, south, east, north = self.config.get("bbox", [-125, 24, -66, 50])
        params = {
            "file": filename,
            "lev_2_m_above_ground": "on",
            "lev_10_m_above_ground": "on",
            "lev_surface": "on",
            "var_TMP": "on",
            "var_DPT": "on",
            "var_UGRD": "on",
            "var_VGRD": "on",
            "var_PRES": "on",
            "subregion": "",
            "leftlon": west,
            "rightlon": east,
            "toplat": north,
            "bottomlat": south,
            "dir": f"/rtma2p5.{date}",
        }
        content, request_url = get_bytes(
            client,
            self.config["filter_endpoint"],
            params=params,
            retries=retries,
            backoff_seconds=backoff,
        )
        if not content.startswith(b"GRIB"):
            raise ValueError("RTMA filter response was not GRIB2")
        return [
            RawPayload(
                f"rtma2p5-{date}-t{cycle}z",
                content,
                request_url,
                "application/x-grib2",
            )
        ]

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[GridField]:
        if not payloads:
            return []
        payload = payloads[0]
        match = re.search(r"rtma2p5-(\d{8})-t(\d{2})z", payload.identifier)
        if match is None:
            raise ValueError(f"Unrecognized RTMA identifier: {payload.identifier}")
        valid_time = datetime.strptime("".join(match.groups()), "%Y%m%d%H").replace(tzinfo=UTC)
        bbox = tuple(self.config.get("bbox", [-125, 24, -66, 50]))
        resolution = float(self.config.get("storage_resolution_degrees", 0.25))
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
                        "sp": "pressure",
                    }.items()
                    if source in dataset
                }
                if mapping:
                    parts.append(sample_dataset(dataset, mapping, bbox, resolution).load())
        if not parts:
            return []
        output = xr.merge(parts, compat="override")
        for variable in ("temperature", "dewpoint"):
            if variable in output:
                output[variable] = output[variable] - 273.15
                output[variable].attrs["units"] = "°C"
        if "pressure" in output:
            output["pressure"] = output.pressure / 100
            output.pressure.attrs["units"] = "hPa"
        if "wind_u" in output and "wind_v" in output:
            output["wind_speed"] = np.hypot(output.wind_u, output.wind_v)
            output["wind_direction"] = (
                270 - np.degrees(np.arctan2(output.wind_v, output.wind_u))
            ) % 360
            for variable in ("wind_u", "wind_v", "wind_speed"):
                output[variable].attrs["units"] = "m/s"
            output.wind_direction.attrs["units"] = "°"
        if "temperature" in output and "dewpoint" in output:
            numerator = np.exp((17.625 * output.dewpoint) / (243.04 + output.dewpoint))
            denominator = np.exp((17.625 * output.temperature) / (243.04 + output.temperature))
            output["humidity"] = (100 * numerator / denominator).clip(0, 100)
            output.humidity.attrs["units"] = "%"

        units = {name: str(array.attrs.get("units", "")) for name, array in output.data_vars.items()}
        provenance = ArtifactProvenance(
            source=self.name,
            product=self.product,
            data_class=self.data_class,
            valid_time=valid_time,
            available_at=ingested_at,
            ingested_at=ingested_at,
            source_id=payload.identifier,
            source_url=payload.request_url,
        )
        return [GridField(output, provenance, units, bbox, "2.5 km native; resampled")]
