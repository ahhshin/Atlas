from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pandas as pd

from world_state.ingest.artifacts import ArtifactProvenance, GridField
from world_state.ingest.base import DataClass, DataSource, RawPayload
from world_state.ingest.grib import grib_datasets, sample_dataset
from world_state.ingest.http import get_bytes

DEFAULT_PRODUCTS = {
    "MergedBaseReflectivityQC": ("radar_reflectivity", "dBZ"),
    "PrecipRate": ("precipitation_rate", "mm/h"),
    "MultiSensor_QPE_01H_Pass2": ("precipitation_1h", "mm"),
    "MESH": ("maximum_expected_hail_size", "mm"),
}


def _valid_time(dataset, fallback: datetime) -> datetime:
    if "valid_time" not in dataset.coords:
        return fallback
    parsed = pd.Timestamp(dataset.valid_time.values).to_pydatetime()
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class MrmsProvider(DataSource):
    name = "mrms"
    product = "mrms-conus-2d"
    data_class = DataClass.ANALYZED

    def configured_products(self) -> dict[str, tuple[str, str]]:
        requested = self.config.get("products", list(DEFAULT_PRODUCTS))
        return {name: DEFAULT_PRODUCTS[name] for name in requested}

    def fetch(self, client: httpx.Client, now: datetime) -> list[RawPayload]:
        del now
        self.fetch_errors = []
        payloads: list[RawPayload] = []
        for product in self.configured_products():
            url = f"{self.config['base_url'].rstrip('/')}/{product}/MRMS_{product}.latest.grib2.gz"
            try:
                content, request_url = get_bytes(
                    client,
                    url,
                    retries=int(self.http_config.get("retries", 3)),
                    backoff_seconds=float(self.http_config.get("backoff_seconds", 0.5)),
                )
                if not content.startswith(b"\x1f\x8b"):
                    raise ValueError(f"{product} response was not gzip")
                payloads.append(
                    RawPayload(product, content, request_url, "application/gzip")
                )
            except (httpx.HTTPError, ValueError) as error:
                self.fetch_errors.append(f"{product}: {error}")
        if not payloads:
            raise RuntimeError("; ".join(self.fetch_errors) or "No MRMS products fetched")
        return payloads

    def normalize(self, payloads: list[RawPayload], ingested_at: datetime) -> list[GridField]:
        fields: list[GridField] = []
        configured = self.configured_products()
        bbox = tuple(self.config.get("bbox", [-125, 24, -66, 50]))
        resolution = float(self.config.get("storage_resolution_degrees", 0.25))
        for payload in payloads:
            if payload.identifier not in configured:
                continue
            canonical_name, unit = configured[payload.identifier]
            with grib_datasets(payload.content, gzip_encoded=True) as datasets:
                dataset = next((value for value in datasets if value.data_vars), None)
                if dataset is None:
                    continue
                source_name = next(iter(dataset.data_vars))
                sampled = sample_dataset(
                    dataset, {source_name: canonical_name}, bbox, resolution
                ).load()
                valid_time = _valid_time(dataset, ingested_at)
            sampled[canonical_name].attrs.update(
                {"units": unit, "source_product": payload.identifier}
            )
            missing_threshold = -90 if canonical_name == "radar_reflectivity" else -900
            sampled[canonical_name] = sampled[canonical_name].where(
                sampled[canonical_name] > missing_threshold
            )
            provenance = ArtifactProvenance(
                source=self.name,
                product=payload.identifier,
                data_class=self.data_class,
                valid_time=valid_time,
                available_at=ingested_at,
                ingested_at=ingested_at,
                source_id=f"{payload.identifier}-{valid_time.isoformat()}",
                source_url=payload.request_url,
            )
            fields.append(
                GridField(
                    sampled,
                    provenance,
                    {canonical_name: unit},
                    bbox,
                    "0.01° native; resampled",
                )
            )
        return fields
