# World State

World State is a personal research platform for collecting environmental observations, retaining
their provenance, and evaluating compact forecasting models. It has two intentionally separate
paths:

- continuously collected public observations for a live world-state view and future point-in-time
  evaluation;
- deterministic synthetic gridded data for offline development, GUI demos, and baseline tests.

Synthetic values are never silently presented as observations.

## Architecture

```text
ECCC / NWS / synthetic
        │
        ▼
provider fetch + normalize
        │
        ├── immutable raw response archive
        ├── normalized point observations (Parquet)
        └── ingestion runs and freshness (DuckDB)
                          │
                          ▼
                    Streamlit GUI

synthetic generator ──► Zarr + forecast Parquet ──► experiments / demo forecast
```

The GUI only reads normalized stores. It never calls a public API during page rendering. The
ingestion worker is a separate process, so a source outage does not prevent the app from loading
the last valid data.

## Data sources

| Source | State | Class | Current scope |
| --- | --- | --- | --- |
| Synthetic | Implemented | `SYNTHETIC` | Deterministic coarse CONUS grid |
| ECCC GeoMet SWOB | Implemented | `OBSERVED` | Five configured Canadian stations |
| NOAA/NWS API | Implemented | `OBSERVED` | Ten configured US stations |
| NOAA RTMA | Planned | `ANALYZED` | CONUS gridded current state |
| NOAA HRRR | Planned | `FORECAST` | Retained cycles at +6/+12/+24/+48 h where available |
| GOES, FIRMS, OpenAQ, USGS, EIA | Planned | Varies | Added incrementally after gridded weather |
| ERA5 | Planned backfill | Reanalysis/`ANALYZED` | Historical ML training, never live |

ECCC and NWS are live only after the ingestion command has succeeded. The Data Feeds page reports
that fact from the ledger; it does not fabricate source status.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Set a contact string for the NWS `User-Agent` header. A repository URL or email is suitable:

```bash
export WORLD_STATE_CONTACT='https://github.com/your-name/world-state'
```

No ECCC or NWS API key is required.

## Run synthetic mode

```bash
world-state-generate
streamlit run app/Home.py
```

In the World State page, select **Synthetic forecast**. Generated artifacts are stored at:

- `data/processed/environment.zarr`
- `data/predictions/forecasts.parquet`
- `data/predictions/metrics.parquet`

## Run live ingestion

Fetch either source manually:

```bash
world-state-ingest fetch eccc
world-state-ingest fetch nws
```

Fetch all enabled implemented sources once:

```bash
world-state-ingest fetch all
# equivalent development worker invocation
python -m world_state.ingest.worker --once
```

Run the continuously scheduled worker in a separate terminal:

```bash
python -m world_state.ingest.worker
```

Then start the GUI independently:

```bash
streamlit run app/Home.py
```

Source cadences, station subsets, endpoints, timeouts, retries, and enabled state live in
`configs/sources.yaml`. The defaults schedule ECCC and NWS every ten minutes. One provider failure
does not stop another provider's run.

Inspect feed health without the GUI:

```bash
world-state-ingest health
```

## Storage and provenance

Raw responses are archived without overwrite under:

```text
data/raw/{source}/YYYY/MM/DD/
```

Normalized point observations are append-only, partitioned Parquet under:

```text
data/normalized/point_observations/source={source}/date=YYYY-MM-DD/
```

The DuckDB ledger is `data/metadata/ingestion.duckdb`. It records each running, successful,
partial, or failed attempt, including byte counts, new-record counts, error text, and the latest
source timestamp.

Every normalized point retains:

- source and source product;
- `OBSERVED`, `ANALYZED`, `FORECAST`, `DERIVED`, or `SYNTHETIC` class;
- measurement valid time;
- when the value was available to this system (ECCC source processing time when supplied);
- local ingestion time;
- station/source identifiers and quality flag;
- location, variable, canonical value, and unit;
- forecast reference time and horizon fields for future forecast providers.

Canonical point units are Celsius, hectopascals, percent, metres per second, angular degrees, and
millimetres. Missing source values are omitted rather than replaced with zeros.

## Tests

```bash
pytest
ruff check .
```

Tests use checked-in response fixtures and mocked HTTP transports; they do not require public API
availability. They cover parsing, missing values, UTC conversion, unit conversion, retry behavior,
duplicate prevention, synthetic leakage checks, and Streamlit page execution.

