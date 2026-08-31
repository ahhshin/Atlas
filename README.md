# Atlas

Atlas is a local-first multimodal environmental data platform and forecasting-research sandbox.
It continuously retains the weather information that was actually available at a given time,
while keeping deterministic synthetic data as an explicit offline development mode.

The project is deliberately small-infrastructure: Python, Streamlit, Parquet, Zarr, and DuckDB.
There is no message broker, orchestrator, or distributed database.

## Current capabilities

| Source | Artifact | Class | Initial bounded scope |
| --- | --- | --- | --- |
| NOAA Aviation Weather METAR | `PointBatch` | `OBSERVED` | Bulk current cache, filtered to North America |
| NOAA RTMA | `GridField` | `ANALYZED` | CONUS temperature, dew point, humidity, pressure and wind |
| NOAA MRMS | `GridField` | `ANALYZED` | Reflectivity, precipitation rate, one-hour QPE and MESH |
| NOAA ProbSevere | `EventCollection` | `ANALYZED` | Current CONUS storm polygons and severe/hail/wind/tornado probabilities |
| NOAA HRRR | `ForecastField` | `FORECAST` | Every collected cycle at +1/+3/+6/+12 h; extended cycles at +24/+48 h |
| NOAA GOES-19 | `GridField` + `EventCollection` | `OBSERVED` | ABI C08/C13 brightness temperature and five-minute GLM flashes |
| USGS NWIS | `PointBatch` | `OBSERVED` | Current streamflow and gauge height at configured research gauges |
| NASA FIRMS | `EventCollection` | `OBSERVED` | VIIRS active-fire detections; requires `FIRMS_MAP_KEY` |
| EPA AirNow | `PointBatch` | `OBSERVED` | Hourly criteria-pollutant observations; requires `AIRNOW_API_KEY` |
| EIA | `PointBatch` | `OBSERVED` | Hourly balancing-authority demand/net generation; requires `EIA_API_KEY` |
| ECCC SWOB | `PointBatch` | `OBSERVED` | Existing configured Canadian stations, retained for compatibility |
| NWS stations | `PointBatch` | `OBSERVED` | Existing adapter retained but disabled in favor of bulk METAR |
| Synthetic generator | `PointBatch` plus lab artifacts | `SYNTHETIC` | Deterministic coarse CONUS development fixture |

OpenAQ and gridded National Water Model data remain follow-on providers; AirNow and USGS provide
the initial air-quality and hydrology modalities.

## Architecture

```text
public source
    │
    ├── fetch source-native bytes ──► immutable raw archive + request metadata
    │
    └── normalize
          ├── PointBatch ──────────► partitioned Parquet
          ├── GridField ───────────► xarray / Zarr
          ├── ForecastField ───────► immutable cycle+horizon Zarr
          └── EventCollection ─────► GeoParquet
                                      │
                                      ▼
                    DuckDB IDs, current state, asset catalog,
                    forecast runs, freshness, and ingestion ledger
                                      │
                                      ▼
                            Atlas State Explorer
```

The Streamlit process never calls a public API. Collection runs independently, so the UI can keep
serving the last valid state during a source outage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

`cfgrib` and the Python `eccodes` wheels decode the bounded NOAA GRIB2 responses; a separate
`wgrib2` installation is not required.

## Ingest live data

Each implemented source has an explicit CLI target:

```bash
world-state-ingest fetch metar
world-state-ingest fetch rtma
world-state-ingest fetch mrms
world-state-ingest fetch probsevere
world-state-ingest fetch hrrr
world-state-ingest fetch goes
world-state-ingest fetch usgs
world-state-ingest fetch firms     # FIRMS_MAP_KEY required
world-state-ingest fetch airnow    # AIRNOW_API_KEY required
world-state-ingest fetch eia       # EIA_API_KEY required
```

Fetch all sources whose `enabled` flag is true:

```bash
world-state-ingest fetch all
```

Run only sources with both `enabled: true` and `scheduled: true` continuously:

```bash
world-state-worker
```

METAR and ProbSevere are scheduled by default. RTMA, MRMS, and HRRR are live-capable but manual by
default because their download volume is meaningful on a personal machine. Enable scheduling in
`configs/sources.yaml` after choosing a suitable cadence and storage budget. NOMADS requests use
server-side geographic, level, and variable filters; Atlas never requests an entire NOAA archive.

Source notes:

- METAR uses Aviation Weather's once-per-minute bulk CSV cache instead of polling a small station
  list.
- RTMA and HRRR use NOAA NOMADS filters for the configured CONUS bounding box and selected fields.
- MRMS downloads only four current two-dimensional products, then stores a research-scale bounded
  grid.
- ProbSevere discovers the newest published GeoJSON collection and preserves both its valid and
  production timestamps.
- HRRR keeps every downloaded run/horizon under a content-stable identity. Regular cycles retain
  +1/+3/+6/+12 h; the latest complete 00/06/12/18 UTC extended cycle also retains +24/+48 h.
- GOES uses the operational GOES-19 public bucket, storing only channels 08/13 and a short GLM
  flash window rather than full satellite archives.
- USGS uses a configured gauge set so the initial local archive stays useful and bounded.
- FIRMS, AirNow, and EIA adapters are implemented but disabled until their corresponding
  environment-variable API key is present.

Endpoints, bounding boxes, variables, cadences, retry behavior, and schedule flags are all visible
in `configs/sources.yaml`.

## Run the Explorer

```bash
streamlit run app/Home.py
```

The Atlas State Explorer is organized by `Atmosphere | Radar | Satellite | Hydrology | Fire | Air |
Energy`. Implemented layers provide:

- analyzed rasters and METAR overlays;
- MRMS rasters and ProbSevere polygons;
- time selection, spatial anomaly, and spatial percentile views;
- source, product, valid, availability, ingestion, run, and horizon provenance;
- HRRR run/horizon selection and comparison with a matching eventual analysis when one exists;
- an explicit synthetic forecast mode that is never presented as live data.

The Data Feeds page and CLI health command derive status from real ingestion attempts:

```bash
world-state-ingest health
world-state-ingest catalog grid_assets
world-state-ingest catalog forecast_runs
```

## Storage and deduplication

```text
data/raw/{source}/YYYY/MM/DD/                         immutable source bytes
data/normalized/point_observations/source=.../       append-only Parquet
data/normalized/grid_fields/source=.../              append-only Zarr
data/normalized/forecast_fields/source=.../          immutable run/horizon Zarr
data/normalized/event_collections/source=.../        GeoParquet
data/metadata/ingestion.duckdb                       ledger and catalogs
```

DuckDB's `point_record_ids` table is the deduplication index. An append asks that compact table
which IDs are new; it does not scan historical Parquet files. `latest_point_state`,
`point_partitions`, `grid_assets`, `forecast_runs`, `event_assets`, and `source_freshness` support
fast UI and research queries while history remains append-only.

Every artifact preserves source/product identity, data class, valid time, source availability time,
local ingestion time, and source URL. Forecasts also preserve reference time and horizon. Raw
responses and normalized forecast cycles are never overwritten.

## Point-in-time research queries

Atlas treats source availability and local ingestion as separate facts. A state query includes an
artifact only when both timestamps are at or before `T`:

```bash
world-state-ingest state-at 2026-08-30T23:45:00Z
```

In Python:

```python
from datetime import UTC, datetime
from world_state.ingest.catalog import AtlasCatalog

state = AtlasCatalog().assets_available_at(datetime(2026, 8, 30, 23, 45, tzinfo=UTC))
# state["points"], state["grids"], state["events"], state["forecasts"]
```

This is the foundation for leakage-safe samples of `state(T) -> state(T + delta)`. Initial targets
are precipitation at +1/+3/+6 h, reflectivity evolution, extreme-precipitation probability, and
severe-storm probability using RTMA + MRMS + GOES + METAR + HRRR.

## Historical Research / Atlas Mini

`Atlas Mini · Extreme Precipitation` is the first supervised experiment built from real historical
data. It asks whether the previous 24 hours of environmental state contain enough signal to
estimate the probability that each 0.25° CONUS grid cell will experience locally extreme
precipitation during the next six hours.

This deliberately small V1 uses ten ERA5 surface channels at three-hour cadence:

- 2 m temperature and dew point;
- surface pressure, 10 m U/V wind, total cloud cover, and skin temperature;
- preceding three-hour total precipitation and downward solar-radiation accumulations;
- top-layer volumetric soil moisture.

Wind speed, relative humidity, recent precipitation totals, pressure tendency, and local 3×3
neighborhood features are derived during loading/training rather than stored as duplicate fields.
ERA5 is accessed anonymously from the public Earthmover Icechunk archive on AWS. Its temporal
layout permits bounded regional reads; Atlas does not retain source archives or global fields.
Every historical store is marked `RETROSPECTIVE_REANALYSIS` and must not be interpreted as an
operational feed representing what was available in real time.

The smoke and full configurations are:

```text
configs/research/precip_smoke.yaml    January 2020
configs/research/precip_v1.yaml       2020-01-01 through 2022-12-31
```

Both use the same grid, cadence, variables, compression, and 15 GiB hard cap. State is partitioned
into one compressed Zarr v2 store per year and chunked for 24-hour sequences and regional crops.
The full estimator reports about 7.1 GiB expected and a conservative 12.3 GiB upper bound. Actual
usage is checked after every yearly partition and again after targets and experiment artifacts.
The completed 2020–2022 build contains 8,768 timestamps on a 105×237 grid with no missing source
fields. It currently uses 3.83 GiB of file content (4.8 GiB allocated on disk), including the fixed
sample caches and full experiment artifacts—well below the 15 GiB cap.

### SSD storage guard

All state, targets, climatology, splits, temporary stores, checkpoints, and experiment results are
written beneath `ATLAS_DATA_ROOT`. The checked-in configurations use the machine's larger SSD:

```bash
export ATLAS_DATA_ROOT=/mnt/games/Atlas/data
```

The backfill verifies that `/mnt/games` is a mounted writable filesystem, prints its resolved path
and free space, and aborts rather than falling back to the repository/system disk. On another
machine, change `required_mount` in the YAML and set `ATLAS_DATA_ROOT` to that machine's data disk.

### Reproduce the smoke experiment

Always estimate first:

```bash
atlas-research estimate configs/research/precip_smoke.yaml
atlas-research backfill configs/research/precip_smoke.yaml
atlas-research validate configs/research/precip_smoke.yaml
```

The complete real-data path is:

```text
public ERA5 subset → normalized yearly Zarr → 24 h state sequence
                  → future 6 h precipitation → local extreme label
                  → model probability → known outcome → loss and metrics
```

Train the minimum baselines and compact neural model:

```bash
atlas-research train configs/research/precip_smoke.yaml --model climatology
atlas-research train configs/research/precip_smoke.yaml --model persistence
atlas-research train configs/research/precip_smoke.yaml --model simple
atlas-research train configs/research/precip_smoke.yaml --model neural
atlas-research evaluate <experiment-id> --data-root /mnt/games/Atlas/data
```

`climatology` uses training event frequency by grid cell. `persistence` fits a calibrated logistic
model from recent precipitation. `simple` is multivariable logistic regression. `neural` is a
small spatial-temporal MLP using sequence statistics and 3×3 neighborhood context; its parameter
count is recorded and remains far below 10 million.

### Target and leakage controls

ERA5 total precipitation is an hourly accumulation ending at its valid time. Atlas sums three
source hours into each state field. For anchor `T`, the target is the sum of the state
precipitation fields ending at `T+3h` and `T+6h`; no future precipitation appears in the input.
The extreme threshold is the grid-cell-specific 95th percentile of next-six-hour precipitation
fit from training timestamps only:

```text
extreme(T) = precipitation(T:T+6h) > training_grid_cell_p95
```

The full chronological split is 2020–2021 train, January–June 2022 validation, and July–December
2022 test. Samples whose target would cross a split boundary are excluded. Split assignments,
feature/target window bounds, and training-only fit metadata are stored explicitly. Missing values
remain NaN, with separate state and target masks; the model pipeline never silently treats them as
physical zeroes. Normalization, thresholds, and climatology are all fit from training data only.

Each run records its git commit, dataset version, variables, target, context, model, parameter
count, configuration, checkpoint, calibration, regional metrics, timeline, spatial Brier skill,
and train/validation/test metrics. Headline evaluation includes Brier score and skill, log loss,
precision, recall, PR-AUC, ROC-AUC, event prevalence, and positive-event loss.

Open **Experiments → Historical · Atlas Mini** to inspect dataset years, dimensions, disk usage,
missingness, split counts, prevalence, model comparisons, calibration, spatial skill, and replayed
predictions beside actual next-six-hour precipitation and labels. The synthetic experiment mode is
still available beside it.

After the smoke pipeline succeeds, build the three-year dataset:

```bash
atlas-research estimate configs/research/precip_v1.yaml
atlas-research backfill configs/research/precip_v1.yaml
atlas-research validate configs/research/precip_v1.yaml
```

Normal validation rechecks structure, cadence, units, leakage boundaries, storage, target
alignment, and the immutable missingness summary saved by backfill. Use
`atlas-research validate configs/research/precip_v1.yaml --deep` to rescan every missing-data mask.

The first complete V1 evaluation uses 299,968 fixed training samples and 94,080 fixed test samples.
The test event prevalence is 5.58%:

| Model | Parameters | Brier | Brier skill | PR-AUC | ROC-AUC | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Climatology | 0 | 0.0527 | 0.000 | 0.055 | 0.495 | 0.000 |
| Persistence | 5 | 0.1534 | -1.912 | 0.327 | 0.856 | 0.549 |
| Logistic | 58 | 0.1381 | -1.621 | 0.387 | 0.890 | 0.810 |
| Compact neural | 3,985 | 0.1225 | -1.326 | 0.464 | 0.923 | 0.875 |

The learned models show useful ranking signal but negative Brier skill because their class-balanced
training probabilities are overconfident. Atlas reports that result directly; calibration is a
future improvement rather than something hidden behind a threshold metric.

Atlas Mini is an initial research experiment for validating data, leakage, training, and evaluation
infrastructure. It is not intended to beat or replace operational numerical weather prediction.

## Latent World Model V0

`Atlas Latent World Model V0` asks whether the existing ten-channel ERA5 state can be compressed
into a reusable, geographically structured representation and predicted three hours forward. It
does not download or duplicate ERA5. Every run reads `atlas-mini-precip-v1` in place and writes
only checkpoints, metadata, selected maps, PCA coordinates, and nearest-neighbor diagnostics under
that dataset's `latent_world/` directory on the configured SSD.

The model is deliberately small:

```text
normalized X(t) + explicit missing masks
  → regional convolutional encoder
  → Z(t) [D × ceil(latitude/4) × ceil(longitude/4)]
  → spatial 3×3 ConvGRU over eight 3-hour latent maps
  → Ẑ(t+3h)
  → convolutional decoder
  → normalized X̂(t+3h)
```

The full 105×237 physical grid is padded to 108×240 internally, producing a 27×60 regional map,
then cropped exactly back to 105×237 after decoding. A token therefore represents approximately a
1°×1° region. Missing physical values are zero-filled only after normalization and are always
accompanied by one explicit mask channel per variable; masked targets do not contribute to loss.

Training is staged. The encoder/decoder first establish a stable reconstruction space. The encoder
and decoder are then frozen while the ConvGRU learns both latent and decoded physical losses.
Finally, the encoder is frozen and a 1×1 probability head tests whether Z(t) supports the existing
next-six-hour extreme-precipitation target. Each stage restores its best validation epoch.

### Install and run

PyTorch is optional so the live platform and precipitation experiment remain lightweight. For a
CPU-only local installation, install the official CPU wheel before the project extra:

```bash
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,latent]'
export ATLAS_DATA_ROOT=/mnt/games/Atlas/data
```

Run the bounded real-data D=32/D=64 smoke ablation:

```bash
atlas-research latent-smoke configs/research/latent_world_smoke.yaml
```

Run individual full-grid stages. Dynamics and probe automatically resume the latest compatible
autoencoder unless `--experiment-id` is supplied:

```bash
atlas-research latent-train configs/research/latent_world_v0.yaml --stage autoencoder
atlas-research latent-train configs/research/latent_world_v0.yaml --stage dynamics
atlas-research latent-train configs/research/latent_world_v0.yaml --stage probe
atlas-research latent-evaluate <experiment-id>
```

`latent_world_v0_d32.yaml` provides the matching D=32 full-grid configuration. The full configs
use chronologically distributed bounded samples because this is a local CPU experiment, not a
claim to exhaustive training. Their spatial extent remains the complete CONUS grid. Selected
samples may be temporarily cached in capped RAM; they are never persisted as a second state
dataset. Estimated persistent output is printed before training. Latent artifacts have a separate
2 GiB cap and still count toward the existing 15 GiB dataset cap.

### Completed V0 results

The checked local run `20260831T221147Z-latent-d64-20c756af` used 64/16/16 chronological
train/validation/test timestamps on the complete grid, D=64, three epochs per stage, and 415,531
total parameters. It stores 27.9 MiB of artifacts; all five completed latent runs together use
about 34.0 MiB.

| Result | V0 D=64 |
| --- | ---: |
| Latent grid | 27×60×64 |
| Physical-to-latent element compression | 2.40× |
| Reconstruction normalized RMSE | 0.626 |
| Physical persistence normalized RMSE at +3h | 0.488 |
| Decoded latent persistence normalized RMSE | 0.692 |
| Decoded learned dynamics normalized RMSE | 0.661 |
| Latent persistence MSE | 0.346 |
| Learned latent prediction MSE | 0.208 |
| Frozen probe PR-AUC / ROC-AUC | 0.156 / 0.685 |
| Engineered-feature reference PR-AUC / ROC-AUC | 0.464 / 0.923 |

The learned dynamics improves decoded RMSE by about 4.6% over latent persistence and reduces
latent MSE by about 40%, demonstrating predictable structure in Z. It does **not** beat direct
physical persistence overall; its normalized RMSE is about 35.5% worse. Downward solar radiation
is a notable exception, with physical RMSE about 26.3% better than persistence. The frozen probe
contains measurable precipitation ranking signal but remains well behind the engineered-feature
model and is poorly calibrated. These are V0 findings, not evidence of a complete Earth model.

The controlled crop smoke ablation keeps all settings identical. D=32 provides 5.0× compression;
D=64 provides 2.5×. D=64 produced a substantially stronger frozen probe in this small run, while
neither size reliably beat physical persistence. That result supports testing representation size
without claiming that more dimensions are universally better.

Open **Experiments → Historical · Latent World Model** for model/config provenance, D=32/D=64
comparison, actual versus reconstructed fields, physical and latent persistence comparisons,
three-hour predictions and error maps, latent magnitude/change/prediction error, PCA projections,
nearest regional states, per-variable physical metrics, and frozen-probe calibration.

## Synthetic lab and tests

```bash
world-state-generate
pytest
ruff check src app tests
```

Tests use fixtures and mocked transports rather than depending on live services. They cover point,
grid, forecast, and event routing; GeoParquet metadata; catalog-based deduplication; point-in-time
availability; provider parsing; HTTP retry behavior; provenance; and Streamlit execution.
