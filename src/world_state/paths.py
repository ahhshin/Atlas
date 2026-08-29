from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OBSERVATIONS_PATH = DATA_DIR / "processed" / "environment.zarr"
FORECASTS_PATH = DATA_DIR / "predictions" / "forecasts.parquet"
METRICS_PATH = DATA_DIR / "predictions" / "metrics.parquet"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
POINT_OBSERVATIONS_DIR = NORMALIZED_DIR / "point_observations"
METADATA_DIR = DATA_DIR / "metadata"
INGESTION_LEDGER_PATH = METADATA_DIR / "ingestion.duckdb"
SOURCES_CONFIG_PATH = PROJECT_ROOT / "configs" / "sources.yaml"
