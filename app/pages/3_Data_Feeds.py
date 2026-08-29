from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.config import load_config
from world_state.ingest.ledger import IngestionLedger

st.set_page_config(page_title="Data feeds", page_icon="◉", layout="wide")
st.title("Data Feeds")
st.caption("Status is derived from the ingestion ledger and configured source cadence.")


@st.cache_data(ttl=30)
def health_table():
    return IngestionLedger().source_health(load_config())


health = health_table()
if health.empty:
    st.info("No data sources are configured.")
else:
    symbols = {
        "healthy": "● Healthy",
        "partial": "◐ Partial",
        "stale": "◐ Stale",
        "failed": "● Failed",
        "unavailable": "○ Unavailable",
        "disabled": "○ Disabled",
    }
    display = health.copy()
    display["status"] = display.status.map(lambda value: symbols.get(value, value))
    for column in ["last_attempt", "last_success", "data_current_through"]:
        display[column] = display[column].map(
            lambda value: (
                "—"
                if value is None or pd.isna(value)
                else pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
            )
        )
    display["error"] = display.error.fillna("—")
    st.dataframe(
        display.rename(
            columns={
                "source": "Source",
                "status": "Status",
                "last_attempt": "Last attempt",
                "last_success": "Last success",
                "data_current_through": "Data current through",
                "records_or_objects": "Records/objects",
                "error": "Latest error",
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.markdown("Run a one-time refresh with `world-state-ingest fetch all`.")
