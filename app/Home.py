from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.paths import OBSERVATIONS_PATH

st.set_page_config(page_title="World State", page_icon="🌎", layout="wide")
st.title("World State")
st.caption("Environmental observations, forecasts, and reproducible experiments")

if not OBSERVATIONS_PATH.exists():
    st.warning("Synthetic demo artifacts are missing. Run `world-state-generate` to create them.")

st.markdown(
    """
World State keeps live public observations and deterministic research fixtures separate.

- **World State** opens in live mode when real ECCC or NWS observations have been ingested.
- **Synthetic forecast** is an explicit offline/demo mode; it is never labelled as real data.
- **Experiments** compares persistence and climatology on the synthetic fixture.
- **Data Feeds** reports attempts, successes, freshness, and errors from the ingestion ledger.

The Streamlit process only reads normalized data. Collection runs independently through
`world-state-ingest` or the scheduled `world-state-worker` process.
"""
)

st.code("world-state-ingest fetch all\nstreamlit run app/Home.py", language="bash")
