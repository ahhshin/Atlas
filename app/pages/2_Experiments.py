from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_state.store import load_metrics
from world_state.synthetic import SYNTHETIC_VARIABLES

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("Experiments")
st.caption("Synthetic baseline evaluation · lower RMSE is better")

metrics = load_metrics()
target = st.selectbox(
    "Target",
    list(SYNTHETIC_VARIABLES),
    format_func=lambda name: SYNTHETIC_VARIABLES[name]["label"],
)
filtered = metrics.loc[metrics.target == target].copy()

fig = px.line(
    filtered,
    x="forecast_horizon_hours",
    y="rmse",
    color="model",
    markers=True,
    labels={
        "forecast_horizon_hours": "Forecast horizon (hours)",
        "rmse": f"RMSE ({SYNTHETIC_VARIABLES[target]['unit']})",
    },
)
fig.update_layout(height=430)
st.plotly_chart(fig, width="stretch")

st.dataframe(
    filtered[["model", "forecast_horizon_hours", "mae", "rmse", "bias", "samples"]],
    hide_index=True,
    width="stretch",
    column_config={
        "forecast_horizon_hours": st.column_config.NumberColumn("Horizon", format="%d h"),
        "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.3f"),
        "bias": st.column_config.NumberColumn("Bias", format="%+.3f"),
    },
)
