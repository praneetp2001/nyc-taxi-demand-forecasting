"""Streamlit dashboard for replaying NYC taxi demand forecasts by zone."""
import datetime as dt

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="NYC Taxi Demand Forecast", layout="wide")
st.title("NYC Taxi Demand Forecast")
st.caption("LightGBM quantile forecasts over a held-out evaluation window. Bubble size and color show P50 demand.")


@st.cache_data(ttl=300)
def get_meta():
    return requests.get(f"{API_BASE}/meta", timeout=10).json()


@st.cache_data(ttl=300)
def get_zones():
    return pd.DataFrame(requests.get(f"{API_BASE}/zones", timeout=10).json())


def get_predictions(timestamp: str):
    r = requests.get(f"{API_BASE}/predict/all", params={"timestamp": timestamp}, timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def get_explanation(zone_id: int, timestamp: str):
    r = requests.post(
        f"{API_BASE}/predict",
        json={"zone_id": zone_id, "timestamp": timestamp},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


try:
    meta = get_meta()
except requests.exceptions.ConnectionError:
    st.error("Cannot reach the FastAPI service. Start it with:\n\n`uvicorn src.api.main:app --port 8000`")
    st.stop()

min_ts = pd.Timestamp(meta["min_timestamp"])
max_ts = pd.Timestamp(meta["max_timestamp"])

hours = pd.date_range(min_ts, max_ts, freq="h")
labels = [h.strftime("%Y-%m-%d %H:00") for h in hours]

col1, col2 = st.columns([3, 1])
with col1:
    selected_label = st.select_slider("Hour", options=labels, value=labels[len(labels) // 2])
selected_ts = pd.Timestamp(dt.datetime.strptime(selected_label, "%Y-%m-%d %H:00"))

zones = get_zones()
preds = get_predictions(selected_ts.isoformat())
merged = zones.merge(preds, on="zone_id", how="inner")
merged["uncertainty"] = merged["p90"] - merged["p10"]

map_col, detail_col = st.columns([2, 1])

with map_col:
    fig = px.scatter_mapbox(
        merged,
        lat="centroid_lat",
        lon="centroid_lon",
        size=merged["p50"].clip(lower=0.5),
        color="p50",
        color_continuous_scale="Viridis",
        hover_name="zone",
        hover_data={"borough": True, "p10": ":.1f", "p50": ":.1f", "p90": ":.1f", "centroid_lat": False, "centroid_lon": False},
        zoom=9.3,
        mapbox_style="carto-positron",
        height=600,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with detail_col:
    st.subheader("Zone detail")
    zone_options = dict(zip(merged["zone"].fillna("Unknown") + " (" + merged["borough"].fillna("?") + ")", merged["zone_id"]))
    selected_zone_label = st.selectbox("Zone", sorted(zone_options.keys()))
    selected_zone_id = zone_options[selected_zone_label]

    row = merged[merged["zone_id"] == selected_zone_id].iloc[0]
    st.metric("P50 forecast", f"{row['p50']:.1f} trips")
    st.write(f"P10 - P90 range: **{row['p10']:.1f} - {row['p90']:.1f}**")

    st.subheader("Why this prediction?")
    explanation = get_explanation(int(selected_zone_id), selected_ts.isoformat())
    contrib_df = pd.DataFrame(explanation["top_features"]).set_index("feature")
    fig2 = px.bar(
        contrib_df.iloc[::-1],
        x="shap_value",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    fig2.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)
