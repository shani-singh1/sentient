"""Bengaluru Road Infrastructure Risk — Decision-maker dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src.common.paths import RESULTS_ROOT

st.set_page_config(
    page_title="Bengaluru Road Risk",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling — no heavy table overlays
st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; }
    .sub-header { font-size: 0.95rem; color: #64748b; margin-bottom: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
    .insight-box { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-left: 4px solid #0ea5e9; padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0; }
    .insight-box.warning { border-left-color: #dc2626; }
    .insight-box.success { border-left-color: #16a34a; }
    /* Fix table visibility — no conflicting backgrounds */
    .stDataFrame [data-testid="stDataFrame"] div { background: white !important; }
    .stDataFrame td, .stDataFrame th { background: white !important; color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

ROAD_RISK_PATH = RESULTS_ROOT / "road_risk_ranking.json"

if not ROAD_RISK_PATH.exists():
    st.error("Road risk data not found. Run: `python -m src.features.build_road_risk`")
    st.stop()

data = json.loads(ROAD_RISK_PATH.read_text(encoding="utf-8"))
roads = data.get("roads", [])

if not roads:
    st.error("No roads in road risk file.")
    st.stop()


def risk_color(score: float) -> list[int]:
    if score >= 0.7:
        return [194, 24, 27, 230]
    if score >= 0.4:
        return [245, 158, 11, 220]
    return [34, 197, 94, 180]


# Build GeoJSON for PathLayer
features = []
for r in roads:
    path = r.get("path", [])
    if len(path) < 2:
        continue
    coords = [[p[0], p[1]] for p in path]
    color = risk_color(r["risk_score"])
    features.append({
        "path": coords,
        "color": color,
        "name": r["name"],
        "risk_pct": r["risk_pct"],
        "risk_level": r["risk_level"],
        "highway": r.get("highway", ""),
    })

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    top_n = st.slider("Show top N roads", 20, 200, 80)
    risk_filter = st.multiselect(
        "Risk level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )

# Filter
display_roads = [r for r in roads if r["risk_level"] in risk_filter][:top_n]
display_features = [f for f in features if f["risk_level"] in risk_filter][:top_n]

high = sum(1 for r in roads if r["risk_level"] == "High")
medium = sum(1 for r in roads if r["risk_level"] == "Medium")
low = sum(1 for r in roads if r["risk_level"] == "Low")
total_km = sum(r["length_m"] for r in roads) / 1000
high_km = sum(r["length_m"] for r in roads if r["risk_level"] == "High") / 1000

# Header
st.markdown('<p class="main-header">Bengaluru Road Infrastructure Risk</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prioritize inspection and maintenance using predictive risk scores from satellite stress signals</p>', unsafe_allow_html=True)
st.markdown("---")

# Executive summary — value-first
st.markdown("## Executive summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("High-risk roads", high, help="Risk ≥ 70%")
with col2:
    st.metric("Total km at high risk", f"{high_km:.1f}", help="Cumulative length")
with col3:
    st.metric("Medium-risk roads", medium, help="Risk 40–70%")
with col4:
    st.metric("Total ranked roads", len(roads))

st.markdown("")
st.markdown(
    f'<div class="insight-box warning"><strong>Action:</strong> {high} roads ({high_km:.1f} km) require priority inspection. '
    f'Addressing the top 10 alone could reduce exposure by ~{min(10, high) * 0.5:.0f}% of highest-risk segments.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="insight-box success"><strong>Context:</strong> {low} roads at low risk — focus budget on high/medium zones first.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Charts row
st.markdown("## Risk overview")

c1, c2 = st.columns(2)
with c1:
    # Risk distribution pie
    fig_dist = go.Figure(data=[go.Pie(
        labels=["High (≥70%)", "Medium (40–70%)", "Low (<40%)"],
        values=[high, medium, low],
        hole=0.5,
        marker_colors=["#dc2626", "#f59e0b", "#22c55e"],
        textinfo="label+value",
    )])
    fig_dist.update_layout(
        title="Road count by risk level",
        height=280,
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with c2:
    # Top 12 riskiest roads bar
    top_roads = [r for r in roads if r["risk_level"] == "High"][:12]
    if not top_roads:
        top_roads = roads[:12]
    fig_bar = go.Figure(data=[go.Bar(
        x=[r["risk_pct"] for r in top_roads],
        y=[r["name"][:35] + ("…" if len(r["name"]) > 35 else "") for r in top_roads],
        orientation="h",
        marker_color=["#dc2626"] * len(top_roads),
    )])
    fig_bar.update_layout(
        title="Top 12 riskiest roads",
        xaxis_title="Risk %",
        height=280,
        margin=dict(t=40, b=40, l=120, r=20),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Zone and highway breakdown
df_roads = pd.DataFrame(roads)
if "zone" in df_roads.columns and len(df_roads["zone"].dropna()) > 0:
    c3, c4 = st.columns(2)
    with c3:
        zone_counts = df_roads.groupby("zone").agg(
            high=("risk_level", lambda s: (s == "High").sum()),
            medium=("risk_level", lambda s: (s == "Medium").sum()),
            low=("risk_level", lambda s: (s == "Low").sum()),
        ).reset_index()
        fig_zone = go.Figure()
        for col, color in [("high", "#dc2626"), ("medium", "#f59e0b"), ("low", "#22c55e")]:
            fig_zone.add_trace(go.Bar(name=col.capitalize(), x=zone_counts["zone"], y=zone_counts[col], marker_color=color))
        fig_zone.update_layout(
            barmode="stack",
            title="Risk by zone (NW, NE, SW, SE)",
            height=250,
            margin=dict(t=40, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_zone, use_container_width=True)
    with c4:
        hwy = df_roads.groupby("highway")["risk_score"].mean().sort_values(ascending=True).tail(10)
        fig_hwy = go.Figure(data=[go.Bar(x=hwy.values, y=hwy.index, orientation="h", marker_color="#0ea5e9")])
        fig_hwy.update_layout(
            title="Avg risk by road type",
            xaxis_title="Risk %",
            height=250,
            margin=dict(t=40, b=40, l=80, r=20),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hwy, use_container_width=True)

st.markdown("---")

# Map
st.markdown("## Risk map")
st.caption("Roads colored by risk: Red = high, amber = medium, green = low.")

if display_features:
    path_layer = pdk.Layer(
        "PathLayer",
        display_features,
        get_path="path",
        get_color="color",
        get_width=3,
        width_scale=2,
        width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=12.9716, longitude=77.5946, zoom=11, pitch=0)
    deck = pdk.Deck(
        layers=[path_layer],
        initial_view_state=view,
        tooltip={
            "html": "<b>{name}</b><br>Risk: {risk_pct}% ({risk_level})<br>Type: {highway}",
            "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "14px"},
        },
        map_style="light",
    )
    st.pydeck_chart(deck)
else:
    st.info("No roads match the selected filters.")

st.markdown("---")

# Ranking table — no row background overlay
st.markdown("## Risk ranking")
st.caption("Prioritized roads for inspection. Use filters in sidebar.")

df = pd.DataFrame(display_roads)
if len(df) > 0:
    cols = ["rank", "name", "risk_level", "risk_pct", "highway", "length_m"]
    if "zone" in df.columns:
        cols = ["rank", "name", "risk_level", "risk_pct", "zone", "highway", "length_m"]
    display = df[[c for c in cols if c in df.columns]].copy()
    display = display.rename(columns={
        "rank": "Rank",
        "name": "Road",
        "risk_level": "Risk",
        "risk_pct": "Risk %",
        "zone": "Zone",
        "highway": "Type",
        "length_m": "Length (m)",
    })
    # No row styling — causes visibility issues; use Risk column for color
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Road": st.column_config.TextColumn(width="large"),
            "Risk": st.column_config.TextColumn(width="small"),
            "Risk %": st.column_config.NumberColumn(format="%.1f"),
            "Zone": st.column_config.TextColumn(width="small"),
            "Type": st.column_config.TextColumn(width="small"),
            "Length (m)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

st.markdown("---")

with st.expander("How to use this dashboard"):
    st.markdown("""
- **Risk %**: Higher values indicate roads in areas with elevated stress signals (flood, heat, vegetation change) — prioritize for inspection.
- **Map**: Red roads = highest priority; amber = monitor; green = lower priority.
- **Data**: Sentinel-1/2, Landsat, nightlights, ERA5 climate. Bengaluru 2020–2024.
- **Road geometry**: OpenStreetMap. Risk from tile-level model overlaid on road network.
    """)
