"""Interactive Streamlit dashboard for Task 2."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import EMPLOYED, PARTICIPATION, RATE, load_data

st.set_page_config(page_title="India Employment Pulse", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #f7fbff 0%, #eef8f3 100%);}
    [data-testid="stMetric"] {background:white; padding:18px; border-radius:16px;
    border:1px solid #deebe5; box-shadow:0 8px 22px rgba(19, 63, 48, .06);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def data():
    return load_data()


historical, recent = data()
st.title("India Employment Pulse")
st.caption("Task 2 · Unemployment Analysis with Python · 2019–2020")

area_options = ["All"] + sorted(historical["Area"].dropna().unique().tolist())
area = st.sidebar.selectbox("Area", area_options)
states = st.sidebar.multiselect(
    "States", sorted(historical["Region"].unique()), placeholder="All states"
)
filtered = historical.copy()
if area != "All":
    filtered = filtered[filtered["Area"] == area]
if states:
    filtered = filtered[filtered["Region"].isin(states)]

monthly = (
    filtered.groupby("Date", as_index=False)
    .agg({RATE: "mean", EMPLOYED: "sum", PARTICIPATION: "mean"})
    .sort_values("Date")
)
peak = monthly.loc[monthly[RATE].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average unemployment", f"{filtered[RATE].mean():.2f}%")
c2.metric("Peak unemployment", f"{peak[RATE]:.2f}%", peak["Date"].strftime("%b %Y"))
c3.metric("Average participation", f"{filtered[PARTICIPATION].mean():.2f}%")
c4.metric("Records analysed", f"{len(filtered):,}")

trend = px.line(
    monthly,
    x="Date",
    y=RATE,
    markers=True,
    title="How unemployment changed over time",
    color_discrete_sequence=["#0f766e"],
)
trend.add_vrect(
    x0="2020-03-01",
    x1="2020-05-31",
    fillcolor="#ef4444",
    opacity=0.12,
    line_width=0,
    annotation_text="COVID-19 lockdown",
)
st.plotly_chart(trend, use_container_width=True)

left, right = st.columns(2)
state_avg = (
    filtered.groupby("Region", as_index=False)[RATE]
    .mean()
    .sort_values(RATE, ascending=False)
)
left.plotly_chart(
    px.bar(
        state_avg.head(12),
        x=RATE,
        y="Region",
        orientation="h",
        title="States with highest average unemployment",
        color=RATE,
        color_continuous_scale="OrRd",
    ).update_layout(yaxis={"categoryorder": "total ascending"}),
    use_container_width=True,
)
area_avg = historical.groupby(["Date", "Area"], as_index=False)[RATE].mean()
right.plotly_chart(
    px.line(
        area_avg,
        x="Date",
        y=RATE,
        color="Area",
        markers=True,
        title="Rural and urban comparison",
        color_discrete_map={"Rural": "#16a34a", "Urban": "#f97316"},
    ),
    use_container_width=True,
)

st.subheader("2020 regional overview")
zone_avg = recent.groupby("Zone", as_index=False)[RATE].mean()
st.plotly_chart(
    px.bar(
        zone_avg,
        x="Zone",
        y=RATE,
        color="Zone",
        title="Average unemployment by zone",
    ),
    use_container_width=True,
)

with st.expander("View filtered data"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False),
        "filtered_unemployment.csv",
        "text/csv",
    )

