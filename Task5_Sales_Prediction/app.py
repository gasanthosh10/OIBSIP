"""Interactive Streamlit sales predictor for Task 5."""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import (
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    load_data,
    train_and_save,
)

st.set_page_config(page_title="AdSpark Forecaster", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:linear-gradient(135deg,#fffaf1,#f4f0ff);}
    [data-testid="stMetric"] {background:white; padding:18px; border-radius:16px;
    border:1px solid #ece4ff; box-shadow:0 8px 24px rgba(81,45,168,.08);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH)


@st.cache_data
def dataset():
    return load_data()


model, metrics = artifacts()
df = dataset()
best = metrics["models"][metrics["best_model"]]

st.title("AdSpark Forecaster")
st.caption("Task 5 · Predict sales from advertising investment")

c1, c2, c3 = st.columns(3)
c1.metric("Selected model", metrics["best_model"])
c2.metric("Test R²", f"{best['r2']:.3f}")
c3.metric("Mean absolute error", f"{best['mae']:.2f} sales units")

left, right = st.columns([0.82, 1.18])
with left:
    st.subheader("Plan an advertising mix")
    tv = st.slider("TV budget", 0.0, 300.0, 150.0, 1.0)
    radio = st.slider("Radio budget", 0.0, 50.0, 25.0, 0.5)
    newspaper = st.slider("Newspaper budget", 0.0, 120.0, 30.0, 1.0)
    sample = pd.DataFrame([[tv, radio, newspaper]], columns=FEATURES)
    prediction = float(model.predict(sample)[0])
    st.metric("Predicted sales", f"{prediction:.2f} units")
    st.caption(
        "Budgets use the same units as the supplied dataset; sales are expressed "
        "in the dataset's sales units."
    )

with right:
    scatter = px.scatter(
        df,
        x="TV",
        y="Sales",
        size="Radio",
        color="Radio",
        hover_data=["Newspaper"],
        title="Sales relationship with TV and radio advertising",
        color_continuous_scale="Purples",
    )
    scatter.add_scatter(
        x=[tv],
        y=[prediction],
        mode="markers",
        marker={"size": 18, "color": "#ef4444", "symbol": "star"},
        name="Your forecast",
    )
    st.plotly_chart(scatter, use_container_width=True)

st.subheader("Model comparison")
comparison = pd.DataFrame(
    [
        {
            "Model": name,
            "R²": values["r2"],
            "MAE": values["mae"],
            "RMSE": values["rmse"],
        }
        for name, values in metrics["models"].items()
    ]
)
st.dataframe(
    comparison.style.format({"R²": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
    hide_index=True,
    use_container_width=True,
)

importance = pd.DataFrame(
    {
        "Channel": list(metrics["feature_importance"]),
        "Influence": list(metrics["feature_importance"].values()),
    }
).sort_values("Influence")
st.plotly_chart(
    px.bar(
        importance,
        x="Influence",
        y="Channel",
        orientation="h",
        title=f"Channel influence in {metrics['best_model']}",
        color="Influence",
        color_continuous_scale="Sunset",
    ),
    use_container_width=True,
)

st.info(
    "The model estimates association from historical advertising data. "
    "It should support—not replace—business judgement and controlled experiments."
)
