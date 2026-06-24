"""Interactive Streamlit spam detector for Task 4."""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from train_model import METRICS_PATH, MODEL_PATH, train_and_save

st.set_page_config(page_title="Inbox Shield", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at top right, #172554, #081426 55%);}
    h1,h2,h3,p,label,.stMarkdown {color:#eef6ff !important;}
    [data-testid="stMetric"] {background:#12243d; padding:18px; border-radius:16px;
    border:1px solid #23476d;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(METRICS_PATH)


model, metrics = get_artifacts()
st.title("Inbox Shield")
st.caption("Task 4 · NLP-powered email and SMS spam detection")

m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Spam precision", f"{metrics['precision']:.1%}")
m3.metric("Spam recall", f"{metrics['recall']:.1%}")

left, right = st.columns([1.35, 0.65])
with left:
    st.subheader("Check a message")
    example = st.selectbox(
        "Try an example",
        [
            "Write my own message",
            "Congratulations! You won a FREE prize. Call now to claim.",
            "Hi, are we still meeting at 6 pm near the library?",
            "URGENT! Your account was selected for a cash reward. Click now.",
        ],
    )
    initial = "" if example == "Write my own message" else example
    message = st.text_area(
        "Message text",
        value=initial,
        height=180,
        placeholder="Paste an email or SMS message here...",
    )
    if st.button("Analyse message", type="primary", use_container_width=True):
        if not message.strip():
            st.warning("Enter a message before analysing it.")
        else:
            spam_probability = float(model.predict_proba([message])[0, 1])
            if spam_probability >= 0.5:
                st.error(f"Likely spam · {spam_probability:.1%} confidence")
                st.write("Be cautious with links, requests for money, and personal details.")
            else:
                st.success(f"Likely legitimate · {1-spam_probability:.1%} confidence")
            st.progress(spam_probability, text=f"Spam probability: {spam_probability:.1%}")

with right:
    st.subheader("How it works")
    st.markdown(
        """
        1. Text is converted into TF-IDF word and phrase features.
        2. Logistic regression estimates the probability of spam.
        3. The 50% decision threshold assigns the final label.

        The model was trained with class balancing because spam is less common
        than legitimate messages.
        """
    )

st.subheader("Evaluation")
cm = pd.DataFrame(
    metrics["confusion_matrix"],
    index=["Actual ham", "Actual spam"],
    columns=["Predicted ham", "Predicted spam"],
)
fig = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    title="Confusion matrix on unseen test messages",
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Educational model: predictions assist judgement but should not replace "
    "a production email security system."
)

