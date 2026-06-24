"""Task 4: Train and evaluate an NLP spam detection model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df.drop_duplicates(inplace=True)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save() -> tuple[Pipeline, dict]:
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df["message"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["ham", "spam"]
        ),
        "test_messages": x_test.tolist(),
        "test_labels": y_test.tolist(),
        "test_probabilities": probabilities.tolist(),
        "training_records": len(x_train),
        "testing_records": len(x_test),
        "spam_share": float(df["target"].mean()),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return model, metrics


if __name__ == "__main__":
    _, result = train_and_save()
    print("Task 4 model trained and saved.")
    print(f"Accuracy:  {result['accuracy']:.3%}")
    print(f"Precision: {result['precision']:.3%}")
    print(f"Recall:    {result['recall']:.3%}")
    print(result["classification_report"])

