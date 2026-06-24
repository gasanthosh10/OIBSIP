"""Task 5: Train and evaluate advertising-driven sales prediction models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Advertising.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sales_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.joblib"
FEATURES = ["TV", "Radio", "Newspaper"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def score_model(model, x_test, y_test) -> dict:
    predictions = model.predict(x_test)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "predictions": predictions.tolist(),
    }


def train_and_save():
    df = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["Sales"], test_size=0.2, random_state=42
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
        ),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        results[name] = score_model(model, x_test, y_test)

    best_name = max(results, key=lambda name: results[name]["r2"])
    best_model = candidates[best_name]
    metrics = {
        "best_model": best_name,
        "models": results,
        "test_actual": y_test.tolist(),
        "test_index": y_test.index.tolist(),
        "feature_correlations": df.corr(numeric_only=True)["Sales"]
        .drop("Sales")
        .to_dict(),
        "training_records": len(x_train),
        "testing_records": len(x_test),
    }
    if hasattr(best_model, "coef_"):
        metrics["feature_importance"] = dict(zip(FEATURES, best_model.coef_.tolist()))
    else:
        metrics["feature_importance"] = dict(
            zip(FEATURES, best_model.feature_importances_.tolist())
        )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)
    return best_model, metrics


if __name__ == "__main__":
    _, result = train_and_save()
    print(f"Best model: {result['best_model']}")
    for name, values in result["models"].items():
        print(
            f"{name}: R²={values['r2']:.3f}, "
            f"MAE={values['mae']:.3f}, RMSE={values['rmse']:.3f}"
        )

