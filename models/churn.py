import os
import joblib
import numpy as np

ARTIFACTS_DIR = "artifacts"

FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "AOV",
    "CancelRate", "Tenure", "UniqueProducts", "ChannelEncoded",
]


def get_churn_results() -> dict:
    cache = joblib.load(os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl"))
    return {
        "top50": cache["churn_top50"],
        "feature_importance": cache["churn_feature_importance"],
        "feature_names": cache["churn_feature_names"],
    }


def predict_churn(features: dict) -> float:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "churn_model.pkl"))
    X = np.array([[features.get(f, 0.0) for f in FEATURE_COLS]])
    return float(model.predict_proba(X)[0][1])
