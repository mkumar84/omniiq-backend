import os
import joblib
import numpy as np

ARTIFACTS_DIR = "artifacts"


def get_segmentation_results() -> list[dict]:
    cache = joblib.load(os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl"))
    return cache["segmentation"]


def predict_segment(recency: float, frequency: float, monetary: float) -> str:
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(ARTIFACTS_DIR, "kmeans_model.pkl"))
    X_scaled = scaler.transform(np.array([[recency, frequency, monetary]]))
    return int(kmeans.predict(X_scaled)[0])
