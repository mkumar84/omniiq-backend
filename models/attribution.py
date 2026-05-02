import os
import joblib

ARTIFACTS_DIR = "artifacts"


def get_attribution_results() -> list[dict]:
    cache = joblib.load(os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl"))
    return cache["attribution"]
