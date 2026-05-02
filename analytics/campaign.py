import os
import joblib

ARTIFACTS_DIR = "artifacts"


def get_campaign_data() -> dict:
    cache = joblib.load(os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl"))
    return {
        "monthly_revenue": cache["campaign_monthly"],
        "top_products": cache["campaign_top_products"],
    }
