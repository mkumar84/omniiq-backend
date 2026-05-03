"""OmniIQ FastAPI server — 8 endpoints."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm.narrator import narrate_attribution, narrate_churn, narrate_segmentation
from llm.query_engine import answer_query
from llm.recommender import recommend_campaigns

ARTIFACTS_DIR = "artifacts"
_cache: dict[str, Any] = {}
_narrators: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache_path = os.path.join(ARTIFACTS_DIR, "analytics_cache.pkl")
    if os.path.exists(cache_path):
        _cache.update(joblib.load(cache_path))
        print("Analytics cache loaded successfully.")
    else:
        print("WARNING: artifacts/analytics_cache.pkl not found — run ingest.py first.")
    yield


app = FastAPI(title="OmniIQ API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_cache():
    if not _cache:
        raise HTTPException(503, "Analytics not ready — run ingest.py first.")


class QueryRequest(BaseModel):
    question: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_loaded": bool(_cache),
        "model": "claude-sonnet-4-6",
        "timestamp": datetime.now(EASTERN).isoformat(),
    }


@app.get("/summary")
def summary():
    _require_cache()
    return {
        "total_revenue": round(_cache["total_revenue"], 2),
        "total_customers": _cache["total_customers"],
        "avg_order_value": round(_cache["avg_order_value"], 2),
        "churn_rate_pct": round(_cache["churn_rate"], 1),
        "top_attribution_channel": _cache["top_channel"],
        "largest_segment": _cache["largest_segment"],
    }


def _safe_narrate(key: str, fn, *args) -> str:
    """Call a narrator function, cache the result, return fallback on error."""
    if key not in _narrators:
        try:
            _narrators[key] = fn(*args)
        except Exception as exc:
            print(f"Narrator '{key}' failed: {exc}")
            _narrators[key] = "AI insight temporarily unavailable."
    return _narrators[key]


@app.get("/segmentation")
def segmentation():
    _require_cache()
    return {
        "segments": _cache["segmentation"],
        "insight": _safe_narrate("segmentation", narrate_segmentation, _cache["segmentation"]),
    }


@app.get("/attribution")
def attribution():
    _require_cache()
    return {
        "channels": _cache["attribution"],
        "insight": _safe_narrate("attribution", narrate_attribution, _cache["attribution"]),
    }


@app.get("/churn")
def churn():
    _require_cache()
    churn_payload = {
        "top50": _cache["churn_top50"],
        "feature_importance": _cache["churn_feature_importance"],
        "feature_names": _cache["churn_feature_names"],
        "churn_rate_pct": round(_cache["churn_rate"], 1),
    }
    return {**churn_payload, "insight": _safe_narrate("churn", narrate_churn, churn_payload)}


@app.get("/campaign")
def campaign():
    _require_cache()
    return {
        "monthly_revenue": _cache["campaign_monthly"],
        "top_products": _cache["campaign_top_products"],
    }


@app.post("/query")
def query(req: QueryRequest):
    _require_cache()
    try:
        answer = answer_query(req.question, _cache)
    except Exception as exc:
        print(f"Query engine error: {exc}")
        answer = "Unable to process query — AI service temporarily unavailable."
    return {
        "question": req.question,
        "answer": answer,
        "timestamp": datetime.now(EASTERN).isoformat(),
    }


@app.get("/recommendations/{segment}")
def recommendations(segment: str):
    _require_cache()
    seg_data = next(
        (s for s in _cache["segmentation"] if s["Segment"].lower() == segment.lower()),
        None,
    )
    if not seg_data:
        valid = [s["Segment"] for s in _cache["segmentation"]]
        raise HTTPException(404, f"Segment '{segment}' not found. Valid: {valid}")
    try:
        recs = recommend_campaigns(seg_data)
    except Exception as exc:
        print(f"Recommender error: {exc}")
        recs = "Campaign recommendations temporarily unavailable."
    return {"segment": segment, "recommendations": recs}
