"""
RevIQ AI: FastAPI prediction service
Endpoints: /predict-churn, /revenue-risk, /health
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, CONFIG
from src.risk.revenue_risk_score import compute_risk_scores, label_risk
from src.explainability.shap_explainer import compute_shap_values, top_drivers_for_customer

app = FastAPI(
    title="RevIQ AI",
    description="SaaS Revenue Intelligence API",
    version="1.0.0",
)

# ── Model loading ────────────────────────────────────────────────────────────

_churn_model = None
_churn_features = None


def _get_churn_model():
    global _churn_model, _churn_features
    if _churn_model is None:
        path = MODELS_DIR / "churn_model.pkl"
        feat_path = MODELS_DIR / "churn_features.pkl"
        if not path.exists():
            raise HTTPException(503, "Model not trained yet. Run train_churn_model.py first.")
        _churn_model = joblib.load(path)
        _churn_features = joblib.load(feat_path)
    return _churn_model, _churn_features


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChurnRequest(BaseModel):
    customer_id: str
    mrr: float = Field(..., gt=0)
    arr: float = Field(..., gt=0)
    months_to_renewal: int = Field(..., ge=1, le=12)
    logins: float = 0
    active_users: float = 0
    feature_adoption: float = Field(0, ge=0, le=1)
    api_calls: float = 0
    tickets: float = 0
    sentiment: float = Field(0.5, ge=0, le=1)
    avg_resolution_time: float = 1.0
    nps: int = Field(7, ge=0, le=10)
    health_score: float = Field(0.7, ge=0, le=1)
    last_touch_days: int = 14
    csm_assigned: bool = False
    segment_Mid_Market: bool = False
    segment_SMB: bool = False


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    revenue_risk_score: float
    arr_at_risk: float
    top_drivers: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        _get_churn_model()
        model_ready = True
    except Exception:
        model_ready = False
    return HealthResponse(status="ok", model_loaded=model_ready, version="1.0.0")


@app.post("/predict-churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest):
    model, features = _get_churn_model()

    row = {f: 0.0 for f in features}
    for field, val in req.model_dump().items():
        key = field.replace("_Mid_Market", "_Mid-Market")
        if key in row:
            row[key] = float(val)

    # Derived features
    row["mrr_change_pct"] = 0.0
    row["logins_roll3m"] = row["logins"]
    row["logins_trend3m"] = 0.0
    row["feature_adoption_roll3m"] = row["feature_adoption"]
    row["tickets_roll3m"] = row["tickets"]
    row["sentiment_roll3m"] = row["sentiment"]
    row["mrr_lag1"] = row["mrr"]
    row["logins_lag1"] = row["logins"]
    row["tickets_lag1"] = row["tickets"]
    row["nps_lag1"] = row["nps"]
    row["health_score_lag1"] = row["health_score"]
    row["csm_assigned"] = float(req.csm_assigned)

    X = pd.DataFrame([row])[features].fillna(0)
    churn_prob = float(model.predict_proba(X)[:, 1][0])

    arr_series = pd.Series([req.arr])
    months_series = pd.Series([req.months_to_renewal])
    prob_series = pd.Series([churn_prob])

    risk_score = float(compute_risk_scores(prob_series, arr_series, months_series).iloc[0])
    risk_level = label_risk(pd.Series([risk_score])).iloc[0]
    arr_at_risk = round(churn_prob * req.arr, 0)

    # SHAP drivers
    try:
        sv = compute_shap_values(model, X)
        drivers = top_drivers_for_customer(sv, features, 0, top_n=5)
    except Exception:
        drivers = []

    return ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=round(churn_prob, 4),
        risk_level=risk_level,
        revenue_risk_score=risk_score,
        arr_at_risk=arr_at_risk,
        top_drivers=drivers,
    )


@app.get("/revenue-risk-summary")
def revenue_risk_summary():
    path = Path("data/reports/customer_risk_scores.csv")
    if not path.exists():
        raise HTTPException(404, "Risk scores not computed yet.")
    df = pd.read_csv(path)
    summary = df.groupby("risk_level").agg(
        customers=("customer_id", "count"),
        total_arr=("arr", "sum"),
        total_arr_at_risk=("arr_at_risk", "sum"),
    ).reset_index()
    return summary.to_dict(orient="records")
