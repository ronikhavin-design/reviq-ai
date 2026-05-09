"""
Computes a composite Revenue Risk Score per customer.
Combines churn probability, ARR exposure, and renewal urgency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CONFIG, PROCESSED_DIR, REPORTS_DIR


def compute_risk_scores(
    churn_probs: pd.Series,
    arr: pd.Series,
    months_to_renewal: pd.Series,
    cfg: dict | None = None,
) -> pd.Series:
    if cfg is None:
        cfg = CONFIG

    w = cfg["risk_score"]["weights"]

    # Normalize ARR exposure to 0–1 across customers
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    # Renewal urgency: high if renewal in <= 3 months
    renewal_urgency = (months_to_renewal <= 3).astype(float)
    # Partial urgency for 4–6 months
    renewal_urgency = renewal_urgency + ((months_to_renewal <= 6) & (months_to_renewal > 3)).astype(float) * 0.5

    score = (
        w["churn_probability"] * churn_probs
        + w["arr_exposure"] * arr_norm
        + w["renewal_urgency"] * renewal_urgency.clip(0, 1)
    )
    return score.clip(0, 1).round(4)


def label_risk(score: pd.Series, cfg: dict | None = None) -> pd.Series:
    if cfg is None:
        cfg = CONFIG
    thresholds = cfg["risk_score"]["thresholds"]
    return score.apply(
        lambda s: "High" if s >= thresholds["high"] else ("Medium" if s >= thresholds["medium"] else "Low")
    )


def build_customer_risk_table(
    churn_df: pd.DataFrame,
    churn_probs: np.ndarray,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Build per-customer risk snapshot using their most recent monthly record."""
    df = churn_df.copy().reset_index(drop=True)
    df["churn_probability"] = churn_probs
    latest = df.sort_values("month").groupby("customer_id").last().reset_index()

    latest["revenue_risk_score"] = compute_risk_scores(
        latest["churn_probability"],
        latest["arr"],
        latest["months_to_renewal"],
        cfg,
    )
    latest["risk_level"] = label_risk(latest["revenue_risk_score"], cfg)
    latest["arr_at_risk"] = (latest["churn_probability"] * latest["arr"]).round(0)

    cols = [
        "customer_id", "mrr", "arr", "months_to_renewal",
        "churn_probability", "revenue_risk_score", "risk_level", "arr_at_risk",
        "nps", "health_score", "last_touch_days",
    ]
    available = [c for c in cols if c in latest.columns]
    result = latest[available].sort_values("revenue_risk_score", ascending=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPORTS_DIR / "customer_risk_scores.csv", index=False)
    logger.info(f"Risk table: {len(result)} customers. High risk: {(result['risk_level']=='High').sum()}")

    return result


if __name__ == "__main__":
    import joblib
    df = pd.read_csv(PROCESSED_DIR / "churn_features.csv")
    from src.features.build_features import get_model_matrix
    X, y = get_model_matrix(df)
    model = joblib.load("models/churn_model.pkl")
    probs = model.predict_proba(X.fillna(0))[:, 1]
    build_customer_risk_table(df, probs)
