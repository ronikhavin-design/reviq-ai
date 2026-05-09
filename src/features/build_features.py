"""
Joins all raw tables and engineers features for churn prediction and ARR forecasting.
Each feature has a clear business interpretation to support SHAP explainability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SYNTHETIC_DIR, PROCESSED_DIR


def load_raw(synthetic_dir: Path = SYNTHETIC_DIR) -> dict[str, pd.DataFrame]:
    tables = ["customers", "subscriptions", "product_usage", "support", "customer_success"]
    return {t: pd.read_csv(synthetic_dir / f"{t}.csv") for t in tables}


def _rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll{w}m"] = (
            df.groupby("customer_id")[col]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"{col}_trend{w}m"] = (
            df.groupby("customer_id")[col]
            .transform(lambda x: x.rolling(w, min_periods=2).apply(
                lambda v: (v.iloc[-1] - v.iloc[0]) / (v.iloc[0] + 1e-6), raw=False
            ))
        )
    return df


def build_churn_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    subs = raw["subscriptions"].copy()
    usage = raw["product_usage"].copy()
    support = raw["support"].copy()
    cs = raw["customer_success"].copy()
    customers = raw["customers"].copy()

    # Rolling features on usage
    usage = _rolling_features(usage, "logins", [3])
    usage = _rolling_features(usage, "feature_adoption", [3])

    # Rolling features on support
    support["tickets_roll3m"] = (
        support.groupby("customer_id")["tickets"]
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
    )
    support["sentiment_roll3m"] = (
        support.groupby("customer_id")["sentiment"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # Join all monthly signals
    merged = subs.merge(usage, on=["customer_id", "month"], how="left")
    merged = merged.merge(support, on=["customer_id", "month"], how="left")
    merged = merged.merge(cs, on=["customer_id", "month"], how="left")
    merged = merged.merge(customers, on="customer_id", how="left")

    # Lag features: what happened last month?
    for col in ["mrr", "logins", "tickets", "nps", "health_score"]:
        merged[f"{col}_lag1"] = merged.groupby("customer_id")[col].shift(1)

    # MRR change
    merged["mrr_change_pct"] = (merged["mrr"] - merged["mrr_lag1"]) / (merged["mrr_lag1"] + 1e-6)

    # Categorical encoding
    merged = pd.get_dummies(merged, columns=["segment", "plan", "industry", "country"], drop_first=True)

    # Boolean to int
    if "csm_assigned" in merged.columns:
        merged["csm_assigned"] = merged["csm_assigned"].astype(int)

    merged = merged.dropna(subset=["logins_lag1"])  # remove first month per customer

    logger.info(f"Churn feature matrix: {merged.shape}")
    return merged


CHURN_FEATURE_COLS = [
    "mrr", "arr", "months_to_renewal",
    "logins", "active_users", "feature_adoption", "api_calls",
    "logins_roll3m", "logins_trend3m", "feature_adoption_roll3m",
    "tickets", "tickets_roll3m", "sentiment", "sentiment_roll3m", "avg_resolution_time",
    "nps", "health_score", "last_touch_days", "csm_assigned",
    "mrr_lag1", "logins_lag1", "tickets_lag1", "nps_lag1", "health_score_lag1",
    "mrr_change_pct", "month",
]

TARGET_COL = "churned"


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in CHURN_FEATURE_COLS if c in df.columns]
    # Include one-hot encoded columns
    dummy_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["segment_", "plan_", "industry_", "country_"]
    )]
    feature_cols = available + dummy_cols
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]
    return X, y


def build_arr_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    targets = pd.read_csv(SYNTHETIC_DIR / "targets.csv")
    subs = raw["subscriptions"].copy()

    monthly_arr = (
        subs[subs["churned"] == 0]
        .groupby("month")["arr"]
        .sum()
        .reset_index()
        .rename(columns={"arr": "total_arr"})
    )

    df = monthly_arr.merge(targets, on="month", how="left")
    df["arr_lag1"] = df["total_arr"].shift(1)
    df["arr_lag2"] = df["total_arr"].shift(2)
    df["arr_lag3"] = df["total_arr"].shift(3)
    df["arr_roll3m"] = df["total_arr"].rolling(3, min_periods=1).mean()
    df["arr_growth_3m"] = (df["total_arr"] - df["arr_lag3"]) / (df["arr_lag3"] + 1e-6)
    df["month_of_year"] = ((df["month"] - 1) % 12) + 1
    df["quarter"] = ((df["month_of_year"] - 1) // 3) + 1

    df = df.dropna(subset=["arr_lag1"])
    logger.info(f"ARR feature matrix: {df.shape}")
    return df


def save_features(churn_df: pd.DataFrame, arr_df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    churn_df.to_csv(PROCESSED_DIR / "churn_features.csv", index=False)
    arr_df.to_csv(PROCESSED_DIR / "arr_features.csv", index=False)
    logger.success("Feature files saved.")


if __name__ == "__main__":
    raw = load_raw()
    churn_df = build_churn_features(raw)
    arr_df = build_arr_features(raw)
    save_features(churn_df, arr_df)
