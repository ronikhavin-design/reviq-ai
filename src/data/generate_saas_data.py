"""
Generates synthetic SaaS company data: customers, subscriptions, usage, support, and CS.
Designed to realistically simulate churn dynamics, ARR movements, and risk signals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import SYNTHETIC_DIR, CONFIG


SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
INDUSTRIES = ["SaaS", "FinTech", "Healthcare", "Retail", "Manufacturing", "Media"]
COUNTRIES = ["US", "UK", "Germany", "Israel", "France", "Canada", "Australia"]
PLANS = ["Starter", "Growth", "Professional", "Enterprise"]

PLAN_MRR = {
    "Starter": (500, 1_500),
    "Growth": (1_500, 5_000),
    "Professional": (5_000, 15_000),
    "Enterprise": (15_000, 80_000),
}

SEGMENT_PLAN_WEIGHTS = {
    "SMB": [0.55, 0.30, 0.12, 0.03],
    "Mid-Market": [0.10, 0.35, 0.40, 0.15],
    "Enterprise": [0.00, 0.05, 0.25, 0.70],
}

BASE_CHURN_BY_SEGMENT = {"SMB": 0.22, "Mid-Market": 0.12, "Enterprise": 0.06}


def _seed(cfg: dict) -> int:
    return cfg["data"]["random_seed"]


def generate_customers(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(cfg))
    n = cfg["data"]["n_customers"]

    segments = rng.choice(SEGMENTS, size=n, p=[0.50, 0.35, 0.15])
    plans = np.array([
        rng.choice(PLANS, p=SEGMENT_PLAN_WEIGHTS[seg]) for seg in segments
    ])
    start_months_ago = rng.integers(1, cfg["data"]["n_months"], size=n)

    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "segment": segments,
        "plan": plans,
        "industry": rng.choice(INDUSTRIES, size=n),
        "country": rng.choice(COUNTRIES, size=n, p=[0.40, 0.15, 0.12, 0.10, 0.10, 0.08, 0.05]),
        "start_months_ago": start_months_ago,
        "csm_assigned": rng.choice([True, False], size=n, p=[0.30, 0.70]),
    })
    return df


def _mrr_for_plan(plan: str, rng: np.random.Generator) -> float:
    lo, hi = PLAN_MRR[plan]
    return round(rng.uniform(lo, hi), -2)


def generate_subscriptions(customers: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(cfg) + 1)
    n_months = cfg["data"]["n_months"]
    rows = []

    for _, cust in customers.iterrows():
        base_mrr = _mrr_for_plan(cust["plan"], rng)
        base_churn_p = BASE_CHURN_BY_SEGMENT[cust["segment"]]
        active_months = n_months - cust["start_months_ago"]
        churned = False
        churned_month = None

        for m in range(1, active_months + 1):
            if churned:
                break

            # Gradual MRR drift (expansion or contraction)
            growth = rng.normal(0.005, 0.015)
            mrr = round(base_mrr * (1 + growth * m), -2)
            arr = mrr * 12

            # Renewal every 12 months
            months_to_renewal = 12 - (m % 12)

            # Churn probability increases near renewal and with low NPS/usage (added later via join)
            churn_p = base_churn_p * (1.5 if months_to_renewal <= 2 else 1.0)
            will_churn = rng.random() < churn_p / 12  # monthly probability

            if will_churn and m > 3:  # no churn in first 3 months
                churned = True
                churned_month = m

            rows.append({
                "customer_id": cust["customer_id"],
                "month": m,
                "mrr": mrr,
                "arr": arr,
                "months_to_renewal": months_to_renewal,
                "churned": int(will_churn and m > 3),
            })

    return pd.DataFrame(rows)


def generate_product_usage(customers: pd.DataFrame, subscriptions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(cfg) + 2)
    rows = []

    sub_summary = subscriptions.groupby("customer_id").agg(
        max_month=("month", "max"),
        ever_churned=("churned", "max"),
    ).reset_index()

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        row = sub_summary[sub_summary["customer_id"] == cid]
        if row.empty:
            continue
        max_month = int(row["max_month"].iloc[0])
        will_churn = bool(row["ever_churned"].iloc[0])

        # Churning customers show declining usage trajectory
        for m in range(1, max_month + 1):
            if will_churn:
                decay = max(0.0, 1.0 - (m / max_month) * rng.uniform(0.3, 0.8))
            else:
                decay = rng.uniform(0.7, 1.1)

            base_logins = {"SMB": 15, "Mid-Market": 40, "Enterprise": 120}[cust["segment"]]
            logins = max(0, int(rng.normal(base_logins * decay, base_logins * 0.15)))
            active_users = max(1, int(logins * rng.uniform(0.3, 0.7)))
            feature_adoption = round(min(1.0, rng.beta(2, 3) * decay), 3)
            api_calls = max(0, int(rng.exponential(500 * decay)))

            rows.append({
                "customer_id": cid,
                "month": m,
                "logins": logins,
                "active_users": active_users,
                "feature_adoption": feature_adoption,
                "api_calls": api_calls,
            })

    return pd.DataFrame(rows)


def generate_support(customers: pd.DataFrame, subscriptions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(cfg) + 3)
    rows = []

    sub_summary = subscriptions.groupby("customer_id").agg(
        max_month=("month", "max"),
        ever_churned=("churned", "max"),
    ).reset_index()

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        row = sub_summary[sub_summary["customer_id"] == cid]
        if row.empty:
            continue
        max_month = int(row["max_month"].iloc[0])
        will_churn = bool(row["ever_churned"].iloc[0])

        for m in range(1, max_month + 1):
            # Churning customers open more tickets
            base_tickets = 0.8 if will_churn else 0.3
            tickets = rng.poisson(base_tickets)
            sentiment = round(rng.beta(3, 2 if not will_churn else 4), 3)  # 0–1, lower = worse
            resolution_time = round(rng.exponential(2.5 if will_churn else 1.2), 1)

            rows.append({
                "customer_id": cid,
                "month": m,
                "tickets": tickets,
                "sentiment": sentiment,
                "avg_resolution_time": resolution_time,
            })

    return pd.DataFrame(rows)


def generate_customer_success(customers: pd.DataFrame, subscriptions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(cfg) + 4)
    rows = []

    sub_summary = subscriptions.groupby("customer_id").agg(
        max_month=("month", "max"),
        ever_churned=("churned", "max"),
    ).reset_index()

    for _, cust in customers.iterrows():
        cid = cust["customer_id"]
        row = sub_summary[sub_summary["customer_id"] == cid]
        if row.empty:
            continue
        max_month = int(row["max_month"].iloc[0])
        will_churn = bool(row["ever_churned"].iloc[0])

        for m in range(1, max_month + 1):
            nps = int(rng.normal(6 if not will_churn else 4, 2))
            nps = max(0, min(10, nps))
            health = round(rng.beta(5 if not will_churn else 2, 2 if not will_churn else 5), 3)
            # Churning customers less likely to have recent CS touchpoint
            last_touch = int(rng.exponential(15 if not will_churn else 45))

            rows.append({
                "customer_id": cid,
                "month": m,
                "nps": nps,
                "health_score": health,
                "last_touch_days": last_touch,
                "csm_assigned": cust["csm_assigned"],
            })

    return pd.DataFrame(rows)


def generate_revenue_targets(cfg: dict) -> pd.DataFrame:
    n_months = cfg["data"]["n_months"]
    base_arr = 30_000_000
    rows = []
    for m in range(1, n_months + 1):
        target = round(base_arr * (1 + 0.02 * m), -3)
        actual = round(target * np.random.uniform(0.88, 1.05), -3)
        rows.append({
            "month": m,
            "target_arr": target,
            "actual_arr": actual,
            "forecast_arr": round(actual * np.random.uniform(0.97, 1.03), -3),
        })
    return pd.DataFrame(rows)


def generate_all(cfg: dict | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    if cfg is None:
        from src.config import CONFIG
        cfg = CONFIG

    logger.info("Generating synthetic SaaS dataset...")

    customers = generate_customers(cfg)
    subscriptions = generate_subscriptions(customers, cfg)
    usage = generate_product_usage(customers, subscriptions, cfg)
    support = generate_support(customers, subscriptions, cfg)
    cs = generate_customer_success(customers, subscriptions, cfg)
    targets = generate_revenue_targets(cfg)

    datasets = {
        "customers": customers,
        "subscriptions": subscriptions,
        "product_usage": usage,
        "support": support,
        "customer_success": cs,
        "targets": targets,
    }

    if save:
        SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in datasets.items():
            path = SYNTHETIC_DIR / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Saved {name}.csv — {len(df):,} rows")

    logger.success("Data generation complete.")
    return datasets


if __name__ == "__main__":
    generate_all()
