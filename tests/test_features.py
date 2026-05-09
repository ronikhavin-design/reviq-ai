import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.data.generate_saas_data import generate_customers, generate_subscriptions, generate_revenue_targets
from src.features.build_features import build_churn_features, get_model_matrix, build_arr_features
from src.risk.revenue_risk_score import compute_risk_scores, label_risk
from src.config import CONFIG


@pytest.fixture
def cfg():
    c = CONFIG.copy()
    c["data"]["n_customers"] = 100
    c["data"]["n_months"] = 12
    return c


@pytest.fixture
def raw_data(cfg):
    from src.data.generate_saas_data import (
        generate_customers, generate_subscriptions,
        generate_product_usage, generate_support, generate_customer_success,
    )
    customers = generate_customers(cfg)
    subs = generate_subscriptions(customers, cfg)
    usage = generate_product_usage(customers, subs, cfg)
    support = generate_support(customers, subs, cfg)
    cs = generate_customer_success(customers, subs, cfg)
    return {
        "customers": customers,
        "subscriptions": subs,
        "product_usage": usage,
        "support": support,
        "customer_success": cs,
    }


def test_customer_generation(cfg):
    df = generate_customers(cfg)
    assert len(df) == cfg["data"]["n_customers"]
    assert "customer_id" in df.columns
    assert df["segment"].isin(["SMB", "Mid-Market", "Enterprise"]).all()


def test_subscription_generation(cfg, raw_data):
    subs = raw_data["subscriptions"]
    assert "mrr" in subs.columns
    assert "churned" in subs.columns
    assert (subs["mrr"] > 0).all()
    assert subs["churned"].isin([0, 1]).all()


def test_churn_features_shape(raw_data):
    df = build_churn_features(raw_data)
    assert len(df) > 0
    assert "churned" in df.columns
    assert "mrr" in df.columns


def test_model_matrix_no_nans(raw_data):
    df = build_churn_features(raw_data)
    X, y = get_model_matrix(df)
    assert not X.isnull().all(axis=None)
    assert y.isin([0, 1]).all()
    assert len(X) == len(y)


def test_risk_score_bounds():
    churn_p = pd.Series([0.0, 0.5, 1.0])
    arr = pd.Series([10000, 50000, 200000])
    months = pd.Series([12, 6, 1])
    scores = compute_risk_scores(churn_p, arr, months)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_risk_label_values():
    scores = pd.Series([0.1, 0.5, 0.9])
    labels = label_risk(scores)
    assert set(labels).issubset({"Low", "Medium", "High"})


def test_revenue_targets_shape(cfg):
    df = generate_revenue_targets(cfg)
    assert len(df) == cfg["data"]["n_months"]
    assert "target_arr" in df.columns
    assert "actual_arr" in df.columns
