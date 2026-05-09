"""
Trains churn prediction models with full MLflow experiment tracking.
Compares Logistic Regression baseline vs XGBoost, logs metrics and artifacts.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR, CONFIG
from src.features.build_features import get_model_matrix


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / "churn_features.csv")
    return get_model_matrix(df)


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


def train_logistic_baseline(X_train, y_train, X_test, y_test, cfg: dict) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=cfg["data"]["random_seed"])
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob)

    with mlflow.start_run(run_name="logistic_baseline", nested=True):
        mlflow.log_params({"model": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    logger.info(f"Logistic baseline: ROC-AUC: {metrics['roc_auc']}, PR-AUC: {metrics['pr_auc']}")
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test, cfg: dict) -> tuple[XGBClassifier, dict]:
    xgb_cfg = cfg["churn_model"]["xgboost"]
    model = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        eval_metric=xgb_cfg["eval_metric"],
        random_state=cfg["data"]["random_seed"],
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_prob)

    with mlflow.start_run(run_name="xgboost_churn", nested=True):
        mlflow.log_params({**xgb_cfg, "model": "xgboost"})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

    logger.info(f"XGBoost: ROC-AUC: {metrics['roc_auc']}, PR-AUC: {metrics['pr_auc']}")
    return model, metrics


def run(cfg: dict | None = None) -> XGBClassifier:
    if cfg is None:
        cfg = CONFIG

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["churn_model"]["test_size"],
        random_state=cfg["churn_model"]["random_state"],
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}, Churn rate: {y.mean():.2%}")

    with mlflow.start_run(run_name="churn_experiment"):
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("churn_rate", round(float(y.mean()), 4))

        train_logistic_baseline(X_train, y_train, X_test, y_test, cfg)
        best_model, best_metrics = train_xgboost(X_train, y_train, X_test, y_test, cfg)

        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

    # Save model locally
    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "churn_features.pkl")
    logger.success("Churn model saved.")

    return best_model


if __name__ == "__main__":
    run()
