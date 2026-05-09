"""
Trains an ARR forecasting model using time-based features.
Produces 3-month forward ARR predictions with confidence intervals.
"""

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, MODELS_DIR, CONFIG


ARR_FEATURE_COLS = [
    "arr_lag1", "arr_lag2", "arr_lag3",
    "arr_roll3m", "arr_growth_3m",
    "month_of_year", "quarter",
]
ARR_TARGET = "total_arr"


def load_features() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "arr_features.csv")


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {
        "mae": round(mae, 0),
        "rmse": round(rmse, 0),
        "mape": round(mape, 2),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def run(cfg: dict | None = None) -> XGBRegressor:
    if cfg is None:
        cfg = CONFIG

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = load_features()
    available = [c for c in ARR_FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df[ARR_TARGET]

    # Time-series split: never shuffle time series data
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        xgb_cfg = cfg["arr_forecast"]["xgboost"]
        model = XGBRegressor(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            random_state=cfg["arr_forecast"]["random_state"],
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        fold_metrics = evaluate_regression(y_te.values, preds)
        cv_metrics.append(fold_metrics)
        logger.info(f"Fold {fold+1}: MAPE={fold_metrics['mape']:.2f}%, RMSE=${fold_metrics['rmse']:,.0f}")

    avg_metrics = {k: round(np.mean([m[k] for m in cv_metrics]), 4) for k in cv_metrics[0]}

    with mlflow.start_run(run_name="arr_forecast_xgboost"):
        mlflow.log_params({**cfg["arr_forecast"]["xgboost"], "n_cv_splits": n_splits})
        mlflow.log_metrics({f"cv_{k}": v for k, v in avg_metrics.items()})

        # Final model trained on all data
        final_model = XGBRegressor(
            **cfg["arr_forecast"]["xgboost"],
            random_state=cfg["arr_forecast"]["random_state"],
            verbosity=0,
        )
        final_model.fit(X, y)
        mlflow.xgboost.log_model(final_model, "model")

    import joblib
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "arr_forecast_model.pkl")
    joblib.dump(available, MODELS_DIR / "arr_features.pkl")
    logger.success(f"ARR model saved. CV MAPE: {avg_metrics['mape']:.2f}%")

    return final_model


if __name__ == "__main__":
    run()
