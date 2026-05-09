"""
Full RevIQ AI pipeline: generate data → features → train models → compute risk scores.
Run: python run_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def main():
    logger.info("=== RevIQ AI Pipeline ===")

    # Step 1: Generate synthetic SaaS data
    logger.info("Step 1/4: Generating synthetic SaaS data...")
    from src.data.generate_saas_data import generate_all
    from src.config import CONFIG
    generate_all(CONFIG)

    # Step 2: Build feature matrices
    logger.info("Step 2/4: Building feature matrices...")
    from src.features.build_features import load_raw, build_churn_features, build_arr_features, save_features
    from src.config import SYNTHETIC_DIR
    raw = load_raw(SYNTHETIC_DIR)
    churn_df = build_churn_features(raw)
    arr_df = build_arr_features(raw)
    save_features(churn_df, arr_df)

    # Step 3: Train models
    logger.info("Step 3/4: Training churn model...")
    from src.models.train_churn_model import run as train_churn
    churn_model = train_churn(CONFIG)

    logger.info("Step 3/4: Training ARR forecast model...")
    from src.models.train_arr_forecast import run as train_arr
    train_arr(CONFIG)

    # Step 4: Compute risk scores
    logger.info("Step 4/4: Computing revenue risk scores...")
    import joblib
    import numpy as np
    from src.config import PROCESSED_DIR, MODELS_DIR
    from src.features.build_features import get_model_matrix
    import pandas as pd

    churn_df_saved = pd.read_csv(PROCESSED_DIR / "churn_features.csv")
    X, y = get_model_matrix(churn_df_saved)
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    probs = model.predict_proba(X.fillna(0))[:, 1]

    from src.risk.revenue_risk_score import build_customer_risk_table
    build_customer_risk_table(churn_df_saved, probs, CONFIG)

    logger.success("=== Pipeline complete. Launch dashboard: streamlit run app/streamlit_app.py ===")


if __name__ == "__main__":
    main()
