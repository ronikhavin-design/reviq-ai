"""
SHAP-based explainability for churn predictions.
Translates model outputs into human-readable business drivers.
"""

import shap
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import REPORTS_DIR, PROCESSED_DIR


FEATURE_LABELS = {
    "mrr": "Monthly Revenue",
    "arr": "Annual Revenue",
    "months_to_renewal": "Months to Renewal",
    "logins": "Monthly Logins",
    "active_users": "Active Users",
    "feature_adoption": "Feature Adoption Rate",
    "api_calls": "API Calls",
    "logins_roll3m": "Avg Logins (3M)",
    "logins_trend3m": "Login Trend (3M)",
    "feature_adoption_roll3m": "Avg Feature Adoption (3M)",
    "tickets": "Support Tickets",
    "tickets_roll3m": "Total Tickets (3M)",
    "sentiment": "Support Sentiment",
    "sentiment_roll3m": "Avg Sentiment (3M)",
    "avg_resolution_time": "Avg Resolution Time (days)",
    "nps": "Net Promoter Score",
    "health_score": "Customer Health Score",
    "last_touch_days": "Days Since Last CS Touch",
    "csm_assigned": "CS Manager Assigned",
    "mrr_change_pct": "MRR Change %",
}


def get_explainer(model, X_sample: pd.DataFrame) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    explainer = get_explainer(model, X)
    return explainer.shap_values(X)


def top_drivers_for_customer(
    shap_values: np.ndarray,
    feature_names: list[str],
    customer_idx: int,
    top_n: int = 5,
) -> list[dict]:
    sv = shap_values[customer_idx]
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [
        {
            "feature": FEATURE_LABELS.get(name, name),
            "raw_feature": name,
            "shap_value": round(float(val), 4),
            "direction": "increases risk" if val > 0 else "decreases risk",
        }
        for name, val in pairs
    ]


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "raw_feature": feature_names,
        "mean_abs_shap": mean_abs.round(4),
    }).sort_values("mean_abs_shap", ascending=False)
    return df


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    save_dir: Path | None = None,
) -> Path:
    """Horizontal bar chart of mean |SHAP| per feature. Saved as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importance = global_feature_importance(shap_values, feature_names).head(top_n)
    labels = importance["feature"].tolist()[::-1]
    values = importance["mean_abs_shap"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * top_n + 1.5)))
    bars = ax.barh(labels, values, color="#2196F3", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (average impact on churn prediction)", fontsize=11)
    ax.set_title(f"Global SHAP Feature Importance — Top {top_n} Churn Drivers", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_dir = (save_dir or REPORTS_DIR) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_global_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def plot_customer_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    customer_idx: int,
    customer_id: str,
    churn_prob: float,
    top_n: int = 10,
    save_dir: Path | None = None,
) -> Path:
    """Diverging bar chart of per-customer SHAP values. Red = increases risk, blue = decreases risk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv = shap_values[customer_idx]
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels = [FEATURE_LABELS.get(name, name) for name, _ in pairs][::-1]
    values = [float(val) for _, val in pairs][::-1]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * top_n + 1.5)))
    ax.barh(labels, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (positive = pushes toward churn, negative = reduces risk)", fontsize=11)
    ax.set_title(
        f"Churn Risk Drivers — Customer {customer_id}  ({churn_prob:.0%} predicted risk)",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_dir = (save_dir or REPORTS_DIR) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shap_customer_{customer_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")
    return out_path


def format_churn_explanation(customer_id: str, churn_prob: float, drivers: list[dict]) -> str:
    lines = [
        f"Customer {customer_id}: {churn_prob:.0%} churn risk",
        "",
        "Main risk drivers:",
    ]
    for i, d in enumerate(drivers, 1):
        sign = "↑" if d["shap_value"] > 0 else "↓"
        lines.append(f"  {i}. {d['feature']}: {sign} {d['direction']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import joblib
    from src.features.build_features import get_model_matrix

    df = pd.read_csv(PROCESSED_DIR / "churn_features.csv")
    X, y = get_model_matrix(df)
    model = joblib.load("models/churn_model.pkl")

    logger.info("Computing SHAP values...")
    sv = compute_shap_values(model, X.fillna(0))

    importance = global_feature_importance(sv, X.columns.tolist())
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    logger.success("SHAP importance saved.")
    print(importance.head(10).to_string(index=False))
