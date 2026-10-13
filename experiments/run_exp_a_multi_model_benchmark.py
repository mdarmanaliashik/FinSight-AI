"""
FinSight AI - Comprehensive Multi-Model Benchmark & Multicollinearity Audit
============================================================================
Kaggle Grandmaster Grade Rigorous Comparison

Evaluates:
1. Multicollinearity: Variance Inflation Factor (VIF) & Correlation Matrix
2. Models Tested:
   - Linear Regression (Ridge L2 Regularized)
   - Decision Tree Regressor (DTR)
   - Support Vector Regressor (SVR - RBF kernel)
   - LightGBM Quantile Forecaster
   - Weighted Stacking Ensemble (Ridge + LightGBM + DTR)
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forecasting.features import build_forecasting_features
from src.forecasting.models import LightGBMQuantileForecaster
from src.forecasting.metrics import calculate_forecasting_metrics

PROCESSED_DIR = r"d:\Others\data\processed"
RESULTS_DIR = r"d:\Others\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def check_multicollinearity(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) for all numerical features.
    Rule of thumb:
    - VIF < 5: Low multicollinearity
    - VIF 5-10: Moderate multicollinearity
    - VIF > 10: High multicollinearity (critical for Linear Models, benign for Trees)
    """
    print("\n" + "="*70)
    print("STEP 1: MULTICOLLINEARITY AUDIT (VARIANCE INFLATION FACTOR - VIF)")
    print("="*70)

    # Standardize to avoid numeric instability during VIF calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif_val = variance_inflation_factor(X_scaled, i)
        except Exception:
            vif_val = np.nan
        vif_data.append({
            "Feature": col,
            "VIF": round(float(vif_val), 2),
            "Status": "High Collinearity (>10)" if vif_val > 10 else ("Moderate (5-10)" if vif_val > 5 else "Low (<5)")
        })

    vif_df = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return vif_df


def run_multi_model_benchmark():
    # Load Olist Platform Series
    olist_path = os.path.join(PROCESSED_DIR, "olist_daily_sales_platform.parquet")
    df_olist = pd.read_parquet(olist_path)

    # Horizon: 7 days
    horizon = 7
    df_feat, feature_cols = build_forecasting_features(
        df=df_olist,
        target_col="sales",
        date_col="Date",
        horizon=horizon
    )

    n_samples = len(df_feat)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    # 1. Run VIF Check
    vif_results = check_multicollinearity(X_train)
    print(vif_results.head(15).to_string(index=False))
    vif_results.to_csv(os.path.join(RESULTS_DIR, "exp_a_vif_report.csv"), index=False)

    print("\n" + "="*70)
    print("STEP 2: MULTI-MODEL BENCHMARK (LR, SVR, DTR, LIGHTGBM, ENSEMBLE)")
    print("="*70)

    y_test_arr = y_test.values
    y_train_arr = y_train.values
    model_records = []

    # -------------------------------------------------------------------------
    # Model 1: Ridge Linear Regression (Handles collinearity via L2 penalty)
    # -------------------------------------------------------------------------
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=100.0, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    preds_lr = np.maximum(0.0, lr_pipe.predict(X_test))
    m_lr = calculate_forecasting_metrics(y_test_arr, preds_lr, y_train_history=y_train_arr)
    model_records.append({
        "Model Family": "Linear",
        "Model Name": "Ridge Linear Regression (LR)",
        "MAE": m_lr["MAE"],
        "RMSE": m_lr["RMSE"],
        "WAPE (%)": m_lr["WAPE_pct"],
        "MASE": m_lr["MASE"]
    })

    # -------------------------------------------------------------------------
    # Model 2: Decision Tree Regressor (DTR)
    # -------------------------------------------------------------------------
    dtr = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    dtr.fit(X_train, y_train)
    preds_dtr = np.maximum(0.0, dtr.predict(X_test))
    m_dtr = calculate_forecasting_metrics(y_test_arr, preds_dtr, y_train_history=y_train_arr)
    model_records.append({
        "Model Family": "Single Tree",
        "Model Name": "Decision Tree Regressor (DTR)",
        "MAE": m_dtr["MAE"],
        "RMSE": m_dtr["RMSE"],
        "WAPE (%)": m_dtr["WAPE_pct"],
        "MASE": m_dtr["MASE"]
    })

    # -------------------------------------------------------------------------
    # Model 3: Support Vector Regressor (SVR)
    # -------------------------------------------------------------------------
    svr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=1000.0, epsilon=50.0, kernel="rbf"))
    ])
    svr_pipe.fit(X_train, y_train)
    preds_svr = np.maximum(0.0, svr_pipe.predict(X_test))
    m_svr = calculate_forecasting_metrics(y_test_arr, preds_svr, y_train_history=y_train_arr)
    model_records.append({
        "Model Family": "Kernel SVM",
        "Model Name": "Support Vector Regressor (SVR)",
        "MAE": m_svr["MAE"],
        "RMSE": m_svr["RMSE"],
        "WAPE (%)": m_svr["WAPE_pct"],
        "MASE": m_svr["MASE"]
    })

    # -------------------------------------------------------------------------
    # Model 4: LightGBM (Gradient Boosted Trees - Our Baseline)
    # -------------------------------------------------------------------------
    lgbm = LightGBMQuantileForecaster(quantiles=(0.10, 0.50, 0.90), n_estimators=300, learning_rate=0.03)
    lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds_lgbm = lgbm.predict(X_test)["p50"]
    m_lgbm = calculate_forecasting_metrics(y_test_arr, preds_lgbm, y_train_history=y_train_arr)
    model_records.append({
        "Model Family": "GBDT Ensemble",
        "Model Name": "LightGBM Quantile (P50)",
        "MAE": m_lgbm["MAE"],
        "RMSE": m_lgbm["RMSE"],
        "WAPE (%)": m_lgbm["WAPE_pct"],
        "MASE": m_lgbm["MASE"]
    })

    # -------------------------------------------------------------------------
    # Model 5: Weighted Stacking Ensemble (LightGBM 60% + Ridge 25% + DTR 15%)
    # -------------------------------------------------------------------------
    preds_ensemble = 0.60 * preds_lgbm + 0.25 * preds_lr + 0.15 * preds_dtr
    m_ens = calculate_forecasting_metrics(y_test_arr, preds_ensemble, y_train_history=y_train_arr)
    model_records.append({
        "Model Family": "Hybrid Ensemble",
        "Model Name": "Weighted Ensemble (LGBM+Ridge+DTR)",
        "MAE": m_ens["MAE"],
        "RMSE": m_ens["RMSE"],
        "WAPE (%)": m_ens["WAPE_pct"],
        "MASE": m_ens["MASE"]
    })

    # Comparative DataFrame
    res_df = pd.DataFrame(model_records).sort_values(by="MAE").reset_index(drop=True)
    print("\n" + "="*80)
    print("FINAL HEAD-TO-HEAD MODEL COMPARISON RESULTS:")
    print("="*80)
    print(res_df.to_string(index=False))

    res_df.to_csv(os.path.join(RESULTS_DIR, "exp_a_multi_model_benchmark.csv"), index=False)
    print(f"\n Results saved to: {os.path.join(RESULTS_DIR, 'exp_a_multi_model_benchmark.csv')}")


if __name__ == "__main__":
    run_multi_model_benchmark()
