"""
FinSight AI - Experiment A: Temporal Sales Forecasting Benchmark (Q1 Protocol)
===============================================================================
5-Seed Purged Walk-Forward Evaluation with 7-Day Embargo & Zero-Leakage Guarantees

Datasets:
- Olist Platform (Macro B2C)
- UCI Online Retail (Wholesale B2B SME OOD)

Horizons:
- 7 Days, 14 Days, 30 Days

Seeds:
- [42, 101, 2024, 777, 999]

Metrics Reported:
- MAE, RMSE, WAPE (%), MASE, 80% Coverage (PICP %), MPIW, DM p-value (Harvey-Leybourne-Newbold corrected)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forecasting.features import build_forecasting_features, FEATURE_COLS
from src.forecasting.models import (
    LastValueNaive,
    SeasonalNaive,
    MovingAverageBaseline,
    LightGBMQuantileForecaster
)
from src.forecasting.metrics import calculate_forecasting_metrics, diebold_mariano_test

PROCESSED_DIR = r"d:\Others\data\processed"
RESULTS_DIR = r"d:\Others\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 101, 2024, 777, 999]


def evaluate_horizon_seeds(
    df_daily: pd.DataFrame,
    dataset_name: str,
    horizon: int
) -> List[Dict]:
    """
    Evaluates a specific dataset and horizon across 5 random seeds with purged walk-forward embargo.
    """
    # 1. Build anti-leakage feature store for target horizon H
    df_feat, feature_cols = build_forecasting_features(
        df=df_daily,
        target_col="sales",
        date_col="Date",
        horizon=horizon
    )

    n_samples = len(df_feat)
    
    # Purged Walk-Forward Split with exact 7-Day Embargo as specified in Section 4.4
    embargo = 7
    test_size = max(35, int(n_samples * 0.15))
    val_size = max(30, int(n_samples * 0.15))
    
    test_start = n_samples - test_size
    val_end = test_start - embargo
    val_start = val_end - val_size
    train_end = val_start - embargo

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[val_start:val_end]
    test_df = df_feat.iloc[test_start:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    y_test_arr = y_test.values
    y_train_arr = y_train.values

    # -------------------------------------------------------------------------
    # Baseline Predictions (Symmetric Rolling Evaluation with identical info set)
    # -------------------------------------------------------------------------
    # A0: Last observed value at origin t = T - horizon (stored in lag_0)
    preds_a0 = test_df["lag_0"].values
    metrics_a0 = calculate_forecasting_metrics(y_test_arr, preds_a0, y_train_history=y_train_arr)

    # A1: Seasonal Naive from preceding week at origin t (stored in lag_6)
    preds_a1 = test_df["lag_6"].values
    metrics_a1 = calculate_forecasting_metrics(y_test_arr, preds_a1, y_train_history=y_train_arr)

    # A2: 28-day Moving Average at origin t (stored in rolling_mean_28)
    preds_a2 = test_df["rolling_mean_28"].values
    metrics_a2 = calculate_forecasting_metrics(y_test_arr, preds_a2, y_train_history=y_train_arr)

    err_a1 = np.abs(y_test_arr - preds_a1)

    seed_records = []

    # Deterministic baselines
    base_rows = [
        {"Model ID": "A0", "Model Name": "Last-Value Naive", "metrics": metrics_a0, "err": np.abs(y_test_arr - preds_a0)},
        {"Model ID": "A1", "Model Name": "Seasonal Naive (7d)", "metrics": metrics_a1, "err": err_a1},
        {"Model ID": "A2", "Model Name": "EMA / Rolling MA (28d)", "metrics": metrics_a2, "err": np.abs(y_test_arr - preds_a2)},
    ]

    for b in base_rows:
        dm_stat, dm_pval = ("Ref", "Ref") if b["Model ID"] == "A1" else diebold_mariano_test(b["err"], err_a1, h=horizon)
        for s in SEEDS:
            seed_records.append({
                "Dataset": dataset_name,
                "Horizon": f"{horizon} Days",
                "Model ID": b["Model ID"],
                "Model Name": b["Model Name"],
                "Seed": s,
                "MAE": b["metrics"]["MAE"],
                "RMSE": b["metrics"]["RMSE"],
                "WAPE (%)": b["metrics"]["WAPE_pct"],
                "sMAPE (%)": b["metrics"]["sMAPE_pct"],
                "MASE": b["metrics"]["MASE"],
                "Coverage (%)": np.nan,
                "MPIW": np.nan,
                "DM Stat": dm_stat if dm_stat == "Ref" else float(dm_stat),
                "DM p-value": dm_pval if dm_pval == "Ref" else float(dm_pval)
            })

    # -------------------------------------------------------------------------
    # Model A3: LightGBM Quantile Forecaster across 5 Seeds
    # -------------------------------------------------------------------------
    for seed in SEEDS:
        lgbm = LightGBMQuantileForecaster(
            quantiles=(0.10, 0.50, 0.90),
            n_estimators=300,
            learning_rate=0.03,
            random_state=seed
        )
        lgbm.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        preds_dict = lgbm.predict(X_test)

        p10 = preds_dict["p10"]
        p50 = preds_dict["p50"]
        p90 = preds_dict["p90"]

        m_a3 = calculate_forecasting_metrics(
            y_true=y_test_arr,
            y_pred=p50,
            y_train_history=y_train_arr,
            y_lower=p10,
            y_upper=p90
        )

        err_a3 = np.abs(y_test_arr - p50)
        dm_stat, dm_pval = diebold_mariano_test(e1=err_a3, e2=err_a1, h=horizon)

        seed_records.append({
            "Dataset": dataset_name,
            "Horizon": f"{horizon} Days",
            "Model ID": "A3",
            "Model Name": "LightGBM Quantile (Ours)",
            "Seed": seed,
            "MAE": m_a3["MAE"],
            "RMSE": m_a3["RMSE"],
            "WAPE (%)": m_a3["WAPE_pct"],
            "sMAPE (%)": m_a3["sMAPE_pct"],
            "MASE": m_a3["MASE"],
            "Coverage (%)": m_a3["Coverage_pct"],
            "MPIW": m_a3["Mean_Interval_Width"],
            "DM Stat": float(dm_stat),
            "DM p-value": float(dm_pval)
        })

    return seed_records


def run_experiment_a():
    print("=" * 80)
    print("FINSIGHT AI — EXPERIMENT A: ZERO-LEAKAGE 5-SEED FORECASTING BENCHMARK")
    print(f"Random Seeds: {SEEDS} | 7-Day Purged Walk-Forward Embargo")
    print("=" * 80)

    olist_path = os.path.join(PROCESSED_DIR, "olist_daily_sales_platform.parquet")
    df_olist = pd.read_parquet(olist_path)

    uci_path = os.path.join(PROCESSED_DIR, "uci_daily_sales.parquet")
    df_uci = pd.read_parquet(uci_path)

    all_seed_records = []
    horizons = [7, 14, 30]

    for h in horizons:
        print(f"\n>> Running Olist Platform (B2C) | Horizon: {h} Days...")
        rec_olist = evaluate_horizon_seeds(df_olist, "Olist Platform (B2C)", h)
        all_seed_records.extend(rec_olist)

        print(f">> Running UCI Online Retail (B2B SME) | Horizon: {h} Days...")
        rec_uci = evaluate_horizon_seeds(df_uci, "UCI (B2B SME OOD)", h)
        all_seed_records.extend(rec_uci)

    df_raw = pd.DataFrame(all_seed_records)
    raw_csv = os.path.join(RESULTS_DIR, "exp_a_forecasting_benchmark_raw.csv")
    df_raw.to_csv(raw_csv, index=False)
    print(f"\n[Raw 5-seed benchmark saved to {raw_csv}]")

    summary_rows = []
    grouped = df_raw.groupby(["Dataset", "Horizon", "Model ID", "Model Name"], as_index=False)

    for (dataset, horizon, model_id, model_name), grp in grouped:
        mae_m, mae_s = grp["MAE"].mean(), grp["MAE"].std()
        rmse_m, rmse_s = grp["RMSE"].mean(), grp["RMSE"].std()
        wape_m, wape_s = grp["WAPE (%)"].mean(), grp["WAPE (%)"].std()
        smape_m, smape_s = grp["sMAPE (%)"].mean(), grp["sMAPE (%)"].std()
        mase_m, mase_s = grp["MASE"].mean(), grp["MASE"].std()

        cov_vals = grp["Coverage (%)"].dropna()
        cov_str = f"{cov_vals.mean():.2f}% ± {cov_vals.std():.2f}%" if len(cov_vals) > 0 else "N/A"

        mpiw_vals = grp["MPIW"].dropna()
        mpiw_str = f"{mpiw_vals.mean():.2f} ± {mpiw_vals.std():.2f}" if len(mpiw_vals) > 0 else "N/A"

        if "Ref" in grp["DM Stat"].values:
            dm_str = "Ref"
        else:
            dm_stat_m = grp["DM Stat"].astype(float).mean()
            dm_p_m = grp["DM p-value"].astype(float).mean()
            dm_str = f"{dm_p_m:.4f} (stat={dm_stat_m:.2f})"

        summary_rows.append({
            "Dataset": dataset,
            "Horizon": horizon,
            "Model ID": model_id,
            "Model Name": model_name,
            "MAE": f"{mae_m:.2f} ± {mae_s:.2f}" if mae_s > 0 else f"{mae_m:.2f}",
            "RMSE": f"{rmse_m:.2f} ± {rmse_s:.2f}" if rmse_s > 0 else f"{rmse_m:.2f}",
            "WAPE (%)": f"{wape_m:.2f}% ± {wape_s:.2f}%" if wape_s > 0 else f"{wape_m:.2f}%",
            "sMAPE (%)": f"{smape_m:.2f}% ± {smape_s:.2f}%" if smape_s > 0 else f"{smape_m:.2f}%",
            "MASE": f"{mase_m:.4f} ± {mase_s:.4f}" if mase_s > 0 else f"{mase_m:.4f}",
            "80% Coverage": cov_str,
            "MPIW": mpiw_str,
            "DM p-value": dm_str
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "exp_a_forecasting_benchmark.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"[Publication summary saved to {summary_csv}]")

    print("\n" + "=" * 90)
    print("Q1 TABLE 1: EXPERIMENT A — MULTI-HORIZON SALES FORECASTING BENCHMARK (5-SEED MEAN ± SD)")
    print("=" * 90)
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    run_experiment_a()
