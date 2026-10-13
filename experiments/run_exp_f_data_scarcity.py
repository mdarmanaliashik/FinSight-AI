"""
FinSight AI - Experiment F: Data Scarcity & Cold-Start Robustness (Q1 Protocol)
==============================================================================
Evaluates LightGBM Quantile Forecaster under restricted historical training windows:
[30 Days, 60 Days, 90 Days, 180 Days, 365 Days]

Populates Table 6 of the Q1 Paper.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.forecasting.features import build_forecasting_features
from src.forecasting.models import LightGBMQuantileForecaster
from src.forecasting.metrics import calculate_forecasting_metrics

RESULTS_DIR = r"d:\Others\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 101, 2024, 777, 999]


def run_experiment_f():
    print("=" * 80)
    print("FINSIGHT AI — EXPERIMENT F: DATA SCARCITY BENCHMARK")
    print("=" * 80)

    data_path = os.path.join(r"d:\Others\data\processed", "olist_daily_sales_platform.parquet")
    df_raw = pd.read_parquet(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    df_feat, feat_cols = build_forecasting_features(df_raw, target_col="sales", date_col="Date", horizon=7)

    # Test set: final 60 days
    test_df = df_feat.iloc[-60:].reset_index(drop=True)
    X_test, y_test = test_df[feat_cols], test_df["target"].values

    history_windows = [30, 60, 90, 180, 365]
    records = []

    for days in history_windows:
        print(f">> Training with past {days} days of history...")
        # Train slice is `days` rows immediately prior to test set (with 7-day embargo)
        end_idx = len(df_feat) - 60 - 7
        start_idx = max(0, end_idx - days)
        train_df = df_feat.iloc[start_idx:end_idx].reset_index(drop=True)

        X_train, y_train = train_df[feat_cols], train_df["target"]

        wape_list, mase_list, cov_list = [], [], []

        for s in SEEDS:
            lgbm = LightGBMQuantileForecaster(
                quantiles=(0.10, 0.50, 0.90),
                n_estimators=200,
                learning_rate=0.03,
                random_state=s
            )
            lgbm.fit(X_train, y_train)
            preds = lgbm.predict(X_test)

            m = calculate_forecasting_metrics(
                y_true=y_test,
                y_pred=preds["p50"],
                y_train_history=y_train.values,
                y_lower=preds["p10"],
                y_upper=preds["p90"]
            )
            wape_list.append(m["WAPE_pct"])
            mase_list.append(m["MASE"])
            cov_list.append(m["Coverage_pct"])

        records.append({
            "Training History": f"{days} Days",
            "WAPE (%)": f"{np.mean(wape_list):.2f}% ± {np.std(wape_list):.2f}%",
            "MASE": f"{np.mean(mase_list):.4f} ± {np.std(mase_list):.4f}",
            "80% Coverage": f"{np.mean(cov_list):.2f}% ± {np.std(cov_list):.2f}%"
        })

    df_f = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "exp_f_data_scarcity_benchmark.csv")
    df_f.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("Q1 TABLE 6: EXPERIMENT F — DATA SCARCITY & COLD START BENCHMARK")
    print("=" * 80)
    print(df_f.to_string(index=False))
    print(f"\n[Saved to {out_csv}]")


if __name__ == "__main__":
    run_experiment_f()
