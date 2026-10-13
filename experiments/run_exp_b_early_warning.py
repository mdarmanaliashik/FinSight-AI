"""
FinSight AI - Experiment B: Cash-Flow Risk & Early Warning Benchmark (Q1 Protocol)
==================================================================================
5-Seed Walk-Forward Evaluation with 100% Zero-Leakage Recursive Rollout

Architectures Evaluated:
- B0: Static Threshold (Reactive, alerts only when current cash < reserve limit)
- B1: Point-Forecast Extrapolation (P50 Median Sales Recursive Rollout)
- B2: P10 Downside Quantile Stress Engine (FinSight AI Core, P10 Recursive Rollout)
- B3: 7-Day Net Cash Burn Run-Rate Baseline

Metrics:
- Precision, Recall, F1-Score, PR-AUC
- Average Early Warning Lead Time (L_warning in days)
- Monthly False Alarms (False Positive Rate per 30 days)
- Shortfall Amount Prediction MAE (R$)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cashflow.engine import SMEConfig, CashFlowEngine
from src.forecasting.features import build_forecasting_features
from src.forecasting.models import LightGBMQuantileForecaster

PROJECT_ROOT = r"d:\Others"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 101, 2024, 777, 999]


def run_experiment_b():
    print("=" * 85)
    print("FINSIGHT AI — EXPERIMENT B: CASH-FLOW RISK & EARLY WARNING BENCHMARK")
    print(f"Random Seeds: {SEEDS} | Zero-Leakage Recursive Autoregressive Rollout")
    print("=" * 85)

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "olist_daily_sales_platform.parquet")
    df_raw = pd.read_parquet(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # 1. 1-Step Horizon Features for Recursive Base Model Training
    df_feat, feat_cols = build_forecasting_features(df_raw, target_col="sales", date_col="Date", horizon=1)
    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:].copy().reset_index(drop=True)

    X_train, y_train = train_df[feat_cols], train_df["target"]
    X_val, y_val = val_df[feat_cols], val_df["target"]

    # 2. SME Configuration for Realistic Retail Merchant
    config = SMEConfig(
        initial_cash=60000.0,
        min_reserve=50000.0,
        monthly_payroll=140000.0,
        monthly_rent=35000.0,
        monthly_utilities_misc=15000.0,
        monthly_debt_service=20000.0,
        cogs_ratio=0.74,
        supplier_terms_lag=15,
        sales_settlement_lag=2
    )

    # 3. Ground-Truth Daily Ledger over Holdout
    cf_engine = CashFlowEngine(config=config)
    true_ledger = cf_engine.simulate(sales_series=test_df["target"], dates=test_df["Date"])
    test_df["true_closing_cash"] = true_ledger["closing_cash"]
    test_df["true_is_breach"] = true_ledger["is_reserve_breach"]
    test_df["true_shortfall"] = true_ledger["shortfall_amount"]

    forecast_horizon = 21
    num_steps = len(test_df) - forecast_horizon

    # Pre-extract full historical sales up to test_df start for recursive warm-up
    warmup_history = list(df_raw[df_raw["Date"] < test_df["Date"].iloc[0]]["sales"].values)

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n[Seed {seed}] Training 1-step Quantile LightGBM...")
        model = LightGBMQuantileForecaster(
            quantiles=(0.10, 0.50, 0.90),
            n_estimators=250,
            learning_rate=0.03,
            random_state=seed
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        res = {
            "B0": {"alerts": [], "actuals": [], "lead_times": [], "shortfall_errs": [], "probs": []},
            "B1": {"alerts": [], "actuals": [], "lead_times": [], "shortfall_errs": [], "probs": []},
            "B2": {"alerts": [], "actuals": [], "lead_times": [], "shortfall_errs": [], "probs": []},
            "B3": {"alerts": [], "actuals": [], "lead_times": [], "shortfall_errs": [], "probs": []}
        }

        for t in range(num_steps):
            curr_cash = float(test_df.loc[t, "true_closing_cash"])
            future_window = test_df.iloc[t+1 : t+1+forecast_horizon].reset_index(drop=True)
            future_dates = future_window["Date"]

            # Ground truth breach occurrence in the next 21 days
            breach_in_window = bool(future_window["true_is_breach"].any())
            actual_first_breach_idx = future_window.index[future_window["true_is_breach"]].tolist()
            actual_lead = float(actual_first_breach_idx[0] + 1) if actual_first_breach_idx else 0.0
            actual_max_shortfall = float(future_window["true_shortfall"].max())

            # Historical sales strictly observed up to day t (Zero Leakage!)
            current_history = warmup_history + list(test_df.loc[:t, "target"].values)

            # Generate 21-day recursive trajectory rollout using ONLY current_history
            rollout = model.predict_trajectory(
                history_sales=current_history,
                start_date=future_dates.iloc[0],
                horizon=forecast_horizon
            )
            pred_p10 = rollout["p10"]
            pred_p50 = rollout["p50"]

            # -----------------------------------------------------------------
            # Model B0: Static Threshold (Alerts only if curr_cash < reserve today)
            # -----------------------------------------------------------------
            b0_alert = (curr_cash < config.min_reserve)
            b0_shortfall = max(0.0, config.min_reserve - curr_cash)
            res["B0"]["alerts"].append(b0_alert)
            res["B0"]["actuals"].append(breach_in_window)
            res["B0"]["probs"].append(1.0 if b0_alert else 0.0)
            if b0_alert and breach_in_window:
                res["B0"]["lead_times"].append(0.0)
            res["B0"]["shortfall_errs"].append(abs(actual_max_shortfall - b0_shortfall))

            # -----------------------------------------------------------------
            # Model B1: Point-Forecast Extrapolation (P50 Recursive Rollout)
            # -----------------------------------------------------------------
            sim_cfg = SMEConfig(
                initial_cash=curr_cash,
                min_reserve=config.min_reserve,
                monthly_payroll=config.monthly_payroll,
                monthly_rent=config.monthly_rent,
                monthly_utilities_misc=config.monthly_utilities_misc,
                monthly_debt_service=config.monthly_debt_service,
                cogs_ratio=config.cogs_ratio,
                supplier_terms_lag=config.supplier_terms_lag,
                sales_settlement_lag=config.sales_settlement_lag
            )
            engine_sim = CashFlowEngine(config=sim_cfg)

            b1_ledger = engine_sim.simulate(sales_series=pd.Series(pred_p50), dates=future_dates)
            b1_alert = bool(b1_ledger["is_reserve_breach"].any())
            b1_indices = b1_ledger.index[b1_ledger["is_reserve_breach"]].tolist()
            b1_lead = float(b1_indices[0] + 1) if b1_indices else 0.0
            b1_shortfall = float(b1_ledger["shortfall_amount"].max())
            res["B1"]["alerts"].append(b1_alert)
            res["B1"]["actuals"].append(breach_in_window)
            res["B1"]["probs"].append(1.0 if b1_alert else 0.0)
            if b1_alert and breach_in_window:
                res["B1"]["lead_times"].append(b1_lead)
            res["B1"]["shortfall_errs"].append(abs(actual_max_shortfall - b1_shortfall))

            # -----------------------------------------------------------------
            # Model B2: P10 Downside Quantile Stress Engine (FinSight AI Core)
            # -----------------------------------------------------------------
            b2_ledger = engine_sim.simulate(sales_series=pd.Series(pred_p10), dates=future_dates)
            b2_alert = bool(b2_ledger["is_reserve_breach"].any())
            b2_indices = b2_ledger.index[b2_ledger["is_reserve_breach"]].tolist()
            b2_lead = float(b2_indices[0] + 1) if b2_indices else 0.0
            b2_shortfall = float(b2_ledger["shortfall_amount"].max())
            res["B2"]["alerts"].append(b2_alert)
            res["B2"]["actuals"].append(breach_in_window)
            res["B2"]["probs"].append(1.0 if b2_alert else 0.0)
            if b2_alert and breach_in_window:
                res["B2"]["lead_times"].append(b2_lead)
            res["B2"]["shortfall_errs"].append(abs(actual_max_shortfall - b2_shortfall))

            # -----------------------------------------------------------------
            # Model B3: 7-Day Net Cash Burn Run-Rate
            # -----------------------------------------------------------------
            lookback_start = max(0, t - 7)
            burn_sample = test_df.iloc[lookback_start:t+1]
            daily_burn = (curr_cash - float(burn_sample["true_closing_cash"].iloc[0])) / max(1, len(burn_sample))
            proj_end_cash_b3 = curr_cash + (daily_burn * forecast_horizon)
            b3_alert = (proj_end_cash_b3 < config.min_reserve)
            b3_shortfall = max(0.0, config.min_reserve - proj_end_cash_b3)
            res["B3"]["alerts"].append(b3_alert)
            res["B3"]["actuals"].append(breach_in_window)
            res["B3"]["probs"].append(1.0 if b3_alert else 0.0)
            if b3_alert and breach_in_window:
                res["B3"]["lead_times"].append(actual_lead)
            res["B3"]["shortfall_errs"].append(abs(actual_max_shortfall - b3_shortfall))

        # Evaluate metrics for current seed
        model_names = {
            "B0": "Static Threshold (Reactive)",
            "B1": "Point-Forecast Extrapolation (P50)",
            "B2": "P10 Downside Quantile Engine (FinSight AI)",
            "B3": "7-Day Net Cash Burn Run-Rate"
        }

        for mid, mname in model_names.items():
            y_true = np.array(res[mid]["actuals"]).astype(int)
            y_pred = np.array(res[mid]["alerts"]).astype(int)
            y_prob = np.array(res[mid]["probs"])

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

            lead = float(np.mean(res[mid]["lead_times"])) if len(res[mid]["lead_times"]) > 0 else 0.0
            # Monthly False Alarms (per 30 days)
            false_positives = np.sum((y_pred == 1) & (y_true == 0))
            monthly_fa = float((false_positives / num_steps) * 30.0)

            shortfall_mae = float(np.mean(res[mid]["shortfall_errs"]))

            all_seed_results.append({
                "Architecture ID": mid,
                "Architecture Name": mname,
                "Seed": seed,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "PR-AUC": pr_auc,
                "Lead Time (Days)": lead,
                "Monthly False Alarms": monthly_fa,
                "Shortfall MAE (R$)": shortfall_mae
            })

    df_raw_b = pd.DataFrame(all_seed_results)
    raw_b_csv = os.path.join(RESULTS_DIR, "exp_b_early_warning_raw.csv")
    df_raw_b.to_csv(raw_b_csv, index=False)
    print(f"\n[Raw Exp B saved to {raw_b_csv}]")

    # 4. Summary Table 2 (Mean ± SD)
    summary_rows = []
    for (mid, mname), grp in df_raw_b.groupby(["Architecture ID", "Architecture Name"], as_index=False):
        prec_m, prec_s = grp["Precision"].mean(), grp["Precision"].std()
        rec_m, rec_s = grp["Recall"].mean(), grp["Recall"].std()
        f1_m, f1_s = grp["F1-Score"].mean(), grp["F1-Score"].std()
        prauc_m, prauc_s = grp["PR-AUC"].mean(), grp["PR-AUC"].std()
        lead_m, lead_s = grp["Lead Time (Days)"].mean(), grp["Lead Time (Days)"].std()
        mfa_m, mfa_s = grp["Monthly False Alarms"].mean(), grp["Monthly False Alarms"].std()
        sf_m, sf_s = grp["Shortfall MAE (R$)"].mean(), grp["Shortfall MAE (R$)"].std()

        summary_rows.append({
            "Architecture ID": mid,
            "Architecture Name": mname,
            "Precision": f"{prec_m:.3f} ± {prec_s:.3f}" if prec_s > 0 else f"{prec_m:.3f}",
            "Recall": f"{rec_m:.3f} ± {rec_s:.3f}" if rec_s > 0 else f"{rec_m:.3f}",
            "F1-Score": f"{f1_m:.3f} ± {f1_s:.3f}" if f1_s > 0 else f"{f1_m:.3f}",
            "PR-AUC": f"{prauc_m:.3f} ± {prauc_s:.3f}" if prauc_s > 0 else f"{prauc_m:.3f}",
            "Lead Time (Days)": f"{lead_m:.1f} ± {lead_s:.1f}" if lead_s > 0 else f"{lead_m:.1f}",
            "Monthly False Alarms": f"{mfa_m:.2f} ± {mfa_s:.2f}" if mfa_s > 0 else f"{mfa_m:.2f}",
            "Shortfall MAE (R$)": f"{sf_m:.2f} ± {sf_s:.2f}" if sf_s > 0 else f"{sf_m:.2f}"
        })

    df_summary_b = pd.DataFrame(summary_rows)
    summary_b_csv = os.path.join(RESULTS_DIR, "exp_b_early_warning_benchmark.csv")
    df_summary_b.to_csv(summary_b_csv, index=False)
    print(f"[Publication summary saved to {summary_b_csv}]")

    print("\n" + "=" * 95)
    print("Q1 TABLE 2: EXPERIMENT B — CASH-FLOW EARLY WARNING BENCHMARK (5-SEED ZERO-LEAKAGE)")
    print("=" * 95)
    print(df_summary_b.to_string(index=False))


if __name__ == "__main__":
    run_experiment_b()
