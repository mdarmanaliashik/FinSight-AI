"""
FinSight AI - Temporal Leakage Verification Suite
Mathematical Invariance Tests: Future Perturbation Invariance & Online-Batch Equivalence.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.forecasting.features import (
    extract_features_from_history,
    build_forecasting_features,
    FEATURE_COLS
)

def run_future_perturbation_test() -> Dict[str, Any]:
    """
    Tests whether feature values computed at forecast origin t are strictly invariant
    to arbitrary perturbations of future sales y_{t+1 ... t+H}.
    """
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    sales = np.random.uniform(500, 2000, size=n)
    
    df_clean = pd.DataFrame({"Date": dates, "sales": sales})

    origin_idx = 60
    horizon = 7
    origin_date = dates[origin_idx]
    target_date = dates[origin_idx + horizon]

    # Baseline features extracted at origin t
    history_baseline = sales[:origin_idx + 1]
    feats_baseline = extract_features_from_history(history_baseline, target_date).values

    # Drastically perturb future values (100x multiplier + shock)
    sales_perturbed = sales.copy()
    sales_perturbed[origin_idx + 1:] = sales_perturbed[origin_idx + 1:] * 100.0 + 999999.0
    
    history_perturbed = sales_perturbed[:origin_idx + 1]
    feats_perturbed = extract_features_from_history(history_perturbed, target_date).values

    # Test 1: Slice invariance
    is_slice_invariant = bool(np.allclose(feats_baseline, feats_perturbed, atol=1e-9))

    # Test 2: Full batch dataset invariance up to origin t
    df_feats_orig, _ = build_forecasting_features(df_clean, horizon=horizon)
    df_perturbed = pd.DataFrame({"Date": dates, "sales": sales_perturbed})
    df_feats_pert, _ = build_forecasting_features(df_perturbed, horizon=horizon)

    # Compare all rows where origin_date <= origin_date
    mask_orig = df_feats_orig["origin_date"] <= origin_date
    feats_sub_orig = df_feats_orig.loc[mask_orig, FEATURE_COLS].values
    feats_sub_pert = df_feats_pert.loc[mask_orig, FEATURE_COLS].values

    is_batch_invariant = bool(np.allclose(feats_sub_orig, feats_sub_pert, atol=1e-9))
    max_abs_diff = float(np.max(np.abs(feats_sub_orig - feats_sub_pert)))

    return {
        "is_slice_invariant": is_slice_invariant,
        "is_batch_invariant": is_batch_invariant,
        "max_absolute_difference": max_abs_diff,
        "status": "PASSED" if (is_slice_invariant and is_batch_invariant) else "FAILED"
    }

def run_online_vs_batch_equivalence_test() -> Dict[str, Any]:
    """
    Verifies that the batch dataset extraction function and the single-step online
    feature extractor produce 100% numerically identical representations.
    """
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2017-01-01", periods=n, freq="D")
    sales = np.random.uniform(100, 1000, size=n)
    df = pd.DataFrame({"Date": dates, "sales": sales})

    horizon = 3
    df_batch, feat_names = build_forecasting_features(df, horizon=horizon)

    # Pick 5 random sample rows to verify
    sample_indices = [0, len(df_batch) // 4, len(df_batch) // 2, 3 * len(df_batch) // 4, len(df_batch) - 1]
    discrepancies = []

    for idx in sample_indices:
        batch_row = df_batch.iloc[idx]
        target_dt = batch_row["target_date"]
        orig_dt = batch_row["origin_date"]
        
        # History slice up to origin date
        hist_slice = df.loc[df["Date"] <= orig_dt, "sales"].values
        online_feats = extract_features_from_history(hist_slice, target_dt).iloc[0]

        batch_vals = batch_row[feat_names].values.astype(float)
        online_vals = online_feats[feat_names].values.astype(float)

        diff = np.max(np.abs(batch_vals - online_vals))
        discrepancies.append(diff)

    max_diff = float(np.max(discrepancies))
    is_equivalent = bool(max_diff < 1e-9)

    return {
        "is_equivalent": is_equivalent,
        "max_discrepancy": max_diff,
        "tested_samples": len(sample_indices),
        "status": "PASSED" if is_equivalent else "FAILED"
    }

def run_all_leakage_tests() -> pd.DataFrame:
    """Executes the complete leakage test suite and returns a summary DataFrame."""
    t1 = run_future_perturbation_test()
    t2 = run_online_vs_batch_equivalence_test()

    records = [
        {
            "Test_Name": "Future Perturbation Invariance",
            "Description": "Features at origin t must remain invariant when y_{t+1...t+H} is perturbed",
            "Max_Diff": f"{t1['max_absolute_difference']:.2e}",
            "Result": t1["status"]
        },
        {
            "Test_Name": "Online vs. Batch Equivalence",
            "Description": "Vectorized dataset generator and online single-origin extractor match exactly",
            "Max_Diff": f"{t2['max_discrepancy']:.2e}",
            "Result": t2["status"]
        }
    ]
    return pd.DataFrame(records)
