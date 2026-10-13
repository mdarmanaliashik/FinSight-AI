"""
FinSight AI - Temporal Validation & Rolling-Origin Walk-Forward CV
Purged and Embargoed Walk-Forward Splitter with Embargo Sensitivity Analysis.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

def get_rolling_origin_splits(
    df: pd.DataFrame,
    date_col: str = "Date",
    min_train_days: int = 180,
    horizon_days: int = 14,
    step_days: int = 28,
    embargo_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Constructs Purged & Embargoed Rolling-Origin Walk-Forward Cross-Validation splits.
    
    Structure:
    - Train Window: [start, origin_t]
    - Embargo Window: (origin_t, origin_t + embargo_days]
    - Test Window: (origin_t + embargo_days, origin_t + embargo_days + horizon_days]
    
    Embargo parameter guarantees that multi-day rolling statistics (up to 28 days) 
    do not contaminate out-of-sample evaluation periods (de Prado, 2018).
    """
    dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    total_days = len(dates)

    splits = []
    current_origin_idx = min_train_days - 1

    fold_idx = 1
    while current_origin_idx + embargo_days + horizon_days < total_days:
        train_end_idx = current_origin_idx
        embargo_end_idx = train_end_idx + embargo_days
        test_end_idx = embargo_end_idx + horizon_days

        train_dates = (dates.iloc[0], dates.iloc[train_end_idx])
        embargo_dates = (dates.iloc[train_end_idx + 1], dates.iloc[embargo_end_idx]) if embargo_days > 0 else None
        test_dates = (dates.iloc[embargo_end_idx + 1], dates.iloc[test_end_idx])

        splits.append({
            "Fold": fold_idx,
            "Train_Start": train_dates[0].strftime("%Y-%m-%d"),
            "Train_End": train_dates[1].strftime("%Y-%m-%d"),
            "Train_Days": train_end_idx + 1,
            "Embargo_Days": embargo_days,
            "Test_Start": test_dates[0].strftime("%Y-%m-%d"),
            "Test_End": test_dates[1].strftime("%Y-%m-%d"),
            "Test_Days": horizon_days
        })
        current_origin_idx += step_days
        fold_idx += 1

    return splits

def analyze_embargo_sensitivity(
    df: pd.DataFrame,
    embargo_options: List[int] = [0, 7, 14, 30]
) -> pd.DataFrame:
    """
    Evaluates split feasibility and test coverage under various embargo constraints (0, 7, 14, 30 days).
    """
    records = []
    for emb in embargo_options:
        splits = get_rolling_origin_splits(df, min_train_days=180, horizon_days=14, step_days=28, embargo_days=emb)
        records.append({
            "Embargo_Days": emb,
            "Available_Folds": len(splits),
            "Min_Train_Days": 180,
            "Test_Horizon_Days": 14,
            "Theoretical_Protection": "None (Potential Leakage)" if emb == 0 else f"{emb}-day autoregressive purge"
        })
    return pd.DataFrame(records)
