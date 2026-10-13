"""
FinSight AI - Temporal Feature Store & Anti-Leakage Feature Engineering
========================================================================
Enhanced Feature Engine:
- Autoregressive Lags: t (lag_0), t-1, t-2, t-6, t-13, t-20, t-27
- Rolling Statistics: Mean, Std, Skewness, Min, Max over 7, 14, 28 days
- Calendar Dimensions: Day of week, Day of month, Month, Is_weekend
- Cyclical Fourier Encodings: sin/cos transformations for weekly and annual cycles
- Economic & Institutional: Payday windows, Statutory Holidays (Brazil/UK), Black Friday / Cyber Monday promo flags

Guarantees:
1. Mathematical Zero-Leakage: At forecast origin t, all features are derived strictly from y <= t.
2. Trajectory Compatibility: Supports direct multi-horizon and recursive rollout.
"""

from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats

# Statutory Public Holidays (Brazil 2016-2018 & UK 2010-2011)
BRAZIL_HOLIDAYS = {
    pd.to_datetime(d).date() for d in [
        "2017-01-01", "2017-02-28", "2017-04-14", "2017-04-21", "2017-05-01",
        "2017-06-15", "2017-09-07", "2017-10-12", "2017-11-02", "2017-11-15", "2017-12-25",
        "2018-01-01", "2018-02-13", "2018-03-30", "2018-04-21", "2018-05-01",
        "2018-05-31", "2018-09-07", "2018-10-12", "2018-11-02", "2018-11-15", "2018-12-25"
    ]
}

UK_BANK_HOLIDAYS = {
    pd.to_datetime(d).date() for d in [
        "2010-12-25", "2010-12-26", "2010-12-27", "2010-12-28",
        "2011-01-01", "2011-01-03", "2011-04-22", "2011-04-25", "2011-04-29",
        "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-25", "2011-12-26", "2011-12-27"
    ]
}

BLACK_FRIDAY_DATES = {
    pd.to_datetime(d).date() for d in [
        "2010-11-26", "2011-11-25", "2017-11-24", "2018-11-23"
    ]
}

CYBER_MONDAY_DATES = {
    pd.to_datetime(d).date() for d in [
        "2010-11-29", "2011-11-28", "2017-11-27", "2018-11-26"
    ]
}

FEATURE_COLS = [
    # 1. Target-Aligned Autoregressive Lags (relative to origin t, perfectly matching target weekday cycle)
    "lag_0", "lag_1", "lag_2", "lag_7", "lag_14", "lag_21", "lag_28",
    # 2. Rolling Statistics (Historical slices up to origin t: 5 stats x 3 windows = 15 features)
    "rolling_mean_7", "rolling_std_7", "rolling_skew_7", "rolling_min_7", "rolling_max_7",
    "rolling_mean_14", "rolling_std_14", "rolling_skew_14", "rolling_min_14", "rolling_max_14",
    "rolling_mean_28", "rolling_std_28", "rolling_skew_28", "rolling_min_28", "rolling_max_28",
    # 3. Calendar Dimensions for Target Horizon Date
    "day_of_week", "day_of_month", "month", "is_weekend",
    "sin_dow", "cos_dow", "sin_month", "cos_month",
    # 4. Institutional & Economic Regime Features
    "is_payday_window", "is_public_holiday", "is_promo_event"
]

FEATURE_GROUPS = {
    "Autoregressive Lags": ["lag_0", "lag_1", "lag_2", "lag_7", "lag_14", "lag_21", "lag_28"],
    "Rolling Statistics": [
        "rolling_mean_7", "rolling_std_7", "rolling_skew_7", "rolling_min_7", "rolling_max_7",
        "rolling_mean_14", "rolling_std_14", "rolling_skew_14", "rolling_min_14", "rolling_max_14",
        "rolling_mean_28", "rolling_std_28", "rolling_skew_28", "rolling_min_28", "rolling_max_28"
    ],
    "Calendar & Fourier": [
        "day_of_week", "day_of_month", "month", "is_weekend",
        "sin_dow", "cos_dow", "sin_month", "cos_month"
    ],
    "Institutional & Economic": ["is_payday_window", "is_public_holiday", "is_promo_event"]
}


def extract_features_from_history(
    history_sales: Union[pd.Series, np.ndarray, List[float]],
    target_date: Union[pd.Timestamp, str],
    country: str = "Brazil"
) -> pd.DataFrame:
    """
    Extracts features for predicting sales on `target_date` using ONLY historical sales observed up to origin t.
    Zero-leakage guarantee: history_sales contains [y_1, y_2, ..., y_t].
    Lags:
    - lag_0: y_t (latest observed revenue at origin t)
    - lag_1, lag_2: y_{t-1}, y_{t-2} (immediate autoregressive continuity)
    - lag_7, lag_14, lag_21, lag_28: y_{t-7}, y_{t-14}, y_{t-21}, y_{t-28} (target weekday aligned cyclical lags)
    """
    y = np.asarray(history_sales, dtype=float)
    if len(y) < 29:
        padding = np.full(29 - len(y), y[0] if len(y) > 0 else 0.0)
        y = np.concatenate([padding, y])

    t_date = pd.to_datetime(target_date).date()
    row: Dict[str, Any] = {}

    # 1. Target-Aligned Autoregressive Lags (relative to origin t, the last element y[-1])
    row["lag_0"] = float(y[-1])
    row["lag_1"] = float(y[-2])
    row["lag_2"] = float(y[-3])
    row["lag_7"] = float(y[-8]) if len(y) >= 8 else float(y[0])
    row["lag_14"] = float(y[-15]) if len(y) >= 15 else float(y[0])
    row["lag_21"] = float(y[-22]) if len(y) >= 22 else float(y[0])
    row["lag_28"] = float(y[-29]) if len(y) >= 29 else float(y[0])

    # 2. Rolling Aggregations over historical window ending at t (5 stats x 3 windows = 15 features)
    for w in [7, 14, 28]:
        w_slice = y[-w:]
        row[f"rolling_mean_{w}"] = float(np.mean(w_slice))
        row[f"rolling_std_{w}"] = float(np.std(w_slice, ddof=1)) if len(w_slice) > 1 else 0.0
        row[f"rolling_min_{w}"] = float(np.min(w_slice))
        row[f"rolling_max_{w}"] = float(np.max(w_slice))
        skew_val = stats.skew(w_slice)
        row[f"rolling_skew_{w}"] = float(skew_val) if not np.isnan(skew_val) else 0.0

    # 3. Calendar Dimensions for TARGET date
    dow = pd.to_datetime(target_date).dayofweek
    dom = pd.to_datetime(target_date).day
    m = pd.to_datetime(target_date).month
    row["day_of_week"] = dow
    row["day_of_month"] = dom
    row["month"] = m
    row["is_weekend"] = int(dow in [5, 6])

    # 4. Cyclical Fourier Encodings
    row["sin_dow"] = float(np.sin(2 * np.pi * dow / 7))
    row["cos_dow"] = float(np.cos(2 * np.pi * dow / 7))
    row["sin_month"] = float(np.sin(2 * np.pi * m / 12))
    row["cos_month"] = float(np.cos(2 * np.pi * m / 12))

    # 5. SME Payday Window Flag (1st-5th and 25th-31st)
    row["is_payday_window"] = int(dom in [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31])

    # 6. Public Holidays
    holidays_set = BRAZIL_HOLIDAYS if country.lower() == "brazil" else UK_BANK_HOLIDAYS
    row["is_public_holiday"] = int(t_date in holidays_set)

    # 7. Promo Event (Black Friday / Cyber Monday)
    row["is_promo_event"] = int(t_date in BLACK_FRIDAY_DATES or t_date in CYBER_MONDAY_DATES)

    df_row = pd.DataFrame([row])[FEATURE_COLS]
    return df_row


def build_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    date_col: str = "Date",
    horizon: int = 1,
    country: str = "Brazil"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs a direct multi-horizon dataset where each row corresponds to target date T = t + horizon.
    All features are derived strictly from history observed up to origin t (i.e. y <= t = T - horizon).
    
    Returns:
        df_feats: DataFrame containing feature columns, target, and dates.
        FEATURE_COLS: List of feature names.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    sales = df[target_col].values
    dates = pd.to_datetime(df[date_col]).values

    n = len(df)
    records = []

    # Requires at least 29 days of history prior to origin t (to populate lag_28 = y_{t-28})
    start_idx = 29 + horizon - 1

    for i in range(start_idx, n):
        target_idx = i
        origin_idx = i - horizon
        
        target_val = sales[target_idx]
        target_dt = dates[target_idx]
        
        history_slice = sales[:origin_idx + 1]
        
        feats = extract_features_from_history(history_slice, target_dt, country=country).iloc[0].to_dict()
        feats["target"] = target_val
        feats["origin_date"] = dates[origin_idx]
        feats["target_date"] = target_dt
        records.append(feats)

    df_feats = pd.DataFrame(records)
    return df_feats, FEATURE_COLS
