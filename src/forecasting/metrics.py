"""
FinSight AI - Temporal Forecasting Metrics & Statistical Significance
======================================================================
Q1 Publication & Competition Grade Evaluation Protocol

Includes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- WAPE (Weighted Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error, scaled against in-sample seasonal naive)
- Empirical Prediction Interval Coverage (PICP %) & Width (MPIW)
- Diebold-Mariano Statistical Significance Test (HLN modified & HAC robust)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.stats as stats


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    power: int = 1
) -> Tuple[float, float]:
    """
    Computes the Diebold-Mariano test statistic and p-value
    comparing two forecast error vectors (e1 vs e2).
    H0: Both models have equal forecast accuracy.
    H1: Model 1 has significantly lower forecast error than Model 2.
    """
    e1_arr = np.asarray(e1, dtype=float)
    e2_arr = np.asarray(e2, dtype=float)
    T = len(e1_arr)
    if T < 5:
        return 0.0, 1.0

    if power == 1:
        d = np.abs(e2_arr) - np.abs(e1_arr)  # positive d means model 1 is better
    else:
        d = (e2_arr ** 2) - (e1_arr ** 2)

    mean_d = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=1))

    # Auto-covariance estimation with Bartlett kernel (Newey-West type truncation)
    gamma_sum = 0.0
    max_lag = min(h, max(1, T // 4))
    for lag in range(1, max_lag):
        if (T - lag) > 3:
            cov_mat = np.cov(d[lag:], d[:-lag])
            if cov_mat.shape == (2, 2) and not np.isnan(cov_mat[0, 1]):
                gamma_k = cov_mat[0, 1]
                gamma_sum += 2 * (1 - lag / max_lag) * gamma_k

    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 1e-12 or np.isnan(var_d):
        dm_stat = 0.0
    else:
        dm_stat = mean_d / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (HLN) small-sample correction
    hln_correction = np.sqrt(max(0.01, (T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm_stat_corrected = dm_stat * hln_correction

    # Two-sided p-value from Student-t distribution with T-1 d.o.f.
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_corrected), df=max(1, T - 1)))

    return round(float(dm_stat_corrected), 4), round(float(p_value), 5)


def calculate_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    seasonal_period: int = 7,
    num_features: int = 1
) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics for time-series forecasting.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    H = len(y_t)

    if H == 0:
        raise ValueError("Ground truth vector cannot be empty.")

    # 1. Absolute & Squared Errors
    abs_err = np.abs(y_t - y_p)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))

    # 2. Percentage Errors: WAPE & sMAPE (Time-Series Industry Standards)
    sum_actuals = float(np.sum(y_t))
    wape = float((np.sum(abs_err) / sum_actuals) * 100) if sum_actuals > 0 else 0.0

    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    denom = np.where(denom == 0, 1e-6, denom)
    smape = float(np.mean(abs_err / denom) * 100)

    # 3. MASE (Mean Absolute Scaled Error — Hyndman & Koehler 2006)
    mase = 1.0
    if y_train_history is not None and len(y_train_history) > seasonal_period:
        y_hist = np.asarray(y_train_history, dtype=float)
        naive_denom = np.mean(np.abs(y_hist[seasonal_period:] - y_hist[:-seasonal_period]))
        if naive_denom > 1e-6:
            mase = float(mae / naive_denom)

    # 4. Uncertainty Calibration (Prediction Interval Coverage & Width)
    coverage = None
    mean_width = None
    if y_lower is not None and y_upper is not None:
        y_l = np.asarray(y_lower, dtype=float)
        y_u = np.asarray(y_upper, dtype=float)
        in_interval = (y_t >= y_l) & (y_t <= y_u)
        coverage = float(np.mean(in_interval) * 100)
        mean_width = float(np.mean(y_u - y_l))

    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE_pct": round(wape, 2),
        "sMAPE_pct": round(smape, 2),
        "MASE": round(mase, 4),
    }

    if coverage is not None:
        metrics["Coverage_pct"] = round(coverage, 2)
        metrics["Mean_Interval_Width"] = round(mean_width, 2)

    return metrics


def calculate_conformal_exceedance_probability(
    target_sales: float,
    p10: float,
    p50: float,
    p90: float
) -> float:
    """
    Computes empirical exceedance probability P(Sales >= target_sales)
    using piecewise-linear CDF interpolation between calibrated quantiles P10, P50, and P90.
    """
    t = float(target_sales)
    m_p10 = float(p10)
    m_p50 = float(p50)
    m_p90 = float(p90)

    if t <= m_p10:
        ratio = max(0.0, t / (m_p10 if m_p10 > 0 else 1.0))
        prob = 90.0 + (1.0 - ratio) * 9.9
    elif t <= m_p50:
        denom = max(1e-6, m_p50 - m_p10)
        prob = 90.0 - ((t - m_p10) / denom) * 40.0
    elif t <= m_p90:
        denom = max(1e-6, m_p90 - m_p50)
        prob = 50.0 - ((t - m_p50) / denom) * 40.0
    else:
        excess = (t - m_p90) / (m_p90 if m_p90 > 0 else 1.0)
        prob = max(0.1, 10.0 * np.exp(-excess * 2.0))

    return round(float(np.clip(prob, 0.1, 99.9)), 2)


def evaluate_conformal_coverage(
    y_true: np.ndarray,
    y_p10: np.ndarray,
    y_p90: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates exact empirical coverage percentage (PICP) and interval width (MPIW)
    for [P10, P90] conformal interval.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_l = np.asarray(y_p10, dtype=float)
    y_u = np.asarray(y_p90, dtype=float)
    in_bounds = (y_t >= y_l) & (y_t <= y_u)
    return {
        "empirical_coverage_pct": round(float(np.mean(in_bounds) * 100.0), 2),
        "mean_interval_width": round(float(np.mean(y_u - y_l)), 2),
        "left_tail_breach_pct": round(float(np.mean(y_t < y_l) * 100.0), 2),
        "right_tail_surge_pct": round(float(np.mean(y_t > y_u) * 100.0), 2),
    }

# Backward compatibility alias
calculate_forecast_metrics = calculate_forecasting_metrics

