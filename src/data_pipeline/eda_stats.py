"""
FinSight AI - Statistical Testing & EDA Engine
Augmented Dickey-Fuller (ADF), KPSS Tests, ACF/PACF, STL Decomposition,
CUSUM Structural Break, and Black Friday Outlier Sensitivity Analysis.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL

def format_kpss_p(p_val: float) -> str:
    """Formats KPSS p-value accounting for lookup table boundaries."""
    if p_val <= 0.01:
        return "< 0.01 (Reject H0)"
    elif p_val >= 0.10:
        return "> 0.10 (Fail to Reject H0)"
    else:
        return f"{p_val:.4f}"

def test_stationarity(series: pd.Series, name: str = "Series") -> pd.DataFrame:
    """
    Executes dual Augmented Dickey-Fuller (ADF) and KPSS tests for unit root and stationarity.
    - ADF H0: Non-stationary (Unit root exists). Reject if p < 0.05.
    - KPSS Level H0: Stationary around constant (regression='c'). Reject if p < 0.05.
    - KPSS Trend H0: Stationary around linear trend (regression='ct'). Reject if p < 0.05.
    """
    s_clean = series.dropna()
    
    # Raw series
    adf_raw = adfuller(s_clean, autolag="AIC")
    kpss_c_raw = kpss(s_clean, regression="c", nlags="auto")
    kpss_ct_raw = kpss(s_clean, regression="ct", nlags="auto")

    # First differenced
    s_diff = s_clean.diff().dropna()
    adf_diff = adfuller(s_diff, autolag="AIC")
    kpss_c_diff = kpss(s_diff, regression="c", nlags="auto")
    kpss_ct_diff = kpss(s_diff, regression="ct", nlags="auto")

    def infer_stationarity(adf_p, kpss_c_p, kpss_ct_p):
        adf_stat = adf_p < 0.05
        kpss_c_stat = kpss_c_p >= 0.05
        kpss_ct_stat = kpss_ct_p >= 0.05
        if adf_stat and kpss_c_stat:
            return "Strictly Stationary (Mean & Variance Reverting)"
        elif adf_stat and not kpss_c_stat and kpss_ct_stat:
            return "Trend-Stationary (Stationary Around Growth Trend)"
        elif adf_stat and not kpss_c_stat and not kpss_ct_stat:
            return "Conflicting / Variance Break (ADF Reject, KPSS Reject)"
        elif not adf_stat and not kpss_c_stat:
            return "Non-Stationary (Unit Root Present)"
        elif not adf_stat and kpss_c_stat:
            return "Stationary by KPSS (ADF Underpowered by Seasonality)"
        else:
            return "Indeterminate"

    records = [
        {
            "Series": f"{name} (Raw)",
            "ADF_Stat": round(adf_raw[0], 4),
            "ADF_pvalue": f"{adf_raw[1]:.4e}",
            "KPSS_Level_Stat": round(kpss_c_raw[0], 4),
            "KPSS_Level_p": format_kpss_p(kpss_c_raw[1]),
            "KPSS_Trend_Stat": round(kpss_ct_raw[0], 4),
            "KPSS_Trend_p": format_kpss_p(kpss_ct_raw[1]),
            "Inference": infer_stationarity(adf_raw[1], kpss_c_raw[1], kpss_ct_raw[1])
        },
        {
            "Series": f"{name} (1st Difference)",
            "ADF_Stat": round(adf_diff[0], 4),
            "ADF_pvalue": f"{adf_diff[1]:.4e}",
            "KPSS_Level_Stat": round(kpss_c_diff[0], 4),
            "KPSS_Level_p": format_kpss_p(kpss_c_diff[1]),
            "KPSS_Trend_Stat": round(kpss_ct_diff[0], 4),
            "KPSS_Trend_p": format_kpss_p(kpss_ct_diff[1]),
            "Inference": infer_stationarity(adf_diff[1], kpss_c_diff[1], kpss_ct_diff[1])
        }
    ]
    return pd.DataFrame(records)

def analyze_outlier_sensitivity(df_daily: pd.DataFrame, outlier_date: str = "2017-11-24") -> pd.DataFrame:
    """
    Computes rigorous sensitivity of statistical moments (Mean, Std, Skewness, Kurtosis)
    when retaining vs trimming genuine macroeconomic shock events (e.g., Black Friday).
    """
    s = df_daily.set_index("Date")["sales"]
    dt = pd.to_datetime(outlier_date)

    s_full = s.copy()
    s_trimmed = s.drop(index=dt) if dt in s.index else s.copy()

    records = [
        {
            "Condition": "Full Empirical Data (Accounting Truth)",
            "Sample_Size": len(s_full),
            "Mean": round(s_full.mean(), 2),
            "Std_Dev": round(s_full.std(), 2),
            "Max": round(s_full.max(), 2),
            "Skewness": round(stats.skew(s_full), 3),
            "Kurtosis_Fisher": round(stats.kurtosis(s_full, fisher=True), 3),
            "Kurtosis_Pearson": round(stats.kurtosis(s_full, fisher=False), 3),
            "Description": "Includes Black Friday 2017 shock (R$ 175,178.46)"
        },
        {
            "Condition": "Trimmed (Excluding Black Friday)",
            "Sample_Size": len(s_trimmed),
            "Mean": round(s_trimmed.mean(), 2),
            "Std_Dev": round(s_trimmed.std(), 2),
            "Max": round(s_trimmed.max(), 2),
            "Skewness": round(stats.skew(s_trimmed), 3),
            "Kurtosis_Fisher": round(stats.kurtosis(s_trimmed, fisher=True), 3),
            "Kurtosis_Pearson": round(stats.kurtosis(s_trimmed, fisher=False), 3),
            "Description": "Artificial statistical artifact removal"
        }
    ]
    return pd.DataFrame(records)

def compute_cusum_structural_break(series: pd.Series) -> Dict[str, Any]:
    """
    Computes cumulative sum of standardized residuals (CUSUM) to detect structural breaks.
    """
    y = series.values
    n = len(y)
    mean_val = np.mean(y)
    std_val = np.std(y, ddof=1)
    
    # Standardized residuals
    standardized = (y - mean_val) / std_val
    cusum = np.cumsum(standardized)
    
    # Boundary threshold (approximate 95% critical value line)
    # Critical boundary: a * sqrt(n) + 2 * a * t / sqrt(n), a=0.948 for 5% level
    t = np.arange(1, n + 1)
    boundary = 0.948 * np.sqrt(n) + 2 * 0.948 * t / np.sqrt(n)
    
    max_cusum_idx = int(np.argmax(np.abs(cusum)))
    has_break = bool(np.any(np.abs(cusum) > boundary))
    
    return {
        "cusum": cusum,
        "boundary": boundary,
        "max_cusum_step": max_cusum_idx,
        "has_break": has_break
    }

def decompose_stl(df_daily: pd.DataFrame, period: int = 7) -> Any:
    """
    Applies Loess-based Seasonal-Trend Decomposition (STL) to isolate weekly seasonality.
    """
    s = df_daily.set_index("Date")["sales"]
    stl = STL(s, period=period, seasonal=7, robust=True)
    res = stl.fit()
    return res
