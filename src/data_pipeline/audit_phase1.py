"""
FinSight AI - Phase 1 Comprehensive Statistical Audit & EDA Script
==================================================================
Performs in-depth data profiling, distribution checks, autocorrelation analysis,
and seasonality decomposition on the generated Phase 1 parquet artifacts.
"""

import os
import json
import numpy as np
import pandas as pd

PROCESSED_DIR = r"d:\Others\data\processed"

def profile_time_series(df, name, date_col="Date", target_col="sales"):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    n_days = len(df)
    min_date = df[date_col].min().strftime("%Y-%m-%d")
    max_date = df[date_col].max().strftime("%Y-%m-%d")
    total_sales = df[target_col].sum()
    
    # Sparsity / Zero analysis
    zero_days = (df[target_col] == 0).sum()
    zero_pct = (zero_days / n_days) * 100
    
    # Distributional statistics
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    median_val = df[target_col].median()
    p25 = df[target_col].quantile(0.25)
    p75 = df[target_col].quantile(0.75)
    p90 = df[target_col].quantile(0.90)
    p99 = df[target_col].quantile(0.99)
    max_val = df[target_col].max()
    skew_val = df[target_col].skew()
    kurt_val = df[target_col].kurtosis()
    
    # Autocorrelation (ACF)
    acf_1 = df[target_col].autocorr(lag=1)
    acf_7 = df[target_col].autocorr(lag=7)
    acf_14 = df[target_col].autocorr(lag=14)
    acf_28 = df[target_col].autocorr(lag=28)
    
    # Day-of-week breakdown
    df["dow"] = df[date_col].dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_means = df.groupby("dow")[target_col].mean().to_dict()
    dow_formatted = {dow_names[k]: round(v, 2) for k, v in dow_means.items()}
    
    profile = {
        "dataset_name": name,
        "n_days": int(n_days),
        "date_range": f"{min_date} to {max_date}",
        "total_revenue": round(float(total_sales), 2),
        "zero_days": int(zero_days),
        "zero_pct": round(float(zero_pct), 2),
        "mean": round(float(mean_val), 2),
        "std": round(float(std_val), 2),
        "median": round(float(median_val), 2),
        "iqr": round(float(p75 - p25), 2),
        "p25": round(float(p25), 2),
        "p75": round(float(p75), 2),
        "p90": round(float(p90), 2),
        "p99": round(float(p99), 2),
        "max": round(float(max_val), 2),
        "skewness": round(float(skew_val), 3),
        "kurtosis": round(float(kurt_val), 3),
        "acf_lag1": round(float(acf_1), 3),
        "acf_lag7": round(float(acf_7), 3),
        "acf_lag14": round(float(acf_14), 3),
        "acf_lag28": round(float(acf_28), 3),
        "day_of_week_mean": dow_formatted
    }
    return profile

def run_audit():
    results = {}
    
    # 1. JD.com Platform Macro Daily Sales
    jd_platform_path = os.path.join(PROCESSED_DIR, "jd_daily_sales_platform.parquet")
    df_jd_plat = pd.read_parquet(jd_platform_path)
    results["jd_platform"] = profile_time_series(df_jd_plat, "JD.com Retail Platform (Macro Daily GMV)")
    
    # 2. JD.com Top SKU Micro-Merchant Daily Sales
    jd_sku_path = os.path.join(PROCESSED_DIR, "jd_daily_sales_top_sku.parquet")
    df_jd_sku = pd.read_parquet(jd_sku_path)
    results["jd_top_sku"] = profile_time_series(df_jd_sku, "JD.com Top SKU 068f4481b3 (Micro-Merchant Series)")

    # 3. Olist Platform Macro Daily Sales
    olist_platform_path = os.path.join(PROCESSED_DIR, "olist_daily_sales_platform.parquet")
    df_olist_plat = pd.read_parquet(olist_platform_path)
    results["olist_platform"] = profile_time_series(df_olist_plat, "Olist Platform (B2C E-Com Macro)")
    
    # 4. Olist Top SME Daily Sales
    olist_top_path = os.path.join(PROCESSED_DIR, "olist_daily_sales_top_seller.parquet")
    df_olist_top = pd.read_parquet(olist_top_path)
    results["olist_top_seller"] = profile_time_series(df_olist_top, "Olist Top Merchant (Single SME Profile)")
    
    print(json.dumps(results, indent=2))
    
    # Save JSON report
    with open(os.path.join(PROCESSED_DIR, "phase1_audit_report.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_audit()
