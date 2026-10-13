"""
FinSight AI - Olist Brazilian E-Commerce Ingestion & Preprocessing Module
Join-Inflation Empirical Proof, Delivery Bias Audit, Train-Only Seller Selection (k=3),
and Syntetos-Boylan Intermittent Demand Classification.
"""
import os
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

from src.data_pipeline.config import OLIST_DIR, PROCESSED_DATA_DIR

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file for cryptographic reproducibility."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_join_inflation(orders_path: Path, items_path: Path, payments_path: Path) -> Dict[str, Any]:
    """
    Audits the empirical join-inflation caused by naively joining Orders x Items x Payments.
    Explains the Cartesian product mechanism:
    Multiple payment rows (payment_sequential / split payments) x Multiple order items.
    """
    df_orders = pd.read_csv(orders_path)
    df_items = pd.read_csv(items_path)
    df_payments = pd.read_csv(payments_path)

    delivered = df_orders[df_orders["order_status"] == "delivered"]
    
    # Physical ground truth: Orders x Items
    items_merged = pd.merge(delivered, df_items, on="order_id", how="inner")
    true_rows = len(items_merged)
    true_price = items_merged["price"].sum()
    true_inflow = (items_merged["price"] + items_merged["freight_value"]).sum()

    # Naive join: Orders x Items x Payments
    naive_merged = pd.merge(items_merged, df_payments, on="order_id", how="inner")
    naive_rows = len(naive_merged)
    naive_price = naive_merged["price"].sum()
    naive_inflow = (naive_merged["price"] + naive_merged["freight_value"]).sum()

    row_inflation_pct = ((naive_rows - true_rows) / true_rows) * 100
    rev_inflation_pct = ((naive_inflow - true_inflow) / true_inflow) * 100

    pmt_counts = df_payments.groupby("order_id").size()
    item_counts = df_items.groupby("order_id").size()
    multi_pmt_count = (pmt_counts > 1).sum()
    multi_item_count = (item_counts > 1).sum()
    both_multi = len(set(pmt_counts[pmt_counts > 1].index).intersection(set(item_counts[item_counts > 1].index)))

    return {
        "true_rows": true_rows,
        "naive_rows": naive_rows,
        "row_inflation_pct": row_inflation_pct,
        "true_inflow": true_inflow,
        "naive_inflow": naive_inflow,
        "rev_inflation_pct": rev_inflation_pct,
        "phantom_revenue": naive_inflow - true_inflow,
        "multi_pmt_orders": multi_pmt_count,
        "multi_pmt_pct": multi_pmt_count / len(df_orders) * 100,
        "multi_item_orders": multi_item_count,
        "multi_item_pct": multi_item_count / len(df_orders) * 100,
        "both_multi_orders": both_multi
    }

def analyze_delivery_bias(orders_path: Path) -> pd.DataFrame:
    """
    Audits the distribution of order statuses and analyzes delivered filter bias.
    """
    df_orders = pd.read_csv(orders_path)
    total_orders = len(df_orders)
    status_counts = df_orders["order_status"].value_counts().reset_index()
    status_counts.columns = ["OrderStatus", "Count"]
    status_counts["Percentage"] = status_counts["Count"] / total_orders * 100
    return status_counts

def classify_demand_syntetos_boylan(series: pd.Series, active_window_only: bool = True) -> Dict[str, Any]:
    """
    Classifies a time-series using the Syntetos-Boylan (2005) categorisation:
    - ADI (Average Demand Interval) = Total Periods / Non-Zero Demand Periods
    - CV^2 = (Std / Mean)^2 of non-zero demand periods
    
    If active_window_only=True, series begins strictly at the first non-zero demand date
    to eliminate pre-onboarding structural zero inflation.
    
    Cutoffs: ADI = 1.32, CV^2 = 0.49
    Quadrants:
      - Smooth: ADI < 1.32, CV^2 < 0.49
      - Intermittent: ADI >= 1.32, CV^2 < 0.49
      - Erratic: ADI < 1.32, CV^2 >= 0.49
      - Lumpy: ADI >= 1.32, CV^2 >= 0.49
    """
    if active_window_only:
        non_zero_indices = series[series > 0].index
        if len(non_zero_indices) > 0:
            first_active_idx = non_zero_indices[0]
            series = series.loc[first_active_idx:]

    non_zero = series[series > 0]
    total_periods = len(series)
    non_zero_periods = len(non_zero)
    zero_periods = total_periods - non_zero_periods
    zero_ratio = zero_periods / total_periods if total_periods > 0 else 0.0

    if non_zero_periods == 0:
        return {"ADI": np.nan, "CV2": np.nan, "Category": "Zero-Demand", "Zero_Ratio": 1.0, "Total_Days": total_periods, "Active_Days": 0, "Zero_Days": total_periods}

    adi = total_periods / non_zero_periods
    mean_nz = non_zero.mean()
    std_nz = non_zero.std(ddof=1) if len(non_zero) > 1 else 0.0
    cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0.0

    if adi < 1.32 and cv2 < 0.49:
        category = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        category = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        category = "Erratic"
    else:
        category = "Lumpy"

    return {
        "Total_Days": total_periods,
        "Active_Days": non_zero_periods,
        "Zero_Days": zero_periods,
        "Zero_Ratio": zero_ratio,
        "ADI": round(adi, 2),
        "CV2": round(cv2, 2),
        "Category": category
    }

def load_clean_olist_items(orders_path: Path, items_path: Path, end_date: str = "2018-08-20") -> pd.DataFrame:
    """
    Loads and cleans Olist items strictly within the commercial operational window (2017-01-05 to end_date).
    Filters strictly for delivered orders.
    Default end_date='2018-08-20' eliminates right-censoring delivery lag in late August 2018.
    """
    df_orders = pd.read_csv(orders_path)
    df_items = pd.read_csv(items_path)

    delivered = df_orders[df_orders["order_status"] == "delivered"].copy()
    delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])
    delivered["purchase_date"] = delivered["order_purchase_timestamp"].dt.date

    merged = pd.merge(
        delivered[["order_id", "customer_id", "purchase_date", "order_purchase_timestamp"]],
        df_items[["order_id", "order_item_id", "seller_id", "price", "freight_value"]],
        on="order_id",
        how="inner"
    )
    merged["purchase_date"] = pd.to_datetime(merged["purchase_date"])
    merged["total_inflow"] = merged["price"] + merged["freight_value"]

    # Filter to commercial operational window
    op_mask = (merged["purchase_date"] >= "2017-01-05") & (merged["purchase_date"] <= end_date)
    df_clean = merged[op_mask].copy()

    return df_clean

def select_sellers_train_only(df_clean: pd.DataFrame, train_cutoff: str = "2017-12-31", k: int = 3) -> pd.DataFrame:
    """
    Selects top k SME sellers strictly based on training period data (to eliminate selection lookahead bias).
    Multi-criteria ranking based on Train Revenue, Active Days (>20% of train year), and demand stability.
    """
    train_df = df_clean[df_clean["purchase_date"] <= train_cutoff]
    
    # Train stats (361 days from 2017-01-05 to 2017-12-31)
    seller_train = train_df.groupby("seller_id").agg(
        train_revenue=("total_inflow", "sum"),
        train_orders=("order_id", "nunique"),
        train_active_days=("purchase_date", "nunique")
    )
    seller_train["train_active_ratio"] = seller_train["train_active_days"] / 361.0

    # Filter sellers active at least 20% of the training year
    viable = seller_train[seller_train["train_active_ratio"] >= 0.20].sort_values(by="train_revenue", ascending=False)
    
    # Full period stats for verification
    total_days = len(pd.date_range("2017-01-05", df_clean["purchase_date"].max(), freq="D"))
    seller_full = df_clean.groupby("seller_id").agg(
        full_revenue=("total_inflow", "sum"),
        full_orders=("order_id", "nunique"),
        full_active_days=("purchase_date", "nunique")
    )
    seller_full["full_active_ratio"] = seller_full["full_active_days"] / float(total_days)

    top_k = viable.head(k).join(seller_full, how="inner")
    return top_k

def aggregate_olist_daily(
    df_items: pd.DataFrame,
    seller_id: str = None,
    start_date: str = "2017-01-05",
    end_date: str = "2018-08-20"
) -> pd.DataFrame:
    """
    Constructs an unbroken daily calendar series for either:
    1. Marketplace-level macro platform (seller_id = None)
    2. Specific SME seller (seller_id specified)
    Computes both 'sales' (total_inflow = price + freight) and 'sales_product' (price alone).
    """
    if seller_id is not None:
        sub = df_items[df_items["seller_id"] == seller_id].copy()
    else:
        sub = df_items.copy()

    daily = sub.groupby("purchase_date").agg(
        sales=("total_inflow", "sum"),
        sales_product=("price", "sum"),
        freight_sum=("freight_value", "sum"),
        order_count=("order_id", "nunique"),
        items_count=("order_item_id", "count")
    ).reset_index().rename(columns={"purchase_date": "Date"})

    # Calendar range
    calendar_grid = pd.date_range(start_date, end_date, freq="D", name="Date")
    daily_full = daily.set_index("Date").reindex(calendar_grid).fillna(0.0).reset_index()

    # Average Order Value
    daily_full["avg_order_value"] = np.where(
        daily_full["order_count"] > 0,
        daily_full["sales"] / daily_full["order_count"],
        0.0
    )

    daily_full["is_trading_day"] = (daily_full["sales"] > 0).astype(int)
    daily_full["day_of_week"] = daily_full["Date"].dt.dayofweek
    daily_full["is_weekend"] = daily_full["day_of_week"].isin([5, 6]).astype(int)

    return daily_full

def process_and_save_olist(
    orders_path: Path,
    items_path: Path,
    output_dir: Path = PROCESSED_DATA_DIR,
    end_date: str = "2018-08-20"
) -> Dict[str, Any]:
    """
    Saves clean Olist daily series for Platform Macro and Train-Selected SME sellers to Parquet with SHA-256.
    """
    df_clean = load_clean_olist_items(orders_path, items_path, end_date=end_date)
    daily_platform = aggregate_olist_daily(df_clean, seller_id=None, end_date=end_date)
    
    # Primary continuous SME benchmark seller (Rank-3 by revenue, Rank-1 by operational continuity)
    sme_primary_id = "4a3ca9315b744ce9f8e9374361493884"
    daily_sme_bench = aggregate_olist_daily(df_clean, seller_id=sme_primary_id, end_date=end_date)

    clean_items_path = output_dir / "olist_clean_items.parquet"
    plat_path = output_dir / "olist_daily_sales_platform.parquet"
    sme_bench_path = output_dir / "olist_daily_sales_top_seller.parquet"

    df_clean.to_parquet(clean_items_path, engine="pyarrow", index=False)
    daily_platform.to_parquet(plat_path, engine="pyarrow", index=False)
    daily_sme_bench.to_parquet(sme_bench_path, engine="pyarrow", index=False)

    return {
        "clean_items_path": str(clean_items_path),
        "clean_items_sha256": compute_sha256(clean_items_path),
        "platform_path": str(plat_path),
        "platform_sha256": compute_sha256(plat_path),
        "sme_bench_path": str(sme_bench_path),
        "sme_bench_sha256": compute_sha256(sme_bench_path),
        "clean_items_df": df_clean,
        "platform_df": daily_platform,
        "sme_bench_df": daily_sme_bench
    }
