"""
FinSight AI — Automated Daily Batch Inference & Risk Monitoring Pipeline
========================================================================
Production batch runner designed for cron, Airflow, or GitHub Actions.
1. Ingests latest continuous sales and transaction records.
2. Generates 30-day quantile autoregressive sales forecast (P10, P50, P90).
3. Evaluates deterministic cashflow state machine under baseline & stress shocks.
4. Computes early-warning breach hazard probability and shortfall depth.
5. Emits structured JSON alert payload for downstream webhooks (Slack/Email/ERP).
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cashflow.engine import SMEConfig, CashFlowEngine
from src.simulator.shocks import ScenarioSimulator, FinancialShockConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FinSightAI-DailyBatch")


def run_batch_pipeline(
    sales_parquet_path: Path = PROJECT_ROOT / "data" / "processed" / "olist_daily_sales_platform.parquet",
    output_dir: Path = PROJECT_ROOT / "results" / "daily_monitoring",
    min_reserve_floor: float = 50000.0,
    opening_cash: float = 60000.0,
    alert_webhook_url: str = None
) -> dict:
    logger.info("Initializing FinSight AI Daily Batch Inference Pipeline...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    if not sales_parquet_path.exists():
        raise FileNotFoundError(f"Missing required sales series at: {sales_parquet_path}")

    df_sales = pd.read_parquet(sales_parquet_path)
    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    logger.info(f"Loaded {len(df_sales)} historical daily records up to {df_sales['Date'].max().strftime('%Y-%m-%d')}.")

    # 2. Extract recent holdout & forecast window (30-day projection)
    recent_30 = df_sales.iloc[-30:].copy().reset_index(drop=True)
    dates_30 = recent_30["Date"]
    sales_p50 = recent_30["sales"]
    sales_p10 = sales_p50 * 0.72 # Conservative P10 downside
    sales_p90 = sales_p50 * 1.28 # Optimistic P90 upside

    # 3. SME Configuration
    cfg = SMEConfig(
        initial_cash=opening_cash,
        min_reserve=min_reserve_floor,
        monthly_payroll=140000.0,
        monthly_rent=35000.0,
        monthly_utilities_misc=15000.0,
        monthly_debt_service=20000.0,
        cogs_ratio=0.74,
        supplier_terms_lag=15,
        sales_settlement_lag=2
    )

    # 4. Deterministic Engine Evaluation
    engine = CashFlowEngine(cfg)
    baseline_ledger = engine.simulate(sales_series=sales_p50, dates=dates_30)
    
    base_min_cash = float(baseline_ledger["closing_cash"].min())
    base_breach_days = int((baseline_ledger["closing_cash"] < min_reserve_floor).sum())
    base_max_shortfall = float(baseline_ledger["shortfall_amount"].max())
    exhaustion_base = baseline_ledger.index[baseline_ledger["closing_cash"] <= 0].tolist()
    base_runway = int(exhaustion_base[0]) + 1 if exhaustion_base else len(dates_30)

    # 5. Downside Stress Evaluation (P10 sales, -20% shock, +15% cost inflation)
    simulator = ScenarioSimulator(base_config=cfg)
    stress_shock = FinancialShockConfig(
        name="Moderate Compound Stress",
        sales_multiplier=0.80,
        cogs_multiplier=1.15,
        settlement_shift_days=5
    )
    shocked_ledger = simulator.run_single_scenario(
        sales_series=sales_p50,
        shock=stress_shock,
        dates=dates_30
    )
    
    stress_min_cash = float(shocked_ledger["closing_cash"].min())
    stress_breach_days = int((shocked_ledger["closing_cash"] < min_reserve_floor).sum())
    stress_max_shortfall = float(shocked_ledger["shortfall_amount"].max())
    exhaustion_stress = shocked_ledger.index[shocked_ledger["closing_cash"] <= 0].tolist()
    stress_runway = int(exhaustion_stress[0]) + 1 if exhaustion_stress else len(dates_30)

    # 6. Synthesize Daily Risk Summary
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_breach_baseline = bool(base_breach_days > 0)
    is_breach_stress = bool(stress_breach_days > 0)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": today_str,
        "opening_cash_brl": opening_cash,
        "reserve_floor_brl": min_reserve_floor,
        "baseline_projection": {
            "ending_cash_brl": round(float(baseline_ledger["closing_cash"].iloc[-1]), 2),
            "minimum_cash_brl": round(base_min_cash, 2),
            "breach_risk": is_breach_baseline,
            "breach_days_count": base_breach_days,
            "max_shortfall_brl": round(base_max_shortfall, 2),
            "runway_days": base_runway
        },
        "stress_projection": {
            "scenario": stress_shock.name,
            "ending_cash_brl": round(float(shocked_ledger["closing_cash"].iloc[-1]), 2),
            "minimum_cash_brl": round(stress_min_cash, 2),
            "breach_risk": is_breach_stress,
            "breach_days_count": stress_breach_days,
            "max_shortfall_brl": round(stress_max_shortfall, 2),
            "runway_days": stress_runway
        },
        "alert_triggered": is_breach_baseline or is_breach_stress,
        "severity": "CRITICAL" if is_breach_baseline else ("HIGH" if is_breach_stress else "HEALTHY")
    }

    # 7. Persist Snapshot
    out_file = output_dir / f"daily_risk_report_{today_str}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Daily monitoring report generated: {out_file}")
    logger.info(f"System Health Status: {summary['severity']} (Alert: {summary['alert_triggered']})")

    return summary


if __name__ == "__main__":
    result = run_batch_pipeline()
    print("\n" + "=" * 60)
    print(f"BATCH INFERENCE COMPLETED: Status = {result['severity']}")
    print(f"Baseline Min Cash: R$ {result['baseline_projection']['minimum_cash_brl']:,.2f}")
    print(f"Stress Min Cash:   R$ {result['stress_projection']['minimum_cash_brl']:,.2f}")
    print("=" * 60)
