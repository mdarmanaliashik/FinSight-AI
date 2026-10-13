"""
FinSight AI — Financial Scenario Simulator (Stress-Testing Engine)
===================================================================
Deterministic macro & micro financial shock simulator.

Supports two shock-application modes:
  • Full-series (legacy): shock applied from day 0 — compares "alternative business".
  • Onset-dated (new): shock starts at a specific index t₀, so pre-shock history
    is identical to baseline.  This is the correct design for evaluating
    early-warning lead-time (Experiment B in Phase 4).

Shocks Simulated:
  C1. Sales Contraction (−20%, −30%)
  C2. Cost Inflation (+20%)
  C3. Settlement Delay (+7d, +14d)
  C4. Compound Crisis (Black Swan: −25% sales, +15% cost, +7d delay)

All scenarios inherit the conservation invariant from the engine
(integer centavo arithmetic, C_T ≡ C_0 + Σ I − Σ O).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.cashflow.engine import SMEConfig, CashFlowEngine


@dataclass
class FinancialShockConfig:
    """
    Parameters for a single financial stress scenario.
    """
    name: str = "Base Case"
    sales_multiplier: float = 1.0          # 0.80 = 20% decline in sales volume
    cogs_multiplier: float = 1.0           # 1.20 = 20% increase in supplier unit costs
    settlement_shift_days: int = 0         # +7 = receivables collection delayed by 7 days
    onset_day_index: Optional[int] = None  # Day index where shock begins (None = from day 0)


class ScenarioSimulator:
    """
    Deterministic Financial Scenario Simulator.
    Calculates multi-horizon stress impact across multiple economic shocks.
    """

    def __init__(self, base_config: Optional[SMEConfig] = None):
        self.base_config = base_config or SMEConfig()
        self.engine = CashFlowEngine(config=self.base_config)

    def run_single_scenario(
        self,
        sales_series: pd.Series,
        dates: pd.Series,
        shock: FinancialShockConfig,
        initial_cash: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Runs a single simulated scenario through the deterministic engine.
        """
        return self.engine.simulate(
            sales_series=sales_series,
            dates=dates,
            initial_cash=initial_cash,
            sales_volume_multiplier=shock.sales_multiplier,
            cogs_cost_multiplier=shock.cogs_multiplier,
            sales_settlement_shift=shock.settlement_shift_days,
            shock_onset_idx=shock.onset_day_index,
        )

    @staticmethod
    def default_scenarios(
        onset_idx: Optional[int] = None,
    ) -> List[FinancialShockConfig]:
        """Return the standard suite of stress scenarios."""
        return [
            FinancialShockConfig(
                name="Base Case (Status Quo)",
                sales_multiplier=1.0, cogs_multiplier=1.0,
                settlement_shift_days=0, onset_day_index=None,
            ),
            FinancialShockConfig(
                name="C1: Sales Contraction (−20%)",
                sales_multiplier=0.80, cogs_multiplier=1.0,
                settlement_shift_days=0, onset_day_index=onset_idx,
            ),
            FinancialShockConfig(
                name="C1: Severe Sales Drop (−30%)",
                sales_multiplier=0.70, cogs_multiplier=1.0,
                settlement_shift_days=0, onset_day_index=onset_idx,
            ),
            FinancialShockConfig(
                name="C2: Supplier Cost Spike (+20%)",
                sales_multiplier=1.0, cogs_multiplier=1.20,
                settlement_shift_days=0, onset_day_index=onset_idx,
            ),
            FinancialShockConfig(
                name="C3: Liquidity Freeze (+7d Delay)",
                sales_multiplier=1.0, cogs_multiplier=1.0,
                settlement_shift_days=7, onset_day_index=onset_idx,
            ),
            FinancialShockConfig(
                name="C3: Severe Delay (+14d Delay)",
                sales_multiplier=1.0, cogs_multiplier=1.0,
                settlement_shift_days=14, onset_day_index=onset_idx,
            ),
            FinancialShockConfig(
                name="C4: Compound Crisis (Black Swan)",
                sales_multiplier=0.75, cogs_multiplier=1.15,
                settlement_shift_days=7, onset_day_index=onset_idx,
            ),
        ]

    def run_stress_suite(
        self,
        sales_series: pd.Series,
        dates: pd.Series,
        initial_cash: Optional[float] = None,
        shock_onset_idx: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute the pre-registered shock suite (Base, C1, C2, C3, C4).
        Returns a comparative summary table and full trajectory ledgers.

        Parameters
        ----------
        shock_onset_idx : int or None
            If set, shocks begin at this day index. Pre-shock history is
            identical to baseline (correct for lead-time evaluation).
            If None, shocks apply from day 0 (legacy full-series mode).
        """
        scenarios = self.default_scenarios(onset_idx=shock_onset_idx)
        summary_rows = []
        trajectories = {}

        for shock in scenarios:
            ledger = self.run_single_scenario(
                sales_series=sales_series,
                dates=dates,
                shock=shock,
                initial_cash=initial_cash,
            )
            trajectories[shock.name] = ledger

            # Key financial stress metrics
            breach_days = int(ledger["is_reserve_breach"].sum())
            insolvent_days = int(ledger["is_insolvent"].sum())
            min_cash = float(ledger["closing_cash"].min())
            final_cash = float(ledger["closing_cash"].iloc[-1])
            max_shortfall = float(ledger["shortfall_amount"].max())
            total_days = len(ledger)
            breach_pct = round(breach_days / total_days * 100, 1) if total_days > 0 else 0.0

            # First breach date
            breach_records = ledger[ledger["is_reserve_breach"] == 1]
            first_breach_date = (
                breach_records["date"].iloc[0].strftime("%Y-%m-%d")
                if len(breach_records) > 0
                else "None"
            )

            # First insolvency date (balance < 0)
            insolvent_records = ledger[ledger["is_insolvent"] == 1]
            first_insolvency_date = (
                insolvent_records["date"].iloc[0].strftime("%Y-%m-%d")
                if len(insolvent_records) > 0
                else "None"
            )

            summary_rows.append({
                "Scenario": shock.name,
                "Sales Impact": f"{(shock.sales_multiplier - 1.0)*100:+.0f}%",
                "COGS Impact": f"{(shock.cogs_multiplier - 1.0)*100:+.0f}%",
                "Lag Shift": f"+{shock.settlement_shift_days}d",
                "Onset": f"Day {shock.onset_day_index}" if shock.onset_day_index else "Day 0",
                "First Breach Date": first_breach_date,
                "Breach Days": breach_days,
                "Breach %": f"{breach_pct}%",
                "First Insolvency": first_insolvency_date,
                "Insolvent Days": insolvent_days,
                "Min Cash (R$)": round(min_cash, 2),
                "Max Shortfall (R$)": round(max_shortfall, 2),
                "Ending Cash (R$)": round(final_cash, 2),
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, trajectories
