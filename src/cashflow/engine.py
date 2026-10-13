"""
FinSight AI — Deterministic Cash-Flow Engine & Risk Detection System
=====================================================================
Cash-basis daily accounting state machine for SME liquidity simulation.

State Equation:
    C_t = C_{t-1} + I_t - O_t

All arithmetic uses integer centavos (BRL) or pence (GBP) internally
to guarantee exact conservation: C_T ≡ C_0 + Σ I_t − Σ O_t
(IEEE-754 floating-point rounding is eliminated).

Where:
    C_t: Closing cash balance on day t (integer centavos)
    I_t: Realized cash inflow (settled sales receivables)
    O_t: Cash outflows (COGS, Payroll, Rent, Utilities, Debt)
    Breach_t: 𝟙(C_t < R)
    Shortfall_t: max(0, R − C_t)

Scope: This module tests H3 (deterministic stress testing).
       H2 (early-warning detection quality) is evaluated in Phase 4.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Helpers: float ↔ integer-centavo conversion
# ---------------------------------------------------------------------------

def _to_centavos(value: float) -> int:
    """Convert a float currency value to integer centavos (round half-up)."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * 100
               if abs(value) < 1e-9 else
               Decimal(str(round(value, 2))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) * 100)


def _to_centavos_array(arr) -> np.ndarray:
    """Convert an array of float values to integer centavos."""
    return np.round(np.asarray(arr, dtype=float) * 100).astype(np.int64)


def _from_centavos(centavos: int) -> float:
    """Convert integer centavos back to float currency value."""
    return centavos / 100.0


def _from_centavos_array(arr: np.ndarray) -> np.ndarray:
    """Convert array of integer centavos to float currency values."""
    return arr.astype(float) / 100.0


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class SMEConfig:
    """
    Cash-basis accounting configuration for an SME business entity.

    Parameters are designed to be calibrated from observed seller data.
    See ``calibrate_from_revenue()`` for data-driven initialization.
    """
    initial_cash: float = 30000.0          # Opening cash balance at t=0
    min_reserve: float = 15000.0           # Minimum operating reserve threshold (R)
    sales_settlement_lag: int = 2          # Days to settle receivables
    cogs_ratio: float = 0.60              # Variable COGS as fraction of sales
    cogs_fixed_fraction: float = 0.0       # Fraction of COGS that is fixed (0=fully variable)
    supplier_terms_lag: int = 15           # Days after sale when supplier invoice is due
    monthly_payroll: float = 8000.0        # Total monthly salary obligations
    payroll_days: Tuple[int, ...] = (1, 15)
    monthly_rent: float = 2500.0
    rent_day: int = 5
    monthly_utilities_misc: float = 1200.0
    utilities_day: int = 10
    monthly_debt_service: float = 1500.0
    debt_day: int = 20

    @property
    def monthly_fixed_opex(self) -> float:
        """Total monthly fixed operating expenses (Payroll + Rent + Utilities + Debt)."""
        return (self.monthly_payroll + self.monthly_rent
                + self.monthly_utilities_misc + self.monthly_debt_service)

    @property
    def daily_avg_fixed_opex(self) -> float:
        """30-day average daily fixed OPEX — used for reserve protocol."""
        return self.monthly_fixed_opex / 30.0

    @staticmethod
    def calibrate_from_revenue(
        daily_sales: pd.Series,
        *,
        opex_revenue_ratio: float = 0.25,
        cogs_ratio: float = 0.55,
        cogs_fixed_fraction: float = 0.0,
        initial_cash_months: float = 0.5,
        settlement_lag: int = 2,
        supplier_lag: int = 15,
    ) -> "SMEConfig":
        """
        Data-driven calibration of SME accounting parameters.

        Parameters
        ----------
        daily_sales : pd.Series
            Daily sales values (active days only recommended).
        opex_revenue_ratio : float
            Total fixed OPEX as fraction of mean monthly revenue (default 25%).
        cogs_ratio : float
            Variable cost of goods sold as fraction of sales.
        cogs_fixed_fraction : float
            Fraction of COGS that is fixed (inventory already purchased).
            0.0 = fully variable, 1.0 = fully fixed.
        initial_cash_months : float
            Initial cash as multiple of monthly revenue (default 0.5 months).
        settlement_lag : int
            Days to settle receivables.
        supplier_lag : int
            Net supplier payment terms in days.

        Returns
        -------
        SMEConfig with calibrated parameters and documented provenance.
        """
        monthly_revenue = float(daily_sales.mean()) * 30.0
        monthly_opex = monthly_revenue * opex_revenue_ratio

        # Allocate fixed OPEX: 50% payroll, 20% rent, 15% utilities, 15% debt
        payroll = round(monthly_opex * 0.50, 2)
        rent = round(monthly_opex * 0.20, 2)
        utilities = round(monthly_opex * 0.15, 2)
        debt = round(monthly_opex * 0.15, 2)
        initial_cash = round(monthly_revenue * initial_cash_months, 2)

        # Protocol: R = 30-day average daily fixed OPEX × 30
        # i.e., min_reserve = 1 month of fixed OPEX
        min_reserve = round(monthly_opex, 2)

        return SMEConfig(
            initial_cash=initial_cash,
            min_reserve=min_reserve,
            sales_settlement_lag=settlement_lag,
            cogs_ratio=cogs_ratio,
            cogs_fixed_fraction=cogs_fixed_fraction,
            supplier_terms_lag=supplier_lag,
            monthly_payroll=payroll,
            payroll_days=(1, 15),
            monthly_rent=rent,
            rent_day=5,
            monthly_utilities_misc=utilities,
            utilities_day=10,
            monthly_debt_service=debt,
            debt_day=20,
        )


# ---------------------------------------------------------------------------
#  Engine
# ---------------------------------------------------------------------------

class CashFlowEngine:
    """
    Deterministic cash-basis financial state engine.

    All internal arithmetic uses integer centavos to guarantee exact
    conservation invariant:  C_T ≡ C_0 + Σ I_t − Σ O_t
    """

    def __init__(self, config: Optional[SMEConfig] = None):
        self.config = config or SMEConfig()

    def simulate(
        self,
        sales_series: pd.Series,
        dates: pd.Series,
        initial_cash: Optional[float] = None,
        custom_inflow_override: Optional[np.ndarray] = None,
        custom_cogs_override: Optional[np.ndarray] = None,
        sales_settlement_shift: int = 0,
        cogs_cost_multiplier: float = 1.0,
        sales_volume_multiplier: float = 1.0,
        shock_onset_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Execute deterministic day-by-day cash-flow accounting.

        Parameters
        ----------
        shock_onset_idx : int or None
            If provided, sales_volume_multiplier and cogs_cost_multiplier
            are applied only from this day index onward (0-based).
            Days before shock_onset_idx use multiplier=1.0.
        """
        n = len(sales_series)
        if n == 0:
            raise ValueError("Sales series cannot be empty.")

        dates_dt = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        raw_sales_float = sales_series.values.astype(float).copy()

        # Apply volume multiplier (optionally from shock onset)
        if shock_onset_idx is not None and shock_onset_idx < n:
            raw_sales_float[:shock_onset_idx] *= 1.0  # no-op pre-shock
            raw_sales_float[shock_onset_idx:] *= sales_volume_multiplier
        else:
            raw_sales_float *= sales_volume_multiplier

        # Convert to integer centavos
        raw_sales = _to_centavos_array(raw_sales_float)

        # 1. Inflow calculation with settlement delay
        effective_lag = max(0, self.config.sales_settlement_lag + sales_settlement_shift)

        if custom_inflow_override is not None:
            cash_inflow = _to_centavos_array(custom_inflow_override)
        else:
            cash_inflow = np.zeros(n, dtype=np.int64)
            if effective_lag < n:
                init_fill = int(np.mean(raw_sales[:min(7, n)]))
                cash_inflow[:effective_lag] = init_fill
                cash_inflow[effective_lag:] = raw_sales[:n - effective_lag]
            else:
                cash_inflow[:] = int(np.mean(raw_sales))

        # 2. Supplier outflow (COGS) with payment terms
        supplier_lag = max(0, self.config.supplier_terms_lag)

        # Compute effective COGS multiplier per day
        cogs_mult = np.ones(n, dtype=float)
        if shock_onset_idx is not None and shock_onset_idx < n:
            cogs_mult[shock_onset_idx:] = cogs_cost_multiplier
        else:
            cogs_mult[:] = cogs_cost_multiplier

        # COGS: variable portion scales with sales × multiplier,
        #        fixed portion = base_cogs × cogs_fixed_fraction (doesn't shrink with sales)
        base_cogs_float = raw_sales_float * self.config.cogs_ratio
        if self.config.cogs_fixed_fraction > 0:
            # Reference COGS (at multiplier=1.0, volume=1.0)
            ref_sales = sales_series.values.astype(float)
            ref_cogs = ref_sales * self.config.cogs_ratio
            fixed_part = ref_cogs * self.config.cogs_fixed_fraction
            variable_part = base_cogs_float * (1.0 - self.config.cogs_fixed_fraction)
            total_cogs_float = (fixed_part + variable_part) * cogs_mult
        else:
            total_cogs_float = base_cogs_float * cogs_mult

        base_cogs = _to_centavos_array(total_cogs_float)

        if custom_cogs_override is not None:
            cogs_outflow = _to_centavos_array(custom_cogs_override)
        else:
            cogs_outflow = np.zeros(n, dtype=np.int64)
            if supplier_lag < n:
                init_cogs = int(np.mean(base_cogs[:min(7, n)]))
                cogs_outflow[:supplier_lag] = init_cogs
                cogs_outflow[supplier_lag:] = base_cogs[:n - supplier_lag]
            else:
                cogs_outflow[:] = int(np.mean(base_cogs))

        # 3. Fixed operating & debt outflow schedules (integer centavos)
        payroll_per = _to_centavos_array(
            [self.config.monthly_payroll / len(self.config.payroll_days)]
        )[0]
        rent_cents = _to_centavos_array([self.config.monthly_rent])[0]
        util_cents = _to_centavos_array([self.config.monthly_utilities_misc])[0]
        debt_cents = _to_centavos_array([self.config.monthly_debt_service])[0]

        payroll_outflow = np.zeros(n, dtype=np.int64)
        rent_outflow = np.zeros(n, dtype=np.int64)
        utilities_outflow = np.zeros(n, dtype=np.int64)
        debt_outflow = np.zeros(n, dtype=np.int64)

        for i, dt in enumerate(dates_dt):
            day = dt.day
            if day in self.config.payroll_days:
                payroll_outflow[i] = payroll_per
            if day == self.config.rent_day:
                rent_outflow[i] = rent_cents
            if day == self.config.utilities_day:
                utilities_outflow[i] = util_cents
            if day == self.config.debt_day:
                debt_outflow[i] = debt_cents

        total_outflow = cogs_outflow + payroll_outflow + rent_outflow + utilities_outflow + debt_outflow
        net_cash_flow = cash_inflow - total_outflow

        # 4. State integration loop (integer centavos — exact)
        init_cash_cents = _to_centavos_array(
            [self.config.initial_cash if initial_cash is None else initial_cash]
        )[0]

        opening_cash = np.zeros(n, dtype=np.int64)
        closing_cash = np.zeros(n, dtype=np.int64)
        curr = init_cash_cents

        for i in range(n):
            opening_cash[i] = curr
            curr = curr + net_cash_flow[i]
            closing_cash[i] = curr

        # 5. Conservation invariant assertion
        expected_final = init_cash_cents + int(np.sum(cash_inflow)) - int(np.sum(total_outflow))
        assert closing_cash[-1] == expected_final, (
            f"Conservation invariant violated: C_T={closing_cash[-1]} ≠ "
            f"C_0 + ΣI − ΣO = {expected_final}"
        )

        # 6. Risk assessment (integer centavos)
        reserve_cents = _to_centavos_array([self.config.min_reserve])[0]
        is_reserve_breach = (closing_cash < reserve_cents).astype(int)
        shortfall_amount = np.maximum(np.int64(0), reserve_cents - closing_cash)
        is_insolvent = (closing_cash < 0).astype(int)
        insolvency_amount = np.maximum(np.int64(0), -closing_cash)

        # Severity tiers
        # 0 = healthy, 1 = caution (< R), 2 = warning (< R/2), 3 = critical (< 0)
        severity = np.zeros(n, dtype=int)
        severity[closing_cash < reserve_cents] = 1
        severity[closing_cash < reserve_cents // 2] = 2
        severity[closing_cash < 0] = 3

        # 7. Build ledger — convert back to float for display
        ledger = pd.DataFrame({
            "date": dates_dt,
            "opening_cash": _from_centavos_array(opening_cash),
            "sales_booked": _from_centavos_array(raw_sales),
            "cash_inflow": _from_centavos_array(cash_inflow),
            "outflow_cogs": _from_centavos_array(cogs_outflow),
            "outflow_payroll": _from_centavos_array(payroll_outflow),
            "outflow_rent": _from_centavos_array(rent_outflow),
            "outflow_utilities": _from_centavos_array(utilities_outflow),
            "outflow_debt": _from_centavos_array(debt_outflow),
            "total_outflow": _from_centavos_array(total_outflow),
            "net_cash_flow": _from_centavos_array(net_cash_flow),
            "closing_cash": _from_centavos_array(closing_cash),
            "is_reserve_breach": is_reserve_breach,
            "shortfall_amount": _from_centavos_array(shortfall_amount),
            "is_insolvent": is_insolvent,
            "insolvency_amount": _from_centavos_array(insolvency_amount),
            "severity_tier": severity,
        })

        # Rolling 7-day average daily burn rate & estimated runway
        rolling_burn = ledger["net_cash_flow"].rolling(window=7, min_periods=1).mean()
        ledger["rolling_burn_7d"] = np.round(rolling_burn, 2)
        ledger["runway_days"] = np.where(
            rolling_burn < 0,
            np.round(np.maximum(0.0, ledger["closing_cash"] / (-rolling_burn)), 1),
            999.0,
        )

        return ledger


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------

def generate_anchored_ledger(
    daily_sales_df: pd.DataFrame,
    config: Optional[SMEConfig] = None,
    sales_col: str = "sales",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Takes an empirical daily sales parquet DataFrame and outputs a complete
    cash-basis historical cash ledger.
    """
    engine = CashFlowEngine(config=config)
    return engine.simulate(
        sales_series=daily_sales_df[sales_col],
        dates=daily_sales_df[date_col],
    )


def calculate_risk_metrics(
    ground_truth_breach: np.ndarray,
    predicted_breach: np.ndarray,
    ground_truth_shortfall: np.ndarray,
    predicted_shortfall: np.ndarray,
    warning_horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Calculates statistical risk detection metrics:
    Precision, Recall, F1, Shortfall MAE, and Mean Early Warning Lead Time.
    """
    gt_b = ground_truth_breach.astype(int)
    pr_b = predicted_breach.astype(int)

    tp = np.sum((gt_b == 1) & (pr_b == 1))
    fp = np.sum((gt_b == 0) & (pr_b == 1))
    fn = np.sum((gt_b == 1) & (pr_b == 0))
    tn = np.sum((gt_b == 0) & (pr_b == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(gt_b)) if len(gt_b) > 0 else 0.0

    shortfall_mae = float(np.mean(np.abs(ground_truth_shortfall - predicted_shortfall)))

    mean_lead_time = float(np.mean(warning_horizons)) if (warning_horizons and len(warning_horizons) > 0) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "shortfall_mae": round(shortfall_mae, 2),
        "mean_lead_time_days": round(mean_lead_time, 2),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
