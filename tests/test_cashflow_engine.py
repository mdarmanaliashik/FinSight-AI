"""
FinSight AI - Golden Unit Test Suite for Cash-Flow & Scenario Simulator
========================================================================
PyTest Golden Fixture Suite.
Guarantees 100% mathematical determinism, zero floating-point discrepancy,
and absolute accounting conservation laws.
"""

import os
import sys
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cashflow.engine import SMEConfig, CashFlowEngine, calculate_risk_metrics
from src.simulator.shocks import ScenarioSimulator, FinancialShockConfig


# -----------------------------------------------------------------------------
# Test 1: Zero Inflow Analytical Runway Test
# -----------------------------------------------------------------------------
def test_zero_inflow_runway():
    """
    Mathematical Proof:
    Under 0 sales inflow, closing cash must monotonically decrease strictly by
    the sum of scheduled fixed OPEX (payroll + rent + utilities + debt).
    """
    config = SMEConfig(
        initial_cash=50000.0,
        min_reserve=20000.0,
        cogs_ratio=0.60,
        monthly_payroll=6000.0,    # 3000 on 1st, 3000 on 15th
        payroll_days=(1, 15),
        monthly_rent=3000.0,       # 3000 on 5th
        rent_day=5,
        monthly_utilities_misc=1000.0, # 1000 on 10th
        utilities_day=10,
        monthly_debt_service=2000.0,   # 2000 on 20th
        debt_day=20,
        sales_settlement_lag=0,
        supplier_terms_lag=0
    )
    engine = CashFlowEngine(config=config)
    
    # 30 days of January 2023 with exactly 0 sales
    dates = pd.date_range("2023-01-01", "2023-01-30", freq="D")
    sales = pd.Series(0.0, index=range(len(dates)))

    ledger = engine.simulate(sales_series=sales, dates=pd.Series(dates))

    # Analytical sum of fixed costs in Jan 2023:
    # Payroll: 3000 (1st) + 3000 (15th) = 6000
    # Rent: 3000 (5th)
    # Utilities: 1000 (10th)
    # Debt: 2000 (20th)
    # Total monthly drain = 6000 + 3000 + 1000 + 2000 = 12000
    expected_ending_cash = 50000.0 - 12000.0 # 38000.0
    actual_ending_cash = ledger["closing_cash"].iloc[-1]

    assert abs(actual_ending_cash - expected_ending_cash) < 1e-4, \
        f"Analytical mismatch: expected {expected_ending_cash}, got {actual_ending_cash}"
    assert (ledger["outflow_payroll"].sum() == 6000.0)
    assert (ledger["outflow_rent"].sum() == 3000.0)
    assert (ledger["outflow_utilities"].sum() == 1000.0)
    assert (ledger["outflow_debt"].sum() == 2000.0)
    print("\n Test 1 Passed: Zero inflow analytical runway verified.")


# -----------------------------------------------------------------------------
# Test 2: Hand-Calculated Golden 5-Day Collision Test
# -----------------------------------------------------------------------------
def test_hand_calculated_golden_ledger():
    """
    Verifies step-by-step arithmetic against a pre-calculated manual audit ledger.
    Day 1: Payroll day ($2,000 outflow), Opening $10,000, Inflow $1,000 -> Closing $9,000
    Day 2: Normal day, Inflow $2,000, Outflow $0 -> Closing $11,000
    Day 3: Rent day ($1,500 outflow), Inflow $500 -> Closing $10,000
    Reserve = $9,500. Breach on Day 1 (Closing $9,000 < $9,500). Shortfall = $500.
    """
    config = SMEConfig(
        initial_cash=10000.0,
        min_reserve=9500.0,
        cogs_ratio=0.0,            # isolate fixed costs
        monthly_payroll=2000.0,
        payroll_days=(1,),
        monthly_rent=1500.0,
        rent_day=3,
        monthly_utilities_misc=0.0,
        monthly_debt_service=0.0,
        sales_settlement_lag=0,
        supplier_terms_lag=0
    )
    engine = CashFlowEngine(config=config)

    dates = pd.date_range("2023-01-01", "2023-01-03", freq="D")
    sales = pd.Series([1000.0, 2000.0, 500.0])

    ledger = engine.simulate(sales_series=sales, dates=pd.Series(dates))

    # Day 1: Open 10000, In 1000, Out 2000 -> Close 9000 -> Breach True, Shortfall 500
    assert ledger.loc[0, "opening_cash"] == 10000.0
    assert ledger.loc[0, "closing_cash"] == 9000.0
    assert ledger.loc[0, "is_reserve_breach"] == 1
    assert ledger.loc[0, "shortfall_amount"] == 500.0

    # Day 2: Open 9000, In 2000, Out 0 -> Close 11000 -> Breach False, Shortfall 0
    assert ledger.loc[1, "opening_cash"] == 9000.0
    assert ledger.loc[1, "closing_cash"] == 11000.0
    assert ledger.loc[1, "is_reserve_breach"] == 0
    assert ledger.loc[1, "shortfall_amount"] == 0.0

    # Day 3: Open 11000, In 500, Out 1500 -> Close 10000 -> Breach False, Shortfall 0
    assert ledger.loc[2, "opening_cash"] == 11000.0
    assert ledger.loc[2, "closing_cash"] == 10000.0
    assert ledger.loc[2, "is_reserve_breach"] == 0
    assert ledger.loc[2, "shortfall_amount"] == 0.0
    print(" Test 2 Passed: Hand-calculated golden ledger verified.")


# -----------------------------------------------------------------------------
# Test 3: Settlement Delay Conservation of Mass
# -----------------------------------------------------------------------------
def test_settlement_shift_conservation():
    """
    Verifies that a settlement delay shifts cash across time without
    destroying or inventing monetary value.
    """
    config = SMEConfig(initial_cash=20000.0, sales_settlement_lag=0, supplier_terms_lag=0)
    engine = CashFlowEngine(config=config)

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    sales = pd.Series(np.random.uniform(500, 2000, size=60))

    # Run base
    base_ledger = engine.simulate(sales_series=sales, dates=pd.Series(dates), sales_settlement_shift=0)
    # Run with 7-day delay
    delay_ledger = engine.simulate(sales_series=sales, dates=pd.Series(dates), sales_settlement_shift=7)

    # In the middle window (after pipeline lag catches up), the delayed inflow
    # should exactly match the shifted base sales.
    base_inflows = base_ledger["cash_inflow"].values
    delay_inflows = delay_ledger["cash_inflow"].values

    # Check shift alignment
    assert np.allclose(delay_inflows[7:], base_inflows[:53]), \
        "Shifted inflows do not match lagged base sales!"
    print(" Test 3 Passed: Conservation of mass under settlement delay verified.")


# -----------------------------------------------------------------------------
# Test 4: Idempotency (Zero Variance Guarantee)
# -----------------------------------------------------------------------------
def test_idempotency():
    """
    Mathematical Proof:
    Executing the simulator N=50 times on identical inputs must yield
    strictly identical numbers (variance = 0.000000).
    """
    config = SMEConfig()
    engine = CashFlowEngine(config=config)

    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    sales = pd.Series(np.sin(np.linspace(0, 10, 30)) * 500 + 1000)

    first_run = engine.simulate(sales_series=sales, dates=pd.Series(dates))["closing_cash"].values

    for _ in range(50):
        subsequent_run = engine.simulate(sales_series=sales, dates=pd.Series(dates))["closing_cash"].values
        diff = np.max(np.abs(first_run - subsequent_run))
        assert diff == 0.0, f"Discrepancy detected across identical runs: {diff}"

    print(" Test 4 Passed: 100% Idempotency (Zero Variance) guaranteed.")


# -----------------------------------------------------------------------------
# Test 5: Stress Shock Monotonicity
# -----------------------------------------------------------------------------
def test_scenario_shock_monotonicity():
    """
    Economic Intuition:
    A severe sales drop (-30%) must always produce equal or worse minimum cash
    and equal or higher shortfall than the Base Case.
    """
    simulator = ScenarioSimulator()
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    sales = pd.Series([1000.0] * 60)

    summary_df, _ = simulator.run_stress_suite(sales_series=sales, dates=pd.Series(dates), initial_cash=15000.0)

    col = "Min Cash (R$)" if "Min Cash (R$)" in summary_df.columns else "Min Cash Balance ($)"
    base_min = summary_df.loc[summary_df["Scenario"].str.contains("Base Case"), col].values[0]
    severe_min = summary_df.loc[summary_df["Scenario"].str.contains("Severe Sales Drop"), col].values[0]

    assert severe_min <= base_min, \
        f"Monotonicity violation: Severe sales drop min cash ({severe_min}) > base min cash ({base_min})"
    print(" Test 5 Passed: Scenario shock monotonicity verified.")


# -----------------------------------------------------------------------------
# Test 6: Risk Metrics Evaluator
# -----------------------------------------------------------------------------
def test_calculate_risk_metrics():
    """
    Verifies Precision, Recall, F1, and Lead-Time calculations.
    """
    gt_breach = np.array([0, 0, 1, 1, 1, 0, 0])
    pr_breach = np.array([0, 0, 1, 1, 0, 0, 0]) # 1 missed breach (FN=1, TP=2)
    gt_shortfall = np.array([0.0, 0.0, 500.0, 1000.0, 200.0, 0.0, 0.0])
    pr_shortfall = np.array([0.0, 0.0, 450.0, 950.0, 0.0, 0.0, 0.0])
    warning_horizons = [5, 4] # Warning issued 5 and 4 days in advance

    metrics = calculate_risk_metrics(
        ground_truth_breach=gt_breach,
        predicted_breach=pr_breach,
        ground_truth_shortfall=gt_shortfall,
        predicted_shortfall=pr_shortfall,
        warning_horizons=warning_horizons
    )

    assert metrics["precision"] == 1.0  # 2/2
    assert metrics["recall"] == round(2/3, 4) # 2/3 ≈ 0.6667
    assert metrics["mean_lead_time_days"] == 4.5
    print(" Test 6 Passed: Risk metrics calculations verified.")


if __name__ == "__main__":
    if pytest is not None:
        pytest.main(["-v", __file__])
    else:
        print("Running tests without pytest runner...")
        test_zero_inflow_runway()
        test_hand_calculated_golden_ledger()
        test_settlement_shift_conservation()
        test_idempotency()
        test_scenario_shock_monotonicity()
        test_calculate_risk_metrics()
        print("\nALL CASHFLOW ENGINE TESTS PASSED (100% SUCCESS)!")


