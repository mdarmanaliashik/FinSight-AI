"""
FinSight AI - Tool-Grounded Copilot Function Calling Contracts
==============================================================
Strict Pydantic schemas and deterministic execution router.
Eliminates LLM mathematical hallucinations by grounding all quantitative
financial calculations in deterministic Python engines.
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.cashflow.engine import SMEConfig, CashFlowEngine
from src.simulator.shocks import ScenarioSimulator, FinancialShockConfig


# -----------------------------------------------------------------------------
# 1. Pydantic Tool Input / Output Schemas
# -----------------------------------------------------------------------------

class GetSalesForecastInput(BaseModel):
    horizon_days: int = Field(30, ge=1, le=90, description="Forecast horizon in days (1 to 90).")
    quantile: float = Field(0.50, description="Quantile level: 0.10 for conservative/downside, 0.50 for median point forecast, 0.90 for optimistic.")


class SalesForecastOutput(BaseModel):
    horizon_days: int
    quantile_label: str
    mean_daily_sales: float
    total_projected_sales: float
    min_daily_sales: float
    max_daily_sales: float
    trajectory_sample: Dict[str, float]


class GetCashFlowProjectionInput(BaseModel):
    days: int = Field(30, ge=1, le=90, description="Projection duration in days (1 to 90).")
    scenario_type: str = Field("median", description="Forecast scenario to use: 'median' (P50), 'downside' (P10), or 'upside' (P90).")


class CashFlowProjectionOutput(BaseModel):
    days: int
    scenario_type: str
    starting_cash: float
    ending_cash: float
    min_cash_balance: float
    min_reserve_limit: float
    is_reserve_breach: bool
    first_breach_day: Optional[int]
    total_inflow: float
    total_outflow: float
    shortfall_amount: float
    runway_days: int


class SimulateScenarioInput(BaseModel):
    sales_shock_pct: float = Field(0.0, description="Percentage change in sales revenue, e.g. -0.20 for 20% drop, +0.10 for 10% surge.")
    cost_shock_pct: float = Field(0.0, description="Percentage change in COGS supplier costs, e.g. +0.15 for 15% inflation.")
    delay_days: int = Field(0, ge=0, le=60, description="Days of delay in customer sales settlement payout.")


class ScenarioSimulationOutput(BaseModel):
    sales_shock_pct: float
    cost_shock_pct: float
    delay_days: int
    baseline_ending_cash: float
    shocked_ending_cash: float
    net_cash_impact: float
    is_reserve_breach: bool
    shortfall_amount: float
    baseline_runway_days: int
    shocked_runway_days: int
    runway_change_days: int


class GetReserveStatusInput(BaseModel):
    pass


class ReserveStatusOutput(BaseModel):
    current_cash: float
    min_reserve: float
    buffer_amount: float
    is_currently_breached: bool
    monthly_fixed_opex: float
    reserve_coverage_months: float


# -----------------------------------------------------------------------------
# 2. Tool Execution Engine
# -----------------------------------------------------------------------------

class FinSightToolKit:
    """
    Deterministic Financial Execution Engine for the LLM Copilot.
    Wraps CashFlowEngine, ScenarioSimulator, and LightGBM Forecasts into
    type-safe function call endpoints.
    """

    def __init__(
        self,
        config: SMEConfig,
        baseline_sales_p50: pd.Series,
        baseline_sales_p10: Optional[pd.Series] = None,
        baseline_sales_p90: Optional[pd.Series] = None,
        dates: Optional[pd.Series] = None
    ):
        self.config = config
        self.engine = CashFlowEngine(config=config)
        self.simulator = ScenarioSimulator(base_config=config)
        
        self.sales_p50 = baseline_sales_p50.reset_index(drop=True)
        self.sales_p10 = (baseline_sales_p10 if baseline_sales_p10 is not None else self.sales_p50 * 0.70).reset_index(drop=True)
        self.sales_p90 = (baseline_sales_p90 if baseline_sales_p90 is not None else self.sales_p50 * 1.30).reset_index(drop=True)
        
        if dates is not None:
            self.dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        else:
            self.dates = pd.date_range(start="2024-01-01", periods=len(self.sales_p50), freq="D")

    def get_sales_forecast(self, horizon_days: int = 30, quantile: float = 0.50) -> Dict[str, Any]:
        """Returns deterministic sales forecast statistics for a given horizon and quantile."""
        h = min(horizon_days, len(self.sales_p50))
        if quantile <= 0.25:
            series = self.sales_p10.iloc[:h]
            label = "P10 (Conservative Downside)"
        elif quantile >= 0.75:
            series = self.sales_p90.iloc[:h]
            label = "P90 (Optimistic Upside)"
        else:
            series = self.sales_p50.iloc[:h]
            label = "P50 (Median Expected)"

        date_strs = [d.strftime("%Y-%m-%d") for d in self.dates.iloc[:h]]
        sample_dict = {date_strs[i]: round(float(series.iloc[i]), 2) for i in range(min(7, h))}

        output = SalesForecastOutput(
            horizon_days=h,
            quantile_label=label,
            mean_daily_sales=round(float(series.mean()), 2),
            total_projected_sales=round(float(series.sum()), 2),
            min_daily_sales=round(float(series.min()), 2),
            max_daily_sales=round(float(series.max()), 2),
            trajectory_sample=sample_dict
        )
        return output.model_dump()

    def get_cashflow_projection(self, days: int = 30, scenario_type: str = "median") -> Dict[str, Any]:
        """Simulates deterministic daily ledger projection and checks reserve compliance."""
        h = min(days, len(self.sales_p50))
        sc_lower = scenario_type.lower()
        if "down" in sc_lower or "p10" in sc_lower or "worst" in sc_lower:
            sales = self.sales_p10.iloc[:h]
            sc_name = "P10 Downside"
        elif "up" in sc_lower or "p90" in sc_lower or "best" in sc_lower:
            sales = self.sales_p90.iloc[:h]
            sc_name = "P90 Upside"
        else:
            sales = self.sales_p50.iloc[:h]
            sc_name = "P50 Median"

        ledger = self.engine.simulate(sales_series=sales, dates=self.dates.iloc[:h])
        
        starting_cash = float(self.config.initial_cash)
        ending_cash = float(ledger["closing_cash"].iloc[-1])
        min_cash = float(ledger["closing_cash"].min())
        is_breach = bool(ledger["is_reserve_breach"].any())
        
        breach_indices = ledger.index[ledger["is_reserve_breach"]].tolist()
        first_breach_day = int(breach_indices[0]) + 1 if breach_indices else None
        
        shortfall = float(ledger["shortfall_amount"].max())
        
        # Calculate runway: days until closing_cash < 0
        exhaustion_indices = ledger.index[ledger["closing_cash"] <= 0].tolist()
        runway = int(exhaustion_indices[0]) + 1 if exhaustion_indices else h

        output = CashFlowProjectionOutput(
            days=h,
            scenario_type=sc_name,
            starting_cash=round(starting_cash, 2),
            ending_cash=round(ending_cash, 2),
            min_cash_balance=round(min_cash, 2),
            min_reserve_limit=round(float(self.config.min_reserve), 2),
            is_reserve_breach=is_breach,
            first_breach_day=first_breach_day,
            total_inflow=round(float(ledger["cash_inflow"].sum()), 2),
            total_outflow=round(float(ledger["total_outflow"].sum()), 2),
            shortfall_amount=round(shortfall, 2),
            runway_days=runway
        )
        return output.model_dump()

    def simulate_scenario(
        self,
        sales_shock_pct: float = 0.0,
        cost_shock_pct: float = 0.0,
        delay_days: int = 0
    ) -> Dict[str, Any]:
        """Applies external shocks and returns exact differential impact against baseline."""
        h = min(30, len(self.sales_p50))
        sales = self.sales_p50.iloc[:h]
        dates = self.dates.iloc[:h]

        baseline_ledger = self.engine.simulate(sales_series=sales, dates=dates)
        
        shock_cfg = FinancialShockConfig(
            name="Custom Scenario",
            sales_multiplier=float(1.0 + sales_shock_pct),
            cogs_multiplier=float(1.0 + cost_shock_pct),
            settlement_shift_days=int(delay_days)
        )
        shocked_ledger = self.simulator.run_single_scenario(
            sales_series=sales,
            shock=shock_cfg,
            dates=dates
        )

        base_end = float(baseline_ledger["closing_cash"].iloc[-1])
        shock_end = float(shocked_ledger["closing_cash"].iloc[-1])
        net_impact = shock_end - base_end
        
        is_breach = bool(shocked_ledger["is_reserve_breach"].any())
        shortfall = float(shocked_ledger["shortfall_amount"].max())

        # Runway calculation
        base_ex = baseline_ledger.index[baseline_ledger["closing_cash"] <= 0].tolist()
        base_runway = int(base_ex[0]) + 1 if base_ex else h

        shock_ex = shocked_ledger.index[shocked_ledger["closing_cash"] <= 0].tolist()
        shock_runway = int(shock_ex[0]) + 1 if shock_ex else h

        output = ScenarioSimulationOutput(
            sales_shock_pct=sales_shock_pct,
            cost_shock_pct=cost_shock_pct,
            delay_days=delay_days,
            baseline_ending_cash=round(base_end, 2),
            shocked_ending_cash=round(shock_end, 2),
            net_cash_impact=round(net_impact, 2),
            is_reserve_breach=is_breach,
            shortfall_amount=round(shortfall, 2),
            baseline_runway_days=base_runway,
            shocked_runway_days=shock_runway,
            runway_change_days=shock_runway - base_runway
        )
        return output.model_dump()

    def get_reserve_status(self) -> Dict[str, Any]:
        """Returns the current liquidity buffer, monthly OPEX, and coverage ratio."""
        curr = float(self.config.initial_cash)
        res = float(self.config.min_reserve)
        buffer_val = curr - res
        monthly_opex = float(self.config.monthly_payroll + self.config.monthly_rent + self.config.monthly_utilities_misc + self.config.monthly_debt_service)
        coverage_months = round(curr / monthly_opex, 2) if monthly_opex > 0 else 999.0

        output = ReserveStatusOutput(
            current_cash=round(curr, 2),
            min_reserve=round(res, 2),
            buffer_amount=round(buffer_val, 2),
            is_currently_breached=(curr < res),
            monthly_fixed_opex=round(monthly_opex, 2),
            reserve_coverage_months=coverage_months
        )
        return output.model_dump()

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an incoming JSON tool call from an LLM to the matching engine."""
        if tool_name == "get_sales_forecast":
            validated = GetSalesForecastInput(**arguments)
            return self.get_sales_forecast(horizon_days=validated.horizon_days, quantile=validated.quantile)
        elif tool_name == "get_cashflow_projection":
            validated = GetCashFlowProjectionInput(**arguments)
            return self.get_cashflow_projection(days=validated.days, scenario_type=validated.scenario_type)
        elif tool_name == "simulate_scenario":
            validated = SimulateScenarioInput(**arguments)
            return self.simulate_scenario(
                sales_shock_pct=validated.sales_shock_pct,
                cost_shock_pct=validated.cost_shock_pct,
                delay_days=validated.delay_days
            )
        elif tool_name == "get_reserve_status":
            return self.get_reserve_status()
        else:
            raise ValueError(f"Unknown tool name: '{tool_name}'")

    @staticmethod
    def get_tools_spec() -> List[Dict[str, Any]]:
        """Returns standard Function Calling tool definitions compatible with Gemini / OpenAI."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_sales_forecast",
                    "description": "Retrieve daily sales forecast statistics, total expected revenue, and trajectory for a given horizon and uncertainty quantile.",
                    "parameters": GetSalesForecastInput.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cashflow_projection",
                    "description": "Simulate deterministic cash balance, ending liquidity, runway days, and reserve breach risks for the next N days.",
                    "parameters": GetCashFlowProjectionInput.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_scenario",
                    "description": "Perform stress testing with revenue shocks, COGS cost inflation, or settlement payment delays, measuring exact net monetary impact and runway shift.",
                    "parameters": SimulateScenarioInput.model_json_schema()
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_reserve_status",
                    "description": "Check current cash balance, mandatory reserve limits, fixed monthly OPEX, and available liquidity buffer.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
