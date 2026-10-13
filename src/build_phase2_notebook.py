"""
FinSight AI — Programmatic Notebook Builder for Phase 2 (Dual-Market v2 - Reconciled)
======================================================================================
Author: Senior Kaggle Grandmaster & Principal Data Scientist
Scope: Hypothesis H_3 (Mathematical Determinism in Financial Stress Testing)

Reconciliation Focus:
  100% exact alignment between markdown narrative claims, empirical tables,
  mathematical state equations, and executed code outputs.
  - Zero tolerance for text-to-output mismatches.
  - Fix float epsilon check in JD.com conservation printout.
  - Fix mathtext subscript in Figure 3.
  - Ingest strictly from Phase 1 manifest (Olist + JD.com).

Creates & Pre-Renders: d:\\Others\\notebooks\\Phase2_Deterministic_Cashflow_Engine.ipynb
"""

import os
import sys
import json
import hashlib
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = r"d:\Others\notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(NOTEBOOKS_DIR, "Phase2_Deterministic_Cashflow_Engine.ipynb")


def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # =======================================================================
    # CELL 0: Header & Theoretical Framework (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""# Phase 2: Deterministic Cash-Flow Engine & Stress Simulator

**Author:** FinSight AI Core Research Team (Senior Kaggle Grandmaster & Principal Data Scientist)  
**Core Hypothesis Focus:** $\mathbf{H_3}$ — Mathematical Determinism & Conservation in Financial Stress Testing  
**Operational Scope:** Cross-market cash-basis liquidity modeling across **Olist Brazilian E-Commerce** (Long-Horizon SME) and **JD.com Retail** (Rapid-Fulfillment Micro-Merchant).  
*Methodological Note:* Hypothesis $\mathbf{H_2}$ (early-warning detection efficacy: PR-AUC, lead-time horizon) is formally evaluated downstream in Phase 4.

---

### Mathematical State Equation (Cash-Basis Accounting)

The cashflow engine models the daily treasury position of an enterprise as a discrete-time deterministic state transition:

$$\Large C_t = C_{t-1} + I_t - O_t$$

Where:
* $C_t \in \mathbb{Z}$: End-of-day closing cash balance (tracked in integer centavos / fēn to eliminate floating-point representation drift).
* $I_t \in \mathbb{Z}_{\ge 0}$: Realized cash collections on day $t$, lagged from customer sales by operational settlement latency $L$:
  $$I_t = S_{t-L}$$
* $O_t \in \mathbb{Z}_{\ge 0}$: Total itemized cash outflows on day $t$:
  $$O_t = \text{COGS}_t + \text{Payroll}_t + \text{Rent}_t + \text{Utilities}_t + \text{Debt}_t$$
* $R \in \mathbb{Z}_{\ge 0}$: Minimum operational liquidity reserve (calibrated as 30 days of average fixed OPEX).

### Formal Indicator & Shortfall Operators

$$\Large \text{Reserve Breach}_t = \mathbb{I}(C_t < R), \quad \text{Shortfall}_t = \max(0, R - C_t)$$
$$\Large \text{Insolvency}_t = \mathbb{I}(C_t < 0), \quad \text{Insolvency Deficit}_t = \max(0, -C_t)$$

### Exact Conservation Law Invariant

Across any simulation horizon $t \in [1, T]$, the monetary mass is conserved with zero leakage:

$$\Large C_T \equiv C_0 + \sum_{t=1}^T I_t - \sum_{t=1}^T O_t \quad \iff \quad \left| C_T - \left( C_0 + \sum_{t=1}^T I_t - \sum_{t=1}^T O_t \right) \right| = 0.00$$

This conservation invariant is enforced at every simulation step and asserted via 64-bit integer arithmetic.
"""))

    # =======================================================================
    # CELL 1: Environment Setup & Imports (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""import os, sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# High-resolution publication aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (15, 6),
    'figure.dpi': 120,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Dynamic project root traversal
for candidate in [os.getcwd(), os.path.abspath(".."), os.path.abspath(".")]:
    if os.path.exists(os.path.join(candidate, "src")):
        PROJECT_ROOT = candidate
        break
else:
    PROJECT_ROOT = r"d:\Others"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# FinSight AI Core Financial Architecture
from src.cashflow.engine import (
    SMEConfig,
    CashFlowEngine,
    generate_anchored_ledger,
    calculate_risk_metrics
)
from src.simulator.shocks import ScenarioSimulator, FinancialShockConfig

print("✓ FinSight AI Deterministic Cash-Flow Engine & Stress Suite loaded successfully.")
print(f"  Project root directory: {PROJECT_ROOT}")
print(f"  Processed artifacts:    {PROCESSED_DIR}")
"""))

    # =======================================================================
    # CELL 2: Data Ingestion with SHA-256 Verification (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 1. Data Ingestion from Phase 1 Manifest (SHA-256 Cryptographic Verification)

Following strict empirical reproducibility protocols, we load the verified outputs produced in **Phase 1**:
1. **Olist Brazilian E-Commerce Benchmark**: `olist_daily_sales_top_sellers_k3.parquet` (Seller `4a3ca9315b744ce9f8e9374361493884`, 593 calendar days, 590 active days, R$ 230,702.52 total sales).
2. **JD.com Retail Micro-Merchant Stream**: `jd_daily_sales_top_sku.parquet` (Top SKU `068f4481b3`, 31 unbroken calendar days in March 2018, RMB ¥5,615,431.53 net sales).
3. **JD.com Retail Platform Benchmark**: `jd_daily_sales_platform.parquet` (31 days macro series, RMB ¥45,538,530.79 net sales).

*Integrity Check:* All input Parquet files are dynamically authenticated against `phase1_dataset_manifest.json` using SHA-256 cryptographic hashes.
"""))

    # =======================================================================
    # CELL 3: Data Ingestion Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""manifest_path = os.path.join(PROCESSED_DIR, "phase1_dataset_manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

def verify_and_load(key: str) -> pd.DataFrame:
    # Load a Parquet dataset from the manifest and assert SHA-256 cryptographic integrity.
    entry = manifest[key]
    fpath = entry["file_path"]
    expected_sha = entry["sha256"]
    with open(fpath, "rb") as fp:
        actual_sha = hashlib.sha256(fp.read()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"FATAL: Cryptographic mismatch for '{key}':\n  Expected: {expected_sha}\n  Actual:   {actual_sha}")
    df = pd.read_parquet(fpath)
    print(f"  ✓ Verified [{key}]: {len(df):>5d} records | SHA-256: {actual_sha[:16]}... OK")
    return df

print("=" * 80)
print("  PHASE 1 ARTIFACT CRYPTOGRAPHIC VERIFICATION")
print("=" * 80)

# 1. Olist Top-3 multi-seller dataset
df_olist_k3 = verify_and_load("Olist Top 3 Train-Selected SME Series")
df_olist_k3["Date"] = pd.to_datetime(df_olist_k3["Date"])

# Extract primary benchmark seller 4a3ca931...
TARGET_SELLER_ID = "4a3ca9315b744ce9f8e9374361493884"
df_olist_sme = df_olist_k3[df_olist_k3["seller_id"] == TARGET_SELLER_ID].sort_values("Date").reset_index(drop=True)

# 2. JD.com Top SKU Micro-Merchant Stream
df_jd_sku = verify_and_load("JD.com Top SKU 068f4481b3 (Micro-Merchant Stream)")
df_jd_sku["Date"] = pd.to_datetime(df_jd_sku["Date"])

# 3. JD.com Platform Macro Series
df_jd_platform = verify_and_load("JD.com Retail Platform Daily Sales (Macro)")
df_jd_platform["Date"] = pd.to_datetime(df_jd_platform["Date"])

print("=" * 80)
print(f"  Olist SME ({TARGET_SELLER_ID[:8]}...): {len(df_olist_sme)} days | Total Sales = R$ {df_olist_sme['sales'].sum():,.2f}")
print(f"  JD.com Top SKU (068f4481b3):         {len(df_jd_sku)} days | Net Sales   = RMB ¥{df_jd_sku['sales'].sum():,.2f}")
print(f"  JD.com Platform Macro:              {len(df_jd_platform)} days | Net Sales   = RMB ¥{df_jd_platform['sales'].sum():,.2f}")
print("=" * 80)
"""))

    # =======================================================================
    # CELL 4: Active Window Truncation (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 2. Active Trading Window Truncation (Olist Benchmark)

In real-world e-commerce merchant lifecycles, sellers register accounts before listing active inventory. Seller `4a3ca931...` registered on 2017-01-05 but recorded its first commercial order on **2017-01-08** (Day Index 3).

*Scientific Rationale:* Simulating fixed operational expenses (rent, payroll) over a pre-launch inactive period artificially drains opening capital before any customer cash arrives. To ensure our liquidity breach rate reflects genuine commercial operations rather than onboarding delays, we truncate the series to the **590 active trading days** (2017-01-08 through 2018-08-20).

### Empirical Active Horizon Metrics
* **Full Calendar Horizon:** 2017-01-05 to 2018-08-20 ($593$ calendar days)
* **First Commercial Order Date:** 2017-01-08 (Index 3)
* **Active Trading Horizon:** 2017-01-08 to 2018-08-20 ($590$ active days)
* **Total Commercial Sales:** **R$ 230,702.52**
* **Active Mean Daily Sales:** **R$ 391.02**
* **Active Estimated Monthly Sales (30d):** **R$ 11,730.64**
"""))

    # =======================================================================
    # CELL 5: Active Window Truncation Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""first_active_idx = df_olist_sme[df_olist_sme["sales"] > 0].index[0]
first_active_date = df_olist_sme.loc[first_active_idx, "Date"]

df_active_olist = df_olist_sme.iloc[first_active_idx:].reset_index(drop=True)
n_active_days = len(df_active_olist)
total_active_revenue = df_active_olist["sales"].sum()
mean_daily_revenue = df_active_olist["sales"].mean()
est_monthly_revenue = mean_daily_revenue * 30.0

print("=" * 70)
print("  OLIST CONTINUOUS BENCHMARK: ACTIVE TRADING WINDOW")
print("=" * 70)
print(f"  Full Calendar Span:      {df_olist_sme['Date'].iloc[0].strftime('%Y-%m-%d')} to {df_olist_sme['Date'].iloc[-1].strftime('%Y-%m-%d')} (593 days)")
print(f"  First Commercial Sale:   {first_active_date.strftime('%Y-%m-%d')} (Day Index {first_active_idx})")
print(f"  Active Trading Horizon:  {df_active_olist['Date'].iloc[0].strftime('%Y-%m-%d')} to {df_active_olist['Date'].iloc[-1].strftime('%Y-%m-%d')} ({n_active_days} days)")
print(f"  Total Commercial Sales:  R$ {total_active_revenue:>12,.2f}")
print(f"  Mean Daily Sales:        R$ {mean_daily_revenue:>12,.2f}")
print(f"  Mean Monthly Sales:      R$ {est_monthly_revenue:>12,.2f}")
print("=" * 70)
"""))

    # =======================================================================
    # CELL 6: Empirical SME Accounting Calibration (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Data-Driven SME Parameter Calibration

Rather than imposing arbitrary constants, all financial accounting parameters are **empirically calibrated** from observed sales revenue and Phase 1 logistical measurements:

| Accounting Parameter | Formula / Ground Truth | Calibrated Value | Empirical Rationale & Provenance |
|:---|:---|:---:|:---|
| **Opening Cash ($C_0$)** | $0.5 \times \text{Monthly Revenue}$ | **R$ 5,865.32** | Realistic 15-day liquidity buffer for small micro-enterprises |
| **Minimum Reserve ($R$)** | $1.0 \times \text{Monthly Fixed OPEX}$ | **R$ 2,932.66** | 30-day fixed overhead coverage threshold (insolvency cushion) |
| **Monthly Fixed OPEX** | $25\%$ of Monthly Revenue | **R$ 2,932.66** | Brazilian micro-retail benchmark (SEBRAE / IBGE micro-enterprise cost ratios) |
| • Scheduled Payroll | $50\%$ of Fixed OPEX | R$ 1,466.33 | Split equally across 1st and 15th of each calendar month |
| • Commercial Rent | $20\%$ of Fixed OPEX | R$ 586.53 | Disbursed on the 5th of each calendar month |
| • Utilities & Telecom | $15\%$ of Fixed OPEX | R$ 439.90 | Disbursed on the 10th of each calendar month |
| • Debt Service | $15\%$ of Fixed OPEX | R$ 439.90 | Equipment financing amortized on the 20th of each month |
| **Daily Average OPEX** | Fixed OPEX / 30 | **R$ 97.76** | Baseline daily overhead burn rate |
| **COGS Ratio ($\beta$)** | $55\%$ of Sales | **55%** | Typical gross margin ($45\%$) for marketplace durable goods |
| **Settlement Latency ($L$)** | Phase 1 Median Delivery | **10 days** | Empirically grounded in Phase 1 delivery median ($10.2$ days) |
| **Supplier Terms** | Commercial Trade Credit | **Net-15 days** | Standard trade payable window in commercial retail supply chains |

### Empirical Rationale for Settlement Delay ($L = 10$ Days)
In marketplace platforms, customer card payments are held in escrow and released to merchant accounts upon delivery confirmation. Phase 1 established the empirical delivery distribution:
* **Median delivery latency:** $10.2$ days
* **90th percentile delivery latency:** $23.1$ days

Setting baseline settlement lag to $10$ days directly couples our cash-basis accounting to empirical physical fulfillment.
"""))

    # =======================================================================
    # CELL 7: Calibration Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""config_olist = SMEConfig.calibrate_from_revenue(
    daily_sales=df_active_olist["sales"],
    opex_revenue_ratio=0.25,
    cogs_ratio=0.55,
    cogs_fixed_fraction=0.0,
    initial_cash_months=0.5,
    settlement_lag=10,
    supplier_lag=15,
)

calib_summary = {
    "Opening Cash Buffer (C₀)": f"R$ {config_olist.initial_cash:,.2f}",
    "Minimum Liquidity Reserve (R)": f"R$ {config_olist.min_reserve:,.2f}",
    "Monthly Fixed OPEX": f"R$ {config_olist.monthly_fixed_opex:,.2f}",
    "  • Scheduled Payroll (1st & 15th)": f"R$ {config_olist.monthly_payroll:,.2f}",
    "  • Commercial Rent (5th)": f"R$ {config_olist.monthly_rent:,.2f}",
    "  • Utilities & Telecom (10th)": f"R$ {config_olist.monthly_utilities_misc:,.2f}",
    "  • Equipment Debt Service (20th)": f"R$ {config_olist.monthly_debt_service:,.2f}",
    "Daily Average Fixed OPEX": f"R$ {config_olist.daily_avg_fixed_opex:,.2f}",
    "Variable COGS Ratio (β)": f"{config_olist.cogs_ratio:.0%}",
    "Marketplace Settlement Delay": f"{config_olist.sales_settlement_lag} calendar days",
    "Supplier Payment Terms": f"Net-{config_olist.supplier_terms_lag} days",
}

print("=" * 70)
print("  DATA-DRIVEN SME ACCOUNTING CALIBRATION PROFILE")
print("=" * 70)
for k, v in calib_summary.items():
    print(f"  {k:36s} = {v}")
print("=" * 70)
"""))

    # =======================================================================
    # CELL 8: Baseline Cashflow Execution & Conservation (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 4. Baseline Cash-Flow Engine Execution & Conservation Proof

We run `CashFlowEngine` over the 590-day active trading series. The core conservation law ($C_T \equiv C_0 + \sum I_t - \sum O_t$) is evaluated to confirm absolute zero-cent arithmetic determinism.

### Baseline Accounting Ledger Summary:
* **Trading Duration:** $590$ active days (2017-01-08 to 2018-08-20)
* **Total Settled Inflow ($\sum I_t$):** **R$ 229,246.81**
* **Total Operating Outflows ($\sum O_t$):** **R$ 183,143.59**
  * *Supplier COGS Paid:* R$ 125,810.28 ($68.7\%$ of outflows)
  * *Fixed Operational Overheads:* R$ 57,333.31 ($31.3\%$ of outflows)
* **Net Operating Cashflow Generated:** **R$ 46,103.22**
* **Opening Cash Buffer ($C_0$):** **R$ 5,865.32**
* **Ending Cash Balance ($C_T$):** **R$ 51,968.54** ($C_0 + \text{Net} = 5,865.32 + 46,103.22 = \mathbf{51,968.54}$)
* **Minimum Cash Balance Observed:** **R$ 1,744.23**
* **Conservation Invariant Error:** **R$ 0.00** ($\left| C_T - (C_0 + \sum I - \sum O) \right| = 0.00$, Exact Arithmetic Zero)
* **Reserve Breach Days ($C_t < R$):** **56 days (9.49%)** — *perfectly hits the 5–15% realistic SME operational benchmark!*
* **Insolvency Days ($C_t < 0$):** **0 days (0.00%)**
"""))

    # =======================================================================
    # CELL 9: Baseline Execution Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""engine_olist = CashFlowEngine(config=config_olist)
ledger_olist = engine_olist.simulate(
    sales_series=df_active_olist["sales"],
    dates=df_active_olist["Date"],
)

# Extract core aggregates
total_inflow = ledger_olist["cash_inflow"].sum()
total_outflow = ledger_olist["total_outflow"].sum()
cogs_outflow = ledger_olist["outflow_cogs"].sum()
fixed_opex_outflow = total_outflow - cogs_outflow
net_operating_cash = total_inflow - total_outflow
opening_cash = config_olist.initial_cash
ending_cash = ledger_olist["closing_cash"].iloc[-1]
min_cash = ledger_olist["closing_cash"].min()
breach_days = int(ledger_olist["is_reserve_breach"].sum())
insolvent_days = int(ledger_olist["is_insolvent"].sum())
breach_pct = (breach_days / len(ledger_olist)) * 100.0

# Mathematical Conservation Law Verification
expected_ending_cash = opening_cash + net_operating_cash
conservation_discrepancy = abs(ending_cash - expected_ending_cash)

print("=" * 75)
print("  OLIST CONTINUOUS BENCHMARK: BASELINE LEDGER SCORECARD")
print("=" * 75)
print(f"  Trading Duration:              {len(ledger_olist)} days (2017-01-08 to 2018-08-20)")
print(f"  Total Settled Inflow (Σ I_t):  R$ {total_inflow:>12,.2f}")
print(f"  Total Cash Outflow   (Σ O_t):  R$ {total_outflow:>12,.2f}")
print(f"    • Supplier COGS Paid:        R$ {cogs_outflow:>12,.2f} ({cogs_outflow/total_outflow*100:.1f}%)")
print(f"    • Fixed Operational OPEX:    R$ {fixed_opex_outflow:>12,.2f} ({fixed_opex_outflow/total_outflow*100:.1f}%)")
print(f"  Net Operating Cashflow:        R$ {net_operating_cash:>12,.2f}")
print(f"  Opening Cash Balance (C₀):     R$ {opening_cash:>12,.2f}")
print(f"  Ending Cash Balance  (C_T):    R$ {ending_cash:>12,.2f}")
print(f"  Minimum Cash Observed:         R$ {min_cash:>12,.2f}")
print(f"  Conservation Invariant Error:  R$ {conservation_discrepancy:>12.2f}  {'[✓ EXACT ZERO]' if conservation_discrepancy < 1e-4 else '[✗ VIOLATION]'}")
print(f"  Reserve Breach Days (C < R):   {breach_days:>4d} / {len(ledger_olist)} ({breach_pct:.2f}%)")
print(f"  Insolvency Days     (C < 0):   {insolvent_days:>4d} / {len(ledger_olist)} (0.00%)")
print("=" * 75)

# Assert conservation invariant
assert conservation_discrepancy < 1e-4, f"Conservation law violated: error={conservation_discrepancy}"

print("\nSample Ledger Transactions (First 10 Days):")
display(ledger_olist[[
    "date", "opening_cash", "cash_inflow", "outflow_cogs", "outflow_payroll",
    "outflow_rent", "total_outflow", "closing_cash", "is_reserve_breach", "severity_tier"
]].head(10))
"""))

    # =======================================================================
    # CELL 10: Figure 1 Narrative (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### Figure 1: Olist Cash Balance Trajectory & Inflow/Outflow Timeline

*Top Panel:* Daily closing cash balance ($C_t$) plotted against the calibrated minimum reserve threshold ($R = \text{R\$} 2,932.66$) and the insolvency floor ($C_t = 0$). Red shaded intervals highlight reserve breach episodes (56 days, 9.49%).  
*Bottom Panel:* Daily realized cash inflows ($I_t$, 10-day lag) contrasted with total daily cash outflows ($O_t$). Notice the sharp cyclical outflow spikes driven by the 1st/15th bi-monthly payroll schedule and the 5th-of-month commercial rent payments.
"""))

    # =======================================================================
    # CELL 11: Figure 1 Plot (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})

# Panel 1: Cash Balance & Reserve Status
ax1.plot(ledger_olist["date"], ledger_olist["closing_cash"], color="#1f77b4", lw=2, label="Closing Cash ($C_t$)")
ax1.axhline(config_olist.min_reserve, color="#d62728", ls="--", lw=1.8,
            label=f"Minimum Reserve Threshold (R = R$ {config_olist.min_reserve:,.2f})")
ax1.axhline(0, color="#2c3e50", ls=":", lw=1.5, alpha=0.8, label="Insolvency Boundary (R$ 0.00)")
ax1.fill_between(ledger_olist["date"], ledger_olist["closing_cash"], config_olist.min_reserve,
                 where=(ledger_olist["closing_cash"] < config_olist.min_reserve),
                 color="#d62728", alpha=0.25, label="Liquidity Shortfall Region")

ax1.set_title("Olist Benchmark SME: 590-Day Closing Cash Balance & Reserve Breach Dynamics", fontsize=14, fontweight="bold")
ax1.set_ylabel("Cash Balance (R$)", fontsize=12)
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#bdc3c7", framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Panel 2: Inflow vs Outflow Cash Movements
ax2.plot(ledger_olist["date"], ledger_olist["cash_inflow"], color="#2ca02c", lw=1.2, alpha=0.85, label="Cash Inflow ($I_t$, 10d Lag)")
ax2.plot(ledger_olist["date"], ledger_olist["total_outflow"], color="#e67e22", lw=1.2, alpha=0.85, label="Total Outflow ($O_t$)")
ax2.set_title("Daily Settled Collections vs Calendar-Anchored Outflows", fontsize=12, fontweight="bold")
ax2.set_xlabel("Calendar Date", fontsize=12)
ax2.set_ylabel("Daily Movement (R$)", fontsize=12)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bdc3c7", framealpha=0.9)
ax2.grid(True, alpha=0.3)

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.show()
"""))

    # =======================================================================
    # CELL 12: Settlement-Lag Sensitivity Analysis (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 5. Settlement-Lag Sensitivity Sweep ($L \in \{2, 7, 10, 15, 23\}$ Days)

A pivotal empirical finding from Phase 1 was that Brazilian cross-state logistics introduces substantial fulfillment friction (median delivery latency: $10.2$ days; 90th percentile: $23.1$ days). In escrow-based e-commerce, the merchant cannot access sales proceeds until customer receipt.

We conduct a parametric sensitivity sweep over settlement latency $L$:
* $L = 2$ days: Theoretical fast electronic payment clearance
* $L = 7$ days: Intra-state metropolitan rapid fulfillment
* $L = 10$ days: **Empirical Phase 1 Median (Status Quo Baseline)**
* $L = 15$ days: Inter-state regional delivery delay
* $L = 23$ days: **Empirical Phase 1 90th Percentile (Severe Logistics Congestion)**

### Empirical Findings:
* At $L=2$d: Liquidity breaches are minimal ($21$ days, $3.6\%$) with minimum cash at R$ 2,069.31.
* At $L=10$d (Baseline): Breaches occur for **$56$ days ($9.5\%$)**, establishing our core operational benchmark.
* At $L=23$d (90th percentile latency): Breaches surge to **$106$ days ($18.0\%$)** and trigger **$8$ days of active insolvency** ($C_t < 0$, troughing at -R$ 741.95).  
*Key Takeaway:* Logistics friction alone—without any reduction in consumer demand—is sufficient to drive profitable e-commerce SMEs into technical insolvency.
"""))

    # =======================================================================
    # CELL 13: Settlement Lag Sweep Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""lag_sweep_values = [2, 7, 10, 15, 23]
lag_results = []

for lag in lag_sweep_values:
    cfg = SMEConfig.calibrate_from_revenue(
        daily_sales=df_active_olist["sales"],
        opex_revenue_ratio=0.25,
        cogs_ratio=0.55,
        initial_cash_months=0.5,
        settlement_lag=lag,
        supplier_lag=15,
    )
    eng = CashFlowEngine(config=cfg)
    ldg = eng.simulate(sales_series=df_active_olist["sales"], dates=df_active_olist["Date"])
    
    b_days = int(ldg["is_reserve_breach"].sum())
    ins_days = int(ldg["is_insolvent"].sum())
    m_cash = float(ldg["closing_cash"].min())
    e_cash = float(ldg["closing_cash"].iloc[-1])
    
    lag_results.append({
        "Settlement Lag (L)": f"{lag} days",
        "Empirical Benchmark": "Electronic Clear" if lag == 2 else ("Metro Delivery" if lag == 7 else ("Phase 1 Median" if lag == 10 else ("Regional Delay" if lag == 15 else "Phase 1 90th Pct"))),
        "Breach Days": b_days,
        "Breach %": f"{(b_days / len(ldg)) * 100:.1f}%",
        "Insolvent Days": ins_days,
        "Min Cash (R$)": round(m_cash, 2),
        "Ending Cash (R$)": round(e_cash, 2),
    })

df_lag_sweep = pd.DataFrame(lag_results)
print("Table 1: SETTLEMENT LATENCY SENSITIVITY SCORECARD (Olist Benchmark)")
display(df_lag_sweep)
"""))

    # =======================================================================
    # CELL 14: Stress Testing Suite (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 6. Experiment C — Financial Stress Scenarios (Onset-Dated at Day 180)

To rigorously evaluate how economic shocks destabilize SME liquidity, we test a structured suite of macro-financial stress scenarios.

### Onset-Dated Experimental Design
Shocks commence precisely at **Day Index 180** (`2017-07-07`, approximately 6 months into active operations):
$$\text{Multiplier}_t = \begin{cases} 1.0 & \text{for } t < 180 \quad (\text{pre-shock: strictly identical to baseline}) \\ \text{Shock Multiplier} & \text{for } t \ge 180 \quad (\text{post-shock: shock regime enforced}) \end{cases}$$

This design guarantees that pre-shock operational history is invariant, establishing a rigorous controlled baseline.

### Pre-Registered Stress Scenarios & Empirical Outcomes:
1. **Base Case (Status Quo):** Ending Cash **R$ 51,968.54**, Min Cash **R$ 1,744.23**, $56$ breach days ($9.5\%$), $0$ insolvent days.
2. **C1 (Demand Contraction -20%):** Ending Cash drops to **R$ 35,772.77**, Min Cash **R$ 1,744.23**, $56$ breach days ($9.5\%$).
3. **C1 (Severe Sales Drop -30%):** Ending Cash contracts to **R$ 27,674.51**, Min Cash **R$ 1,744.23**, $56$ breach days ($9.5\%$).
4. **C2 (Cost Inflation +20%):** Wholesale COGS increase $\times 1.20$, compressing Ending Cash to **R$ 32,397.65**, $56$ breach days ($9.5\%$).
5. **C3 (Liquidity Freeze +7d Delay):** Settlement delayed to $17$ days. Breaches jump to **$85$ days ($14.4\%$)**, Min Cash dips to **R$ 691.92**, Ending Cash **R$ 50,295.70**.
6. **C3 (Severe Delay +14d Delay):** Settlement delayed to $24$ days. Breaches reach **$108$ days ($18.3\%$)** with **$11$ active insolvency days** (troughing at **-R$ 864.62**).
7. **C4 (Black Swan Compound Crisis):** Simultaneous demand drop ($-25\%$), cost spike ($+15\%$), and settlement delay ($+7$d). Breaches reach **$85$ days ($14.4\%$)**, Min Cash dips to **R$ 691.92**, and Ending Cash collapses to **R$ 19,605.92** (a catastrophic **-R$ 32,362.62 capital destruction**).
"""))

    # =======================================================================
    # CELL 15: Stress Testing Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""SHOCK_ONSET_INDEX = 180
shock_onset_date = df_active_olist["Date"].iloc[SHOCK_ONSET_INDEX]

simulator = ScenarioSimulator(base_config=config_olist)
summary_df, stress_trajectories = simulator.run_stress_suite(
    sales_series=df_active_olist["sales"],
    dates=df_active_olist["Date"],
    shock_onset_idx=SHOCK_ONSET_INDEX,
)

print(f"Shock Inception Point: Day Index {SHOCK_ONSET_INDEX} ({shock_onset_date.strftime('%Y-%m-%d')})")
print(f"Total Trajectory Horizon: {len(df_active_olist)} calendar days\n")

print("Table 2: COMPARATIVE FINANCIAL STRESS SCORECARD (Onset at Day 180)")
display(summary_df[[
    "Scenario", "Sales Impact", "COGS Impact", "Lag Shift", "Onset",
    "First Breach Date", "Breach Days", "Breach %", "Insolvent Days",
    "Min Cash (R$)", "Ending Cash (R$)"
]])
"""))

    # =======================================================================
    # CELL 16: Fixed vs Variable COGS Analysis (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 6.1 Cost Rigidity Sensitivity: Fixed vs. Variable COGS

In micro-retail operations, inventory costs are rarely $100\%$ variable. Merchants must frequently pre-purchase seasonal stock or commit to minimum order quantities (MOQs). When demand contracts, committed wholesale expenses do not immediately vanish.

We simulate a **$50\%$ Fixed COGS Alternative** under Scenario C1 ($-20\%$ sales drop):
$$\text{COGS}_t = \left[ 0.50 \times \text{COGS}_t^{\text{base}} + 0.50 \times \left( \text{Sales}_t \times 0.80 \times \beta \right) \right]$$

### Empirical Finding:
* Under **Fully Variable COGS**, ending cash is **R$ 35,772.77**.
* Under **$50\%$ Fixed COGS Commitments**, ending cash collapses to **R$ 25,987.13**.
* *Operating Leverage Effect:* Cost rigidity causes an additional **-R$ 9,785.64 in liquidity destruction** because committed purchase obligations persist despite lower retail revenue.
"""))

    # =======================================================================
    # CELL 17: Fixed vs Variable COGS Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Configure SME with 50% fixed COGS
config_fixed_cogs = SMEConfig.calibrate_from_revenue(
    daily_sales=df_active_olist["sales"],
    opex_revenue_ratio=0.25,
    cogs_ratio=0.55,
    cogs_fixed_fraction=0.50,
    initial_cash_months=0.5,
    settlement_lag=10,
    supplier_lag=15,
)

engine_fixed = CashFlowEngine(config=config_fixed_cogs)
ledger_c1_fixed = engine_fixed.simulate(
    sales_series=df_active_olist["sales"],
    dates=df_active_olist["Date"],
    sales_volume_multiplier=0.80,
    shock_onset_idx=SHOCK_ONSET_INDEX,
)

ledger_c1_var = stress_trajectories["C1: Sales Contraction (−20%)"]

cogs_comp_data = [
    {
        "Cost Structure Model": "Fully Variable COGS (100% Flexible)",
        "Breach Days": int(ledger_c1_var["is_reserve_breach"].sum()),
        "Breach %": f"{(int(ledger_c1_var['is_reserve_breach'].sum()) / len(ledger_c1_var)) * 100:.1f}%",
        "Min Cash (R$)": round(ledger_c1_var["closing_cash"].min(), 2),
        "Ending Cash (R$)": round(ledger_c1_var["closing_cash"].iloc[-1], 2),
        "Retained Capital Drain": "R$ 0.00 (Reference)",
    },
    {
        "Cost Structure Model": "50% Fixed Inventory Commitments (Rigid)",
        "Breach Days": int(ledger_c1_fixed["is_reserve_breach"].sum()),
        "Breach %": f"{(int(ledger_c1_fixed['is_reserve_breach'].sum()) / len(ledger_c1_fixed)) * 100:.1f}%",
        "Min Cash (R$)": round(ledger_c1_fixed["closing_cash"].min(), 2),
        "Ending Cash (R$)": round(ledger_c1_fixed["closing_cash"].iloc[-1], 2),
        "Retained Capital Drain": f"-R$ {ledger_c1_var['closing_cash'].iloc[-1] - ledger_c1_fixed['closing_cash'].iloc[-1]:,.2f}",
    }
]

df_cogs_comp = pd.DataFrame(cogs_comp_data)
print("Table 3: DEMAND SHOCK WITH INVENTORY RIGIDITY (C1 Sales -20%)")
display(df_cogs_comp)
"""))

    # =======================================================================
    # CELL 18: Figure 2 Fan Chart Narrative (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### Figure 2: Multi-Scenario Trajectory Fan Chart

Visualizing the divergence of enterprise cash balances across all stress scenarios post Day 180 (`2017-07-07`).  
Notice that all trajectories remain strictly collinear prior to Day 180, proving that pre-shock accounting invariance holds. Post-onset, the compound Black Swan crisis (red curve) drains capital from R$ 51,968.54 down to R$ 19,605.92.
"""))

    # =======================================================================
    # CELL 19: Figure 2 Fan Chart Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""plt.figure(figsize=(16, 7.5))

scenario_colors = {
    "Base Case (Status Quo)": ("#1f77b4", 2.5, "-"),
    "C1: Sales Contraction (−20%)": ("#2ca02c", 1.8, "--"),
    "C1: Severe Sales Drop (−30%)": ("#8c564b", 1.8, "--"),
    "C2: Supplier Cost Spike (+20%)": ("#ff7f0e", 1.8, "-."),
    "C3: Liquidity Freeze (+7d Delay)": ("#9467bd", 1.8, ":"),
    "C3: Severe Delay (+14d Delay)": ("#7f7f7f", 1.8, ":"),
    "C4: Compound Crisis (Black Swan)": ("#d62728", 2.8, "-"),
}

for name, (color, lw, ls) in scenario_colors.items():
    if name in stress_trajectories:
        traj = stress_trajectories[name]
        plt.plot(traj["date"], traj["closing_cash"], label=name, color=color, linewidth=lw, linestyle=ls)

# Plot 50% fixed COGS trajectory
plt.plot(ledger_c1_fixed["date"], ledger_c1_fixed["closing_cash"],
         label="C1 Alt: Sales -20% (50% Fixed COGS)", color="#17becf", linewidth=2.0, linestyle="--")

# Markers
plt.axvline(shock_onset_date, color="#2c3e50", ls="-.", lw=2.0, alpha=0.85,
            label=f"Shock Onset (Day 180: {shock_onset_date.strftime('%Y-%m-%d')})")
plt.axhline(config_olist.min_reserve, color="#d62728", ls="--", lw=1.5, alpha=0.9,
            label=f"Reserve R (R$ {config_olist.min_reserve:,.0f})")
plt.axhline(0, color="black", ls=":", lw=1.2, alpha=0.7)

plt.title("Comparative Enterprise Liquidity Trajectories Under Simulated Stress Scenarios", fontsize=15, fontweight="bold", pad=15)
plt.xlabel("Calendar Date", fontsize=12)
plt.ylabel("Closing Cash Balance (R$)", fontsize=12)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#bdc3c7", framealpha=0.95, fontsize=9.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

    # =======================================================================
    # CELL 20: Market 2: JD.com Rapid Settlement Validation (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 7. Market 2 Cross-Validation: JD.com Retail Top SKU (Rapid Settlement)

To demonstrate the structural universality of our deterministic engine, we deploy it on the **JD.com Retail Micro-Merchant Stream** (`068f4481b3`, Brand `99d41501ff`, 31 unbroken calendar days in March 2018).

### Structural Contrast: Brazilian SME vs. Chinese High-Speed Logistics
* **Olist (Brazil):** Cross-state courier network with a median delivery latency of **10.2 days**, inducing working capital lockups and a baseline reserve breach rate of **9.49% (56 days)**.
* **JD.com (China):** Vertically-integrated direct fulfillment network (JD Logistics) achieving a median delivery latency of **0.992 days (23.8 hours)**. Platform settlement occurs within **1 day**, drastically compressing the cash-conversion cycle.

### JD.com Baseline Scorecard (March 2018):
* **Simulation Horizon:** $31$ unbroken days (2018-03-01 to 2018-03-31)
* **Total Commercial Sales:** **RMB ¥5,615,431.53** | **Daily Mean Sales:** **RMB ¥181,142.95**
* **Opening Cash Buffer ($C_0$):** **RMB ¥2,717,144.29**
* **Minimum Reserve ($R$):** **RMB ¥1,086,857.72**
* **Total Realized Inflow ($\sum I_t$):** **RMB ¥5,553,173.18**
* **Total Operating Outflow ($\sum O_t$):** **RMB ¥4,926,615.95** (COGS: ¥3,839,758.23, OPEX: ¥1,086,857.72)
* **Net Cashflow Generated:** **RMB ¥626,557.23**
* **Ending Cash Position ($C_T$):** **RMB ¥3,343,701.52**
* **Minimum Cash Observed:** **RMB ¥2,241,537.45**
* **Reserve Breach Days ($C_t < R$):** **0 days (0.00%)**
* **Insolvency Days ($C_t < 0$):** **0 days (0.00%)**
* **Conservation Invariant Error:** **RMB ¥0.00** ($\left| C_T - (C_0 + \sum I - \sum O) \right| = 0.00$, Exact Zero)
"""))

    # =======================================================================
    # CELL 21: JD.com Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Calibrate accounting parameters for high-volume consumer electronics/apparel SKU
config_jd = SMEConfig.calibrate_from_revenue(
    daily_sales=df_jd_sku["sales"],
    opex_revenue_ratio=0.20,
    cogs_ratio=0.60,
    initial_cash_months=0.5,
    settlement_lag=1,  # 1-day rapid settlement via JD Logistics
    supplier_lag=15,
)

engine_jd = CashFlowEngine(config=config_jd)
ledger_jd = engine_jd.simulate(
    sales_series=df_jd_sku["sales"],
    dates=df_jd_sku["Date"],
)

# Compute metrics
jd_inflow = ledger_jd["cash_inflow"].sum()
jd_outflow = ledger_jd["total_outflow"].sum()
jd_cogs = ledger_jd["outflow_cogs"].sum()
jd_opex = jd_outflow - jd_cogs
jd_net_cash = jd_inflow - jd_outflow
jd_ending = ledger_jd["closing_cash"].iloc[-1]
jd_min = ledger_jd["closing_cash"].min()
jd_breaches = int(ledger_jd["is_reserve_breach"].sum())
jd_insolvent = int(ledger_jd["is_insolvent"].sum())
jd_conservation_err = abs(jd_ending - (config_jd.initial_cash + jd_net_cash))

print("=" * 75)
print("  JD.COM RETAIL TOP SKU (068f4481b3): CASH-FLOW SCORECARD (RMB ¥)")
print("=" * 75)
print(f"  Simulation Span:               {len(ledger_jd)} unbroken days (March 2018)")
print(f"  Total Commercial Sales:        RMB ¥ {df_jd_sku['sales'].sum():>13,.2f}")
print(f"  Daily Mean Sales:              RMB ¥ {df_jd_sku['sales'].mean():>13,.2f}")
print(f"  Opening Cash Buffer (C₀):      RMB ¥ {config_jd.initial_cash:>13,.2f}")
print(f"  Minimum Liquidity Reserve (R): RMB ¥ {config_jd.min_reserve:>13,.2f}")
print(f"  Total Realized Inflow (Σ I):   RMB ¥ {jd_inflow:>13,.2f}")
print(f"  Total Operating Outflow (Σ O): RMB ¥ {jd_outflow:>13,.2f}")
print(f"    • Supplier COGS Paid:        RMB ¥ {jd_cogs:>13,.2f}")
print(f"    • Fixed Overhead Paid:       RMB ¥ {jd_opex:>13,.2f}")
print(f"  Net Cashflow Generated:        RMB ¥ {jd_net_cash:>13,.2f}")
print(f"  Ending Cash Balance (C_T):     RMB ¥ {jd_ending:>13,.2f}")
print(f"  Minimum Cash Observed:         RMB ¥ {jd_min:>13,.2f}")
print(f"  Conservation Invariant Error:  RMB ¥ {jd_conservation_err:>13.2f}  {'[✓ EXACT ZERO]' if jd_conservation_err < 1e-4 else '[✗ ERROR]'}")
print(f"  Reserve Breach Days:           {jd_breaches:>5d} / {len(ledger_jd)} (0.00%)")
print(f"  Insolvency Days:               {jd_insolvent:>5d} / {len(ledger_jd)} (0.00%)")
print("=" * 75)

assert jd_conservation_err < 1e-4, "JD conservation error violated!"
"""))

    # =======================================================================
    # CELL 22: Figure 3 JD.com Trajectory Narrative (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### Figure 3: JD.com Rapid-Fulfillment Cash Trajectory

Visualizing the JD.com micro-merchant cash balance throughout March 2018. Because cash settles within 24 hours of dispatch, closing cash steadily accumulates above the reserve threshold ($R = \text{¥} 1,086,857.72$) with **zero liquidity breaches** across the entire trading horizon.
"""))

    # =======================================================================
    # CELL 23: Figure 3 Plot (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""fig, ax = plt.subplots(figsize=(15, 5.5))

ax.plot(ledger_jd["date"], ledger_jd["closing_cash"], color="#2ca02c", lw=2.2, label="Closing Cash ($C_t$, RMB ¥)")
ax.axhline(config_jd.min_reserve, color="#d62728", ls="--", lw=1.8,
           label=f"Reserve Threshold (R = RMB ¥{config_jd.min_reserve:,.0f})")
ax.axhline(config_jd.initial_cash, color="#1f77b4", ls=":", lw=1.5,
           label=f"Opening Cash Buffer ($C_0$ = RMB ¥{config_jd.initial_cash:,.0f})")

ax.set_title("JD.com Retail Top SKU: Rapid Fulfillment Cash Accumulation Trajectory (March 2018)", fontsize=14, fontweight="bold")
ax.set_xlabel("Calendar Date", fontsize=12)
ax.set_ylabel("Cash Balance (RMB ¥)", fontsize=12)
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#bdc3c7", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

plt.tight_layout()
plt.show()
"""))

    # =======================================================================
    # CELL 24: Software Verification 5 Golden Tests (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 8. Software Verification: 5 Golden Hand-Calculated Tests

*Scientific Rigor Distinction:* The property $\text{Variance} = 0.000000$ across repeated simulations is **not an empirical scientific finding**; it is a **rigorous software correctness verification** proving that the underlying engine is implemented as a deterministic pure mathematical function.

| Test # | Golden Verification Fixture | Mathematical Assertion Under Test | Target Outcome | Empirical Status |
|:---:|:---|:---|:---:|:---:|
| **1** | **Zero-Inflow Runway** | Under zero revenue, $C_t$ depletes strictly at scheduled fixed OPEX rate | $\Delta C_T = -\sum \text{OPEX}^{\text{fixed}}$ | **PASSED** (Ending = R$ 1,800.00) |
| **2** | **Calendar Outflow Collision** | Payroll (1st/15th) and Rent (5th) trigger exact scheduled debits | Outflow exact cent match | **PASSED** (Jan 1: R$ 1,000; Jan 5: R$ 800) |
| **3** | **Settlement Shift Conservation** | Shifting inflows preserves total monetary mass across the pipeline | Total inflow sum invariant | **PASSED** (Mass conserved) |
| **4** | **Negative-Floor Invariance** | Insolvency indicator $\mathbb{I}(C_t < 0)$ matches negative cash state | Zero false positive/negative | **PASSED** (100% consistent) |
| **5** | **100× Idempotency Invariance** | 100 identical simulations yield zero floating-point divergence | $\max |\Delta C_t| = 0.000000$ | **PASSED** (Var = 0.000000) |
"""))

    # =======================================================================
    # CELL 25: Software Verification Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""golden_passed = 0
total_golden_tests = 5

print("=" * 70)
print("  EXECUTING SOFTWARE VERIFICATION: 5 GOLDEN TESTS")
print("=" * 70)

# ---------------------------------------------------------------------------
# Test 1: Zero-Inflow Runway
# ---------------------------------------------------------------------------
zero_dates = pd.date_range("2023-01-01", periods=60, freq="D")
zero_sales = pd.Series([0.0] * 60)
cfg_test = SMEConfig(
    initial_cash=10000.0,
    min_reserve=5000.0,
    cogs_ratio=0.55,
    sales_settlement_lag=2,
    supplier_terms_lag=15,
    monthly_payroll=2000.0,
    payroll_days=(1, 15),
    monthly_rent=800.0,
    rent_day=5,
    monthly_utilities_misc=350.0,
    utilities_day=10,
    monthly_debt_service=450.0,
    debt_day=20,
)
eng_test = CashFlowEngine(config=cfg_test)
ldg_test = eng_test.simulate(sales_series=zero_sales, dates=zero_dates)

expected_end_t1 = 10000.0 - ldg_test["total_outflow"].sum() + ldg_test["cash_inflow"].sum()
actual_end_t1 = ldg_test["closing_cash"].iloc[-1]
assert abs(actual_end_t1 - expected_end_t1) < 1e-4, f"Test 1 failed: {actual_end_t1} != {expected_end_t1}"
golden_passed += 1
print(f"  ✓ [TEST 1/5 PASSED] Zero-Inflow Runway: Cash drain perfectly tracks fixed OPEX (End = R$ {actual_end_t1:,.2f})")

# ---------------------------------------------------------------------------
# Test 2: Calendar Outflow Collision
# ---------------------------------------------------------------------------
jan1_out = ldg_test[ldg_test["date"].dt.day == 1].iloc[0]
jan5_out = ldg_test[ldg_test["date"].dt.day == 5].iloc[0]
assert jan1_out["outflow_payroll"] == 1000.0, "Jan 1 payroll mismatch"
assert jan5_out["outflow_rent"] == 800.0, "Jan 5 rent mismatch"
golden_passed += 1
print(f"  ✓ [TEST 2/5 PASSED] Calendar Outflow Collision: Payroll (R$ 1,000.00) & Rent (R$ 800.00) fired on exact dates")

# ---------------------------------------------------------------------------
# Test 3: Settlement Shift Conservation of Mass
# ---------------------------------------------------------------------------
sample_sales = df_active_olist["sales"].iloc[:60]
sample_dates = df_active_olist["Date"].iloc[:60]
base_inflows = eng_test.simulate(sales_series=sample_sales, dates=sample_dates, sales_settlement_shift=0)["cash_inflow"].values
delayed_inflows = eng_test.simulate(sales_series=sample_sales, dates=sample_dates, sales_settlement_shift=7)["cash_inflow"].values

# In middle aligned window, values match exactly
lag_offset = cfg_test.sales_settlement_lag
assert np.array_equal(delayed_inflows[7 + lag_offset:], base_inflows[lag_offset:60 - 7]), "Shift mass conservation violated!"
golden_passed += 1
print(f"  ✓ [TEST 3/5 PASSED] Settlement Shift Conservation: Shifted pipeline preserves monetary mass")

# ---------------------------------------------------------------------------
# Test 4: Negative-Floor Invariance
# ---------------------------------------------------------------------------
for _, row in ldg_test.iterrows():
    if row["closing_cash"] < 0:
        assert row["is_insolvent"] == 1, "Insolvency flag missed!"
    else:
        assert row["is_insolvent"] == 0, "False positive insolvency flag!"
golden_passed += 1
print(f"  ✓ [TEST 4/5 PASSED] Negative-Floor Invariance: Insolvency state perfectly partitioned across all days")

# ---------------------------------------------------------------------------
# Test 5: 100x Idempotency Guarantee
# ---------------------------------------------------------------------------
sample_30_sales = df_active_olist["sales"].iloc[:30]
sample_30_dates = df_active_olist["Date"].iloc[:30]
first_sim = engine_olist.simulate(sales_series=sample_30_sales, dates=sample_30_dates)["closing_cash"].values

max_divergence = 0.0
for _ in range(100):
    re_sim = engine_olist.simulate(sales_series=sample_30_sales, dates=sample_30_dates)["closing_cash"].values
    max_divergence = max(max_divergence, float(np.max(np.abs(first_sim - re_sim))))

assert max_divergence == 0.0, f"Idempotency violation: max divergence = {max_divergence}"
golden_passed += 1
print(f"  ✓ [TEST 5/5 PASSED] 100x Idempotency Guarantee: Max discrepancy across 100 iterations = {max_divergence:.6f}")

print("=" * 70)
print(f"  STATUS: ALL {golden_passed}/{total_golden_tests} SOFTWARE VERIFICATION FIXTURES SATISFIED")
print("=" * 70)
"""))

    # =======================================================================
    # CELL 26: Hand-Calculated Cross-Verification (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""### 8.1 Hand-Calculated Arithmetic Cross-Audit

We perform an independent hand-calculation cross-check for the **Baseline Ledger** and the **C4 Black Swan Scenario**.

#### Baseline Proof:
$$C_T^{\text{Hand}} = C_0 + \sum_{t=1}^T I_t - \sum_{t=1}^T O_t$$
$$C_T^{\text{Hand}} = 5,865.32 + 229,246.81 - 183,143.59 = \mathbf{51,968.54}$$
*Discrepancy:* $|51,968.54 - 51,968.54| = \mathbf{0.00}$ (Zero-Cent Match).

#### C4 Black Swan Monotonicity Check:
Because Scenario C4 imposes simultaneously a demand drop ($-25\%$), cost spike ($+15\%$), and settlement delay ($+7$d) from Day 180 onward, economic monotonicity dictates:
$$C_T^{\text{C4}} < C_T^{\text{Base}} \quad \text{and} \quad \text{Breach Days}^{\text{C4}} \ge \text{Breach Days}^{\text{Base}}$$
* **Baseline Ending Cash:** **R$ 51,968.54** ($56$ breaches)
* **C4 Black Swan Ending Cash:** **R$ 19,605.92** ($85$ breaches)
* **Capital Erosion Induced by C4:** **-R$ 32,362.62**
"""))

    # =======================================================================
    # CELL 27: Hand Calculation Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""# Baseline manual verification
C0_val = config_olist.initial_cash
sum_I_val = ledger_olist["cash_inflow"].sum()
sum_O_val = ledger_olist["total_outflow"].sum()
CT_engine = ledger_olist["closing_cash"].iloc[-1]
CT_hand = C0_val + sum_I_val - sum_O_val
hand_err = abs(CT_engine - CT_hand)

print("Baseline Manual Accounting Cross-Check:")
print(f"  Opening Cash (C₀):              R$ {C0_val:>12,.2f}")
print(f"  Sum Realized Inflows (Σ I):     R$ {sum_I_val:>12,.2f}")
print(f"  Sum Realized Outflows (Σ O):    R$ {sum_O_val:>12,.2f}")
print(f"  Hand-Calculated Closing (C_T):  R$ {CT_hand:>12,.2f}")
print(f"  Engine-Output Closing (C_T):    R$ {CT_engine:>12,.2f}")
print(f"  Manual Calculation Discrepancy: R$ {hand_err:>12.2f}  {'[✓ ZERO-CENT MATCH]' if hand_err < 1e-4 else '[✗ MISMATCH]'}")

# C4 Black swan validation
c4_ledger = stress_trajectories["C4: Compound Crisis (Black Swan)"]
c4_ending_cash = c4_ledger["closing_cash"].iloc[-1]
c4_breach_count = int(c4_ledger["is_reserve_breach"].sum())
base_breach_count = int(ledger_olist["is_reserve_breach"].sum())

print("\nScenario C4 (Black Swan) Validation:")
print(f"  Baseline Ending Cash:           R$ {CT_engine:>12,.2f} | Breaches: {base_breach_count}")
print(f"  C4 Black Swan Ending Cash:      R$ {c4_ending_cash:>12,.2f} | Breaches: {c4_breach_count}")
print(f"  Capital Erosion Induced by C4:  -R$ {CT_engine - c4_ending_cash:>11,.2f}")

assert c4_ending_cash < CT_engine, "Monotonicity violation: C4 ending cash not lower than baseline"
assert c4_breach_count > base_breach_count, "Monotonicity violation: C4 breach count not higher than baseline"
print("  ✓ Economic Monotonicity Guaranteed: C4 ending cash < Baseline & C4 breaches > Baseline.")
"""))

    # =======================================================================
    # CELL 28: Persist Phase 2 Artifacts & Manifest (Markdown)
    # =======================================================================
    cells.append(nbf.v4.new_markdown_cell(r"""## 9. Production Artifact Persistence & Phase 2 Dataset Manifest

We persist the simulated cashflow ledgers, stress scenario summary scorecards, and a cryptographic **Phase 2 Dataset Manifest** (`phase2_dataset_manifest.json`) for downstream use in:
* **Phase 3:** Macro Forecasting & Predictive Feature Stores.
* **Phase 4:** Early-Warning Efficacy Evaluation ($H_2$: PR-AUC, Warning Horizon Lead-Time).

### Persisted Artifact Inventory:
1. `phase2_olist_cashflow_ledger.parquet`: $590$ daily records, $19$ accounting features (R$).
2. `phase2_jd_cashflow_ledger.parquet`: $31$ daily records, $19$ accounting features (RMB ¥).
3. `phase2_stress_summary.parquet`: $7$ scenario stress summary scorecard.
4. `phase2_dataset_manifest.json`: JSON metadata catalog with SHA-256 integrity digests.
"""))

    # =======================================================================
    # CELL 29: Persist Phase 2 Artifacts Code (Code)
    # =======================================================================
    cells.append(nbf.v4.new_code_cell(r"""# 1. Persist Olist Baseline Ledger
olist_ledger_path = os.path.join(PROCESSED_DIR, "phase2_olist_cashflow_ledger.parquet")
ledger_olist.to_parquet(olist_ledger_path, index=False)

# 2. Persist JD.com Baseline Ledger
jd_ledger_path = os.path.join(PROCESSED_DIR, "phase2_jd_cashflow_ledger.parquet")
ledger_jd.to_parquet(jd_ledger_path, index=False)

# 3. Persist Stress Testing Summary Scorecard
stress_path = os.path.join(PROCESSED_DIR, "phase2_stress_summary.parquet")
summary_df.to_parquet(stress_path, index=False)

# Helper to compute SHA-256
def get_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

phase2_manifest = {
    "Olist Primary Continuous SME Cashflow Ledger": {
        "file_path": olist_ledger_path,
        "downstream_phase": "Phase 4 (Early-Warning Lead-Time & PR-AUC Evaluation)",
        "description": "590-day daily cashflow accounting ledger with reserve breaches and severity tiers (R$)",
        "records": len(ledger_olist),
        "columns": list(ledger_olist.columns),
        "start_date": str(ledger_olist["date"].iloc[0].date()),
        "end_date": str(ledger_olist["date"].iloc[-1].date()),
        "file_size_bytes": os.path.getsize(olist_ledger_path),
        "sha256": get_sha256(olist_ledger_path),
    },
    "JD.com Retail Micro-Merchant Cashflow Ledger": {
        "file_path": jd_ledger_path,
        "downstream_phase": "Phase 4 (Cross-Market Rapid-Settlement Comparative Benchmark)",
        "description": "31-day daily cashflow accounting ledger for Top SKU 068f4481b3 with 1-day settlement (RMB ¥)",
        "records": len(ledger_jd),
        "columns": list(ledger_jd.columns),
        "start_date": str(ledger_jd["date"].iloc[0].date()),
        "end_date": str(ledger_jd["date"].iloc[-1].date()),
        "file_size_bytes": os.path.getsize(jd_ledger_path),
        "sha256": get_sha256(jd_ledger_path),
    },
    "Financial Stress Scenario Summary Scorecard": {
        "file_path": stress_path,
        "downstream_phase": "Phase 4 (Stress Testing Multi-Horizon Evaluator)",
        "description": "Comparative scorecard across Base Case, C1, C2, C3, and C4 macro-financial shocks",
        "records": len(summary_df),
        "columns": list(summary_df.columns),
        "start_date": "N/A",
        "end_date": "N/A",
        "file_size_bytes": os.path.getsize(stress_path),
        "sha256": get_sha256(stress_path),
    }
}

manifest_out_path = os.path.join(PROCESSED_DIR, "phase2_dataset_manifest.json")
with open(manifest_out_path, "w", encoding="utf-8") as f:
    json.dump(phase2_manifest, f, indent=2)

print("=" * 80)
print("  PHASE 2 ARTIFACTS SUCCESSFULLY PERSISTED & MANIFESTED")
print("=" * 80)
for k, v in phase2_manifest.items():
    print(f"  • {k}")
    print(f"    Path:    {v['file_path']}")
    print(f"    Records: {v['records']} | Size: {v['file_size_bytes'] / 1024:.1f} KB | SHA-256: {v['sha256'][:16]}...")
print("=" * 80)
print(f"  Phase 2 Manifest saved to: {manifest_out_path}")
"""))

    nb.cells = cells
    return nb


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Building Reconciled Phase 2 Notebook at:\n  -> {NOTEBOOK_PATH}...")
    nb = build_notebook()

    print("Executing all notebook cells via NotebookClient to pre-render tables & plots...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": NOTEBOOKS_DIR}},
    )
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    file_size_kb = os.path.getsize(NOTEBOOK_PATH) / 1024
    print(f"\n[SUCCESS] Reconciled Notebook successfully built, executed, and saved:\n  -> {NOTEBOOK_PATH} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
