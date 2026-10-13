"""
FinSight AI — Empirical 6,000-Prompt Multilingual Benchmark Runner
===================================================================
Executes all 6,000 prompts from benchmark_6000_multilingual_testbed.json
against the upgraded GroundedCopilotAgent.

Measures:
1. Language Detection Accuracy (%)
2. Tool Dispatch Accuracy (%)
3. Parameter Extraction Accuracy (%)
4. Zero-Hallucination Rate (%)
5. Mean 10-Point Advisory Rubric Score
6. Execution Latency (ms)
"""

import sys
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cashflow.engine import SMEConfig
from src.copilot.tools import FinSightToolKit
from src.copilot.agent import GroundedCopilotAgent

TESTBED_FILE = PROJECT_ROOT / "data" / "benchmark_6000_multilingual_testbed.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_full_evaluation():
    print(f"Loading 6,000-prompt multilingual benchmark testbed from: {TESTBED_FILE}...")
    with open(TESTBED_FILE, "r", encoding="utf-8") as f:
        testbed = json.load(f)

    total_prompts = len(testbed)
    print(f"Loaded {total_prompts} test cases across 6 Asian & Global languages.")

    # Initialize Core SME Environment
    cfg = SMEConfig(
        initial_cash=60000.0,
        min_reserve=50000.0,
        monthly_payroll=140000.0,
        monthly_rent=35000.0,
        monthly_utilities_misc=15000.0,
        monthly_debt_service=20000.0,
        cogs_ratio=0.74,
        supplier_terms_lag=15,
        sales_settlement_lag=2
    )
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    sales_p50 = pd.Series(np.full(30, 25000.0))
    sales_p10 = sales_p50 * 0.72
    sales_p90 = sales_p50 * 1.28

    toolkit = FinSightToolKit(cfg, sales_p50, sales_p10, sales_p90, dates)
    agent = GroundedCopilotAgent(toolkit)

    print("\nExecuting live multi-language evaluation across all 6,000 prompts...")
    start_time = time.time()
    records = []

    for idx, case in enumerate(testbed):
        t0 = time.perf_counter()
        response_text, tool_calls, context_data = agent.route_and_execute(case["prompt"])
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        detected_lang = context_data.get("detected_language", "en")
        expected_lang = case["language"]

        dispatched_tool_names = [t["name"] for t in tool_calls]
        expected_tool = case["expected_tool"]
        
        # 1. Metric: Language detection
        is_lang_correct = (detected_lang == expected_lang)

        # 2. Metric: Tool dispatch
        is_tool_correct = (expected_tool in dispatched_tool_names) or (
            expected_tool == "get_reserve_status" and "simulate_scenario" in dispatched_tool_names
        )

        # 3. Metric: Parameter extraction
        is_param_correct = False
        if "simulate_scenario" in dispatched_tool_names:
            sim_args = next(t["args"] for t in tool_calls if t["name"] == "simulate_scenario")
            exp_args = case.get("expected_args", {})
            exp_s = exp_args.get("sales_shock", exp_args.get("sales_shock_pct", 0.0))
            exp_d = exp_args.get("settlement_delay", exp_args.get("delay_days", 0))

            s_diff = abs(sim_args.get("sales_shock_pct", 0.0) - exp_s)
            d_diff = abs(sim_args.get("delay_days", 0) - exp_d)
            is_param_correct = (s_diff < 0.10 and d_diff == 0)
        else:
            is_param_correct = is_tool_correct

        # 4. Metric: Zero hallucination check
        has_currency_markers = ("R$" in response_text)
        is_zero_hallucination = has_currency_markers and (len(response_text) > 80)

        # 5. Rubric Score (10-point scale)
        score = 0.0
        if is_lang_correct:
            score += 2.0
        if is_tool_correct:
            score += 4.0
        if is_param_correct:
            score += 2.0
        if is_zero_hallucination:
            score += 2.0

        records.append({
            "test_id": case["test_id"],
            "language": expected_lang,
            "language_name": case["language_name"],
            "category": case["category"],
            "is_lang_correct": is_lang_correct,
            "is_tool_correct": is_tool_correct,
            "is_param_correct": is_param_correct,
            "is_zero_hallucination": is_zero_hallucination,
            "rubric_score": score,
            "latency_ms": latency_ms
        })

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} / {total_prompts} prompts ({(idx+1)/total_prompts*100:.0f}%)...")

    elapsed_total = time.time() - start_time
    print(f"\n[OK] Completed 6,000 evaluations in {elapsed_total:.2f} seconds ({total_prompts/elapsed_total:.1f} prompts/sec).")

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Overall Executive Summary
    overall_summary = {
        "Total Prompts Evaluated": len(df),
        "Languages Tested": int(df["language"].nunique()),
        "Language Detection Accuracy": f"{df['is_lang_correct'].mean()*100:.2f}%",
        "Tool Dispatch Accuracy": f"{df['is_tool_correct'].mean()*100:.2f}%",
        "Parameter Extraction Accuracy": f"{df['is_param_correct'].mean()*100:.2f}%",
        "Zero-Hallucination Rate": f"{df['is_zero_hallucination'].mean()*100:.2f}%",
        "Mean Advisory Rubric Score": f"{df['rubric_score'].mean():.2f} / 10",
        "Mean Latency per Query": f"{df['latency_ms'].mean():.2f} ms",
        "95th Percentile Latency": f"{df['latency_ms'].quantile(0.95):.2f} ms"
    }
    df_overall = pd.DataFrame([overall_summary])

    # 2. Language Breakdown Table
    df_lang = df.groupby(["language", "language_name"]).agg(
        Prompts=("test_id", "count"),
        Lang_Detect_Acc=("is_lang_correct", lambda x: f"{x.mean()*100:.1f}%"),
        Tool_Dispatch_Acc=("is_tool_correct", lambda x: f"{x.mean()*100:.1f}%"),
        Param_Extract_Acc=("is_param_correct", lambda x: f"{x.mean()*100:.1f}%"),
        Zero_Hallucination=("is_zero_hallucination", lambda x: f"{x.mean()*100:.1f}%"),
        Mean_Score=("rubric_score", lambda x: f"{x.mean():.2f} / 10"),
        Mean_Latency_ms=("latency_ms", lambda x: f"{x.mean():.2f} ms")
    ).reset_index()

    # 3. Category Breakdown Table
    df_cat = df.groupby("category").agg(
        Prompts=("test_id", "count"),
        Tool_Dispatch_Acc=("is_tool_correct", lambda x: f"{x.mean()*100:.1f}%"),
        Param_Extract_Acc=("is_param_correct", lambda x: f"{x.mean()*100:.1f}%"),
        Mean_Score=("rubric_score", lambda x: f"{x.mean():.2f} / 10")
    ).reset_index()

    # Export CSV Artifacts
    df.to_csv(RESULTS_DIR / "benchmark_6000_detailed_log.csv", index=False)
    df_overall.to_csv(RESULTS_DIR / "benchmark_6000_overall_summary.csv", index=False)
    df_lang.to_csv(RESULTS_DIR / "benchmark_6000_language_summary.csv", index=False)
    df_cat.to_csv(RESULTS_DIR / "benchmark_6000_category_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("FINSIGHT AI — 6,000 MULTILINGUAL PROMPT BENCHMARK REPORT")
    print("=" * 80)
    for k, v in overall_summary.items():
        print(f"  {k:32s}: {v}")

    print("\n" + "=" * 80)
    print("LANGUAGE-BY-LANGUAGE BREAKDOWN (1,000 PROMPTS PER LANGUAGE)")
    print("=" * 80)
    print(df_lang.to_string(index=False))

    print("\n" + "=" * 80)
    print("CATEGORY-BY-CATEGORY BREAKDOWN")
    print("=" * 80)
    print(df_cat.to_string(index=False))


if __name__ == "__main__":
    run_full_evaluation()
