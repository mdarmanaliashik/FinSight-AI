# FinSight AI: Neuro-Symbolic SME Financial Solvency, Runway Forecasting, and Multilingual Grounded Decision Intelligence

[![Live Deployment](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-00c7b7.svg)](https://finsight-ai-qwca.onrender.com)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests: 15 Passed](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)](tests/)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination%20Rate-0.0%25-success.svg)](results/)
[![Architecture: Neuro-Symbolic](https://img.shields.io/badge/Architecture-Neuro--Symbolic%20Solvency%20Framework-4a90d9.svg)](notebooks/)

> **Live Production Deployment:** [https://finsight-ai-qwca.onrender.com](https://finsight-ai-qwca.onrender.com)

---

## Abstract

More than 60% of e-commerce Small and Medium Enterprises (SMEs) fail within their first three operational years — not due to product-market misalignment, but due to **silent cash-flow insolvency**. Unpredictable demand volatility, multi-day platform settlement delays (typically 10–23 days), and rigid supplier liability schedules collectively erode working capital before conventional accounting systems raise alerts.

**FinSight AI** addresses this critical gap through a **Neuro-Symbolic Sovereign Intelligence Architecture**: integrating gradient-boosted machine learning forecasting with a formally verified GAAP-compliant double-entry accounting state machine, a continuous logistic hazard early-warning engine, and a zero-hallucination, tool-grounded AI copilot supporting six languages.

---

## System Architecture Overview

```
                       +-----------------------------------------------+
                       |      FinSight AI System Architecture          |
                       +-----------------------------------------------+
                                               |
       +---------------------------------------+------------------------------------------+
       v                                                                                  v
+-------------------------------+                              +-------------------------------+
|     NEURAL / ML STREAM        |                              |    SYMBOLIC / GAAP STREAM     |
| * Adaptive CatBoost (Champion)|                              | * Integer-Cent Double-Entry   |
| * Stacking & LightGBM Ensembl.|                              | * Exact R$ 0.00 Discrepancy   |
| * P10/P50/P90 Multi-Quantiles |                              | * Settlement Lag Sensitivity  |
| * Zero-Shot JD.com Transfer   |                              | * 30-Day Solvency Stress Sim  |
+---------------+---------------+                              +---------------+---------------+
                |                                                              |
                +------------------------------+-------------------------------+
                                               v
                               +-------------------------------+
                               |   LOGISTIC HAZARD ENGINE      |
                               | * 97.7% Breach Recall         |
                               | * 1.5-Day Early Warning Lead  |
                               +---------------+---------------+
                                               v
                               +-------------------------------+
                               |  ZERO-HALLUCINATION COPILOT   |
                               | * FinSightToolKit Contracts   |
                               | * 6 Global/Asian Languages    |
                               | * 0.0% Critical Hallucination |
                               +-------------------------------+
```

---

## Interactive Executive Decision Dashboard

- **Live Cloud Deployment:** [https://finsight-ai-qwca.onrender.com](https://finsight-ai-qwca.onrender.com)
- **Local Execution (Primary Web Server):**
  ```bash
  python server.py
  ```
  Navigate to `http://localhost:8501` to access the decision intelligence dashboard, featuring interactive scenario sliders, Chart.js cash-flow ledger visualisations, and the multilingual tool-grounded AI copilot.
- **Alternative Streamlit Interface:**
  ```bash
  streamlit run app.py
  ```

---

## Research Notebook Suite (Pre-Rendered)

All research notebooks are fully executed and pre-rendered with interactive Plotly visual analytics, statistically annotated benchmark tables, and formal mathematical derivations:

| Phase | Notebook | Size | Core Methodology, Benchmark Results & Findings |
| :---: | :--- | :---: | :--- |
| **Phase 1** | [`Phase1_Data_Ingestion_And_EDA.ipynb`](notebooks/Phase1_Data_Ingestion_And_EDA.ipynb) | `1.7 MB` | **Dual Real-World Macro Ecosystem Analysis:** Micro-pricing characterisation across **JD.com Retail** ($550\text{k}$ records, direct $0.99\text{d}$ fulfilment, China) and **Olist Brazilian E-Commerce** ($112\text{k}$ items, carrier delivery $10.2\text{d}$, Brazil). Mathematical proof and empirical elimination of **Join-Inflation** ($+4.50\%$ phantom revenue); zero-leakage chronological holdout partition; Syntetos–Boylan intermittent demand classification ($ADI=1.35$, $CV^2=1.12$); cryptographic SHA-256 data integrity manifest. |
| **Phase 2** | [`Phase2_Deterministic_Cashflow_Engine.ipynb`](notebooks/Phase2_Deterministic_Cashflow_Engine.ipynb) | `1.3 MB` | **GAAP-Exact Double-Entry Accounting State Machine:** Integer-cent accounting ledger satisfying the conservation law invariant ($\Delta \text{Cash} = \text{Inflow} - \text{Outflow}$, verified discrepancy $R\$\ 0.00$). Calibrated $5.9\%$ baseline reserve-breach prevalence; settlement latency sweep ($L \in \{2, 7, 10, 15, 23\}\text{d}$); onset-dated stress testing ($t_0=180$, $30\%$ demand collapse, $25\%$ COGS spike); 5 golden hand-calculated arithmetic cross-audits. |
| **Phase 3** | [`Phase3_Sales_Forecasting_Benchmark.ipynb`](notebooks/Phase3_Sales_Forecasting_Benchmark.ipynb) | `1.7 MB` | **Multi-Model Forecasting Benchmark & Cross-Market Transfer Evaluation:** Verified champion — **Adaptive Trend-Calibrated CatBoost** (MAE $218.93\text{ BRL}$, WAPE $59.80\%$, Directional Accuracy $71.1\%$, $p < 0.001$, $+25.81\%$ relative improvement over naïve baseline); Upgraded Stacking Ensemble (CatBoost + LightGBM + Ridge); Platform Cumulative Inflow ($H=7$) — CatBoost achieves **WAPE $0.93\%$** on Olist and **WAPE $7.43\%$ ($R^2 = 0.9966$)** on JD.com; Cross-Market Zero-Shot Domain Transfer achieving **WAPE $18.80\%$ ($R^2 = +0.4708$)** with zero fine-tuning; Foundation model benchmarks (Chronos, MOIRAI, TimeGPT); calibrated multi-quantile uncertainty ($P_{10}/P_{50}/P_{90}$, PICP $67.11\%$). |
| **Phase 4** | [`Phase4_Risk_Warning_And_Copilot.ipynb`](notebooks/Phase4_Risk_Warning_And_Copilot.ipynb) | `620 KB` | **Hazard Early-Warning Engine & Tool-Grounded Copilot Evaluation:** Integration of Phase 3 quantile stream into the $B_2$ early-warning engine; continuous logistic hazard risk scoring ($97.7\%$ breach recall, $1.5\text{d}$ lead-time advantage over reactive thresholding); strict `FinSightToolKit` zero-hallucination function-calling contracts; blinded 60-prompt multi-category rubric evaluation ($9.60/10$ vs $3.60/10$ for unaugmented LLMs, $0.0\%$ critical hallucination rate); scaled 6,000-prompt multilingual benchmark across 6 languages. |
| **Phase 5** | [`Phase5_System_Ablation_And_EndToEnd_Benchmark.ipynb`](notebooks/Phase5_System_Ablation_And_EndToEnd_Benchmark.ipynb) | `684 KB` | **Systematic Ablation Study ($E_0 \dots E_4$) & Frontier Model Evaluation:** Rigorous marginal ablation quantifying the cumulative contribution of each architectural component; cold-start data scarcity analysis ($30\dots 365\text{d}$); scaled 6,000-prompt stress evaluation across 6 languages; tri-model head-to-head frontier comparison (Gemini 1.5 Pro vs Qwen 2.5 vs GPT-4o); formal peer-review defence matrix. |

---

## Phase 3 Benchmark Results

### 1. Primary Forecasting Task: Micro-SME Tactical Daily Sales (Holdout Horizon $H = 225$ Days)

The following table presents a systematic comparative evaluation of all candidate forecasting models, ranked by Mean Absolute Error (MAE) on the chronologically held-out test partition. Statistical significance is assessed via the Diebold–Mariano test against the historical mean baseline.

| Model Architecture | MAE (R$) | RMSE (R$) | WAPE (%) | Directional Accuracy | Diebold–Mariano $p$-value | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive CatBoost (Champion)** | **218.93** | 358.37 | **59.80%** | **71.1%** | **< 0.001** | **+25.81%** |
| **Upgraded Stacking Ensemble** | 231.26 | **295.40** | 63.17% | 68.4% | < 0.001 | +21.63% |
| **Walk-Forward LightGBM** | 238.12 | 344.20 | 65.04% | 69.8% | < 0.001 | +19.31% |
| **CatBoost (Raw Feature Baseline)** | 240.50 | 360.10 | 65.69% | 68.0% | < 0.001 | +18.50% |
| **Support Vector Regression (RBF Kernel)** | 260.15 | 390.45 | 71.06% | 62.5% | 0.012 | +11.84% |
| **LSTM Recurrent Neural Network** | 275.40 | 410.20 | 75.22% | 58.2% | 0.045 | +6.68% |
| **Zero-Predictor Hurdle Baseline** | 366.11 | 498.70 | 100.00% | 0.0% | — | −24.06% |
| **Historical Mean Baseline** | 295.10 | 425.80 | 80.61% | 50.0% | — | Reference |

### 2. Platform-Level Macro Cumulative Inflow Forecasting ($H = 7$ Horizon)

| Dataset & Scope | Model | Daily Sales WAPE | Cumulative Inflow ($H=7$) WAPE | Inflow $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Olist Brazil (Full Marketplace)** | CatBoost Platform Macro | 25.72% | **0.93%** | **0.9994** |
| **JD.com Retail (China High-Frequency)** | CatBoost Platform Macro | 31.45% | **7.43%** | **0.9966** |

### 3. Cross-Market Zero-Shot Domain Transfer (Olist Brazil $\to$ JD.com China)

| Model & Formulation | Input Representation | Transfer WAPE | Transfer $R^2$ | Operational Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **Scale-Invariant Ratio LightGBM** | Moving-Average Ratio Space | **18.80%** | **+0.4708** | **Viable — Direct Zero-Shot Transfer** |
| **Uncalibrated Raw LightGBM** | Raw Monetary Units (R$) | 134.20% | −1.8410 | Failed — Severe Currency-Scale Collapse |

---

## Multilingual Copilot Evaluation ($N = 6{,}000$ Prompt Stress Suite)

The following table summarises the performance of the tool-grounded AI copilot across a scaled, language-balanced evaluation suite comprising 6,000 prompts distributed equally across six languages.

| Language | Script Family | Prompts | Language Detection Accuracy | Tool Dispatch Accuracy | Zero-Hallucination Rate | Mean Rubric Score | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **English** | Latin | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.50 / 10** | 4.34 ms |
| **Bengali (বাংলা)** | Bengali | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.05 / 10** | 6.00 ms |
| **Chinese (中文)** | Hanzi | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.05 / 10** | 6.17 ms |
| **Japanese (日本語)** | Kana / Kanji | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.00 / 10** | 5.59 ms |
| **Hindi (हिन्दी)** | Devanagari | 1,000 | **100.0%** | **75.0%** | **100.0%** | **8.50 / 10** | 3.42 ms |
| **Urdu (اردو)** | Nastaliq | 1,000 | **100.0%** | **75.0%** | **100.0%** | **8.06 / 10** | 4.68 ms |
| **Aggregate (Full Suite)** | **6 Scripts** | **6,000** | **100.0%** | **91.67%** | **100.0%** | **8.86 / 10** | **5.04 ms** |

---

## Serialised Model Artefacts (`models/`)

All trained model artefacts are serialised using `joblib` and are cryptographically tracked in `data/processed/phase3_dataset_manifest.json`:

```
models/
├── olist_best_catboost_ratio.joblib       # Champion: Adaptive Trend-Calibrated CatBoost (WAPE 59.80%)
├── olist_best_stacking_ratio.joblib       # Upgraded Stacking Ensemble (CatBoost + LightGBM + Ridge)
├── olist_best_lightgbm_ratio.joblib       # Scale-Invariant Ratio-Space LightGBM
├── olist_best_catboost_raw.joblib         # Uncalibrated Baseline CatBoost
├── olist_platform_cb_ratio.joblib         # Macro Platform CatBoost (Cumulative Inflow WAPE 0.93%)
└── olist_platform_lgb_ratio.joblib        # Macro Platform LightGBM
```

---

## Reproducibility & Environment Setup

### 1. Environment Configuration

```bash
# Clone the repository
git clone https://github.com/mdarmanaliashik/FinSight-AI.git
cd FinSight-AI

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install all declared dependencies
pip install -r requirements.txt
```

### 2. Executing the Research Notebooks

Launch Jupyter Lab or open any notebook in VS Code:

```bash
jupyter lab notebooks/
```

All notebooks (Phase 1 through Phase 5) are pre-rendered with pre-executed cells, interactive Plotly visualisations, and inline mathematical derivations formatted for publication readability.

### 3. Loading Trained Model Artefacts

```python
import joblib
import pandas as pd

# Load the Phase 3 champion model
champion_model = joblib.load('models/olist_best_catboost_ratio.joblib')

# Inference: predict ratio-space multiplier from 40 engineered features
# pred_ratio = champion_model.predict(features)
# forecasted_sales = pred_ratio * rolling_trend_ma7
```

### 4. Launching the Dashboard Interfaces

```bash
# Primary executive decision intelligence web server
python server.py

# Alternative Streamlit application interface
streamlit run app.py
```

Navigate to `http://localhost:8501` in a web browser. The live cloud deployment is accessible at [https://finsight-ai-qwca.onrender.com](https://finsight-ai-qwca.onrender.com).

---

## Data Provenance & Licensing

- **Olist Brazilian E-Commerce Dataset:** Olist & André Sionek. DOI: [10.5281/zenodo.10515152](https://doi.org/10.5281/zenodo.10515152). Licensed under CC BY-NC-SA 4.0.
- **JD.com Retail Dataset:** Asian e-commerce transaction repository comprising 550,000 verified purchase records.
- **Codebase Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
