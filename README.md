# FinSight AI: Sovereign SME Financial Solvency, Runway Forecasting & Multilingual Grounded Copilot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests: 15 Passed](https://img.shields.io/badge/Tests-15%2F15%20Passing-brightgreen.svg)](tests/)
[![Hallucination Rate: 0.0%](https://img.shields.io/badge/Hallucination%20Rate-0.0%25-success.svg)](results/)

> **FinSight AI** is an enterprise-grade neuro-symbolic decision support system for Small and Medium Enterprises (SMEs) and cross-border e-commerce merchants. It couples **purged walk-forward LightGBM sales forecasting**, a **deterministic GAAP-exact double-entry accounting state machine**, and a **zero-hallucination tool-grounded AI copilot** natively supporting 6 Asian & Global languages.

---

## 📚 Complete Pre-Rendered Research Notebook Suite

All research notebooks are fully executed and pre-rendered with interactive Plotly visualizations, styled tables, and formal mathematical proofs:

| Phase | Notebook Path | File Size | Core Contribution & Innovation |
| :---: | :--- | :---: | :--- |
| **Phase 1** | [`Phase1_Data_Ingestion_And_EDA.ipynb`](notebooks/Phase1_Data_Ingestion_And_EDA.ipynb) | `752 KB` | **Dual Primary Datasets:** JD.com Retail ($550\\text{k}$ items, China) & Olist E-Commerce ($112\\text{k}$ items, Brazil); Exact micro-pricing discount reconciliation ($100.0\\%$ error $< 10^{-13}$); Fulfillment latency dichotomy (JD direct $0.99\\text{d}$ vs Olist carrier $10.2\\text{d}$); Join-inflation empirical proof ($+4.50\\%$ phantom revenue); Zero-leakage invariance test; Syntetos-Boylan demand classification; Delivery right-censoring cutoff at 2018-08-20 ($98.2\\%$ complete); Cryptographic SHA-256 data manifest. |
| **Phase 2** | [`Phase2_Deterministic_Cashflow_Engine.ipynb`](notebooks/Phase2_Deterministic_Cashflow_Engine.ipynb) | `542 KB` | Integer-cent double-entry accounting engine with exact $R\$\ 0.00$ discrepancy; calibrated $5.9\%$ breach prevalence; onset-dated shocks ($t_0=180$); settlement lag sensitivity ($2, 10, 15, 23\text{d}$). |
| **Phase 3** | [`Phase3_Sales_Forecasting_Benchmark.ipynb`](notebooks/Phase3_Sales_Forecasting_Benchmark.ipynb) | `419 KB` | Stochastic 5-seed LightGBM with true $\sigma > 0$; target-aligned lags ($T=t+H, y_{T-7}=y_t$); multi-series evaluation across 4 dynamic regimes; Harvey-Leybourne-Newbold Diebold-Mariano test ($p=0.0002$). |
| **Phase 4** | [`Phase4_Risk_Warning_And_Copilot.ipynb`](notebooks/Phase4_Risk_Warning_And_Copilot.ipynb) | `377 KB` | Logistic Hazard early warning ($97.7\%$ breach recall, $1.5\text{d}$ lead time); 60-prompt grounded copilot benchmark ($9.60/10$ score vs $3.60/10$ for unaugmented LLMs, $0.0\%$ critical hallucination). |
| **Phase 5** | [`Phase5_System_Ablation_And_EndToEnd_Benchmark.ipynb`](notebooks/Phase5_System_Ablation_And_EndToEnd_Benchmark.ipynb) | `647 KB` | **Master System Ablation ($E_0 \dots E_4$) & Scaled Multilingual Frontier Benchmark:** Cold-start data scarcity curve ($30\dots 365\text{d}$); live multi-script query walkthrough; **6,000-prompt stress testbed** across 6 Asian/Global languages; Tri-Model head-to-head frontier evaluation (Gemini 1.5 Pro vs Qwen 2.5 vs GPT-4o); 4-panel publication figure; structured reviewer defense matrix. |

---

## 🌐 6-Language Multilingual Performance ($N=6,000$)

| Language | Script | Prompts | Language Detection | Tool Dispatch Accuracy | Zero-Hallucination Rate | Mean Rubric Score | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🇬🇧 **English** | Latin | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.50 / 10** | 4.34 ms |
| 🇧🇩 **Bengali (বাংলা)** | Bengali | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.05 / 10** | 6.00 ms |
| 🇨🇳 **Chinese (中文)** | Hanzi | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.05 / 10** | 6.17 ms |
| 🇯🇵 **Japanese (日本語)** | Kana/Kanji | 1,000 | **100.0%** | **100.0%** | **100.0%** | **9.00 / 10** | 5.59 ms |
| 🇮🇳 **Hindi (हिन्दी)** | Devanagari | 1,000 | **100.0%** | **75.0%** | **100.0%** | **8.50 / 10** | 3.42 ms |
| 🇵🇰 **Urdu (اردو)** | Nastaliq | 1,000 | **100.0%** | **75.0%** | **100.0%** | **8.06 / 10** | 4.68 ms |
| **Overall Suite** | **6 Languages** | **6,000** | **100.0%** | **91.67%** | **100.0%** | **8.86 / 10** | **5.04 ms** |

---

## ⚡ Quickstart

### 1. Run Interactive Dashboard (Streamlit)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch web application
streamlit run app.py
```
Open `http://localhost:8501` to test interactive scenario sliders and multilingual copilot quick prompts.

### 2. Run Test Suite
```bash
pytest tests -v
```
All **15 unit tests** validate double-entry conservation, settlement shifts, and multilingual intent dispatch.

### 3. Docker Deployment
```bash
docker compose up -d
```

---

## 📄 Provenance & Licensing
- **UCI Online Retail Dataset:** London South Bank University (DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33), CC BY 4.0).
- **Olist Brazilian E-Commerce:** Olist & André Sionek (DOI: [10.5281/zenodo.10515152](https://doi.org/10.5281/zenodo.10515152), CC BY-NC-SA 4.0).
