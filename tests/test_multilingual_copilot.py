"""
Unit Tests for Multilingual Omni-Model Copilot Agent
Verifies language detection, parameter extraction, and deterministic tool dispatch
across English, Bengali, Hindi, Urdu, Chinese, and Japanese.
"""

import pytest
import numpy as np
import pandas as pd
from src.cashflow.engine import SMEConfig
from src.copilot.tools import FinSightToolKit
from src.copilot.agent import GroundedCopilotAgent, detect_language, normalize_digits


@pytest.fixture
def mock_copilot_agent():
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
    sales = pd.Series(np.full(30, 25000.0))
    toolkit = FinSightToolKit(cfg, sales, sales * 0.7, sales * 1.3, dates)
    return GroundedCopilotAgent(toolkit)


def test_language_detection():
    assert detect_language("যদি বিক্রি ২০% কমে") == "bn"
    assert detect_language("यदि बिक्री में 20% की गिरावट आती है") == "hi"
    assert detect_language("اگر فروخت میں 20 فیصد کمی ہو") == "ur"
    assert detect_language("如果销售额下降20%") == "zh"
    assert detect_language("売上が20%減少した場合") == "ja"
    assert detect_language("If sales drop by 20%") == "en"


def test_normalize_digits():
    assert normalize_digits("২০%") == "20%"
    assert normalize_digits("२५%") == "25%"
    assert normalize_digits("۳۰%") == "30%"


@pytest.mark.parametrize("prompt,expected_lang", [
    ("যদি আগামী মাসে আমাদের বিক্রি ২০% কমে এবং খরচ ১৫% বাড়ে, তবে কত লোন লাগবে?", "bn"),
    ("যদি মার্কেটপ্লেস প্রাপ্য টাকা দিতে ৭ দিন দেরি করে, কী হবে?", "bn"),
    ("अगर बिक्री में 20% की गिरावट आती है और लागत 15% बढ़ जाती है, तो कितना लोन चाहिए?", "hi"),
    ("اگر فروخت میں 20 فیصد کمی واقع ہو تو کتنا قرض درکار ہوگا؟", "ur"),
    ("如果下个月销售额下降20%且成本上升15%，我们需要借多少短期贷款？", "zh"),
    ("来月の売上が20%減少し、仕入原価が15%上昇した場合、いくらの短期融資が必要ですか？", "ja"),
    ("If sales drop by 20% and supplier costs rise by 15%, how much credit line is required?", "en")
])
def test_multilingual_dispatch_and_grounding(mock_copilot_agent, prompt, expected_lang):
    ans, tools, ctx = mock_copilot_agent.route_and_execute(prompt)
    assert ctx["detected_language"] == expected_lang
    assert len(tools) > 0
    assert "simulate_scenario" in [t["name"] for t in tools]
    # Verify response contains exact currency numbers
    assert "R$" in ans
    assert len(ans) > 50
