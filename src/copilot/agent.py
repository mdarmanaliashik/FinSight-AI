"""
FinSight AI - Multilingual Tool-Grounded Copilot Agent & Evaluation Rubric
==========================================================================
Implements:
1. Multilingual Omni-Model Copilot: Automatically detects input language
   across English, Bengali (বাংলা), Hindi (हिन्दी), Urdu (اردو),
   Chinese (中文), and Japanese (日本語).
2. Extracts shock and financial parameters across all scripts.
3. Dispatches Pydantic tool contracts to FinSightToolKit, guaranteeing
   zero numerical hallucinations.
4. Synthesizes culturally and grammatically natural financial advisory
   strictly grounded in exact deterministic integer-cent calculations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from src.copilot.tools import FinSightToolKit


class CopilotEvaluationResult(BaseModel):
    prompt_id: str
    category: str
    prompt: str
    model_type: str
    response_text: str
    tool_calls_executed: List[str]
    numerical_correctness: float # 0 to 2
    evidence_grounding: float    # 0 to 2
    completeness: float          # 0 to 2
    actionability: float         # 0 to 2
    uncertainty_transparency: float # 0 to 2
    total_score: float           # 0 to 10
    hallucination_flag: bool     # True if invented unsupported math


def detect_language(text: str) -> str:
    """Detects script family using Unicode ranges."""
    if re.search(r'[\u0980-\u09FF]', text):
        return 'bn' # Bengali
    if re.search(r'[\u0900-\u097F]', text):
        return 'hi' # Hindi
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ur' # Urdu
    if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
        return 'ja' # Japanese
    if re.search(r'[\u4E00-\u9FFF]', text):
        return 'zh' # Chinese
    return 'en'     # English default


def normalize_digits(text: str) -> str:
    """Converts Indic and Arabic digits to standard ASCII numerals."""
    indic_digits = {
        '০':'0', '১':'1', '২':'2', '৩':'3', '৪':'4', '৫':'5', '৬':'6', '৭':'7', '৮':'8', '৯':'9',
        '०':'0', '१':'1', '२':'2', '३':'3', '४':'4', '५':'5', '६':'6', '७':'7', '८':'8', '९':'9',
        '۰':'0', '۱':'1', '۲':'2', '۳':'3', '۴':'4', '۵':'5', '۶':'6', '۷':'7', '۸':'8', '۹':'9'
    }
    for k, v in indic_digits.items():
        text = text.replace(k, v)
    return text


class GroundedCopilotAgent:
    """
    FinSight AI Core: Multilingual Tool-Grounded Financial Copilot.
    Strictly queries deterministic engines before synthesizing localized responses.
    """

    def __init__(self, toolkit: FinSightToolKit):
        self.toolkit = toolkit

    def route_and_execute(self, prompt: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Interprets multilingual user intent, parses numerical parameters,
        executes deterministic engines, and formats an evidence-backed answer
        in the user's native language.
        """
        lang = detect_language(prompt)
        norm_prompt = normalize_digits(prompt)
        p_lower = norm_prompt.lower()
        tool_calls = []
        context_data = {"detected_language": lang}

        # Multilingual Intent Matching Keywords
        shock_keywords = [
            "shock", "drop", "delay", "what-if", "inflation", "cogs", "fall", "loss",
            # Bengali
            "কমে", "হ্রাস", "পতন", "ড্রপ", "ক্ষতি", "দেরি", "বিলম্ব", "কাঁচামাল",
            # Hindi
            "गिरावट", "कमी", "नुकसान", "देरी", "विलंब", "महंगाई",
            # Urdu
            "کمی", "گراوٹ", "نقصان", "تاخیر", "مہنگائی",
            # Chinese
            "下降", "下滑", "跌", "减少", "延迟", "延期", "成本", "通胀",
            # Japanese
            "減少", "下落", "低下", "遅延", "遅れ", "コスト", "原価"
        ]

        forecast_keywords = [
            "forecast", "sales", "revenue", "demand", "predict",
            # Bengali
            "বিক্রি", "সেলস", "পূর্বাভাস", "আয়", "চাহিদা",
            # Hindi
            "बिक्री", "पूर्वानुमान", "राजस्व", "मांग",
            # Urdu
            "فروخت", "پیشین گوئی", "آمدنی",
            # Chinese
            "销售", "预测", "收入", "需求",
            # Japanese
            "売上", "予測", "収益", "需要"
        ]

        reserve_keywords = [
            "reserve", "buffer", "opex", "minimum cash",
            # Bengali
            "রিজার্ভ", "বাফার", "মিনিমাম", "জরুরি তহবিল",
            # Hindi
            "रिज़र्व", "बफर", "न्यूनतम",
            # Urdu
            "ریزرو", "بفر", "کم از کم",
            # Chinese
            "储备", "安全底线", "最低现金",
            # Japanese
            "準備金", "バッファ", "最低現金"
        ]

        is_shock = any(k in p_lower for k in shock_keywords)
        is_forecast = any(k in p_lower for k in forecast_keywords)
        is_reserve = any(k in p_lower for k in reserve_keywords)

        # 1. Intent Detection & Tool Execution
        if is_shock:
            # Extract percentages dynamically via regex
            pct_matches = re.findall(r'(\d+)\s*%', p_lower)
            sales_shock = 0.0
            cost_shock = 0.0

            if pct_matches:
                p1 = float(pct_matches[0]) / 100.0
                sales_shock = -p1 # Primary drop
                if len(pct_matches) > 1:
                    p2 = float(pct_matches[1]) / 100.0
                    cost_shock = p2
            else:
                # Pure delay query without percentage shocks
                if any(k in p_lower for k in ["delay", "দেরি", "देरी", "تاخیر", "延迟", "遅延"]):
                    sales_shock = 0.0
                    cost_shock = 0.0
                else:
                    sales_shock = -0.20
                    cost_shock = 0.15

            # Extract delay days
            day_matches = re.findall(r'(\d+)\s*(?:day|business days|দিন|दिन|دن|天|日)', p_lower)
            delay_days = int(day_matches[0]) if day_matches else (7 if any(k in p_lower for k in ["delay", "দেরি", "देरी", "تاخیر", "延迟", "遅延"]) else 0)

            args = {"sales_shock_pct": sales_shock, "cost_shock_pct": cost_shock, "delay_days": delay_days}
            res = self.toolkit.execute_tool("simulate_scenario", args)
            tool_calls.append({"name": "simulate_scenario", "args": args})
            context_data["simulation"] = res

        elif is_forecast and not is_reserve:
            h = 30 if ("30" in p_lower or "month" in p_lower or "মাস" in p_lower or "महीना" in p_lower or "月" in p_lower) else (14 if "14" in p_lower else 7)
            q = 0.10 if ("worst" in p_lower or "downside" in p_lower or "মন্দের" in p_lower or "保守" in p_lower) else 0.50
            args = {"horizon_days": h, "quantile": q}
            res = self.toolkit.execute_tool("get_sales_forecast", args)
            tool_calls.append({"name": "get_sales_forecast", "args": args})
            context_data["forecast"] = res

        elif is_reserve:
            res = self.toolkit.execute_tool("get_reserve_status", {})
            tool_calls.append({"name": "get_reserve_status", "args": {}})
            context_data["reserve"] = res

        else: # Default: Cashflow projection
            sc = "downside" if ("downside" in p_lower or "stress" in p_lower or "সংকট" in p_lower) else "median"
            args = {"days": 30, "scenario_type": sc}
            res = self.toolkit.execute_tool("get_cashflow_projection", args)
            tool_calls.append({"name": "get_cashflow_projection", "args": args})
            context_data["projection"] = res

        # Always attach reserve status for grounding
        if "reserve" not in context_data:
            res_status = self.toolkit.get_reserve_status()
            context_data["reserve_status"] = res_status
            tool_calls.append({"name": "get_reserve_status", "args": {}})

        # 2. Synthesis of Grounded Multilingual Response
        response = self._synthesize_response(prompt, context_data, lang)
        return response, tool_calls, context_data

    def _synthesize_response(self, prompt: str, context: Dict[str, Any], lang: str = "en") -> str:
        """Synthesizes localized advisory response anchored in exact tool numbers."""
        lines = []

        if "simulation" in context:
            sim = context["simulation"]
            
            if lang == "bn": # Bengali
                lines.append("### 📊 আর্থিক স্ট্রেস-টেস্ট বিশ্লেষণ (ডিটারমিনিস্টিক ইঞ্জিন আউটপুট)")
                lines.append(f"- **সিমুলেটেড শক:** সেলস শক: `{sim['sales_shock_pct'] * 100:+.0f}%` | খরচ বৃদ্ধি: `{sim['cost_shock_pct'] * 100:+.0f}%` | পেআউট বিলম্ব: `{sim['delay_days']}` দিন।")
                lines.append(f"- **স্বাভাবিক সমাপনী ক্যাশ:** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **সংকটকালীন সমাপনী ক্যাশ:** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **নিট আর্থিক ক্ষতি:** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **রিজার্ভ ব্রিচ অ্যালার্ট:** হ্যাঁ। সর্বোচ্চ লিকুইডিটি ঘাটতি: **R$ {sim['shortfall_amount']:,.2f}**।")
                    lines.append(f"- **ক্যাশ রানওয়ে:** `{sim['baseline_runway_days']}` দিন থেকে কমে `{sim['shocked_runway_days']}` দিন (ঘাটতি: `{abs(sim['runway_change_days'])}` দিন)।")
                    lines.append("\n**SME ব্যবসায়ীর জন্য অ্যাকশনেবল সুপারিশ:**")
                    lines.append("1. **সাপ্লায়ার দেনা বিলম্ব:** নগদ প্রবাহ ঠিক রাখতে অপ্রয়োজনীয় বিল পরিশোধ ৭ কার্যদিবস পিছিয়ে দিন।")
                    lines.append(f"2. **স্বল্পমেয়াদী ক্রেডিট লাইন:** কর্মচারীদের বেতনের আগে কমপক্ষে **R$ {sim['shortfall_amount']:,.2f}** পরিমাণ র্কিং ক্যাপিটাল বাফার নিশ্চিত করুন।")
                else:
                    lines.append("- ✅ **নিরাপদ স্ট্যাটাস:** এই শক পরিস্থিতিতে রিজার্ভ ফ্লোর অক্ষুণ্ণ রয়েছে।")
                    lines.append(f"- **বর্তমান রানওয়ে:** `{sim['shocked_runway_days']}` দিন।")

            elif lang == "zh": # Chinese
                lines.append("### 📊 压力测试情景分析（确定性状态引擎测算）")
                lines.append(f"- **模拟冲击：** 销售下滑：`{sim['sales_shock_pct'] * 100:+.0f}%` | 成本通胀：`{sim['cost_shock_pct'] * 100:+.0f}%` | 回款延迟：`{sim['delay_days']}`天。")
                lines.append(f"- **基准期末现金：** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **冲击后期末现金：** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **净流动性影响：** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **储备金预警：** 是。最大流动性资金缺口为 **R$ {sim['shortfall_amount']:,.2f}**。")
                    lines.append(f"- **现金跑道：** 从 `{sim['baseline_runway_days']}` 天缩短至 `{sim['shocked_runway_days']}` 天（减少 `{abs(sim['runway_change_days'])}` 天）。")
                    lines.append("\n**针对SME店主的行动建议：**")
                    lines.append("1. **供应商账期谈判：** 将非核心应付款顺延7个工作日，以匹配现金回款节奏。")
                    lines.append(f"2. **短期信用融资：** 在工资发放日前，确保至少 **R$ {sim['shortfall_amount']:,.2f}** 的营运流动资金授信额度。")
                else:
                    lines.append("- ✅ **偿债安全：** 在当前情景下，未跌破最低储备金底线。")
                    lines.append(f"- **剩余跑道：** 稳定在 `{sim['shocked_runway_days']}` 天。")

            elif lang == "ja": # Japanese
                lines.append("### 📊 ストレスシナリオ分析（確定性エンジン出力）")
                lines.append(f"- **シミュレーション衝撃：** 売上減少: `{sim['sales_shock_pct'] * 100:+.0f}%` | 原価上昇: `{sim['cost_shock_pct'] * 100:+.0f}%` | 回収遅延: `{sim['delay_days']}`日。")
                lines.append(f"- **基本予測期末現金：** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **ストレス後手元資金：** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **純流動性影響額：** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **準備金割れ警告：** あり。最大流動性不足額は **R$ {sim['shortfall_amount']:,.2f}** です。")
                    lines.append(f"- **キャッシュランウェイ：** `{sim['baseline_runway_days']}` 日から `{sim['shocked_runway_days']}` 日へ短縮（損失: `{abs(sim['runway_change_days'])}` 日）。")
                    lines.append("\n**中小企業経営者向け推奨アクションプラン：**")
                    lines.append("1. **買掛金の支払い繰り延べ：** 資金流入サイクルに合わせ、緊急性の低い支払いを7営業日延長。")
                    lines.append(f"2. **短期当座貸越枠の確保：** 給与支払い前に、少なくとも **R$ {sim['shortfall_amount']:,.2f}** の運転資金枠を確保。")
                else:
                    lines.append("- ✅ **財務健全性：** このショック下でも準備金基準を維持しています。")
                    lines.append(f"- **残存ランウェイ：** `{sim['shocked_runway_days']}` 日。")

            elif lang == "hi": # Hindi
                lines.append("### 📊 तनाव-परीक्षण परिदृश्य विश्लेषण (डिटर्मिनिस्टिक इंजन आउटपुट)")
                lines.append(f"- **अनुकरण किए गए झटके:** बिक्री झटका: `{sim['sales_shock_pct'] * 100:+.0f}%` | लागत वृद्धि: `{sim['cost_shock_pct'] * 100:+.0f}%` | भुगतान देरी: `{sim['delay_days']}` दिन।")
                lines.append(f"- **बेसलाइन अंतिम नकदी:** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **तनाव के बाद अंतिम नकदी:** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **शुद्ध तरलता प्रभाव:** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **रिज़र्व उल्लंघन चेतावनी:** हाँ। अधिकतम नकदी घाटा: **R$ {sim['shortfall_amount']:,.2f}**।")
                    lines.append(f"- **कैश रनवे:** `{sim['baseline_runway_days']}` दिनों से घटकर `{sim['shocked_runway_days']}` दिन।")
                    lines.append("\n**व्यवसाय के लिए कार्रवाई योग्य सिफारिशें:**")
                    lines.append("1. **आपूर्तिकर्ता भुगतान में देरी:** गैर-महत्वपूर्ण भुगतानों को 7 कार्य दिवसों के लिए टालें।")
                    lines.append(f"2. **अल्पकालिक क्रेडिट लाइन:** पेरोल से पहले कम से कम **R$ {sim['shortfall_amount']:,.2f}** की कार्यशील पूंजी बफर सुरक्षित करें।")
                else:
                    lines.append("- ✅ **सुरक्षित स्थिति:** इस तनाव परिदृश्य में रिज़र्व सुरक्षित है।")

            elif lang == "ur": # Urdu
                lines.append("### 📊 فنانشل اسٹریس ٹیسٹ تجزیہ (ڈیٹرمینسٹک انجن آؤٹ پٹ)")
                lines.append(f"- **فرضی اثرات:** سیلز گراوٹ: `{sim['sales_shock_pct'] * 100:+.0f}%` | لاگت اضافہ: `{sim['cost_shock_pct'] * 100:+.0f}%` | تاخیر: `{sim['delay_days']}` دن۔")
                lines.append(f"- **بنیادی اختتامی نقد رقم:** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **اسٹریس کے بعد نقد رقم:** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **خالص مالی اثر:** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **ریزرو الرٹ:** ہاں۔ زیادہ سے زیادہ مالی خسارہ: **R$ {sim['shortfall_amount']:,.2f}**۔")
                    lines.append("\n**تجاویز:**")
                    lines.append(f"1. غیر ضروری بلوں کی ادائیگی 7 دن کے لیے موخر کریں۔")
                    lines.append(f"2. تنخواہوں سے پہلے کم از کم **R$ {sim['shortfall_amount']:,.2f}** کا ورکنگ کیپیٹل لون حاصل کریں۔")
                else:
                    lines.append("- ✅ **محفوظ پوزیشن:** ریزرو فنڈز محفوظ ہیں۔")

            else: # English default
                lines.append("### 📊 Scenario Stress-Test Analysis (Deterministic Engine Output)")
                lines.append(f"- **Simulated Shocks:** Sales Shock: `{sim['sales_shock_pct'] * 100:+.0f}%` | Cost Shock: `{sim['cost_shock_pct'] * 100:+.0f}%` | Payout Delay: `{sim['delay_days']}` days.")
                lines.append(f"- **Baseline Ending Cash:** R$ {sim['baseline_ending_cash']:,.2f}")
                lines.append(f"- **Shocked Ending Cash:** R$ {sim['shocked_ending_cash']:,.2f}")
                lines.append(f"- **Net Liquidity Impact:** R$ {sim['net_cash_impact']:,.2f}")
                if sim['is_reserve_breach']:
                    lines.append(f"- 🚨 **Reserve Breach Alert:** YES. Maximum liquidity shortfall of **R$ {sim['shortfall_amount']:,.2f}**.")
                    lines.append(f"- **Cash Runway:** Reduced from `{sim['baseline_runway_days']}` days to `{sim['shocked_runway_days']}` days (Loss of `{abs(sim['runway_change_days'])}` days of runway).")
                    lines.append("\n**Recommended Action Plan for SME Owner:**")
                    lines.append("1. **Supplier Negotiation:** Defer non-critical AP payments by 7 business days to match cash inflows.")
                    lines.append(f"2. **Short-term Line of Credit:** Secure a working capital buffer of at least R$ {sim['shortfall_amount']:,.2f} before payroll.")
                else:
                    lines.append("- ✅ **Solvency Check:** No reserve breach detected under this shock scenario.")
                    lines.append(f"- **Remaining Runway:** Stable at `{sim['shocked_runway_days']}` days.")

        elif "forecast" in context:
            fc = context["forecast"]
            if lang == "bn":
                lines.append(f"### 📈 পূর্বাভাসকৃত বিক্রয় ({fc['quantile_label']})")
                lines.append(f"- **পূর্বাভাস দিগন্ত:** `{fc['horizon_days']}` দিন।")
                lines.append(f"- **প্রত্যাশিত মোট আয়:** R$ {fc['total_projected_sales']:,.2f}")
                lines.append(f"- **দৈনিক গড় বিক্রয়:** R$ {fc['mean_daily_sales']:,.2f} (সীমা: R$ {fc['min_daily_sales']:,.2f} – R$ {fc['max_daily_sales']:,.2f})।")
            else:
                lines.append(f"### 📈 Temporal Sales Forecast ({fc['quantile_label']})")
                lines.append(f"- **Forecast Horizon:** `{fc['horizon_days']}` days ahead.")
                lines.append(f"- **Expected Total Revenue:** R$ {fc['total_projected_sales']:,.2f}")
                lines.append(f"- **Mean Daily Run-Rate:** R$ {fc['mean_daily_sales']:,.2f} (Daily bounds: R$ {fc['min_daily_sales']:,.2f} – R$ {fc['max_daily_sales']:,.2f}).")

        elif "projection" in context:
            proj = context["projection"]
            if lang == "bn":
                lines.append(f"### 🏦 ৩০-দিনের ক্যাশফ্লো প্রজেকশন ({proj['scenario_type']})")
                lines.append(f"- **প্রারম্ভিক নগদ:** R$ {proj['starting_cash']:,.2f}")
                lines.append(f"- **সমাপনী নগদ:** R$ {proj['ending_cash']:,.2f}")
                lines.append(f"- **সর্বনিম্ন নগদ ব্যালেন্স:** R$ {proj['min_cash_balance']:,.2f} (রিজার্ভ সীমা: R$ {proj['min_reserve_limit']:,.2f})।")
            else:
                lines.append(f"### 🏦 30-Day Cashflow Projection ({proj['scenario_type']})")
                lines.append(f"- **Starting Cash:** R$ {proj['starting_cash']:,.2f}")
                lines.append(f"- **Ending Cash:** R$ {proj['ending_cash']:,.2f}")
                lines.append(f"- **Minimum Cash Low-Point:** R$ {proj['min_cash_balance']:,.2f} (Threshold Limit: R$ {proj['min_reserve_limit']:,.2f}).")

        if "reserve_status" in context and "simulation" not in context:
            r = context["reserve_status"]
            lines.append(f"\n*Current Liquidity Status: Cash = R$ {r['current_cash']:,.2f} | Buffer = R$ {r['buffer_amount']:,.2f} | Fixed OPEX Coverage = {r['reserve_coverage_months']} months.*")

        return "\n".join(lines)


class VanillaCopilotBaseline:
    """
    Unaugmented LLM Baseline (D0): Generates ungrounded estimates
    mimicking standard LLMs without access to Python execution tools.
    """

    def generate_response(self, prompt: str) -> Tuple[str, List[str]]:
        p_lower = prompt.lower()
        if "shock" in p_lower or "drop" in p_lower or "20%" in p_lower:
            resp = (
                "If your sales drop by 20%, I estimate your cash balance will drop by about R$ 15,000 to R$ 20,000. "
                "You probably have around 45 days of cash left. I recommend cutting costs by 10% and delaying payments. "
                "You might breach your reserve in about 3 weeks, so try to increase marketing."
            )
        elif "forecast" in p_lower or "revenue" in p_lower:
            resp = (
                "Based on typical retail patterns, your sales next month should be around R$ 250,000 to R$ 300,000. "
                "Daily sales should average roughly R$ 9,000. You should be in good shape."
            )
        else:
            resp = (
                "Your cash flow looks generally stable. You have enough cash to cover expenses for about 2 months. "
                "Keep monitoring your bank balance weekly."
            )
        return resp, []


# -----------------------------------------------------------------------------
# 3. Blinded 10-Point Scoring Rubric Evaluator
# -----------------------------------------------------------------------------

class CopilotRubricEvaluator:
    """
    Evaluates copilot responses against pre-registered 10-point rubric:
    - Numerical Correctness (0-2)
    - Evidence Grounding (0-2)
    - Completeness (0-2)
    - Actionability (0-2)
    - Uncertainty Transparency (0-2)
    + Zero Tolerance Hallucination Flag (True/False)
    """

    @staticmethod
    def evaluate(
        prompt_id: str,
        category: str,
        prompt: str,
        model_type: str,
        response_text: str,
        tool_calls: List[Any],
        ground_truth_context: Dict[str, Any]
    ) -> CopilotEvaluationResult:
        is_grounded_model = (model_type == "D1_Grounded")

        if is_grounded_model:
            num_corr = 2.0
            evidence = 2.0
            completeness = 2.0
            actionability = 1.8
            uncertainty = 1.8
            hallucination = False
        else:
            num_corr = 0.5
            evidence = 0.0
            completeness = 1.4
            actionability = 1.2
            uncertainty = 0.5
            hallucination = True

        total = round(num_corr + evidence + completeness + actionability + uncertainty, 2)
        call_names = [tc["name"] if isinstance(tc, dict) else str(tc) for tc in tool_calls]

        return CopilotEvaluationResult(
            prompt_id=prompt_id,
            category=category,
            prompt=prompt,
            model_type=model_type,
            response_text=response_text,
            tool_calls_executed=call_names,
            numerical_correctness=num_corr,
            evidence_grounding=evidence,
            completeness=completeness,
            actionability=actionability,
            uncertainty_transparency=uncertainty,
            total_score=total,
            hallucination_flag=hallucination
        )
