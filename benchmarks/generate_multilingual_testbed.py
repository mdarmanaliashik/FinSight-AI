"""
FinSight AI — 6,000 Multilingual Prompt Testbed Generator
=========================================================
Generates a 6,000-prompt parameterized benchmark suite across 6 major Asian languages:
- Bengali (বাংলা)      : 1,000 prompts
- Hindi (हिन्दी)       : 1,000 prompts
- Urdu (اردو)         : 1,000 prompts
- Chinese (中文)      : 1,000 prompts
- Japanese (日本語)   : 1,000 prompts
- English (Lingua)    : 1,000 prompts

Each test case contains:
- Unique test_id
- Language code
- Linguistic prompt with natural phrasing
- Target tool call schema & expected arguments
- Ground-truth financial parameters for deterministic Python validation
"""

import json
import itertools
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "benchmark_6000_multilingual_testbed.json"

LANGUAGES = {
    "bn": "Bengali (বাংলা)",
    "hi": "Hindi (हिन्दी)",
    "ur": "Urdu (اردو)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "en": "English (Reference)"
}

TEMPLATES = {
    "bn": [
        "যদি আগামী মাসে আমাদের বিক্রি {sales_shock_pct}% কমে এবং সাপ্লায়ার খরচ {cost_shock_pct}% বাড়ে, তবে রিজার্ভ অক্ষুণ্ণ রাখতে কত টাকা লোন নিতে হবে?",
        "মার্কেটপ্লেস যদি প্রাপ্য টাকা পরিশোধে {delay_days} দিন দেরি করে, তাহলে আমাদের ক্যাশ ব্যালেন্স কি রিজার্ভের নিচে নেমে যাবে?",
        "মন্দার কারণে যদি বিক্রি {sales_shock_pct}% কমে যায়, তাহলে ১৫ তারিখে কর্মচারীদের বেতন দেওয়ার মতো টাকা থাকবে কি?",
        "সাপ্লায়ার কাঁচামালের দাম {cost_shock_pct}% বাড়িয়ে দিলে এবং বিক্রি {sales_shock_pct}% কমলে পরবর্তী ৩০ দিনে আমাদের সর্বোচ্চ ঘাটতি কত হবে?"
    ],
    "hi": [
        "यदि अगले महीने हमारी बिक्री में {sales_shock_pct}% की गिरावट आती है और आपूर्तिकर्ता की लागत {cost_shock_pct}% बढ़ जाती है, तो रिज़र्व बनाए रखने के लिए कितने ऋण की आवश्यकता होगी?",
        "यदि मार्केटप्लेस से भुगतान मिलने में {delay_days} दिनों की देरी होती है, तो क्या हमारा कैश न्यूनतम रिज़र्व से नीचे चला जाएगा?",
        "यदि बिक्री {sales_shock_pct}% गिर जाती है, तो क्या हमारे पास कर्मचारियों के वेतन का भुगतान करने के लिए पर्याप्त नकदी होगी?",
        "आपूर्तिकर्ता लागत {cost_shock_pct}% बढ़ने और बिक्री {sales_shock_pct}% घटने पर अगले 30 दिनों में हमारा अधिकतम घाटा कितना होगा?"
    ],
    "ur": [
        "اگر اگلے ماہ ہماری فروخت میں {sales_shock_pct}% کی کمی واقع ہو اور سپلائر لاگت میں {cost_shock_pct}% کا اضافہ ہو تو ریزرو برقرار رکھنے کے لیے کتنا قرض درکار ہوگا؟",
        "اگر مارکیٹ پلیس کی ادائیگی میں {delay_days} دن کی تاخیر ہو تو کیا ہمارا کیش بیلنس کم از کم ریزرو سے نیچے گر جائے گا؟",
        "اگر فروخت میں {sales_shock_pct}% کی کمی ہو تو کیا ہمارے پاس ملازمین کی تنخواہیں ادا کرنے کے لیے نقد رقم موجود ہوگی؟",
        "اگر خام مال کی لاگت {cost_shock_pct}% بڑھ جائے اور فروخت {sales_shock_pct}% کم ہو تو اگلے 30 دنوں میں ہمارا کل مالی خسارہ کتنا ہوگا؟"
    ],
    "zh": [
        "如果下个月销售额下降{sales_shock_pct}%且供应商成本上升{cost_shock_pct}%，我们需要借入多少短期贷款以避免跌破最低储备金？",
        "如果电商平台的结算回款延迟{delay_days}天，我们的现金储备是否会跌破安全底线？",
        "在销售额下滑{sales_shock_pct}%的情况下，我们是否有足够的流动资金在15号按时发放员工工资？",
        "如果采购成本上涨{cost_shock_pct}%且销售下滑{sales_shock_pct}%，未来30天内的最大资金缺口是多少？"
    ],
    "ja": [
        "来月の売上が{sales_shock_pct}%減少し、仕入原価が{cost_shock_pct}%上昇した場合、最低準備金を維持するためにいくらの短期融资が必要ですか？",
        "プラットフォームからの売掛金回収が{delay_days}日間遅延した場合、手元資金は準備金基準を下回りますか？",
        "売上が{sales_shock_pct}%減少した場合、15日の給与支払いを履行する十分な現金余力はありますか？",
        "仕入コストが{cost_shock_pct}%上昇し売上が{sales_shock_pct}%減少した場合、今後30日間の最大流動性不足額はいくらになりますか？"
    ],
    "en": [
        "If our sales drop by {sales_shock_pct}% and supplier costs increase by {cost_shock_pct}%, how much credit line is required to protect our operating reserve?",
        "If marketplace receivables are delayed by {delay_days} business days, will our cash balance breach the reserve threshold?",
        "Under a {sales_shock_pct}% downturn in sales volume, will we have sufficient liquidity to cover monthly payroll?",
        "If supplier COGS inflates by {cost_shock_pct}% and sales fall by {sales_shock_pct}%, what is the maximum liquidity shortfall over the next 30 days?"
    ]
}


def generate_benchmark_suite(prompts_per_lang: int = 1000) -> list:
    """Generates balanced test cases across parameter grids and languages."""
    np.random.seed(42)
    testbed = []
    
    # Parameter variation grid
    sales_shocks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cost_shocks = [5, 10, 15, 20, 25, 30, 35]
    delays = [3, 5, 7, 10, 14, 21, 28]
    initial_cash_levels = [40000.0, 60000.0, 80000.0, 100000.0, 150000.0]
    reserve_floors = [30000.0, 50000.0, 70000.0]
    
    global_id = 1
    
    for lang_code, lang_name in LANGUAGES.items():
        templates = TEMPLATES[lang_code]
        count = 0
        
        while count < prompts_per_lang:
            template_idx = count % len(templates)
            template = templates[template_idx]
            
            s_shock = int(np.random.choice(sales_shocks))
            c_shock = int(np.random.choice(cost_shocks))
            delay = int(np.random.choice(delays))
            c_init = float(np.random.choice(initial_cash_levels))
            r_floor = float(np.random.choice(reserve_floors))
            
            # Format text prompt
            prompt_text = template.format(
                sales_shock_pct=s_shock,
                cost_shock_pct=c_shock,
                delay_days=delay
            )
            
            # Determine expected tool and target parameters
            if template_idx == 0:
                expected_tool = "simulate_scenario"
                expected_args = {"sales_shock": -round(s_shock / 100.0, 2), "cost_shock": round(c_shock / 100.0, 2), "settlement_delay": 0}
            elif template_idx == 1:
                expected_tool = "simulate_scenario"
                expected_args = {"sales_shock": 0.0, "cost_shock": 0.0, "settlement_delay": delay}
            elif template_idx == 2:
                expected_tool = "get_reserve_status"
                expected_args = {"downside_stress": True}
            else:
                expected_tool = "simulate_scenario"
                expected_args = {"sales_shock": -round(s_shock / 100.0, 2), "cost_shock": round(c_shock / 100.0, 2), "settlement_delay": 0}
            
            case = {
                "test_id": f"TEST_{global_id:05d}",
                "language": lang_code,
                "language_name": lang_name,
                "category": f"Category_{template_idx + 1}",
                "prompt": prompt_text,
                "expected_tool": expected_tool,
                "expected_args": expected_args,
                "context": {
                    "initial_cash": c_init,
                    "reserve_floor": r_floor
                }
            }
            testbed.append(case)
            global_id += 1
            count += 1

    return testbed


def save_testbed(testbed: list, output_path: Path = OUTPUT_FILE):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(testbed, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(testbed)} test cases saved to: {output_path}")


if __name__ == "__main__":
    suite = generate_benchmark_suite(prompts_per_lang=1000)
    save_testbed(suite)
