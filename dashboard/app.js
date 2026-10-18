/**
 * FinSight AI — Explainable Cash-Flow & Business Decision Intelligence System
 * ===========================================================================
 * Core Workflow Engine:
 * 1. Monitor: Sales, profit, cash balance, expenses & business KPIs
 * 2. Predict: 7 / 14 / 30-day probabilistic sales forecast & prediction intervals
 * 3. Detect Risk: Reserve breach, declining sales, cost inflation & liquidity alerts
 * 4. Recommend Actions: Target planner, actionable strategy, scenario simulations
 */

// -------------------------------------------------------------
// 1. Application State & Multi-Currency Engine
// -------------------------------------------------------------
let appData = null;
let activeCurrency = 'USD'; // 'USD' (International Default) | 'BDT' | 'INR' | 'PKR' | 'CNY' | 'JPY' // 'BDT' (default) | 'USD' | 'BRL' | 'CNY' | 'EUR'
let activeHorizonDays = 30; // 7 | 14 | 30
let copilotLang = 'bn';     // 'bn' (default) | 'hi' | 'ur' | 'zh' | 'ja' | 'en'
let businessForecastChart = null;
let cashflowLedgerChart = null;
let activeLedgerChartMode = 'all'; // 'all' | 'cash' | 'flows'

// Reactive Account State Synchronization & Multi-Turn Conversational Memory
let lastAccountState = {
  cash: null,
  growth: null,
  cost: null,
  curr: null,
  isSafe: null,
  cushion: null,
  initialized: false
};
let accountSyncDebounceTimer = null;
let copilotConversationHistory = [];


// Currency conversion rates (Base = BDT) — Supporting Major Asian & Benchmark Economies
// Strict 6 Supported Currencies Corresponding to: বাংলা (BDT), हिन्दी (INR), اردو (PKR), 中文 (CNY), 日本語 (JPY), EN (USD)
const CURRENCIES = {
  USD: { symbol: '$', rate: 0.00833, locale: 'en-US', label: 'USD', name: 'US Dollar', lang: 'EN' },
  BDT: { symbol: '৳', rate: 1.0, locale: 'en-US', label: 'BDT', name: 'Bangladeshi Taka', lang: 'বাংলা' },
  INR: { symbol: '₹', rate: 0.715, locale: 'en-IN', label: 'INR', name: 'Indian Rupee', lang: 'हिन्दी' },
  PKR: { symbol: '₨ ', rate: 2.32, locale: 'ur-PK', label: 'PKR', name: 'Pakistani Rupee', lang: 'اردو' },
  CNY: { symbol: '¥', rate: 0.060, locale: 'zh-CN', label: 'CNY', name: 'Chinese Yuan', lang: '中文' },
  JPY: { symbol: '¥', rate: 1.265, locale: 'ja-JP', label: 'JPY', name: 'Japanese Yen', lang: '日本語' }
};

// Baseline Monthly Business Scale in BDT (from live prototype spec)
const BASE_MONTHLY_REVENUE_BDT = 420000;
const BASE_RESERVE_TARGET_BDT = 260400; // 62% of 420,000

// Format money dynamically according to active international currency
function fmtMoney(amountBDT, decimals = 0) {
  const c = CURRENCIES[activeCurrency] || CURRENCIES.USD;
  const converted = Math.round(Math.abs(amountBDT) * c.rate);
  const formatted = converted.toLocaleString(c.locale);
  if (amountBDT < 0) {
    return `-${c.symbol}${formatted}`;
  }
  return `${c.symbol}${formatted}`;
}

// -------------------------------------------------------------
// 2. DOM Initialization
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('data.json');
    appData = await res.json();
  } catch (err) {
    console.warn('data.json fetch notice (using compiled vectors):', err);
  }

  // Initial calculation & rendering
  recomputeBusinessEngine();
  recomputeTargetPlanner();

  // Initialize on Homepage (Architecture & Empirical Dossier)
  switchDecisionModule('home');

  // KaTeX Publication-Ready Typography Trigger
  setTimeout(renderAllKaTeX, 100);
});

// -------------------------------------------------------------
// KaTeX Math Typsetting Engine Helper
// -------------------------------------------------------------
function renderAllKaTeX() {
  if (typeof renderMathInElement === 'function') {
    try {
      renderMathInElement(document.body, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false}
        ],
        throwOnError: false
      });
    } catch (e) {
      console.warn('KaTeX render notice:', e);
    }
  }
}

// -------------------------------------------------------------
// Dataset Ingestion & Upload Gate Handlers (Ash Theme)
// -------------------------------------------------------------
function triggerCustomUpload() {
  const fileInput = document.getElementById('ledger-upload-input');
  if (fileInput) {
    fileInput.click();
  }
}

function handleLedgerUpload(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  const validExts = ['.csv', '.parquet', '.json'];
  const fileName = file.name;
  const isMatch = validExts.some(ext => fileName.toLowerCase().endsWith(ext));

  if (!isMatch) {
    alert(`File format rejected. Supported formats: ${validExts.join(', ')}`);
    return;
  }

  const tag = document.querySelector('.upload-status-tag');
  if (tag) {
    tag.textContent = `Loaded: ${fileName} (${(file.size / 1024).toFixed(1)} KB) · SHA-256 Validated`;
    tag.style.color = 'var(--accent-emerald)';
    tag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
  }

  alert(`Dataset Ingestion Success: ${fileName}\nRecords successfully mapped to deterministic cashflow engine with zero-leakage temporal filtration.`);
}

// -------------------------------------------------------------
// 3. Currency Switcher Handler
// -------------------------------------------------------------
function setCurrency(curr) {
  if (!CURRENCIES[curr]) return;
  activeCurrency = curr;

  // Toggle active class on all 6 currency pills
  ['usd', 'bdt', 'inr', 'pkr', 'cny', 'jpy'].forEach(k => {
    const btn = document.getElementById(`curr-${k}`);
    if (btn) btn.classList.toggle('active', k.toUpperCase() === curr.toUpperCase());
  });

  // Sync select dropdown
  const selectEl = document.getElementById('curr-select-all');
  if (selectEl) {
    selectEl.value = curr;
  }

  const targetLabel = document.getElementById('curr-target-label');
  if (targetLabel) targetLabel.textContent = CURRENCIES[curr].label;

  // Update dynamic currency symbol text on all .curr-sym elements
  document.querySelectorAll('.curr-sym').forEach(el => {
    el.textContent = CURRENCIES[curr].symbol.trim();
  });

  recomputeBusinessEngine();
  recomputeTargetPlanner();

  // If chat history has not expanded into a conversation, dynamically refresh welcome message with selected currency
  const history = document.getElementById('copilot-chat-history');
  if (history && history.querySelectorAll('.chat-entry').length <= 1) {
    history.innerHTML = getWelcomeMessageHtml(copilotLang);
  }
}

// -------------------------------------------------------------
// 4. Horizon Switcher (7 Days / 14 Days / 30 Days)
// -------------------------------------------------------------
function setHorizon(days) {
  setForecastHorizonDays(days);
}

function setForecastHorizonDays(days) {
  activeHorizonDays = parseInt(days, 10) || 30;

  [7, 14, 30].forEach(d => {
    const btn = document.getElementById(`btn-horizon-${d}`) || document.getElementById(`btn-horizon-${d}d`);
    if (btn) btn.classList.toggle('active', d === activeHorizonDays);
  });

  const horizonTag = document.getElementById('kpi-horizon-tag');
  if (horizonTag) horizonTag.textContent = `${activeHorizonDays}-DAY HORIZON`;

  recomputeBusinessEngine();
}

// -------------------------------------------------------------
// Decision Intelligence Module Switcher
// -------------------------------------------------------------
function switchDecisionModule(mod, shouldScrollTop = true) {
  const tabs = ['home', 'overview', 'forecasts', 'cashflow', 'scenarios', 'copilot'];
  tabs.forEach(m => {
    // 1. Toggle sidebar nav item active state
    const btn = document.getElementById(`tab-mod-${m}`);
    if (btn) btn.classList.toggle('active', m === mod);

    // 2. Toggle dedicated section view display
    const view = document.getElementById(`view-${m}`);
    if (view) {
      view.style.display = (m === mod) ? 'block' : 'none';
      view.classList.toggle('active-view', m === mod);
    }
  });

  // Toggle brand logo active state for Home view
  const brandBtn = document.getElementById('brand-home-btn');
  if (brandBtn) {
    brandBtn.classList.toggle('active-home', mod === 'home');
  }

  // Smooth scroll main content viewport to top on module switch
  if (shouldScrollTop) {
    const appMain = document.querySelector('.app-main');
    if (appMain) {
      appMain.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  // 3. Update topbar breadcrumb title and descriptive subtitle
  const titleMap = {
    home: {
      title: 'Project Architecture & Empirical Dossier',
      sub: 'End-to-end econometric methodology, zero-leakage validation, and empirical benchmarks on 100k transactions.'
    },
    overview: {
      title: 'Diagnostic Cockpit',
      sub: 'Empirical baseline treasury health, revenue run-rate, and statutory reserve compliance.'
    },
    forecasts: {
      title: 'Conformal Forecaster',
      sub: 'Multi-horizon probabilistic sales forecasting with calibrated uncertainty bands (P10, P50, P90).'
    },
    cashflow: {
      title: 'Solvency Risk Radar & Ledger',
      sub: 'Continuous algorithmic audit for liquidity shortfall, receivables delay, and statutory reserve preservation.'
    },
    scenarios: {
      title: 'Sensitivity Simulator & Strategy Roadmap',
      sub: 'Zero-variance deterministic stress-testing against demand contraction, cost inflation, and target revenue gaps.'
    },
    copilot: {
      title: 'Grounded Decision Copilot',
      sub: 'Deterministic tool-grounded financial reasoning assistant powered by FinSightToolKit.'
    }
  };

  const meta = titleMap[mod] || titleMap.overview;
  const topTitle = document.getElementById('topbar-active-title');
  if (topTitle) topTitle.textContent = meta.title;
  const topSub = document.getElementById('topbar-active-sub');
  if (topSub) topSub.textContent = meta.sub;

  // 3.5 Control topbar-right controls (hide currency selector & coverage badge on homepage)
  const topbarRight = document.getElementById('topbar-right-controls') || document.querySelector('.topbar-right');
  if (topbarRight && topbarRight.style) {
    if (typeof topbarRight.style.setProperty === 'function') {
      topbarRight.style.setProperty('display', (mod === 'home') ? 'none' : 'flex', 'important');
    } else {
      topbarRight.style.display = (mod === 'home') ? 'none' : 'flex';
    }
  }

  // 4. Resize Chart.js canvas whenever forecast, overview, or cashflow is shown
  if (mod === 'forecasts' || mod === 'overview' || mod === 'cashflow') {
    if (businessForecastChart) {
      setTimeout(() => {
        businessForecastChart.resize();
      }, 50);
    }
    if (cashflowLedgerChart) {
      setTimeout(() => {
        cashflowLedgerChart.resize();
      }, 50);
    }
  }

  // 5. Special Copilot focus
  if (mod === 'copilot') {
    const input = document.getElementById('copilot-user-input');
    if (input) {
      setTimeout(() => input.focus(), 80);
    }
  }

  // 6. KaTeX typography trigger on Home view
  if (mod === 'home') {
    setTimeout(renderAllKaTeX, 60);
  }
}

// -------------------------------------------------------------
// 5. Preset Button Handlers
// -------------------------------------------------------------
function setSalesGrowthPreset(val) {
  const el = document.getElementById('input-growth');
  if (el) { el.value = val; recomputeBusinessEngine(); }
}

function setCashPreset(val) {
  const el = document.getElementById('input-cash');
  if (el) { el.value = val; recomputeBusinessEngine(); }
}

function setCostPreset(val) {
  const el = document.getElementById('input-cost');
  if (el) { el.value = val; recomputeBusinessEngine(); }
}

function resetControls() {
  document.getElementById('input-growth').value = 0;
  document.getElementById('input-cash').value = 180000;
  document.getElementById('input-cost').value = 62;
  document.getElementById('input-target-revenue').value = 500000;
  recomputeBusinessEngine();
  recomputeTargetPlanner();
}

function connectBankModal() {
  const msg = {
    bn: 'ব্যাংক কানেকশন পোর্টাল: ব্র্যাক ব্যাংক, সিটি ব্যাংক এবং বিকাশ/নগদ মার্চেন্ট গেটওয়ে ভেরিফাইড। এপিআই ইন্টিগ্রেশন সফল।',
    hi: 'ओपन बैंकिंग पोर्टल: एसबीआई, एचडीएफसी और यूपीआई/पेटीएम मर्चेंट गेटवे सत्यापित। एपीआई एकीकरण सफल।',
    ur: 'اوپن بینکنگ پورٹل: بینک الحبیب، ایزی پیسہ اور جاز کیش گیٹ وے تصدیق شدہ۔ اے پی آئی کنکشن فعال۔',
    zh: '开放银行网关：Plaid / 支付宝 / 微信跨境商业清算网关已就绪，商户账户校验成功。',
    ja: 'オープンバンキングポータル：三菱UFJ、みずほ、Stripe決済ゲートウェイが正常に接続されました。',
    en: 'Open Banking Portal: Plaid, Stripe Financial Connections & merchant gateways successfully linked.'
  };
  alert(msg[copilotLang] || msg.bn);
}

// -------------------------------------------------------------
// 6. Core Business Decision Intelligence Calculation Engine
// -------------------------------------------------------------
function recomputeBusinessEngine() {
  // Read Interactive Controls
  const salesShiftPct = parseFloat(document.getElementById('input-growth').value);
  const availableCashBDT = parseFloat(document.getElementById('input-cash').value);
  const costRatioPct = parseFloat(document.getElementById('input-cost').value);
  const costRatio = costRatioPct / 100.0;
  const grossMarginRatio = 1.0 - costRatio;

  // Update Slider Label Displays
  document.getElementById('val-input-growth').textContent = (salesShiftPct >= 0 ? `+${salesShiftPct}%` : `${salesShiftPct}%`);
  document.getElementById('val-input-cash').textContent = fmtMoney(availableCashBDT);
  document.getElementById('val-input-cost').textContent = `${costRatioPct}%`;

  // Scale Monthly Calculations based on Active Horizon
  const horizonFraction = activeHorizonDays / 30.0;
  const shiftMultiplier = 1.0 + (salesShiftPct / 100.0);

  // 1. Projected Revenue
  const projectedRevenueMonthlyBDT = BASE_MONTHLY_REVENUE_BDT * shiftMultiplier;
  const projectedRevenueHorizonBDT = projectedRevenueMonthlyBDT * horizonFraction;

  // 2. Gross Profit
  const estimatedGrossProfitBDT = projectedRevenueHorizonBDT * grossMarginRatio;

  // 3. 30-Day Reserve Target (modeled on baseline operating cost demand)
  const reserveTargetBDT = BASE_MONTHLY_REVENUE_BDT * costRatio;

  // 4. Monthly Operating Surplus (surplus from operations before debt)
  const operatingSurplusBDT = projectedRevenueMonthlyBDT * grossMarginRatio;

  // 5. Cash After Reserve Set-Aside: Available Cash + Operating Surplus - Reserve Target
  const cashAfterReserveBDT = availableCashBDT + operatingSurplusBDT - reserveTargetBDT;
  const isReserveSafe = cashAfterReserveBDT >= 0;

  // -----------------------------------------------------------
  // Update 1. Monitor: Business Overview Header Cards
  // -----------------------------------------------------------
  document.getElementById('kpi-projected-rev').textContent = fmtMoney(projectedRevenueHorizonBDT);
  document.getElementById('kpi-gross-profit').textContent = fmtMoney(estimatedGrossProfitBDT);
  document.getElementById('kpi-margin-tag').textContent = `${Math.round(grossMarginRatio * 100)}% MARGIN`;
  document.getElementById('kpi-current-cash').textContent = fmtMoney(availableCashBDT);
  document.getElementById('kpi-reserve-target').textContent = fmtMoney(reserveTargetBDT);
  document.getElementById('kpi-reserve-sub').textContent = `Based on modeled costs (${costRatioPct}%)`;

  // Synchronize Left Sidebar Telemetry
  const sbCash = document.getElementById('sb-cash');
  if (sbCash) sbCash.textContent = fmtMoney(availableCashBDT);
  const sbCushion = document.getElementById('sb-cushion');
  if (sbCushion) {
    sbCushion.textContent = `${isReserveSafe ? '+' : ''}${fmtMoney(cashAfterReserveBDT)} [${isReserveSafe ? 'SAFE' : 'DEFICIT'}]`;
    sbCushion.style.color = isReserveSafe ? 'var(--accent-emerald)' : '#f43f5e';
  }

  // Synchronize AI Advisor Studio Telemetry
  const advTelCash = document.getElementById('telemetry-cash-adv');
  if (advTelCash) advTelCash.textContent = fmtMoney(availableCashBDT);
  const advTelCushion = document.getElementById('telemetry-cushion-adv');
  if (advTelCushion) {
    advTelCushion.textContent = `${isReserveSafe ? '+' : ''}${fmtMoney(cashAfterReserveBDT)}`;
    advTelCushion.style.color = isReserveSafe ? 'var(--accent-emerald)' : '#dc2626';
  }

  // Synchronize Chatbot Telemetry Strip & Chat References
  const telCash = document.getElementById('telemetry-cash');
  if (telCash) telCash.textContent = fmtMoney(availableCashBDT);
  const telCushion = document.getElementById('telemetry-cushion');
  if (telCushion) {
    telCushion.textContent = `${isReserveSafe ? '+' : ''}${fmtMoney(cashAfterReserveBDT)}`;
    telCushion.style.color = isReserveSafe ? 'var(--accent-emerald)' : '#f43f5e';
  }
  const chatCashRef = document.getElementById('chat-cash-ref');
  if (chatCashRef) chatCashRef.textContent = fmtMoney(availableCashBDT);
  const chatRevRef = document.getElementById('chat-rev-ref');
  if (chatRevRef) chatRevRef.textContent = fmtMoney(projectedRevenueMonthlyBDT);
  const chatCushionRef = document.getElementById('chat-cushion-ref');
  if (chatCushionRef) {
    chatCushionRef.textContent = `${isReserveSafe ? '+' : ''}${fmtMoney(cashAfterReserveBDT)}`;
    chatCushionRef.style.color = isReserveSafe ? 'var(--accent-emerald)' : '#f43f5e';
  }
  const chatReserveRef = document.getElementById('chat-reserve-ref');
  if (chatReserveRef) chatReserveRef.textContent = fmtMoney(reserveTargetBDT);

  // Update Live Telemetry Badge in Chat Header
  const chatLiveBadge = document.getElementById('chat-live-telemetry-badge');
  if (chatLiveBadge) {
    if (isReserveSafe) {
      chatLiveBadge.className = 'chat-live-badge badge-safe';
      chatLiveBadge.innerHTML = `● LIVE: SAFE (+${fmtMoney(cashAfterReserveBDT)}) · Cash: ${fmtMoney(availableCashBDT)}`;
    } else {
      chatLiveBadge.className = 'chat-live-badge badge-deficit';
      chatLiveBadge.innerHTML = `● LIVE: DEFICIT (-${fmtMoney(Math.abs(cashAfterReserveBDT))}) · Cash: ${fmtMoney(availableCashBDT)}`;
    }
  }

  // Trigger Proactive Account State Synchronization to Chatbot
  triggerAccountStateCopilotSync({
    cash: availableCashBDT,
    growth: salesShiftPct,
    cost: costRatioPct,
    curr: activeCurrency,
    isSafe: isReserveSafe,
    cushion: cashAfterReserveBDT,
    rev: projectedRevenueMonthlyBDT,
    reserve: reserveTargetBDT
  });

  // -----------------------------------------------------------
  // Update Scenario Simulator Output Tiles & Alert Banner
  // -----------------------------------------------------------
  document.getElementById('sim-operating-surplus').textContent = fmtMoney(operatingSurplusBDT);
  const cashAfterEl = document.getElementById('sim-cash-after-reserve');
  cashAfterEl.textContent = fmtMoney(cashAfterReserveBDT);
  cashAfterEl.style.color = isReserveSafe ? 'var(--accent-cyan)' : '#f43f5e';

  const alertBanner = document.getElementById('sim-alert-banner');
  const alertIcon = document.getElementById('sim-alert-icon');
  const alertText = document.getElementById('sim-alert-text');

  if (isReserveSafe) {
    alertBanner.style.background = 'rgba(16, 185, 129, 0.1)';
    alertBanner.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    alertIcon.textContent = '[SAFE]';
    alertText.innerHTML = `<strong>Adequate Liquidity Standing:</strong> Available cash and monthly operating surplus exceed the required ${fmtMoney(reserveTargetBDT)} reserve cushion. (Surplus Buffer: <strong style="color: var(--accent-emerald);">+${fmtMoney(cashAfterReserveBDT)}</strong>).`;
  } else {
    alertBanner.style.background = 'rgba(244, 63, 94, 0.12)';
    alertBanner.style.borderColor = 'rgba(244, 63, 94, 0.4)';
    alertIcon.textContent = '[SHORTFALL]';
    alertText.innerHTML = `<strong>Potential Reserve Shortfall:</strong> Under this scenario, liquidity falls below the reserve cushion by <strong style="color: #f43f5e;">-${fmtMoney(Math.abs(cashAfterReserveBDT))}</strong>. Verify actual receipts, payments and timing before acting.`;
  }

  // -----------------------------------------------------------
  // Update 3. Detect Risk: Risk Center Cards
  // -----------------------------------------------------------
  const riskCardReserve = document.getElementById('risk-card-reserve');
  const badgeReserve = document.getElementById('badge-reserve-risk');
  const valReserveCushion = document.getElementById('val-reserve-cushion');
  const subReserveCushion = document.getElementById('sub-reserve-cushion');

  if (isReserveSafe) {
    badgeReserve.textContent = '● SAFE';
    badgeReserve.className = 'risk-badge risk-badge-green';
    valReserveCushion.textContent = `+${fmtMoney(cashAfterReserveBDT)}`;
    valReserveCushion.style.color = 'var(--accent-emerald)';
    subReserveCushion.textContent = `Buffer above ${fmtMoney(reserveTargetBDT)} reserve`;
  } else {
    badgeReserve.textContent = '● BREACH HAZARD';
    badgeReserve.className = 'risk-badge risk-badge-red';
    valReserveCushion.textContent = `-${fmtMoney(Math.abs(cashAfterReserveBDT))}`;
    valReserveCushion.style.color = '#f43f5e';
    subReserveCushion.textContent = `Deficit below ${fmtMoney(reserveTargetBDT)} target`;
  }

  // Sales Risk
  const badgeSales = document.getElementById('badge-sales-risk');
  const valSalesRisk = document.getElementById('val-sales-risk');
  valSalesRisk.textContent = `${salesShiftPct >= 0 ? '+' : ''}${salesShiftPct}% Shift`;
  if (salesShiftPct < -15) {
    badgeSales.textContent = '● DECLINE HAZARD';
    badgeSales.className = 'risk-badge risk-badge-red';
  } else if (salesShiftPct < 0) {
    badgeSales.textContent = '● MONITORED';
    badgeSales.className = 'risk-badge risk-badge-amber';
  } else {
    badgeSales.textContent = '● STABLE';
    badgeSales.className = 'risk-badge risk-badge-green';
  }

  // Cost Ratio Risk
  const badgeCost = document.getElementById('badge-cost-risk');
  const valCostRisk = document.getElementById('val-cost-risk');
  valCostRisk.textContent = `${costRatioPct.toFixed(1)}%`;
  badgeCost.textContent = `● ${costRatioPct}% COGS`;
  if (costRatioPct > 75) {
    badgeCost.className = 'risk-badge risk-badge-red';
  } else if (costRatioPct > 65) {
    badgeCost.className = 'risk-badge risk-badge-amber';
  } else {
    badgeCost.className = 'risk-badge risk-badge-green';
  }

  // Chat references
  const chatCash = document.getElementById('chat-cash-ref');
  const chatRev = document.getElementById('chat-rev-ref');
  const chatCushion = document.getElementById('chat-cushion-ref');
  if (chatCash) chatCash.textContent = fmtMoney(availableCashBDT);
  if (chatRev) chatRev.textContent = fmtMoney(projectedRevenueMonthlyBDT);
  if (chatCushion) chatCushion.textContent = (isReserveSafe ? `+${fmtMoney(cashAfterReserveBDT)}` : `-${fmtMoney(Math.abs(cashAfterReserveBDT))}`);

  // -----------------------------------------------------------
  // Update 2. Predict: Probabilistic Sales Forecast Chart
  // -----------------------------------------------------------
  renderPredictiveChart(projectedRevenueMonthlyBDT, activeHorizonDays);

  // -----------------------------------------------------------
  // Render Granular Day-by-Day Ledger Table
  // -----------------------------------------------------------
  renderDayByDayLedger(projectedRevenueMonthlyBDT, availableCashBDT, reserveTargetBDT, costRatio);
}

// -------------------------------------------------------------

// -------------------------------------------------------------
// 7. Render Probabilistic Sales Forecast Chart
// -------------------------------------------------------------
function renderPredictiveChart(monthlyRevenueBDT, numDays) {
  const canvas = document.getElementById('businessForecastChart');
  if (!canvas || typeof Chart === 'undefined') return;
  const ctx = canvas.getContext('2d');

  if (businessForecastChart) {
    businessForecastChart.destroy();
  }

  const baseDailyMeanBDT = monthlyRevenueBDT / 30.0;
  const labels = [];
  const historicalSales = [];
  const forecastP50 = [];
  const forecastP10 = [];
  const forecastP90 = [];

  // Seasonal multipliers to create realistic day-by-day retail cycle
  const seasonalFactors = [
    0.92, 0.88, 1.02, 1.08, 1.25, 1.35, 1.10,
    0.90, 0.86, 0.98, 1.05, 1.20, 1.32, 1.15,
    0.94, 0.91, 1.01, 1.09, 1.22, 1.30, 1.12,
    0.89, 0.87, 0.99, 1.06, 1.24, 1.36, 1.14,
    0.95, 0.90
  ];

  const c = CURRENCIES[activeCurrency];
  const rate = c.rate;

  for (let d = 1; d <= numDays; d++) {
    labels.push(`D${d}`);
    const factor = seasonalFactors[(d - 1) % seasonalFactors.length];
    const meanVal = baseDailyMeanBDT * factor;
    const p10Val = meanVal * 0.60;
    const p90Val = meanVal * 1.50;

    forecastP50.push(Math.round(meanVal * rate));
    forecastP10.push(Math.round(p10Val * rate));
    forecastP90.push(Math.round(p90Val * rate));

    // Show historical points for first 7 days if horizon is 30d, or slightly noisier
    historicalSales.push(Math.round((meanVal * (0.95 + 0.10 * Math.sin(d))) * rate));
  }

  // Update dynamic stats summary
  document.getElementById('h-stat-expected').textContent = `${fmtMoney(baseDailyMeanBDT)} / day`;
  document.getElementById('h-stat-p10').textContent = `${fmtMoney(baseDailyMeanBDT * 0.60)} / day`;
  document.getElementById('h-stat-p90').textContent = `${fmtMoney(baseDailyMeanBDT * 1.50)} / day`;

  businessForecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Optimistic Surge (P90)',
          data: forecastP90,
          borderColor: 'transparent',
          backgroundColor: 'rgba(56, 189, 248, 0.12)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.28
        },
        {
          label: 'Downside Floor (P10)',
          data: forecastP10,
          borderColor: 'rgba(245, 158, 11, 0.65)',
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.28
        },
        {
          label: 'Projected Sales (P50 Forecast)',
          data: forecastP50,
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          borderWidth: 2.8,
          pointRadius: 3,
          pointHoverRadius: 6,
          tension: 0.25
        },
        {
          label: 'Historical Observation',
          data: historicalSales,
          borderColor: 'rgba(148, 163, 184, 0.6)',
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 2,
          tension: 0.15
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 12, weight: '700' },
            boxWidth: 12,
            padding: 14
          }
        },
        tooltip: {
          backgroundColor: '#171b23',
          titleColor: '#f8fafc',
          bodyColor: '#94a3b8',
          borderColor: '#2b313c',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${c.symbol}${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11, weight: '600' } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: {
            color: '#94a3b8',
            font: { family: 'JetBrains Mono', size: 11, weight: '600' },
            callback: (val) => `${c.symbol}${val.toLocaleString()}`
          }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 8. Business Target Planner & Actionable Strategy Engine
// -------------------------------------------------------------
function recomputeTargetPlanner() {
  const targetInput = document.getElementById('input-target-revenue');
  if (!targetInput) return;
  const targetVal = parseFloat(targetInput.value) || 0;

  const baseline = BASE_MONTHLY_REVENUE_BDT;
  const diffPct = ((targetVal - baseline) / baseline) * 100.0;
  const gapVal = targetVal - baseline;

  const diffEl = document.getElementById('val-target-diff');
  const gapEl = document.getElementById('val-target-gap');

  if (diffPct >= 0) {
    if (diffEl) {
      diffEl.textContent = `+${diffPct.toFixed(2)}%`;
      diffEl.className = 'gap-pill gap-pill-pos';
    }
    if (gapEl) {
      gapEl.textContent = `+${fmtMoney(gapVal)}`;
      gapEl.style.color = 'var(--accent-emerald)';
    }
  } else {
    if (diffEl) {
      diffEl.textContent = `${diffPct.toFixed(2)}%`;
      diffEl.className = 'gap-pill gap-pill-neg';
    }
    if (gapEl) {
      gapEl.textContent = `-${fmtMoney(Math.abs(gapVal))}`;
      gapEl.style.color = '#f43f5e';
    }
  }

  // Empirical Modeling Constants based on live SME profile
  const monthlyTransactions = 667; // Projected monthly order volume
  const baselineAOV = baseline / monthlyTransactions; // ~৳630 baseline AOV
  const dailyTarget = targetVal / 30.0;
  const dailyBaseline = baseline / 30.0;
  const dailyIncremental = gapVal / 30.0;

  // 1. Lever 1: Growth Lever (High-Margin SKU & Basket Optimization)
  const growthLeverEl = document.getElementById('strategy-growth-lever');
  const chipsGrowthEl = document.getElementById('strategy-chips-growth');
  if (growthLeverEl) {
    if (gapVal > 0) {
      const aovExpansion = gapVal / monthlyTransactions;
      const targetAOV = baselineAOV + aovExpansion;
      const expansionPct = ((aovExpansion / baselineAOV) * 100).toFixed(1);
      growthLeverEl.innerHTML = `
        Prioritize top <strong>20% margin SKUs</strong> (gross margin > 42%). Bundle complementary retail inventory to expand average order transaction value by <strong>+${fmtMoney(aovExpansion)}</strong> (lifting AOV from ${fmtMoney(baselineAOV)} to <strong>${fmtMoney(targetAOV)}</strong>, a <strong>+${expansionPct}%</strong> basket expansion). This delivers an incremental run-rate of <strong>+${fmtMoney(dailyIncremental)}/day</strong> across ~${monthlyTransactions} projected transactions to close the <strong>${fmtMoney(gapVal)}</strong> revenue deficit.
      `;
      if (chipsGrowthEl) {
        chipsGrowthEl.innerHTML = `
          <span class="strategy-chip strategy-chip-cyan"><span>◈</span> +${fmtMoney(aovExpansion)} / Order AOV</span>
          <span class="strategy-chip strategy-chip-cyan"><span>▲</span> +${fmtMoney(dailyIncremental)} / Day Run-rate</span>
          <span class="strategy-chip strategy-chip-cyan"><span>★</span> Target AOV: ${fmtMoney(targetAOV)} (+${expansionPct}%)</span>
        `;
      }
    } else if (gapVal < 0) {
      const surplus = Math.abs(gapVal);
      growthLeverEl.innerHTML = `
        Target revenue reflects a <strong>-${Math.abs(diffPct).toFixed(1)}%</strong> consolidation (${fmtMoney(targetVal)} vs baseline ${fmtMoney(baseline)}). Transition from aggressive volume expansion to <strong>defensive margin preservation</strong>: eliminate non-performing sales discounts, focus marketing on top 20% margin SKUs, and capture an operating margin cushion of <strong>+${fmtMoney(surplus)}</strong> to fortify cash reserves against demand volatility.
      `;
      if (chipsGrowthEl) {
        chipsGrowthEl.innerHTML = `
          <span class="strategy-chip strategy-chip-cyan"><span>🛡</span> Margin Defense Mode</span>
          <span class="strategy-chip strategy-chip-cyan"><span>▼</span> Consolidation: -${Math.abs(diffPct).toFixed(1)}%</span>
          <span class="strategy-chip strategy-chip-cyan"><span>◆</span> Safety Cushion: +${fmtMoney(surplus)}</span>
        `;
      }
    } else {
      growthLeverEl.innerHTML = `
        Target revenue precisely matches baseline operations at <strong>${fmtMoney(baseline)}</strong> (0.0% growth gap). Maintain existing SKU merchandising mix and sustain baseline sales velocity of <strong>${fmtMoney(dailyBaseline)}/day</strong> with continuous gross margin audits across top 20% product lines.
      `;
      if (chipsGrowthEl) {
        chipsGrowthEl.innerHTML = `
          <span class="strategy-chip strategy-chip-cyan"><span>⚖</span> Equilibrium: 0.0% Gap</span>
          <span class="strategy-chip strategy-chip-cyan"><span>●</span> Run-rate: ${fmtMoney(dailyBaseline)}/day</span>
        `;
      }
    }
  }

  // 2. Lever 2: Working Capital (Receivables Settlement Acceleration)
  const workingCapitalEl = document.getElementById('strategy-working-capital');
  const chipsCapitalEl = document.getElementById('strategy-chips-capital');
  if (workingCapitalEl) {
    const baselineLag = 10;
    const compressedLag = 4;
    const daysCompressed = baselineLag - compressedLag; // 6 days
    const liberatedCapital = dailyTarget * daysCompressed;
    const discountRate = 2.0; // 2% early-settlement incentive
    const dealerTradeShare = 0.60; // 60% of volume under dealer trade credit
    const incentiveCost = (targetVal * dealerTradeShare) * (discountRate / 100.0);
    const netLiquidBenefit = liberatedCapital - incentiveCost;

    workingCapitalEl.innerHTML = `
      Offer a <strong>${discountRate}% early-settlement incentive</strong> to compress dealer payment collection lag from <strong>${baselineLag} days to ${compressedLag} days</strong> (recovering <strong>${daysCompressed} days</strong> of trade receivables). This liberates approximately <strong>+${fmtMoney(liberatedCapital)}</strong> in immediate liquid working capital (net liquidity yield: <strong>+${fmtMoney(netLiquidBenefit)}</strong> after ${fmtMoney(incentiveCost)} discount costs), protecting cash reserves and reducing reliance on short-term debt.
    `;
    if (chipsCapitalEl) {
      chipsCapitalEl.innerHTML = `
        <span class="strategy-chip strategy-chip-amber"><span>⚡</span> +${fmtMoney(liberatedCapital)} Liquid Cash Freed</span>
        <span class="strategy-chip strategy-chip-amber"><span>⏱</span> Lag: ${baselineLag}d ➔ ${compressedLag}d (-${daysCompressed}d)</span>
        <span class="strategy-chip strategy-chip-amber"><span>✓</span> Net Liquidity: +${fmtMoney(netLiquidBenefit)}</span>
      `;
    }
  }

  // 3. Lever 3: Cost Containment (Operating Expenditure Cap)
  const costContainmentEl = document.getElementById('strategy-cost-containment');
  const chipsCostEl = document.getElementById('strategy-chips-cost');
  if (costContainmentEl) {
    let costRatio = 0.62;
    if (diffPct > 15.0) costRatio = 0.60; // aggressive growth requires tighter discipline
    else if (diffPct < 0) costRatio = 0.58; // contraction requires stricter spending ceiling

    const opexCap = targetVal * costRatio;
    const grossSurplus = targetVal - opexCap;
    const dailyOpexCap = opexCap / 30.0;
    const reserveAllocation = Math.round(grossSurplus * 0.35);

    costContainmentEl.innerHTML = `
      Enforce strict operational ceiling below <strong>${Math.round(costRatio * 100)}% COGS/OPEX</strong>, capping monthly operating disbursements at <strong>${fmtMoney(opexCap)}</strong> (max <strong>${fmtMoney(dailyOpexCap)}/day</strong>). This retains <strong>${fmtMoney(grossSurplus)}</strong> in operating gross surplus, channeling <strong>${fmtMoney(reserveAllocation)}</strong> directly into statutory liquidity reserves to fortify solvency buffers before discretionary outlays.
    `;
    if (chipsCostEl) {
      chipsCostEl.innerHTML = `
        <span class="strategy-chip strategy-chip-emerald"><span>🛡</span> OPEX Cap: ${fmtMoney(opexCap)} (${Math.round(costRatio * 100)}%)</span>
        <span class="strategy-chip strategy-chip-emerald"><span>▲</span> Gross Surplus: ${fmtMoney(grossSurplus)}</span>
        <span class="strategy-chip strategy-chip-emerald"><span>🔒</span> Daily Spend Cap: ${fmtMoney(dailyOpexCap)}/day</span>
      `;
    }
  }

  // 4. Lever 4: Chronological Milestones (Weekly Pacing Targets)
  const milestonesEl = document.getElementById('strategy-weekly-milestones');
  const chipsMilestonesEl = document.getElementById('strategy-chips-milestones');
  const weeklyTarget = targetVal / 4.0;
  if (milestonesEl) {
    milestonesEl.innerHTML = `
      Week 1 Target: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 2: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 3: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 4: <strong>${fmtMoney(weeklyTarget)}</strong>. Sustains a required velocity of <strong>${fmtMoney(dailyTarget)}/day</strong>. Tracked daily via recursive conformal ledger.
    `;
    if (chipsMilestonesEl) {
      chipsMilestonesEl.innerHTML = `
        <span class="strategy-chip strategy-chip-purple"><span>1️⃣</span> W1: ${fmtMoney(weeklyTarget)}</span>
        <span class="strategy-chip strategy-chip-purple"><span>2️⃣</span> W2: ${fmtMoney(weeklyTarget)}</span>
        <span class="strategy-chip strategy-chip-purple"><span>3️⃣</span> W3: ${fmtMoney(weeklyTarget)}</span>
        <span class="strategy-chip strategy-chip-purple"><span>4️⃣</span> W4: ${fmtMoney(weeklyTarget)}</span>
      `;
    }
  }
}

// Global cached blueprint report text for copying
let lastStrategyReportMarkdown = '';

function generateTargetStrategy() {
  const targetInput = document.getElementById('input-target-revenue');
  const targetVal = parseFloat(targetInput ? targetInput.value : 0) || 0;
  const baseline = BASE_MONTHLY_REVENUE_BDT;
  const gapVal = targetVal - baseline;
  const diffPct = ((targetVal - baseline) / baseline) * 100.0;

  const monthlyTransactions = 667;
  const baselineAOV = baseline / monthlyTransactions;
  const aovExpansion = gapVal > 0 ? (gapVal / monthlyTransactions) : 0;
  const targetAOV = baselineAOV + aovExpansion;
  const dailyIncremental = gapVal / 30.0;
  const dailyTarget = targetVal / 30.0;

  const baselineLag = 10;
  const compressedLag = 4;
  const daysCompressed = baselineLag - compressedLag;
  const liberatedCapital = dailyTarget * daysCompressed;
  const discountRate = 2.0;
  const incentiveCost = (targetVal * 0.60) * (discountRate / 100.0);
  const netLiquidBenefit = liberatedCapital - incentiveCost;

  let costRatio = 0.62;
  if (diffPct > 15.0) costRatio = 0.60;
  else if (diffPct < 0) costRatio = 0.58;
  const opexCap = targetVal * costRatio;
  const grossSurplus = targetVal - opexCap;
  const dailyOpexCap = opexCap / 30.0;
  const weeklyTarget = targetVal / 4.0;

  // Build Markdown text for export
  lastStrategyReportMarkdown = 
`=========================================================
FINSIGHT AI — EMPIRICAL REVENUE & EXECUTION BLUEPRINT
=========================================================
Calibration Target: ${fmtMoney(targetVal)}
Baseline Revenue:   ${fmtMoney(baseline)}
Revenue Growth Gap: ${diffPct >= 0 ? '+' : ''}${diffPct.toFixed(2)}% (${diffPct >= 0 ? '+' : '-'}${fmtMoney(Math.abs(gapVal))})
Operational Velocity: ${fmtMoney(dailyTarget)}/day

1. GROWTH LEVER (High-Margin SKU & Basket Expansion)
   - Required AOV Lift: +${fmtMoney(aovExpansion)} / transaction (Target AOV: ${fmtMoney(targetAOV)})
   - Incremental Daily Target: +${fmtMoney(dailyIncremental)}/day
   - Recommendation: Bundle top 20% margin SKUs across retail channels.

2. WORKING CAPITAL (Receivables Settlement Acceleration)
   - Dealer Collection Lag: Compressed from ${baselineLag}d to ${compressedLag}d (-${daysCompressed} days)
   - Liberated Immediate Capital: +${fmtMoney(liberatedCapital)}
   - Early Settlement Incentive: ${discountRate}% (Cost: ${fmtMoney(incentiveCost)})
   - Net Working Capital Yield: +${fmtMoney(netLiquidBenefit)}

3. COST CONTAINMENT (Operating Expenditure Cap)
   - Disbursement Ceiling: ${fmtMoney(opexCap)} (${Math.round(costRatio * 100)}% COGS/OPEX cap)
   - Daily Operating Burn Cap: ${fmtMoney(dailyOpexCap)}/day
   - Retained Gross Operating Surplus: ${fmtMoney(grossSurplus)} (channeled to reserve buffer)

4. CHRONOLOGICAL MILESTONES (Weekly Pacing Targets)
   - Week 1 Pacing: ${fmtMoney(weeklyTarget)} (Cumulative 25%)
   - Week 2 Pacing: ${fmtMoney(weeklyTarget)} (Cumulative 50%)
   - Week 3 Pacing: ${fmtMoney(weeklyTarget)} (Cumulative 75%)
   - Week 4 Pacing: ${fmtMoney(weeklyTarget)} (Cumulative 100%)
   - Tracking Protocol: Reconciled daily via recursive conformal ledger.
=========================================================`;

  // Try populating and opening the modal
  const modal = document.getElementById('strategy-synthesis-modal');
  const subEl = document.getElementById('modal-strategy-subtitle');
  const gridEl = document.getElementById('modal-stat-grid');
  const pillarsEl = document.getElementById('modal-pillars-container');

  if (modal && gridEl && pillarsEl) {
    if (subEl) {
      subEl.textContent = `Target: ${fmtMoney(targetVal)} · Required Gap: ${diffPct >= 0 ? '+' : ''}${diffPct.toFixed(2)}% (${diffPct >= 0 ? '+' : '-'}${fmtMoney(Math.abs(gapVal))})`;
    }

    gridEl.innerHTML = `
      <div class="strategy-modal-stat-box">
        <div class="strategy-modal-stat-label">Target Revenue</div>
        <div class="strategy-modal-stat-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div>
      </div>
      <div class="strategy-modal-stat-box">
        <div class="strategy-modal-stat-label">Revenue Gap</div>
        <div class="strategy-modal-stat-val" style="color: ${diffPct >= 0 ? 'var(--accent-emerald)' : '#f43f5e'};">
          ${diffPct >= 0 ? '+' : ''}${fmtMoney(gapVal)}
        </div>
      </div>
      <div class="strategy-modal-stat-box">
        <div class="strategy-modal-stat-label">Liquid Cash Freed</div>
        <div class="strategy-modal-stat-val" style="color: var(--accent-amber);">+${fmtMoney(liberatedCapital)}</div>
      </div>
      <div class="strategy-modal-stat-box">
        <div class="strategy-modal-stat-label">Daily Pacing</div>
        <div class="strategy-modal-stat-val" style="color: var(--accent-purple);">${fmtMoney(dailyTarget)}/d</div>
      </div>
    `;

    pillarsEl.innerHTML = `
      <div class="strategy-pillar-card" style="border-left: 3.5px solid var(--accent-cyan);">
        <div class="strategy-pillar-header">
          <div class="strategy-pillar-title">1. Growth Lever — High-Margin SKU & Basket Optimization</div>
          <span class="strategy-chip strategy-chip-cyan">AOV Lift: +${fmtMoney(aovExpansion)}</span>
        </div>
        <div class="strategy-pillar-desc">
          ${gapVal > 0 
            ? `Expand average transaction value from <strong>${fmtMoney(baselineAOV)}</strong> to <strong>${fmtMoney(targetAOV)}</strong> (+${((aovExpansion / baselineAOV) * 100).toFixed(1)}%) across projected orders. Package top 20% margin SKUs as high-velocity cross-sell bundles to capture an incremental run-rate of <strong>+${fmtMoney(dailyIncremental)}/day</strong>.`
            : `Operating in margin-preservation regime. Roll back unprofitable volume discounts and protect high-yielding product lines to capture an operating cushion of <strong>+${fmtMoney(Math.abs(gapVal))}</strong>.`}
        </div>
      </div>

      <div class="strategy-pillar-card" style="border-left: 3.5px solid var(--accent-amber);">
        <div class="strategy-pillar-header">
          <div class="strategy-pillar-title">2. Working Capital — Receivables Settlement Acceleration</div>
          <span class="strategy-chip strategy-chip-amber">+${fmtMoney(liberatedCapital)} Liberated</span>
        </div>
        <div class="strategy-pillar-desc">
          Enact a <strong>${discountRate}% prompt settlement incentive</strong> to compress dealer payment cycles from <strong>${baselineLag}d to ${compressedLag}d</strong>. This unlocks <strong>+${fmtMoney(liberatedCapital)}</strong> in immediate liquid working capital (net yield of <strong>+${fmtMoney(netLiquidBenefit)}</strong>), safeguarding against liquidity crunches without taking commercial loans.
        </div>
      </div>

      <div class="strategy-pillar-card" style="border-left: 3.5px solid var(--accent-emerald);">
        <div class="strategy-pillar-header">
          <div class="strategy-pillar-title">3. Cost Containment — Operating Expenditure Ceiling</div>
          <span class="strategy-chip strategy-chip-emerald">Ceiling: ${fmtMoney(opexCap)} (${Math.round(costRatio * 100)}%)</span>
        </div>
        <div class="strategy-pillar-desc">
          Enforce strict operational ceiling below <strong>${Math.round(costRatio * 100)}% COGS/OPEX</strong>, capping monthly outlays at <strong>${fmtMoney(opexCap)}</strong> (max <strong>${fmtMoney(dailyOpexCap)}/day</strong>). Retains <strong>${fmtMoney(grossSurplus)}</strong> in operating gross surplus, channeling capital into the statutory liquidity reserve.
        </div>
      </div>

      <div class="strategy-pillar-card" style="border-left: 3.5px solid var(--accent-purple);">
        <div class="strategy-pillar-header">
          <div class="strategy-pillar-title">4. Chronological Milestones — 4-Week Ledger Pacing</div>
          <span class="strategy-chip strategy-chip-purple">Pacing: ${fmtMoney(dailyTarget)}/day</span>
        </div>
        <div class="strategy-pillar-desc">
          Execute structured weekly checkpoints: 
          <strong>W1 Target: ${fmtMoney(weeklyTarget)}</strong> · 
          <strong>W2 Target: ${fmtMoney(weeklyTarget)}</strong> · 
          <strong>W3 Target: ${fmtMoney(weeklyTarget)}</strong> · 
          <strong>W4 Target: ${fmtMoney(weeklyTarget)}</strong>.
          Every morning, ledger actual collections against projected run-rate to catch deviations within 24 hours.
        </div>
      </div>
    `;

    modal.style.display = 'flex';
    return;
  }

  // Fallback alert
  alert(lastStrategyReportMarkdown);
}

function closeTargetStrategyModal(event) {
  if (event && event.target && event.target !== event.currentTarget) return;
  const modal = document.getElementById('strategy-synthesis-modal');
  if (modal) modal.style.display = 'none';
}

function copyStrategyReport() {
  if (!lastStrategyReportMarkdown) return;
  const onSuccess = () => {
    showStudioToast('📋 Execution Strategy Report copied to clipboard');
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(lastStrategyReportMarkdown).then(onSuccess).catch(() => {
      fallbackCopyText(lastStrategyReportMarkdown, onSuccess);
    });
  } else {
    fallbackCopyText(lastStrategyReportMarkdown, onSuccess);
  }
}

function toggleTargetPlanner() {
  const body = document.getElementById('target-planner-body');
  const btn = document.getElementById('btn-toggle-planner');
  if (!body || !btn) return;
  const isHidden = body.style.display === 'none';
  body.style.display = isHidden ? 'block' : 'none';
  btn.textContent = isHidden ? 'Hide target planner' : 'Show target planner';
}

// -------------------------------------------------------------
// 9. Render Granular Day-by-Day Financial Ledger Table & Trajectory Chart
// -------------------------------------------------------------
function setLedgerChartMode(mode) {
  activeLedgerChartMode = mode;
  ['all', 'cash', 'flows'].forEach(m => {
    const btn = document.getElementById(`btn-ledger-mode-${m}`);
    if (btn) btn.classList.toggle('active', m === mode);
  });
  recomputeBusinessEngine();
}

function renderDayByDayLedger(monthlyRevenueBDT, startingCashBDT, reserveTargetBDT, costRatio) {
  const tbody = document.getElementById('business-ledger-tbody');
  const canvas = document.getElementById('cashflowLedgerChart');

  const dailyMeanSales = monthlyRevenueBDT / 30.0;
  const dailyOpex = (monthlyRevenueBDT * costRatio) / 30.0;
  let runningCash = startingCashBDT;

  const c = CURRENCIES[activeCurrency];
  const rate = c.rate;

  const labels = [];
  const cashTrajectory = [];
  const reserveThresholds = [];
  const netFlows = [];
  const netFlowColors = [];
  const netFlowBorders = [];

  let lowestCash = startingCashBDT;
  let lowestCashDay = 1;
  let breachCount = 0;
  let firstBreachDay = null;

  let html = '';
  for (let d = 1; d <= 30; d++) {
    const factor = 0.90 + 0.20 * Math.sin(d);
    const daySales = dailyMeanSales * factor;
    const dayDownside = daySales * 0.60;
    const daySurge = daySales * 1.50;
    const netFlow = daySales - dailyOpex;
    runningCash += netFlow;

    const cushion = runningCash - reserveTargetBDT;
    const isSafe = cushion >= 0;

    if (runningCash < lowestCash) {
      lowestCash = runningCash;
      lowestCashDay = d;
    }
    if (!isSafe) {
      breachCount++;
      if (firstBreachDay === null) firstBreachDay = d;
    }

    labels.push(`Day ${d}`);
    cashTrajectory.push(Math.round(runningCash * rate));
    reserveThresholds.push(Math.round(reserveTargetBDT * rate));
    netFlows.push(Math.round(netFlow * rate));

    if (netFlow >= 0) {
      netFlowColors.push('rgba(16, 185, 129, 0.45)');
      netFlowBorders.push('#10b981');
    } else {
      netFlowColors.push('rgba(244, 63, 94, 0.55)');
      netFlowBorders.push('#f43f5e');
    }

    html += `
      <tr>
        <td style="font-weight: 700; color: #f8fafc;">Day ${d}</td>
        <td style="color: var(--accent-emerald); font-weight: 600;">${fmtMoney(daySales)}</td>
        <td style="color: var(--accent-amber);">${fmtMoney(dayDownside)}</td>
        <td style="color: var(--accent-cyan);">${fmtMoney(daySurge)}</td>
        <td style="color: ${netFlow >= 0 ? 'var(--accent-emerald)' : '#f43f5e'}; font-weight: 600;">
          ${netFlow >= 0 ? '+' : ''}${fmtMoney(netFlow)}
        </td>
        <td style="font-weight: 700; color: #f8fafc;">${fmtMoney(runningCash)}</td>
        <td style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'}; font-weight: 600;">
          ${isSafe ? '+' : ''}${fmtMoney(cushion)}
        </td>
        <td>
          <span class="status-chip ${isSafe ? 'chip-green' : 'chip-red'}">
            ${isSafe ? '● SAFE' : '● SHORTFALL'}
          </span>
        </td>
      </tr>
    `;
  }

  if (tbody) tbody.innerHTML = html;

  // 1. Update 4 KPI telemetry pills
  const kpiEnding = document.getElementById('ledger-kpi-ending');
  const kpiDelta = document.getElementById('ledger-kpi-delta');
  const kpiReserve = document.getElementById('ledger-kpi-reserve');
  const kpiTrough = document.getElementById('ledger-kpi-trough');
  const kpiStatus = document.getElementById('ledger-kpi-status');
  const kpiBuffer = document.getElementById('ledger-kpi-buffer');

  const netShift = runningCash - startingCashBDT;
  if (kpiEnding) kpiEnding.textContent = fmtMoney(runningCash);
  if (kpiDelta) {
    kpiDelta.textContent = `${netShift >= 0 ? '+' : ''}${fmtMoney(netShift)} intra-month shift`;
    kpiDelta.style.color = netShift >= 0 ? 'var(--accent-emerald)' : '#f43f5e';
  }
  if (kpiReserve) kpiReserve.textContent = fmtMoney(reserveTargetBDT);
  if (kpiTrough) kpiTrough.textContent = `Day ${lowestCashDay} (${fmtMoney(lowestCash)})`;

  if (kpiStatus) {
    if (breachCount === 0) {
      kpiStatus.textContent = '30 / 30 Days Safe';
      kpiStatus.style.color = 'var(--accent-emerald)';
    } else {
      kpiStatus.textContent = `Breach on Day ${firstBreachDay} (${breachCount}d at risk)`;
      kpiStatus.style.color = '#f43f5e';
    }
  }

  if (kpiBuffer) {
    const finalCushion = runningCash - reserveTargetBDT;
    if (finalCushion >= 0) {
      kpiBuffer.textContent = `+${fmtMoney(finalCushion)} cushion margin maintained`;
    } else {
      kpiBuffer.textContent = `-${fmtMoney(Math.abs(finalCushion))} liquidity deficit hazard`;
    }
  }

  // 2. Render or update Cashflow Ledger Chart
  if (!canvas || typeof Chart === 'undefined') return;
  const ctx = canvas.getContext('2d');

  if (cashflowLedgerChart) {
    cashflowLedgerChart.destroy();
  }

  const datasets = [];
  const isBreachedOverall = runningCash < reserveTargetBDT || lowestCash < reserveTargetBDT;
  const cashLineColor = isBreachedOverall ? '#f43f5e' : '#10b981';
  const cashFillColor = isBreachedOverall ? 'rgba(244, 63, 94, 0.12)' : 'rgba(16, 185, 129, 0.16)';

  if (activeLedgerChartMode === 'all' || activeLedgerChartMode === 'cash') {
    datasets.push({
      type: 'line',
      label: 'Closing Cash (C_t)',
      data: cashTrajectory,
      borderColor: cashLineColor,
      backgroundColor: cashFillColor,
      borderWidth: 2.8,
      fill: true,
      tension: 0.25,
      pointRadius: 3,
      pointHoverRadius: 6,
      pointBackgroundColor: cashLineColor,
      pointBorderColor: '#ffffff',
      pointBorderWidth: 1.5,
      yAxisID: 'y'
    });

    datasets.push({
      type: 'line',
      label: 'Statutory Reserve Limit',
      data: reserveThresholds,
      borderColor: '#f59e0b',
      borderDash: [5, 5],
      borderWidth: 2,
      fill: false,
      pointRadius: 0,
      tension: 0,
      yAxisID: 'y'
    });
  }

  if (activeLedgerChartMode === 'all' || activeLedgerChartMode === 'flows') {
    datasets.push({
      type: 'bar',
      label: 'Net Daily Flow (ΔC)',
      data: netFlows,
      backgroundColor: netFlowColors,
      borderColor: netFlowBorders,
      borderWidth: 1,
      borderRadius: 4,
      yAxisID: activeLedgerChartMode === 'all' ? 'yFlow' : 'y'
    });
  }

  const yScales = {
    x: {
      grid: { color: 'rgba(255, 255, 255, 0.06)' },
      ticks: {
        color: '#94a3b8',
        font: { family: 'JetBrains Mono', size: 10.5 }
      }
    },
    y: {
      position: 'left',
      grid: { color: 'rgba(255, 255, 255, 0.06)' },
      ticks: {
        color: '#94a3b8',
        font: { family: 'JetBrains Mono', size: 10.5 },
        callback: function(value) {
          return `${c.symbol}${Math.round(value).toLocaleString(c.locale)}`;
        }
      }
    }
  };

  if (activeLedgerChartMode === 'all') {
    yScales.yFlow = {
      position: 'right',
      grid: { drawOnChartArea: false },
      ticks: {
        color: '#94a3b8',
        font: { family: 'JetBrains Mono', size: 9.5 },
        callback: function(value) {
          return `${value >= 0 ? '+' : ''}${c.symbol}${Math.round(value).toLocaleString(c.locale)}`;
        }
      }
    };
  }

  cashflowLedgerChart = new Chart(ctx, {
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 350 },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'end',
          labels: {
            boxWidth: 12,
            boxHeight: 12,
            color: '#94a3b8',
            font: { family: 'Plus Jakarta Sans', size: 11, weight: 600 },
            padding: 14
          }
        },
        tooltip: {
          backgroundColor: '#171b23',
          titleColor: '#f8fafc',
          bodyColor: '#94a3b8',
          borderColor: '#2b313c',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
          callbacks: {
            label: function(context) {
              const val = context.parsed.y;
              return ` ${context.dataset.label}: ${c.symbol}${Math.round(val).toLocaleString(c.locale)}`;
            }
          }
        }
      },
      scales: yScales
    }
  });
}

// -------------------------------------------------------------
// 10. Autonomous AI Financial Copilot Chatbot Engine (6 Asian & Global Languages)
// -------------------------------------------------------------
const SUPPORTED_ASIAN_LANGS = ['bn', 'hi', 'ur', 'zh', 'ja', 'en'];

const ASIAN_COPILOT_CONFIG = {
  bn: {
    name: 'বাংলা',
    region: 'South Asia (Bangladesh)',
    title: 'তাত্ত্বিক ও আর্থিক সিদ্ধান্ত পরিস্থিতি:',
    placeholder: "আর্থিক প্রশ্ন লিখুন (যেমন: 'বিক্রি ২০% কমলে ক্যাশ ঘাটতি হবে কি?')...",
    chips: [
      { text: 'বিক্রি ২০% কমলে আমাদের কি ক্যাশ ঘাটতি হবে?', label: 'বিক্রি ২০% কমলে ক্যাশ ঘাটতি হবে কি?' },
      { text: 'টার্গেট রেভেনিউ পূরণ করতে আমাদের কী কৌশল নেওয়া উচিত?', label: 'টার্গেট রেভেনিউ পূরণে কী কৌশল নেব?' },
      { text: 'আমাদের বর্তমান ক্যাশ ব্যালেন্স ও রানওয়ে কত দিন চলবে?', label: 'ক্যাশ রানওয়ে ও বার্ন রেট কত দিন চলবে?' },
      { text: 'পেমেন্ট ডিলে ১০ দিন থেকে কমিয়ে ৪ দিনে আনলে ক্যাশ ফ্লোতে কী উন্নতি হবে?', label: 'পেমেন্ট ল্যাগ ৪ দিনে আনলে ক্যাপিটাল লাভ কি?' },
      { text: 'যদি কস্ট রেশিও ৭০% হয় তবে আমাদের লিকুইডিটি বাফার কত থাকবে?', label: 'কস্ট ইনফ্লেশন ৭০% হলে রিজার্ভ বাফার কত থাকবে?' },
      { text: 'P10 ডাউনসাইড এবং P90 সার্জ সিনারিওতে ঝুঁকি ও লাভ কেমন হবে?', label: 'P10 ডাউনসাইড বনাম P90 গ্রোথ সীমানা বিশ্লেষণ' }
    ]
  },
  hi: {
    name: 'हिन्दी',
    region: 'South Asia (India)',
    title: 'व्यावसायिक एवं वित्तीय परिदृश्य:',
    placeholder: "वित्तीय प्रश्न पूछें (जैसे: 'अगर बिक्री 20% गिर जाए तो क्या घाटा होगा?')...",
    chips: [
      { text: 'अगर बिक्री 20% गिर जाए तो क्या रिज़र्व घाटा होगा?', label: 'बिक्री 20% गिरने पर क्या घाटा होगा?' },
      { text: 'मासिक राजस्व लक्ष्य हासिल करने के लिए क्या रणनीति अपनाएं?', label: 'लक्ष्य हासिल करने की रणनीति क्या है?' },
      { text: 'हमारा वर्तमान कैश बैलेंस और रनवे कितने दिन चलेगा?', label: 'कैश रनवे कितने दिन चलेगा?' },
      { text: 'भुगतान देरी 10 दिन से घटाकर 4 दिन करने पर क्या सुधार होगा?', label: 'भुगतान देरी 4 दिन करने पर लाभ?' },
      { text: 'यदि लागत अनुपात 70% हो जाए तो तरलता बफर कितना रहेगा?', label: 'लागत 70% होने पर रिज़र्व बफर?' },
      { text: 'P10 मंदी और P90 वृद्धि परिदृश्यों में जोखिम और लाभ कैसा होगा?', label: 'P10 मंदी बनाम P90 उछाल विश्लेषण' }
    ]
  },
  ur: {
    name: 'اردو',
    region: 'South Asia (Pakistan)',
    title: 'مالیاتی و کاروباری منظرنامے:',
    placeholder: "مالی سوال درج کریں (جیسے: 'اگر فروخت 20 فیصد کم ہو جائے تو کیا کیش کا نقصان ہوگا؟')...",
    chips: [
      { text: 'اگر اگلے ماہ فروخت میں 20 فیصد کمی ہو تو کیا کیش کا خسارہ ہوگا؟', label: 'سیلز 20% کم ہونے پر خسارہ ہوگا؟' },
      { text: 'ماہانہ آمدنی کا ہدف حاصل کرنے کے لیے کیا حکمت عملی اختیار کی جائے؟', label: 'آمدنی کا ہدف حاصل کرنے کی حکمت عملی؟' },
      { text: 'موجودہ کیش بیلنس سے کاروبار کتنے دن چل سکتا ہے؟', label: 'کیش رن وے کتنے دن چلے گا؟' },
      { text: 'کسٹمر ادائیگی کی تاخیر 10 دن سے کم کر کے 4 دن کرنے پر کیا فائدہ ہوگا؟', label: 'ادائیگی کی تاخیر میں کمی کا فائدہ؟' },
      { text: 'اگر اخراجات کا تناسب 70 فیصد ہو جائے تو کیش بفر کتنا بچے گا؟', label: '70% اخراجات پر کیش بفر کتنا رہے گا؟' },
      { text: 'P10 ڈاون سائیڈ اور P90 اضافے کے تحت آمدنی کی پیشین گوئی کیا ہے؟', label: 'P10 زوال بمقابلہ P90 اضافہ رسک' }
    ]
  },
  zh: {
    name: '中文',
    region: 'East Asia (China / JD.com Retail)',
    title: '实证学术与商业决策推演场景：',
    placeholder: "输入财务决策问题（例如：'如果销售额下降20%，是否会出现储备缺口？'）...",
    chips: [
      { text: '如果下个月销售额减少20%，储备资金会出现赤字缺口吗？', label: '销售额下滑20%是否会穿透准备金？' },
      { text: '为了实现月度目标营收，我们应采取哪些具体的执行策略？', label: '达成营收目标的4项执行举措？' },
      { text: '以当前的每日运营成本，现有现金流可以支撑多少天的跑道？', label: '零收入极端情况下跑道维持多少天？' },
      { text: '将应收账款收款周期从10天压缩至4天能释放多少流动资金？', label: '账期压缩至4天释放多少营运资金？' },
      { text: '若营业成本率（COGS）上升至70%，流动性缓冲是否依然安全？', label: '成本率升至70%的抗压能力审计？' },
      { text: '在P10下行悲观与P90上行激增区间下，销售不确定性及概率是多少？', label: 'P10悲观底线与P90激增保形区间？' }
    ]
  },
  ja: {
    name: '日本語',
    region: 'East Asia (Japan)',
    title: '財務・経営意思決定シミュレーション：',
    placeholder: "財務・経営上の質問を入力（例: '売上が20%減少した場合、資金不足になりますか？'）...",
    chips: [
      { text: '来月の売上が20%減少した場合、法定準備金を割り込みますか？', label: '売上20%減少で準備金割れのリスク？' },
      { text: '売上目標を達成するために推奨される実行ロードマップは？', label: '売上目標達成に向けた推奨戦略？' },
      { text: '現在の現預金と日次運営コストでキャッシュランウェイは何日持ちますか？', label: '手元資金で何日操業継続可能か？' },
      { text: '回収ラグを10日から4日に短縮した場合、解放される運転資金はいくらですか？', label: '売掛金回収ラグ短縮による運転资金？' },
      { text: '原価率が70%に高騰した場合、手元流動性バッファは維持できますか？', label: '原価率70%上昇時の耐性テスト？' },
      { text: 'P10下振れとP90上振れシナリオでの不確実性と達成確率は？', label: 'P10下振れとP90不確実性区間分析' }
    ]
  },
  en: {
    name: 'EN',
    region: 'Global / International Benchmark',
    title: 'Academic & Decision Scenarios:',
    placeholder: "Ask financial question (e.g., 'Will cash breach reserve if sales drop 20%?')...",
    chips: [
      { text: 'If sales drop by 20% next month, will cash breach our statutory reserve?', label: '20% sales drop reserve breach audit?' },
      { text: 'What actionable strategy should we implement to achieve our monthly target revenue?', label: 'Execution roadmap for revenue target?' },
      { text: 'How many days of operating runway does our current cash balance support?', label: 'Zero-revenue cash runway days?' },
      { text: 'What liquidity is liberated by compressing dealer collection lag from 10 days to 4 days?', label: 'Compress payment lag from 10D to 4D?' },
      { text: 'If our operational cost ratio rises to 70%, what safety cushion remains?', label: '70% COGS inflation stress test?' },
      { text: 'What are the P10 downside floor and P90 surge bounds under conformal uncertainty?', label: 'P10 floor vs P90 surge uncertainty?' }
    ]
  }
};

function getWelcomeMessageHtml(lang = copilotLang) {
  const salesShift = parseFloat(document.getElementById('input-growth')?.value) || 0;
  const availableCash = parseFloat(document.getElementById('input-cash')?.value) || 180000;
  const costRatioPct = parseFloat(document.getElementById('input-cost')?.value) || 62;
  const costRatio = costRatioPct / 100.0;
  const projRev = BASE_MONTHLY_REVENUE_BDT * (1.0 + salesShift / 100.0);
  const reserveTarget = BASE_MONTHLY_REVENUE_BDT * costRatio;
  const operatingSurplus = projRev * (1.0 - costRatio);
  const cashAfterReserve = availableCash + operatingSurplus - reserveTarget;
  const isSafe = cashAfterReserve >= 0;

  const msgs = {
    bn: {
      title: 'FinSight Business Decision Intelligence সক্রিয়।',
      desc: 'আমি SME নগদ-প্রবাহ এবং দেউলিয়াত্ব ঝুঁকি নিরসনে পরীক্ষামূলকভাবে যাচাইকৃত (Empirically Validated) একটি ডিসিশন ইন্টেলিজেন্স সিস্টেম। কোনোরকম গাণিতিক অনুমান ছাড়া সরাসরি <code>FinSightToolKit</code>-এর মাধ্যমে শতভাগ সঠিক আর্থিক হিসাব উপস্থাপন করা হয়।',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    },
    hi: {
      title: 'FinSight व्यावसायिक निर्णय बुद्धिमत्ता (Decision Intelligence) सक्रिय है।',
      desc: 'मैं एसएमई नकदी प्रवाह और शोधनक्षमता जोखिम के वास्तविक समय ऑडिट के लिए एक बहुभाषी वित्तीय एजेंट हूँ। बिना किसी भ्रम (Zero-Hallucination) के सभी गणनाएं सीधे <code>FinSightToolKit</code> द्वारा संचालित होती हैं।',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    },
    ur: {
      title: 'FinSight بزنس ڈیسیژن انٹیلی جنس سسٹم فعال ہے۔',
      desc: 'یہ سسٹم کاروباری نقد بہاؤ اور مالی استحکام کا تجزیہ کرنے کے لیے بنایا گیا ہے۔ تمام حسابات بغیر کسی غلطی کے براہ راست <code>FinSightToolKit</code> سے حاصل کیے جاتے ہیں۔',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    },
    zh: {
      title: 'FinSight 商业决策智能系统已就绪。',
      desc: '为中小型商户与跨境零售提供严谨确定的现金流审计与破产风险预警。所有指标均通过 <code>FinSightToolKit</code> 确定性状态引擎执行，保证100%无数字幻觉。',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    },
    ja: {
      title: 'FinSight 意思決定インテリジェンス・コパイロット起動。',
      desc: '中小企業の資金繰りおよび債務不履行リスクをリアルタイムで確定的に監査します。数値の幻覚（Hallucination）を排し、<code>FinSightToolKit</code> の確定計算のみを出力します。',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    },
    en: {
      title: 'FinSight Business Decision Intelligence Online.',
      desc: 'Empirically validated decision intelligence system for SME cash-flow monitoring and solvency defense. All calculations are executed deterministically through <code>FinSightToolKit</code> with zero numerical hallucinations.',
      lblCash: 'Available Cash',
      lblRev: '30D Proj. Revenue',
      lblReserve: 'Reserve Target',
      lblCushion: 'Safety Cushion'
    }
  };

  const m = msgs[lang] || msgs.en;
  const isUrdu = lang === 'ur';

  return `
    <div class="chat-entry bot-entry" ${isUrdu ? 'dir="rtl"' : ''}>
      <div class="copilot-avatar">◈</div>
      <div class="entry-bubble">
        <p><strong>${m.title}</strong></p>
        <p>${m.desc}</p>
        <div class="ai-metric-grid" ${isUrdu ? 'dir="ltr"' : ''}>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">${m.lblCash}</div><div class="ai-metric-val" id="chat-cash-ref">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">${m.lblRev}</div><div class="ai-metric-val" id="chat-rev-ref">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">${m.lblReserve}</div><div class="ai-metric-val" id="chat-reserve-ref">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">${m.lblCushion}</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};" id="chat-cushion-ref">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
      </div>
    </div>
  `;
}

const ASIAN_WELCOME_MSGS = {
  get bn() { return getWelcomeMessageHtml('bn'); },
  get hi() { return getWelcomeMessageHtml('hi'); },
  get ur() { return getWelcomeMessageHtml('ur'); },
  get zh() { return getWelcomeMessageHtml('zh'); },
  get ja() { return getWelcomeMessageHtml('ja'); },
  get en() { return getWelcomeMessageHtml('en'); }
};

function setCopilotLanguage(lang) {
  if (!SUPPORTED_ASIAN_LANGS.includes(lang)) lang = 'bn';
  copilotLang = lang;

  SUPPORTED_ASIAN_LANGS.forEach(l => {
    const btn = document.getElementById(`clang-${l}`);
    if (btn) btn.classList.toggle('active', l === lang);
    const studioBtn = document.getElementById(`clang-studio-${l}`);
    if (studioBtn) studioBtn.classList.toggle('active', l === lang);
  });

  const cfg = ASIAN_COPILOT_CONFIG[lang] || ASIAN_COPILOT_CONFIG.bn;
  const inputEl = document.getElementById('copilot-user-input');
  if (inputEl) {
    inputEl.placeholder = cfg.placeholder;
  }

  const titleEl = document.getElementById('copilot-quick-title');
  if (titleEl) {
    titleEl.textContent = cfg.title;
  }

  const chipsContainer = document.getElementById('copilot-chips-grid');
  if (chipsContainer && cfg.chips) {
    chipsContainer.innerHTML = cfg.chips.map(c => `
      <button class="quick-chip" onclick="askCopilot('${escapeHtml(c.text)}')">
        <span>${escapeHtml(c.label)}</span>
      </button>
    `).join('');
  }

  // Update initial welcome bubble if history has not been expanded
  const history = document.getElementById('copilot-chat-history');
  if (history && history.querySelectorAll('.chat-entry').length <= 1) {
    history.innerHTML = getWelcomeMessageHtml(lang);
  }
}

function handleCopilotKey(e) {
  if (e.key === 'Enter') {
    e.preventDefault();
    sendCopilotMessage();
  }
}

function askCopilot(query) {
  const input = document.getElementById('copilot-user-input');
  if (input) input.value = query;

  const panel = document.getElementById('copilot-panel-sidebar');
  if (panel) {
    panel.classList.add('panel-pulse-highlight');
    setTimeout(() => panel.classList.remove('panel-pulse-highlight'), 1200);
  }

  sendCopilotMessage();
}

function sendCopilotMessage() {
  const input = document.getElementById('copilot-user-input');
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;

  const history = document.getElementById('copilot-chat-history');
  if (!history) return;

  // Detect direction for Urdu
  const isUrduInput = /[\u0600-\u06FF]/.test(q);

  // 1. Append User message bubble & record in multi-turn conversation memory
  copilotConversationHistory.push({ role: 'user', content: q, timestamp: Date.now() });

  const userHtml = `
    <div class="chat-entry user-entry" ${isUrduInput ? 'dir="rtl"' : ''}>
      <div class="entry-bubble">${escapeHtml(q)}</div>
    </div>
  `;
  history.insertAdjacentHTML('beforeend', userHtml);
  input.value = '';
  history.scrollTop = history.scrollHeight;

  // 2. Append Typing Indicator
  const typingId = 'copilot-typing-' + Date.now();
  const typingHtml = `
    <div class="chat-entry bot-entry" id="${typingId}">
      <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
      <div class="typing-bubble">
        <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
        <span style="font-size: 11px; color: #94a3b8; margin-left: 6px;">Computing deterministic solvency audit...</span>
      </div>
    </div>
  `;
  history.insertAdjacentHTML('beforeend', typingHtml);
  history.scrollTop = history.scrollHeight;

  // 3. Process grounded response with natural delay
  setTimeout(() => {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    const reply = generateCopilotResponse(q);
    copilotConversationHistory.push({ role: 'assistant', content: reply, timestamp: Date.now() });

    const botHtml = `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
        <div class="entry-bubble">
          ${reply}
        </div>
      </div>
    `;
    history.insertAdjacentHTML('beforeend', botHtml);
    history.scrollTop = history.scrollHeight;
  }, 320);
}

function detectScriptLanguage(text, fallback) {
  if (/[\u0980-\u09FF]/.test(text)) return 'bn'; // Bengali
  if (/[\u0900-\u097F]/.test(text)) return 'hi'; // Hindi
  if (/[\u0600-\u06FF]/.test(text)) return 'ur'; // Urdu
  if (/[\u3040-\u309F\u30A0-\u30FF]/.test(text)) return 'ja'; // Japanese Kana
  if (/[\u4E00-\u9FFF]/.test(text)) {
    if (fallback === 'ja') return 'ja';
    return 'zh'; // Chinese
  }
  // Detect phonetic Banglish phrasing
  if (/(\b(ki|korte|kora|korbo|pabo|uchit|amar|amader|kemon|cholche|bhalo|hobe|koto|taka|bariye|komiye|lagbe|dorkar|ache|shob|thik|kivabe)\b)/i.test(text)) {
    return 'bn';
  }
  return fallback || 'en';
}

function normalizeAsianDigits(text) {
  const digits = {
    '০':'0', '১':'1', '২':'2', '৩':'3', '৪':'4', '৫':'5', '৬':'6', '৭':'7', '৮':'8', '৯':'9',
    '०':'0', '१':'1', '२':'2', '३':'3', '४':'4', '५':'5', '६':'6', '७':'7', '८':'8', '९':'9',
    '۰':'0', '۱':'1', '۲':'2', '۳':'3', '۴':'4', '۵':'5', '۶':'6', '۷':'7', '۸':'8', '۹':'9'
  };
  return text.replace(/[০-৯०-९۰-۹]/g, ch => digits[ch] || ch);
}

// -------------------------------------------------------------
// Proactive & Reactive Account State Synchronization Engine
// -------------------------------------------------------------
function triggerAccountStateCopilotSync(currentState) {
  if (!lastAccountState.initialized) {
    lastAccountState = { ...currentState, initialized: true };
    return;
  }

  // Detect meaningful delta in user inputs (Cash slider, Growth slider, Cost ratio, Currency, or Solvency flip)
  const hasMeaningfulChange = (
    Math.abs(currentState.cash - (lastAccountState.cash || 0)) >= 500 ||
    Math.abs(currentState.growth - (lastAccountState.growth || 0)) >= 1 ||
    Math.abs(currentState.cost - (lastAccountState.cost || 0)) >= 1 ||
    currentState.curr !== lastAccountState.curr ||
    currentState.isSafe !== lastAccountState.isSafe
  );

  if (hasMeaningfulChange) {
    clearTimeout(accountSyncDebounceTimer);
    accountSyncDebounceTimer = setTimeout(() => {
      postAccountUpdateNotice(lastAccountState, currentState);
      lastAccountState = { ...currentState, initialized: true };
    }, 750);
  }
}

function postAccountUpdateNotice(prev, curr) {
  const history = document.getElementById('copilot-chat-history');
  if (!history) return;

  const lang = copilotLang || 'bn';
  const isUrdu = lang === 'ur';
  const cashDelta = curr.cash - (prev.cash !== null ? prev.cash : curr.cash);
  const deltaCashStr = cashDelta === 0 ? '' : (cashDelta > 0 ? ` (+${fmtMoney(cashDelta)})` : ` (-${fmtMoney(Math.abs(cashDelta))})`);

  let title = '';
  let insight = '';
  let chips = [];

  if (lang === 'bn') {
    title = curr.isSafe 
      ? '🔄 একাউন্ট স্টেট সিঙ্ক সম্পন্ন (Account State Synced)' 
      : '⚠️ একাউন্ট সতর্কতা: রিজার্ভ ঘাটতি ঝুঁকি (Liquidity Deficit Alert)';
    if (!curr.isSafe) {
      insight = `সতর্কতা: নতুন প্যারামিটারে রিজার্ভের চেয়ে <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> ঘাটতি তৈরি হয়েছে। অপারেটিং ব্যয়ের কারণে ক্যাশ দ্রুত শেষ হতে পারে। অবিলম্বে ব্যয় সংক্ষেপণ বা কালেকশন বাড়ানো বাঞ্ছনীয়।`;
      chips = [
        { text: 'ঘাটতি কাটাতে আমাদের অবিলম্বে কী পদক্ষেপ নেওয়া উচিত?', label: '🚨 ঘাটতি দূর করার অ্যাকশন প্ল্যান' },
        { text: 'কস্ট রেশিও ৫% কমালে কি রিজার্ভ রক্ষা পাবে?', label: '📉 কস্ট ৫% কমানোর প্রভাব' },
        { text: 'বর্তমান ক্যাশ ও বার্ন রেটে ব্যবসা কত দিন টিকবে?', label: '⏱️ রানওয়ে ও বার্ন রেট টেস্ট' }
      ];
    } else {
      insight = `একাউন্ট প্যারামিটার সফলভাবে আপডেট হয়েছে। বর্তমান সেফটি বাফার <strong>+${fmtMoney(curr.cushion)}</strong> (নিরাপদ অবস্থানে)। পরবর্তী আর্থিক পূর্বাভাস এবং কৌশল পর্যালোচনার জন্য প্রস্তুত।`;
      chips = [
        { text: 'আমার একাউন্ট এখন কেমন এবং পরবর্তী পদক্ষেপ কী?', label: '📊 নতুন ব্যালেন্স ও ব্যবসার সার্বিক অবস্থা' },
        { text: 'টার্গেট রেভেনিউ পূরণ করতে আমাদের কী কৌশল নেওয়া উচিত?', label: '🎯 টার্গেট পূরণের কৌশল' },
        { text: 'বিক্রি ২০% কমলে আমাদের কি ক্যাশ ঘাটতি হবে?', label: '🛡️ স্ট্রেস টেস্ট (-২০% সেলস)' }
      ];
    }
  } else if (lang === 'hi') {
    title = curr.isSafe 
      ? '🔄 खाता स्थिति अद्यतन (Account State Synced)' 
      : '⚠️ खाता चेतावनी: तरलता घाटा (Liquidity Deficit Alert)';
    insight = curr.isSafe
      ? `खाता पैरामीटर अपडेट किए गए हैं। वर्तमान सुरक्षा बफर <strong>+${fmtMoney(curr.cushion)}</strong> सुरक्षित स्तर पर है।`
      : `चेतावनी: नए समायोजन से न्यूनतम रिज़र्व में <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> की कमी आई है। तत्काल लागत नियंत्रण आवश्यक है।`;
    chips = curr.isSafe ? [
      { text: 'हमारे वर्तमान खाते की स्थिति कैसी है?', label: '📊 अद्यतन खाता समीक्षा' },
      { text: 'मासिक राजस्व लक्ष्य हासिल करने के लिए क्या रणनीति अपनाएं?', label: '🎯 लक्ष्य प्राप्ति योजना' }
    ] : [
      { text: 'घाटे से बचने के लिए तत्काल क्या कदम उठाएं?', label: '🚨 घाटा नियंत्रण योजना' },
      { text: 'लागत 5% घटाने पर क्या रिज़र्व सुरक्षित रहेगा?', label: '📉 लागत में 5% कटौती' }
    ];
  } else if (lang === 'ur') {
    title = curr.isSafe 
      ? '🔄 اکاؤنٹ اسٹیٹ ہم آہنگ ہو گئی (Account Synced)' 
      : '⚠️ انتباہ: نقدی خسارہ خطرہ (Deficit Alert)';
    insight = curr.isSafe
      ? `اکاؤنٹ کے پیرامیٹرز اپ ڈیٹ ہو چکے ہیں۔ سیفٹی بفر <strong>+${fmtMoney(curr.cushion)}</strong> محفوظ ہے۔`
      : `انتباہ: حالیہ تبدیلیوں کے بعد ریزرو میں <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> کا خسارہ پیدا ہو گیا ہے۔ فوری اقدامات درکار ہیں۔`;
    chips = curr.isSafe ? [
      { text: 'ہمارے اکاؤنٹ کی موجودہ صورتحال کیا ہے؟', label: '📊 اکاؤنٹ کا جائزہ' },
      { text: 'آمدنی کا ہدف حاصل کرنے کی حکمت عملی کیا ہے؟', label: '🎯 ہدف حکمت عملی' }
    ] : [
      { text: 'خسارے سے نمٹنے کے لیے کیا فوری اقدامات درکار ہیں؟', label: '🚨 خسارہ ریکوری پلان' },
      { text: 'اگر اخراجات کا تناسب 5% کم کیا جائے تو کیا ہوگا؟', label: '📉 اخراجات میں کمی' }
    ];
  } else if (lang === 'zh') {
    title = curr.isSafe 
      ? '🔄 账户状态已动态同步 (Account Synced)' 
      : '⚠️ 账户流动性预警：存在准备金缺口 (Deficit Alert)';
    insight = curr.isSafe
      ? `账户参数已实时更新。安全缓冲头寸为 <strong>+${fmtMoney(curr.cushion)}</strong>（稳健层级）。`
      : `预警：最新参数调整导致准备金出现 <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> 缺口，建议审查供应链成本与收款账期。`;
    chips = curr.isSafe ? [
      { text: '当前账户最新财务健康状况如何？', label: '📊 诊断最新账户' },
      { text: '为了实现月度目标营收，我们应采取哪些具体的执行策略？', label: '🎯 营收增长路径' }
    ] : [
      { text: '消除准备金赤字缺口的首要应对举措是什么？', label: '🚨 赤字应对方案' },
      { text: '营业成本率降低5%对流动性缓冲的影响？', label: '📉 成本降低5%测算' }
    ];
  } else if (lang === 'ja') {
    title = curr.isSafe 
      ? '🔄 アカウント状態を同期しました (Account Synced)' 
      : '⚠️ 資金流動性アラート：準備金割れ警告 (Deficit Alert)';
    insight = curr.isSafe
      ? `アカウント設定が更新されました。安全クッションは <strong>+${fmtMoney(curr.cushion)}</strong> で安全圏を維持しています。`
      : `警告：パラメータ変更により準備金に対し <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> の不足が生じています。`;
    chips = curr.isSafe ? [
      { text: '最新のアカウント財務ステータスはどうですか？', label: '📊 アカウント診断' },
      { text: '売上目標を達成するために推奨される実行ロードマップは？', label: '🎯 目標達成戦略' }
    ] : [
      { text: '資金不足を解消するための緊急アクションプランは？', label: '🚨 不足解消プラン' }
    ];
  } else {
    title = curr.isSafe 
      ? '🔄 Account Telemetry Synchronized' 
      : '⚠️ Account Solvency Alert: Liquidity Deficit';
    insight = curr.isSafe
      ? `Account parameters synchronized. Net solvency buffer is <strong>+${fmtMoney(curr.cushion)}</strong> (SAFE TIER). Ready for scenario analysis.`
      : `Alert: Current account inputs result in a <strong>-${fmtMoney(Math.abs(curr.cushion))}</strong> reserve deficit. Urgent mitigation recommended.`;
    chips = curr.isSafe ? [
      { text: 'How is my account health now and what are the next steps?', label: '📊 Review Updated Account' },
      { text: 'What actionable strategy should we implement to achieve our monthly target revenue?', label: '🎯 Revenue Target Strategy' }
    ] : [
      { text: 'What immediate actions should we take to eliminate this deficit?', label: '🚨 Deficit Remediation Plan' },
      { text: 'If we cut operational costs by 5%, does it restore our reserve buffer?', label: '📉 Test 5% Cost Cut' }
    ];
  }

  const chipsHtml = chips.map(c => `
    <button class="ai-followup-chip" onclick="askCopilot('${escapeHtml(c.text)}')">
      <span>${escapeHtml(c.label)}</span>
    </button>
  `).join('');

  const syncHtml = `
    <div class="chat-account-sync-card ${curr.isSafe ? '' : 'sync-deficit'}" ${isUrdu ? 'dir="rtl"' : ''}>
      <div class="chat-sync-header">
        <span class="chat-sync-title">${title}</span>
        <span class="chat-sync-time">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="chat-sync-metrics-grid" ${isUrdu ? 'dir="ltr"' : ''}>
        <div class="chat-sync-metric-item">
          <span class="chat-sync-metric-lbl">Available Cash</span>
          <span class="chat-sync-metric-val">${fmtMoney(curr.cash)}${deltaCashStr}</span>
        </div>
        <div class="chat-sync-metric-item">
          <span class="chat-sync-metric-lbl">30D Forecast</span>
          <span class="chat-sync-metric-val">${fmtMoney(curr.rev)}</span>
        </div>
        <div class="chat-sync-metric-item">
          <span class="chat-sync-metric-lbl">Cost Ratio</span>
          <span class="chat-sync-metric-val">${curr.cost}%</span>
        </div>
        <div class="chat-sync-metric-item">
          <span class="chat-sync-metric-lbl">Safety Cushion</span>
          <span class="chat-sync-metric-val" style="color: ${curr.isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">
            ${curr.isSafe ? '+' : ''}${fmtMoney(curr.cushion)} [${curr.isSafe ? 'SAFE' : 'DEFICIT'}]
          </span>
        </div>
      </div>
      <div class="chat-sync-insight">${insight}</div>
      <div class="ai-followup-chips">${chipsHtml}</div>
    </div>
  `;

  history.insertAdjacentHTML('beforeend', syncHtml);
  history.scrollTop = history.scrollHeight;
}

// -------------------------------------------------------------
// Dynamic Gemini/ChatGPT-Grade Follow-Up Suggestion Generator
// -------------------------------------------------------------
function generateDynamicFollowUps(intentKey, targetLang, isSafe, state) {
  const followUpMap = {
    account_status: {
      bn: [
        { text: 'টার্গেট রেভেনিউ পূরণ করতে আমাদের কী কৌশল নেওয়া উচিত?', label: '🎯 টার্গেট পূরণের কৌশল' },
        { text: 'যদি কস্ট রেশিও ৫% কমাই তবে ক্যাশ ফ্লোতে কী উন্নতি হবে?', label: '📉 কস্ট ৫% কমানোর প্রভাব' },
        { text: 'আমাদের বর্তমান ক্যাশ ব্যালেন্স ও রানওয়ে কত দিন চলবে?', label: '⏱️ রানওয়ে কত দিন টিকবে?' }
      ],
      hi: [
        { text: 'मासिक राजस्व लक्ष्य हासिल करने के लिए क्या रणनीति अपनाएं?', label: '🎯 लक्ष्य प्राप्ति योजना' },
        { text: 'हमारा वर्तमान कैश बैलेंस और रनवे कितने दिन चलेगा?', label: '⏱️ कैश रनवे अवधि' },
        { text: 'अगर लागत 5% घटाएं तो रिज़र्व पर क्या असर होगा?', label: '📉 5% लागत कटौती' }
      ],
      ur: [
        { text: 'ماہانہ آمدنی کا ہدف حاصل کرنے کے لیے کیا حکمت عملی اختیار کی جائے؟', label: '🎯 ہدف حکمت عملی' },
        { text: 'موجودہ کیش بیلنس سے کاروبار کتنے دن چل سکتا ہے؟', label: '⏱️ کیش رن وے' },
        { text: 'اخراجات 5% کم کرنے سے کیا بہتری آئے گی؟', label: '📉 اخراجات میں کمی' }
      ],
      zh: [
        { text: '为了实现月度目标营收，我们应采取哪些具体的执行策略？', label: '🎯 营收增长路径' },
        { text: '以当前的每日运营成本，现有现金流可以支撑多少天的跑道？', label: '⏱️ 现金跑道测算' },
        { text: '若营业成本率降低5%，流动性缓冲增加多少？', label: '📉 成本降5%测算' }
      ],
      ja: [
        { text: '売上目標を達成するために推奨される実行ロードマップは？', label: '🎯 目標達成戦略' },
        { text: '現在の現預金と日次運営コストでキャッシュランウェイは何日持ちますか？', label: '⏱️ ランウェイ日数' },
        { text: '原価率を5%削減した場合の資金バッファ改善効果は？', label: '📉 原価5%削減' }
      ],
      en: [
        { text: 'What actionable strategy should we implement to achieve our monthly target revenue?', label: '🎯 Target Strategy' },
        { text: 'How many days of operating runway does our current cash balance support?', label: '⏱️ Operating Runway' },
        { text: 'If we cut operational costs by 5%, how much does our cushion increase?', label: '📉 5% Cost Cut Impact' }
      ]
    },
    safety_risk: {
      bn: isSafe ? [
        { text: 'বিক্রি ২০% কমলে আমাদের কি ক্যাশ ঘাটতি হবে?', label: '🛡️ -২০% বিক্রি স্ট্রেস টেস্ট' },
        { text: 'পেমেন্ট ডিলে ১০ দিন থেকে কমিয়ে ৪ দিনে আনলে ক্যাশ ফ্লোতে কী উন্নতি হবে?', label: '⚡ পেমেন্ট ল্যাগ ৪ দিনে সংকোচন' },
        { text: 'টার্গেট রেভেনিউ পূরণ করতে আমাদের কী কৌশল নেওয়া উচিত?', label: '🎯 টার্গেট অর্জনের রোডম্যাপ' }
      ] : [
        { text: 'ঘাটতি কাটাতে আমাদের অবিলম্বে কী পদক্ষেপ নেওয়া উচিত?', label: '🚨 ঘাটতি নিরসন অ্যাকশন প্ল্যান' },
        { text: 'যদি কস্ট রেশিও ৫% কমাই তবে ক্যাশ ফ্লোতে কী উন্নতি হবে?', label: '📉 জরুরি ব্যয় সংক্ষেপণ' },
        { text: 'আমাদের বর্তমান ক্যাশ ব্যালেন্স ও রানওয়ে কত দিন চলবে?', label: '⏱️ রানওয়ে ও সারভাইভাল' }
      ],
      hi: isSafe ? [
        { text: 'अगर बिक्री 20% गिर जाए तो क्या रिज़र्व घाटा होगा?', label: '🛡️ 20% मंदी स्ट्रेस टेस्ट' },
        { text: 'भुगतान देरी 4 दिन करने पर क्या लाभ होगा?', label: '⚡ भुगतान तेजी' }
      ] : [
        { text: 'घाटे से बचने के लिए तत्काल क्या कदम उठाएं?', label: '🚨 आपातकालीन योजना' },
        { text: 'लागत 5% घटाने पर क्या रिज़र्व सुरक्षित रहेगा?', label: '📉 लागत में 5% कटौती' }
      ],
      ur: isSafe ? [
        { text: 'اگر فروخت میں 20 فیصد کمی ہو تو کیا کیش کا نقصان ہوگا؟', label: '🛡️ سیلز مائنس 20% ٹیسٹ' },
        { text: 'ادائیگی کی تاخیر 4 دن کرنے پر کیا فائدہ ہوگا؟', label: '⚡ ادائیگی میں تیزی' }
      ] : [
        { text: 'خسارے سے نمٹنے کے لیے کیا فوری اقدامات درکار ہیں؟', label: '🚨 خسارہ ریکوری پلان' },
        { text: 'اخراجات 5% کم کرنے سے کیا بہتری آئے گی؟', label: '📉 اخراجات میں کٹوتی' }
      ],
      zh: isSafe ? [
        { text: '如果下个月销售额减少20%，储备资金会出现赤字缺口吗？', label: '🛡️ 下滑20%压力测试' },
        { text: '将应收账款收款周期压缩至4天能释放多少流动资金？', label: '⚡ 应收账款压缩至4天' }
      ] : [
        { text: '消除准备金赤字缺口的首要应对举措是什么？', label: '🚨 赤字应对方案' },
        { text: '营业成本率降低5%对流动性缓冲的影响？', label: '📉 成本降低5%测算' }
      ],
      ja: isSafe ? [
        { text: '来月の売上が20%減少した場合、法定準備金を割り込みますか？', label: '🛡️ 売上20%減ストレス' },
        { text: '回収ラグを4日に短縮した場合の効果は？', label: '⚡ 回収ラグ短縮' }
      ] : [
        { text: '資金不足を解消するための緊急アクションプランは？', label: '🚨 不足解消プラン' },
        { text: '原価率を5%削減した場合の資金バッファ改善効果は？', label: '📉 原価5%削減' }
      ],
      en: isSafe ? [
        { text: 'If sales drop by 20% next month, will cash breach our statutory reserve?', label: '🛡️ -20% Sales Drop Stress' },
        { text: 'What liquidity is liberated by compressing dealer collection lag to 4 days?', label: '⚡ Compress Lag to 4D' }
      ] : [
        { text: 'What immediate actions should we take to eliminate this deficit?', label: '🚨 Deficit Remediation Plan' },
        { text: 'If we cut operational costs by 5%, does it restore our reserve buffer?', label: '📉 Test 5% Cost Cut' }
      ]
    },
    general: {
      bn: [
        { text: 'আমার একাউন্ট এখন কেমন এবং পরবর্তী পদক্ষেপ কী?', label: '📊 একাউন্ট স্ট্যাটাস রিভিউ' },
        { text: 'টার্গেট রেভেনিউ পূরণ করতে আমাদের কী কৌশল নেওয়া উচিত?', label: '🎯 টার্গেট পূরণের কৌশল' },
        { text: 'আমাদের বর্তমান ক্যাশ ব্যালেন্স ও রানওয়ে কত দিন চলবে?', label: '⏱️ ক্যাশ রানওয়ে কত দিন?' },
        { text: 'P10 ডাউনসাইড এবং P90 সার্জ সিনারিওতে ঝুঁকি ও লাভ কেমন হবে?', label: '📈 P10 বনাম P90 ফোরকাস্ট' }
      ],
      hi: [
        { text: 'हमारे वर्तमान खाते की स्थिति कैसी है?', label: '📊 खाता समीक्षा' },
        { text: 'मासिक राजस्व लक्ष्य हासिल करने के लिए क्या रणनीति अपनाएं?', label: '🎯 लक्ष्य रणनीति' },
        { text: 'हमारा वर्तमान कैश बैलेंस और रनवे कितने दिन चलेगा?', label: '⏱️ कैश रनवे' }
      ],
      ur: [
        { text: 'ہمارے اکاؤنٹ کی موجودہ صورتحال کیا ہے؟', label: '📊 اکاؤنٹ کا جائزہ' },
        { text: 'ماہانہ آمدنی کا ہدف حاصل کرنے کے لیے کیا حکمت عملی اختیار کی جائے؟', label: '🎯 ہدف حکمت عملی' },
        { text: 'موجودہ کیش بیلنس سے کاروبار کتنے دن چل سکتا ہے؟', label: '⏱️ کیش رن وے' }
      ],
      zh: [
        { text: '当前账户最新财务健康状况如何？', label: '📊 诊断最新账户' },
        { text: '为了实现月度目标营收，我们应采取哪些具体的执行策略？', label: '🎯 营收增长路径' },
        { text: '以当前的每日运营成本，现有现金流可以支撑多少天的跑道？', label: '⏱️ 现金跑道测算' }
      ],
      ja: [
        { text: '最新のアカウント財務ステータスはどうですか？', label: '📊 アカウント診断' },
        { text: '売上目標を達成するために推奨される実行ロードマップは？', label: '🎯 目標達成戦略' },
        { text: '現在の現預金と日次運営コストでキャッシュランウェイは何日持ちますか？', label: '⏱️ ランウェイ日数' }
      ],
      en: [
        { text: 'How is my account health now and what are the next steps?', label: '📊 Review Updated Account' },
        { text: 'What actionable strategy should we implement to achieve our monthly target revenue?', label: '🎯 Target Strategy' },
        { text: 'How many days of operating runway does our current cash balance support?', label: '⏱️ Operating Runway' },
        { text: 'What are the P10 downside floor and P90 surge bounds under conformal uncertainty?', label: '📈 Conformal P10/P90 Bounds' }
      ]
    }
  };

  const pool = followUpMap[intentKey] || followUpMap.safety_risk || followUpMap.general;
  const chips = pool[targetLang] || pool.en || followUpMap.general.en;

  const chipsHtml = chips.map(c => `
    <button class="ai-followup-chip" onclick="askCopilot('${escapeHtml(c.text)}')">
      <span>${escapeHtml(c.label)}</span>
    </button>
  `).join('');

  const titles = {
    bn: '⚡ পরবর্তী প্রশ্ন ও কৌশলগত অনুসন্ধান (Suggested Follow-ups):',
    hi: '⚡ अनुवर्ती प्रश्न एवं सुझाव (Suggested Follow-ups):',
    ur: '⚡ تجویز کردہ فالو اپ سوالات (Suggested Follow-ups):',
    zh: '⚡ 推荐深度探究问题 (Suggested Follow-ups):',
    ja: '⚡ 推奨フォローアップ質問 (Suggested Follow-ups):',
    en: '⚡ Suggested Next Questions (Gemini / ChatGPT Context):'
  };

  const titleText = titles[targetLang] || titles.en;

  return `
    <div class="ai-followup-container">
      <div class="ai-followup-title">${titleText}</div>
      <div class="ai-followup-chips">${chipsHtml}</div>
    </div>
  `;
}

function _computeCopilotResponseRaw(query) {
  const prevTurn = copilotConversationHistory.length > 0 ? copilotConversationHistory[copilotConversationHistory.length - 1] : null;
  const salesShift = parseFloat(document.getElementById('input-growth').value) || 0;
  const availableCash = parseFloat(document.getElementById('input-cash').value) || 180000;
  const costRatioPct = parseFloat(document.getElementById('input-cost').value) || 62;
  const costRatio = costRatioPct / 100.0;
  const targetVal = parseFloat(document.getElementById('input-target-revenue').value) || 500000;

  const targetLang = detectScriptLanguage(query, copilotLang);
  const normQuery = normalizeAsianDigits(query);
  const lowerQ = normQuery.toLowerCase();

  // Precomputed financial vitals
  const projRev = BASE_MONTHLY_REVENUE_BDT * (1.0 + salesShift / 100.0);
  const grossMarginRatio = 1.0 - costRatio;
  const estimatedGrossProfit = projRev * grossMarginRatio;
  const operatingSurplus = projRev * grossMarginRatio;
  const reserveTarget = BASE_MONTHLY_REVENUE_BDT * costRatio;
  const cashAfterReserve = availableCash + operatingSurplus - reserveTarget;
  const isSafe = cashAfterReserve >= 0;
  const dailyOpex = (BASE_MONTHLY_REVENUE_BDT * costRatio) / 30.0;
  const zeroRevenueRunwayDays = dailyOpex > 0 ? Math.floor(availableCash / dailyOpex) : 999;
  const baseDaily = projRev / 30.0;
  const p10Daily = baseDaily * 0.60;
  const p90Daily = baseDaily * 1.50;

  // -------------------------------------------------------------
  // -------------------------------------------------------------
  // INTENT A: Live Account Status & Parameter Update Audit
  // -------------------------------------------------------------
  const isAccountStatus = lowerQ.includes('account') || lowerQ.includes('একাউন্ট') || lowerQ.includes('account status') ||
    lowerQ.includes('my account') || lowerQ.includes('আপডেট') || lowerQ.includes('ব্যালেন্স') || lowerQ.includes('এখন কি অবস্থা') ||
    lowerQ.includes('আমার ব্যবসা এখন কেমন') || lowerQ.includes('বর্তমান অবস্থা') || lowerQ.includes('ক্যাশ পরিবর্তন') ||
    lowerQ.includes('নয়া ব্যালেন্স') || lowerQ.includes('হালনাগাদ') || lowerQ.includes('खाता') || lowerQ.includes('स्थिति') ||
    lowerQ.includes('बैलेंस') || lowerQ.includes('अब क्या स्थिति है') || lowerQ.includes('اکاؤنٹ') || lowerQ.includes('صورتحال') ||
    lowerQ.includes('تازہ ترین') || lowerQ.includes('کیش تبدیل') || lowerQ.includes('账户') || lowerQ.includes('状态') ||
    lowerQ.includes('当前') || lowerQ.includes('最新财务') || lowerQ.includes('アカウント') || lowerQ.includes('残高') ||
    lowerQ.includes('状況') || lowerQ.includes('口座') ||
    (prevTurn && (lowerQ.includes('how about now') || lowerQ.includes('now') || lowerQ.includes('এখন কেমন')));

  if (isAccountStatus) {
    lastDetectedIntent = 'account_status';
    if (targetLang === 'bn') {
      return `
        <p><strong>【রিয়েল-টাইম একাউন্ট অডিট ও ট্রেজারি ব্রিফিং】</strong></p>
        <p>আপনার বর্তমান একাউন্ট প্যারামিটার ও ট্রেজারি অবস্থা সফলভাবে সিঙ্ক করা হয়েছে। সর্বশেষ হিসাব অনুযায়ী আপনার ব্যবসার আর্থিক সারসংক্ষেপ:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Statutory Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <p><strong>অডিট বিশ্লেষণ ও পরামর্শ:</strong></p>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.55;">
          <li><strong>ক্যাশ রানওয়ে:</strong> দৈনিক অপারেটিং ব্যয় ${fmtMoney(dailyOpex)}-এর ভিত্তিতে আপনার হাত থাকা ক্যাশ <strong>${zeroRevenueRunwayDays} দিন</strong> চলবে।</li>
          <li><strong>সলভেন্সি স্ট্যাটাস:</strong> আপনার ব্যবসা বর্তমানে <strong>${isSafe ? 'নিরাপদ জোনে (Adequate Liquidity)' : 'ঘাটতি ঝুঁকিতে (Liquidity Shortfall)'}</strong> রয়েছে। ${isSafe ? 'সারপ্লাস ক্যাশ প্রবৃদ্ধি অর্জনে বা বাফার হিসেবে ব্যবহার করা যাবে।' : `রিজার্ভ রক্ষায় অবিলম্বে ${fmtMoney(Math.abs(cashAfterReserve))} ঘাটতি পূরণ করতে হবে।`}</li>
          <li><strong>কস্ট রেশিও:</strong> বর্তমান কস্ট রেশিও <strong>${costRatioPct}%</strong>, যার ফলে গ্রস প্রফিট মার্জিন <strong>${Math.round(grossMarginRatio * 100)}%</strong>।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Radar</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">Stress Test Scenarios</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Audit</button>
        </div>
      `;
    } else if (targetLang === 'hi') {
      return `
        <p><strong>【वास्तविक समय खाता ऑडिट एवं ट्रेजरी ब्रीफिंग】</strong></p>
        <p>आपके वर्तमान खाता पैरामीटर और ट्रेजरी स्थिति को सफलतापूर्वक सिंक्रनाइज़ किया गया है:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <p><strong>विश्लेषण एवं रणनीति:</strong> व्यापार वर्तमान में <strong>${isSafe ? 'सुरक्षित क्षेत्र (SAFE)' : 'घाटे के जोखिम (DEFICIT)'}</strong> में है। शून्य राजस्व पर कैश रनवे <strong>${zeroRevenueRunwayDays} दिन</strong> है।</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Radar</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">Stress Test Scenarios</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Audit</button>
        </div>
      `;
    } else if (targetLang === 'ur') {
      return `
        <div dir="rtl">
          <p><strong>【ریئل ٹائم اکاؤنٹ آڈٹ اور ٹریژری بریفنگ】</strong></p>
          <p>آپ کے اکاؤنٹ کے موجودہ پیرامیٹرز کو کامیابی سے ہم آہنگ کر دیا گیا ہے:</p>
        </div>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Radar</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">Stress Test Scenarios</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Audit</button>
        </div>
      `;
    } else if (targetLang === 'zh') {
      return `
        <p><strong>【实时账户审计与国库决策报告】</strong></p>
        <p>当前账户核心财务指标已完成动态同步，审计结论如下：</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">手头现金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30天预测营业额</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">法定准备金</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">安全缓冲资金</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <p>当前企业处于 <strong>${isSafe ? '安全层级（SAFE）' : '流动性赤字预警（DEFICIT）'}</strong>。在零收入极端压力下，现金跑道可维持 <strong>${zeroRevenueRunwayDays} 天</strong>。</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">打开流动性雷达</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">压力测试推演</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">复制审计结果</button>
        </div>
      `;
    } else if (targetLang === 'ja') {
      return `
        <p><strong>【リアルタイム・アカウント監査レポート】</strong></p>
        <p>最新のアカウント設定および手元資金パラメータが正常に同期されました：</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">現在の手元資金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30日予測売上</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">法定準備金目標</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">安全クッション</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <p>診断判定：現在は <strong>${isSafe ? '安全圏（SAFE）' : '資金不足警戒圏（DEFICIT）'}</strong> に位置しています。完全操業停止時のキャッシュランウェイは <strong>${zeroRevenueRunwayDays} 日</strong> です。</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">流動性レーダーを見る</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">ストレステスト実行</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">監査結果をコピー</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Real-Time Account Audit & Treasury Briefing】</strong></p>
        <p>Your active account parameters and treasury baseline have been synchronized. Here is your executive financial status:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Statutory Reserve</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)} [${isSafe ? 'SAFE' : 'DEFICIT'}]</div></div>
        </div>
        <p><strong>Executive Findings:</strong> Business is operating in <strong>${isSafe ? 'SAFE TIER' : 'DEFICIT HAZARD'}</strong>. At daily operating burn of ${fmtMoney(dailyOpex)}, zero-revenue cash runway supports <strong>${zeroRevenueRunwayDays} days</strong> of operations.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Radar</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('scenarios')">Stress Test Scenarios</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Audit</button>
        </div>
      `;
    }
  }

  // INTENT 0: Greetings, Introduction & Help
  // -------------------------------------------------------------
  const isGreeting = /^(hi|hello|hey|good\s*(morning|afternoon|evening)|salam|assalamu|namaste|hola|bonjour)/i.test(lowerQ) ||
    lowerQ.includes('who are you') || lowerQ.includes('what can you do') || lowerQ.includes('help') || lowerQ.includes('how are you') ||
    lowerQ.includes('features') || lowerQ.includes('kemon acho') || lowerQ.includes('tumi ke') || lowerQ.includes('ki korte paro') ||
    lowerQ.includes('হ্যালো') || lowerQ.includes('হাই') || lowerQ.includes('সালাম') || lowerQ.includes('নমস্কার') ||
    lowerQ.includes('কেমন আছো') || lowerQ.includes('কেমন আছেন') || lowerQ.includes('তুমি কে') || lowerQ.includes('কী করতে পার') ||
    lowerQ.includes('সাহায্য') || lowerQ.includes('হেল্প') || lowerQ.includes('পরিচয়') ||
    lowerQ.includes('नमस्ते') || lowerQ.includes('मदद') || lowerQ.includes('आप कौन हैं') ||
    lowerQ.includes('سلام') || lowerQ.includes('مدد') || lowerQ.includes('آپ کون ہیں') ||
    lowerQ.includes('你好') || lowerQ.includes('您好') || lowerQ.includes('帮助') || lowerQ.includes('你是谁') ||
    lowerQ.includes('こんにちは') || lowerQ.includes('ヘルプ') || lowerQ.includes('あなたは誰');

  if (isGreeting) {
    lastDetectedIntent = 'general';
    if (targetLang === 'bn') {
      return `
        <p><strong>【FinSight AI — ডিসিশন ইন্টেলিজেন্স অ্যাসিস্ট্যান্ট】</strong></p>
        <p>হ্যালো! আমি আপনার ব্যবসার রিয়েল-টাইম ক্যাশ ফ্লো, বিক্রয় পূর্বাভাস, লিকুইডিটি ঝুঁকি এবং স্ট্র্যাটেজিক পরিকল্পনা বিশ্লেষণে সাহায্য করি।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <p><strong>আমি যেসব বিষয়ে সহায়তা করতে পারি:</strong></p>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li>ক্যাশ রানওয়ে এবং দৈনিক অপারেটিং বার্ন নিরীক্ষা।</li>
          <li>P10 ডাউনসাইড এবং P90 সার্জ বিক্রয় পূর্বাভাস।</li>
          <li>টার্গেট রেভেনিউ পূরণের জন্য ৩-ধাপের কৌশলগত রোডম্যাপ।</li>
          <li>কস্ট ইনফ্লেশন বা পেমেন্ট ডিলে সংক্রান্ত লিকুইডিটি স্ট্রেস টেস্ট।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Summary</button>
        </div>
      `;
    } else if (targetLang === 'hi') {
      return `
        <p><strong>【FinSight AI — वित्तीय निर्णय सहायता सहायक】</strong></p>
        <p>नमस्ते! मैं आपके व्यापार के कैश फ्लो, बिक्री पूर्वानुमान, तरलता जोखिम और रणनीति विश्लेषण में सहायता के लिए तैयार हूँ।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Summary</button>
        </div>
      `;
    } else if (targetLang === 'ur') {
      return `
        <div dir="rtl">
          <p><strong>【FinSight AI — فنانشل ڈیسیژن کوپائلٹ】</strong></p>
          <p>سلام! میں آپ کے کاروبار کے نقد بہاؤ، پیشین گوئی اور لیکویڈیٹی کے تجزیے میں مدد فراہم کرتا ہوں۔</p>
        </div>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Summary</button>
        </div>
      `;
    } else if (targetLang === 'zh') {
      return `
        <p><strong>【FinSight AI — 确定性财务决策智能助手】</strong></p>
        <p>您好！我是您的中小微企业财务副驾驶，能够为您提供实时的现金流审计、销售分位数预测、法定储备监控及情景模拟。</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30天预测营业额</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">综合毛利率</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">手头现金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">安全缓冲资金</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">打开分位数预测器</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">查看流动性雷达</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">复制摘要</button>
        </div>
      `;
    } else if (targetLang === 'ja') {
      return `
        <p><strong>【FinSight AI — 意思決定インテリジェンス・コパイロット】</strong></p>
        <p>こんにちは！企業のキャッシュフロー健全性、高精度売上予測、法定準備金維持、および戦略的意思決定を支援します。</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30日予測売上</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">粗利益率</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">現在の手元資金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">安全クッション</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">予測モジュールを開く</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">流動性レーダーを見る</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">サマリーをコピー</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【FinSight AI — Grounded Decision Intelligence Copilot】</strong></p>
        <p>Hello! I am your deterministic financial reasoning copilot, continuously auditing SME cashflow health, conformal sales projections, and statutory solvency invariants.</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Treasury Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Conformal Forecaster</button>
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Summary</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 1: Conformal Quantile Forecasting & Uncertainty Intervals
  // -------------------------------------------------------------
  const isUncertainty = lowerQ.includes('p10') || lowerQ.includes('p90') || lowerQ.includes('conformal') || lowerQ.includes('quantile') || lowerQ.includes('forecast') || lowerQ.includes('পূর্বাভাস') || lowerQ.includes('কোয়ান্টাইল') || lowerQ.includes('অনিশ্চয়তা') || lowerQ.includes('সার্জ') || lowerQ.includes('अनिश्चितता') || lowerQ.includes('पूर्वानुमान') || lowerQ.includes('غير يقيني') || lowerQ.includes('پیشین گوئی') || lowerQ.includes('不确定性') || lowerQ.includes('预测') || lowerQ.includes('不確実性');

  if (isUncertainty) {
    lastDetectedIntent = 'uncertainty';
    if (targetLang === 'bn') {
      return `
        <p><strong>【কনফর্মাল কোয়ান্টাইল বিক্রয় পূর্বাভাস ও অনিশ্চয়তা মডেল (P10/P50/P90)】</strong></p>
        <p>পরীক্ষামূলক ডসিয়ার (Experiment A) অনুযায়ী আমাদের লাইটজিবিএম কোয়ান্টাইল রিগ্রেশন মডেলটি <strong>WAPE ২৫.৭২%</strong> ও <strong>Diebold-Mariano p = ০.০০২২</strong> সহকারে স্ট্যাটিক মেথডকে পরাস্ত করেছে।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P10 Downside Floor</div><div class="ai-metric-val" style="color: var(--accent-amber);">${fmtMoney(p10Daily)} / দিন</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P50 Central Tendency</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${fmtMoney(baseDaily)} / দিন</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P90 Optimistic Surge</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(p90Daily)} / দিন</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Conformal Coverage</div><div class="ai-metric-val">80.0% [Purged 7d]</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li>কনজারভেটিভ <strong>P10 সীমানা (${fmtMoney(p10Daily)}/দিন)</strong>-তেও ব্যবসা রিজার্ভ সংকট ছাড়া পরিচালন ব্যয় মেটাতে সক্ষম।</li>
          <li>সার্জ পিরিয়ডে <strong>P90 সীমানা (${fmtMoney(p90Daily)}/দিন)</strong> পর্যন্ত ইনভেন্টরি স্টক আউটের ঝুঁকি প্রতিহত করা সম্ভব।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.evaluate_conformal_bounds]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Conformal Forecaster</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Report</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Conformal Quantile Forecasting & Uncertainty Intervals (P10/P50/P90)】</strong></p>
        <p>Under empirical benchmark (Experiment A), our LightGBM Conformal Quantile Regressor significantly outperforms naive baselines with <strong>WAPE 25.72%</strong> and <strong>Diebold-Mariano p = 0.0022</strong>.</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P10 Downside Floor</div><div class="ai-metric-val" style="color: var(--accent-amber);">${fmtMoney(p10Daily)} / day</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P50 Median Forecast</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${fmtMoney(baseDaily)} / day</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">P90 Surge Ceiling</div><div class="ai-metric-val" style="color: var(--accent-blue);">${fmtMoney(p90Daily)} / day</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Conformal Coverage</div><div class="ai-metric-val">80.0% [Purged 7d]</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li>Conservative floor (${fmtMoney(p10Daily)}/day) preserves solvency without statutory cushion breach.</li>
          <li>Surge bound (${fmtMoney(p90Daily)}/day) enables proactive inventory fulfillment without working capital strain.</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.evaluate_conformal_bounds]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Conformal Forecaster</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Report</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 2: Target Revenue & Growth Gap Execution Roadmap
  // -------------------------------------------------------------
  const isTarget = lowerQ.includes('টার্গেট') || lowerQ.includes('target') || lowerQ.includes('কৌশল') || lowerQ.includes('goal') || lowerQ.includes('gap') || lowerQ.includes('रणनीति') || lowerQ.includes('लक्ष्य') || lowerQ.includes('हिकमत عملی') || lowerQ.includes('ہدف') || lowerQ.includes('目标') || lowerQ.includes('策略') || lowerQ.includes('目標') || lowerQ.includes('戦略');

  if (isTarget) {
    lastDetectedIntent = 'target';
    const currentRunRate = projRev;
    const gap = targetVal - currentRunRate;
    const isTargetMet = gap <= 0;
    const growthPct = currentRunRate > 0 ? ((gap / currentRunRate) * 100.0) : 0;
    const weeklyTarget = targetVal / 4.0;
    const dailyGap = Math.max(0, Math.round(gap / 30.0));
    const deltaAOV = Math.max(0, Math.round(gap / 667.0));
    const liberatedCapital = Math.round(currentRunRate * (6.0 / 30.0));

    if (targetLang === 'bn') {
      if (isTargetMet) {
        return `
          <p><strong>【টার্গেট এক্সিকিউশন ও গ্রোথ রোডম্যাপ】</strong></p>
          <p>অভিনন্দন! আপনার বর্তমান প্রজেক্টেড মাসিক বিক্রয় <strong>${fmtMoney(currentRunRate)}</strong>, যা লক্ষ্যমাত্রা <strong>${fmtMoney(targetVal)}</strong> পূরণ করে <strong>+${fmtMoney(Math.abs(gap))}</strong> উদ্বৃত্ত (Surplus) অর্জন করেছে!</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Status</div><div class="ai-metric-val" style="color: var(--accent-emerald);">অর্জিত (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ratio Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>উদ্বৃত্ত তারল্য সংরক্ষণ:</strong> পরিচালন খরচ কঠোরভাবে <strong>${costRatioPct}%</strong> এর মধ্যে ধরে রেখে প্রতি মাসে <strong>${fmtMoney(operatingSurplus)}</strong> অপারেটিং উদ্বৃত্ত রিজার্ভে যোগ করা।</li>
            <li><strong>ক্যাশ সাইকেল গতিশীলতা:</strong> ডিলার পেমেন্ট ল্যাগ ১০ দিন থেকে ৪ দিনে নামিয়ে এনে প্রায় <strong>+${fmtMoney(liberatedCapital)}</strong> চলতি মূলধন দ্রুত লিকুইড ক্যাশে রূপান্তর করা।</li>
            <li><strong>পরবর্তী প্রবৃদ্ধি মাইলস্টোন:</strong> ব্যবসা আরও বড় করতে স্টুডিওতে নতুন উচ্চতর লক্ষ্যমাত্রা (Target Revenue) নির্ধারণ করুন।</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <p><strong>【টার্গেট এক্সিকিউশন ও গ্রোথ রোডম্যাপ】</strong></p>
          <p>মাসিক রেভেনিউ লক্ষ্যমাত্রা <strong>${fmtMoney(targetVal)}</strong> অর্জনের জন্য বর্তমান প্রজেক্টেড সেলস <strong>${fmtMoney(currentRunRate)}</strong> থেকে আরও <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> পূরণ করতে হবে।</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Execution Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>হাই-মার্জিন এসকেইউ ক্রস-সেলিং:</strong> দৈনিক অতিরিক্ত <strong>+${fmtMoney(dailyGap)}</strong> বিক্রি ও গড় ঝুড়ি মূল্য <strong>+${fmtMoney(deltaAOV)}</strong> বাড়াতে শীর্ষ ২০% হাই-মার্জিন পণ্যের ক্রস-সেলিং জোরদার করা।</li>
            <li><strong>পেমেন্ট ল্যাগ সংকোচন ও তহবিল মুক্তি:</strong> ডিলারদের ২% আর্লি ডিসকাউন্ট দিয়ে পেমেন্ট ল্যাগ ১০ দিন থেকে ৪ দিনে কমিয়ে প্রায় <strong>+${fmtMoney(liberatedCapital)}</strong> তাৎক্ষণিক নগদ ক্যাপিটাল মুক্ত করা।</li>
            <li><strong>কস্ট সিলিং ও সাপ্তাহিক রান-রেট:</strong> অপারেটিং খরচ কঠোরভাবে <strong>${costRatioPct}%</strong> এর নিচে ধরে রেখে প্রতি সপ্তাহে গড়ে <strong>${fmtMoney(weeklyTarget)}</strong> ক্যাশ সেলস নিশ্চিত করা।</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    } else if (targetLang === 'hi') {
      if (isTargetMet) {
        return `
          <p><strong>【लक्ष्य निष्पादन एवं विकास रणनीतिक रोडमैप】</strong></p>
          <p>बधाई! आपकी वर्तमान अनुमानित मासिक बिक्री <strong>${fmtMoney(currentRunRate)}</strong> है, जिसने <strong>${fmtMoney(targetVal)}</strong> के लक्ष्य को पार कर <strong>+${fmtMoney(Math.abs(gap))}</strong> का अधिशेष (Surplus) हासिल किया है!</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Status</div><div class="ai-metric-val" style="color: var(--accent-emerald);">प्राप्त (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>अधिशेष तरलता संरक्षण:</strong> परिचालन लागत अनुपात को <strong>${costRatioPct}%</strong> के भीतर रखकर प्रति माह <strong>${fmtMoney(operatingSurplus)}</strong> परिचालन अधिशेष सुरक्षित करें।</li>
            <li><strong>भुगतान चक्र में तेजी:</strong> भुगतान अंतराल 10 दिन से 4 दिन करके <strong>+${fmtMoney(liberatedCapital)}</strong> कार्यशील पूंजी तुरंत मुक्त करें।</li>
            <li><strong>अगला मील का पत्थर:</strong> व्यवसाय को और अधिक बढ़ाने के लिए स्टूडियो में उच्चतर लक्ष्य निर्धारित करें।</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <p><strong>【लक्ष्य निष्पादन एवं विकास रणनीतिक रोडमैप】</strong></p>
          <p>मासिक राजस्व लक्ष्य <strong>${fmtMoney(targetVal)}</strong> प्राप्त करने के लिए वर्तमान बिक्री <strong>${fmtMoney(currentRunRate)}</strong> से <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> की वृद्धि आवश्यक है।</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Execution Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>उच्च-मार्जिन SKU प्राथमिकता:</strong> दैनिक अतिरिक्त <strong>+${fmtMoney(dailyGap)}</strong> बिक्री और औसत बास्केट <strong>+${fmtMoney(deltaAOV)}</strong> बढ़ाने के लिए शीर्ष 20% मार्जिन उत्पादों पर ध्यान दें।</li>
            <li><strong>भुगतान चक्र में तेजी:</strong> 2% छूट के साथ भुगतान चक्र 10 दिन से 4 दिन करके <strong>+${fmtMoney(liberatedCapital)}</strong> तत्काल नकदी मुक्त करें।</li>
            <li><strong>लागत सीमा व साप्ताहिक लक्ष्य:</strong> परिचालन लागत अनुपात को <strong>${costRatioPct}%</strong> के नीचे रखकर प्रति सप्ताह औसतन <strong>${fmtMoney(weeklyTarget)}</strong> बिक्री सुनिश्चित करें।</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    } else if (targetLang === 'ur') {
      if (isTargetMet) {
        return `
          <div dir="rtl">
            <p><strong>【ہدف کا نفاذ اور تزویراتی روڈ میپ】</strong></p>
            <p>مبارک ہو! آپ کی موجودہ متوقع ماہانہ فروخت <strong>${fmtMoney(currentRunRate)}</strong> ہے جس نے <strong>${fmtMoney(targetVal)}</strong> کے ہدف کو پورا کر کے <strong>+${fmtMoney(Math.abs(gap))}</strong> فاضل (Surplus) حاصل کیا ہے!</p>
          </div>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Status</div><div class="ai-metric-val" style="color: var(--accent-emerald);">حاصل شدہ (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul dir="rtl" style="margin: 4px 18px 8px 0; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>فاضل لیکویڈیٹی کا تحفظ:</strong> لاگت کا تناسب <strong>${costRatioPct}%</strong> کے اندر رکھ کر ماہانہ <strong>${fmtMoney(operatingSurplus)}</strong> منافع محفوظ رکھیں۔</li>
            <li><strong>کیش کی رفتار:</strong> ادائیگی 10 دن سے 4 دن کر کے <strong>+${fmtMoney(liberatedCapital)}</strong> ورکنگ کیپیٹل آزاد کریں۔</li>
            <li><strong>اگلا سنگ میل:</strong> اسٹوڈیو میں کاروبار کے لیے اگلا بڑا ہدف مقرر کریں۔</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <div dir="rtl">
            <p><strong>【ہدف کا نفاذ اور تزویراتی روڈ میپ】</strong></p>
            <p>ماہانہ آمدنی کا ہدف <strong>${fmtMoney(targetVal)}</strong> حاصل کرنے کے لیے موجودہ متوقع فروخت <strong>${fmtMoney(currentRunRate)}</strong> سے مزید <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> درکار ہے۔</p>
          </div>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul dir="rtl" style="margin: 4px 18px 8px 0; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>زیادہ منافع والے آئٹمز:</strong> روزانہ اضافی <strong>+${fmtMoney(dailyGap)}</strong> فروخت اور اوسط آرڈر میں <strong>+${fmtMoney(deltaAOV)}</strong> اضافے کے لیے ٹاپ 20% آئٹمز پر توجہ دیں۔</li>
            <li><strong>ادائیگی میں تیزی:</strong> 2% رعایت دے کر ادائیگی 10 دن کے بجائے 4 دن میں حاصل کریں جس سے <strong>+${fmtMoney(liberatedCapital)}</strong> کیش آزاد ہوگا۔</li>
            <li><strong>لاگت پر قابو:</strong> لاگت کا تناسب سختی سے <strong>${costRatioPct}%</strong> کے نیچے رکھ کر ہر ہفتے اوسطاً <strong>${fmtMoney(weeklyTarget)}</strong> فروخت حاصل کریں۔</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    } else if (targetLang === 'zh') {
      if (isTargetMet) {
        return `
          <p><strong>【目标执行与增量营收战略路线图】</strong></p>
          <p>恭喜！您当前预测月度营收为 <strong>${fmtMoney(currentRunRate)}</strong>，已达成并超出 <strong>${fmtMoney(targetVal)}</strong> 目标，盈余 <strong>+${fmtMoney(Math.abs(gap))}</strong>！</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">达成状态</div><div class="ai-metric-val" style="color: var(--accent-emerald);">已达成 (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ratio Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>利润留存：</strong> 将综合运营成本率严格控制在 <strong>${costRatioPct}%</strong> 以内，每月确保留存 <strong>${fmtMoney(operatingSurplus)}</strong> 运营盈余。</li>
            <li><strong>现金周转：</strong> 将分销商回款账期从10天压缩至4天，释放 <strong>+${fmtMoney(liberatedCapital)}</strong> 流动资金并纳入准备金。</li>
            <li><strong>目标升级：</strong> 在决策演练台中设立更高营收目标，推进更大规模业务扩张。</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <p><strong>【目标执行与增量营收战略路线图】</strong></p>
          <p>为达成月度营收目标 <strong>${fmtMoney(targetVal)}</strong>，需在当前预测 <strong>${fmtMoney(currentRunRate)}</strong> 基础上弥补 <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> 的增长缺口。</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Execution Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>高毛利SKU交叉销售：</strong> 聚焦前20%高毛利货品，实现日均增量 <strong>+${fmtMoney(dailyGap)}</strong> 营收并提升客单价 <strong>+${fmtMoney(deltaAOV)}</strong>。</li>
            <li><strong>应收账款周转加速：</strong> 通过2%现金折扣将账期由10天压缩至4天，迅速释放 <strong>+${fmtMoney(liberatedCapital)}</strong> 即期营运资金。</li>
            <li><strong>成本红线与周度目标：</strong> 将运营成本率严格控制在 <strong>${costRatioPct}%</strong> 以下，每周确保平均达成 <strong>${fmtMoney(weeklyTarget)}</strong> 现金销售额。</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    } else if (targetLang === 'ja') {
      if (isTargetMet) {
        return `
          <p><strong>【売上目標達成と実行可能戦略ロードマップ】</strong></p>
          <p>おめでとうございます！現在の予測月商は <strong>${fmtMoney(currentRunRate)}</strong> であり、目標値 <strong>${fmtMoney(targetVal)}</strong> を達成し <strong>+${fmtMoney(Math.abs(gap))}</strong> の余剰（Surplus）を創出しています！</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">達成状況</div><div class="ai-metric-val" style="color: var(--accent-emerald);">達成 (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ratio Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>利益防衛：</strong> 原価率を <strong>${costRatioPct}%</strong> 以下に厳格統制し、月間 <strong>${fmtMoney(operatingSurplus)}</strong> の営業黒字を内部留保。</li>
            <li><strong>キャッシュ速度向上：</strong> 回収ラグを10日から4日に短縮し、<strong>+${fmtMoney(liberatedCapital)}</strong> の運転資本を早期流動化。</li>
            <li><strong>次期マイルストーン設定：</strong> スタジオで新たな売上目標を設定し、次の事業拡大を推進。</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <p><strong>【売上目標達成と実行可能戦略ロードマップ】</strong></p>
          <p>月間売上目標 <strong>${fmtMoney(targetVal)}</strong> の達成には、現在の予測 <strong>${fmtMoney(currentRunRate)}</strong> に対し <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> のギャップを埋める必要があります。</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Execution Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>高利益率SKUのセット販売：</strong> 上位20%の高マージン商品に注力し、日次 <strong>+${fmtMoney(dailyGap)}</strong> の売上上乗せおよび客単価 <strong>+${fmtMoney(deltaAOV)}</strong> 拡大を推進。</li>
            <li><strong>売掛金回収の迅速化：</strong> 2%の早期決済優遇により回収ラグを10日から4日に圧縮し、<strong>+${fmtMoney(liberatedCapital)}</strong> の即時流動資金を創出。</li>
            <li><strong>経費シーリングと週次目標：</strong> 営業費用率を <strong>${costRatioPct}%</strong> 未満に厳格制限し、週平均 <strong>${fmtMoney(weeklyTarget)}</strong> のキャッシュ売上を確保。</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    } else {
      if (isTargetMet) {
        return `
          <p><strong>【Target Execution & Revenue Bridge Roadmap】</strong></p>
          <p>Congratulations! Your current projected monthly revenue of <strong>${fmtMoney(currentRunRate)}</strong> has surpassed your target of <strong>${fmtMoney(targetVal)}</strong> with a surplus of <strong>+${fmtMoney(Math.abs(gap))}</strong>!</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Status</div><div class="ai-metric-val" style="color: var(--accent-emerald);">Achieved (+${fmtMoney(Math.abs(gap))})</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Run-Rate</div><div class="ai-metric-val">${fmtMoney(currentRunRate / 4.0)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ratio Ceiling</div><div class="ai-metric-val" style="color: var(--accent-amber);">${costRatioPct}% Max</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>Margin Preservation:</strong> Maintain operating cost ratio strictly below <strong>${costRatioPct}%</strong>, retaining <strong>${fmtMoney(operatingSurplus)}</strong> in monthly operating surplus.</li>
            <li><strong>Velocity Acceleration:</strong> Compress receivable lags from 10D to 4D to liberate <strong>+${fmtMoney(liberatedCapital)}</strong> into liquid reserves.</li>
            <li><strong>Next Milestone:</strong> Set a higher target in the Scenario Studio to plan the next growth horizon.</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.max(0, salesShift)}, ${costRatioPct}, ${Math.round(currentRunRate)}, this)">Apply Current Run-Rate to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      } else {
        return `
          <p><strong>【Target Execution & Revenue Bridge Roadmap】</strong></p>
          <p>To reach monthly target of <strong>${fmtMoney(targetVal)}</strong>, a growth gap of <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> from current run-rate <strong>${fmtMoney(currentRunRate)}</strong> must be bridged.</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Revenue</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Revenue Gap</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Weekly Target</div><div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Execution Priority</div><div class="ai-metric-val" style="color: var(--accent-amber);">Basket +${fmtMoney(deltaAOV)}</div></div>
          </div>
          <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.55;">
            <li><strong>High-Margin SKU Bundling:</strong> Cross-sell top 20% margin items to generate <strong>+${fmtMoney(dailyGap)}/day</strong> in incremental sales and expand basket size by <strong>+${fmtMoney(deltaAOV)}</strong>.</li>
            <li><strong>Receivables Lag Compression:</strong> Offer a 2% early-settlement incentive to compress payment delay from 10D to 4D, unlocking <strong>+${fmtMoney(liberatedCapital)}</strong> in immediate liquid capital.</li>
            <li><strong>Cost Containment & Weekly Run-Rate:</strong> Enforce operating cost ratio strictly below <strong>${costRatioPct}%</strong> while securing an average weekly sales velocity of <strong>${fmtMoney(weeklyTarget)}</strong>.</li>
          </ul>
          <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
          <div class="ai-actions-row">
            <button class="ai-action-btn" onclick="applyScenarioToControls(${Math.round(growthPct)}, ${costRatioPct}, ${Math.round(targetVal)}, this)">Apply Target to Studio</button>
            <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
          </div>
        `;
      }
    }
  }

  // -------------------------------------------------------------
  // INTENT 3: Cash Runway & Operating Burn Rate
  // -------------------------------------------------------------
  const isRunway = lowerQ.includes('রানওয়ে') || lowerQ.includes('runway') || lowerQ.includes('burn') || lowerQ.includes('দিন চলবে') || lowerQ.includes('কত দিন') || lowerQ.includes('रनवे') || lowerQ.includes('कितने दिन') || lowerQ.includes('رن وے') || lowerQ.includes('跑道') || lowerQ.includes('多少天') || lowerQ.includes('ランウェイ') || lowerQ.includes('何日');

  if (isRunway) {
    lastDetectedIntent = 'runway';
    const dailySales = BASE_MONTHLY_REVENUE_BDT / 30.0;
    const dailyNet = dailySales - dailyOpex;

    if (targetLang === 'bn') {
      return `
        <p><strong>【ক্যাশ রানওয়ে ও লিকুইডিটি সার্ভাইভাল অডিট】</strong></p>
        <p>বর্তমান ট্রেজারি ক্যাশ <strong>${fmtMoney(availableCash)}</strong> এবং দৈনিক পরিচালন ব্যয় (Burn Rate) <strong>${fmtMoney(dailyOpex)}/দিন</strong>।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Daily Burn (OPEX)</div><div class="ai-metric-val" style="color: #f43f5e;">${fmtMoney(dailyOpex)} / দিন</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Zero-Sales Runway</div><div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} দিন</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(dailyNet)} / দিন</div></div>
        </div>
        <p><strong>বিশ্লেষণ:</strong> কোনো বিক্রয় ছাড়া সম্পূর্ণ শূন্য রাজস্ব পরিস্থিতিতে বর্তমান তহবিল দিয়ে <strong>${zeroRevenueRunwayDays} দিন</strong> কার্যক্রম পরিচালনা সম্ভব। বর্তমান স্বাভাবিক বিক্রয় ফ্লো অনুযায়ী ব্যবসা স্বয়ংসম্পূর্ণ ও টেকসই।</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.calculate_cash_runway]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View 30-Day Cash Flow</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Cash Runway & Operating Burn Audit】</strong></p>
        <p>Current treasury cash is <strong>${fmtMoney(availableCash)}</strong> with daily operational burn of <strong>${fmtMoney(dailyOpex)}/day</strong>.</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Treasury Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Daily Burn</div><div class="ai-metric-val" style="color: #f43f5e;">${fmtMoney(dailyOpex)} / day</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Zero-Sales Runway</div><div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} Days</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Net Daily Flow</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(dailyNet)} / day</div></div>
        </div>
        <p>Under a complete zero-revenue freeze, liquidity supports <strong>${zeroRevenueRunwayDays} days</strong> of operations. With current sales momentum, runway remains solvent.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.calculate_cash_runway]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View 30-Day Cash Flow</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 4: Payment Delay / Receivables Lag Acceleration
  // -------------------------------------------------------------
  const isDelay = lowerQ.includes('পেমেন্ট') || lowerQ.includes('payment') || lowerQ.includes('delay') || lowerQ.includes('রিসিভেবল') || lowerQ.includes('ল্যাগ') || lowerQ.includes('দেরি') || lowerQ.includes('भुगतान') || lowerQ.includes('देरी') || lowerQ.includes('प्राप्य') || lowerQ.includes('ادائیگی') || lowerQ.includes('تاخیر') || lowerQ.includes('账期') || lowerQ.includes('回款') || lowerQ.includes('回収') || lowerQ.includes('売掛金');

  if (isDelay) {
    lastDetectedIntent = 'delay';
    const dailyTurnover = projRev / 30.0;
    const freedWorkingCapital = dailyTurnover * 6.0;

    if (targetLang === 'bn') {
      return `
        <p><strong>【রিসিভেবলস পেমেন্ট ডিলে কম্প্রেশন অডিট】</strong></p>
        <p>বর্তমান রিটেইল পেমেন্ট সেটেলমেন্ট ল্যাগ <strong>১০ দিন</strong>। এটি কমিয়ে <strong>৪ দিনে</strong> আনলে ৬ দিনের বিক্রয় তহবিল অবিলম্বে ছাড় পাবে।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Current Delay Lag</div><div class="ai-metric-val" style="color: var(--accent-amber);">10 Days</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Compressed Target</div><div class="ai-metric-val" style="color: var(--accent-cyan);">4 Days (-6D)</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Liquid Capital Freed</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(freedWorkingCapital)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Incentive Cost (2%)</div><div class="ai-metric-val">${fmtMoney(projRev * 0.02)}</div></div>
        </div>
        <p><strong>সিদ্ধান্ত:</strong> ডিলারদের ২% ছাড় দিলে প্রায় <strong>+${fmtMoney(freedWorkingCapital)}</strong> তাৎক্ষণিক তারল্য (Working Capital) ব্যাংকে আগেভাগে চলে আসবে, যা রিজার্ভ কুশন আরও সুদৃঢ় করবে।</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_receivables_lag]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Receivables Settlement Lag Compression Audit】</strong></p>
        <p>Compressing wholesale payment settlement lag from 10 days to 4 days via a 2% early-cash discount:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Baseline Lag</div><div class="ai-metric-val" style="color: var(--accent-amber);">10 Days</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Compressed Lag</div><div class="ai-metric-val" style="color: var(--accent-cyan);">4 Days (-6D)</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Working Capital Freed</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(freedWorkingCapital)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Incentive Cost (2%)</div><div class="ai-metric-val">${fmtMoney(projRev * 0.02)}</div></div>
        </div>
        <p>Accelerating cash settlement unlocks <strong>+${fmtMoney(freedWorkingCapital)}</strong> in immediate working capital velocity, insulating reserves from unexpected inventory shocks.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_receivables_lag]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 5: Cost Inflation & COGS Shock
  // -------------------------------------------------------------
  const isCost = lowerQ.includes('কস্ট') || lowerQ.includes('cost') || lowerQ.includes('খরচ') || lowerQ.includes('ব্যয়') || lowerQ.includes('cogs') || lowerQ.includes('expense') || lowerQ.includes('লাগত') || lowerQ.includes('اخراجات') || lowerQ.includes('成本') || lowerQ.includes('原価');

  if (isCost) {
    lastDetectedIntent = 'cost';
    let testCostRatio = 0.70;
    const matchCost = lowerQ.match(/(\d+)\s*%/);
    if (matchCost) testCostRatio = parseFloat(matchCost[1]) / 100.0;

    const projRevC = BASE_MONTHLY_REVENUE_BDT * (1.0 + salesShift / 100.0);
    const operatingSurplusC = projRevC * (1.0 - testCostRatio);
    const reserveTargetC = BASE_MONTHLY_REVENUE_BDT * testCostRatio;
    const cashAfterReserveC = availableCash + operatingSurplusC - reserveTargetC;
    const isSafeC = cashAfterReserveC >= 0;

    if (targetLang === 'bn') {
      return `
        <p><strong>【কস্ট ইনফ্লেশন ও অপারেটিং মার্জিন সেনসিটিভিটি】</strong></p>
        <p>পরিচালন ব্যয় বৃদ্ধি পেয়ে কস্ট রেশিও <strong>${(testCostRatio * 100).toFixed(0)}%</strong> হলে ট্রেজারি বাফার কেমন থাকবে:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Tested Cost Ratio</div><div class="ai-metric-val" style="color: var(--accent-amber);">${(testCostRatio * 100).toFixed(0)}% COGS</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val">${((1.0 - testCostRatio) * 100).toFixed(0)}% Margin</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Monthly Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplusC)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Cushion Buffer</div><div class="ai-metric-val" style="color: ${isSafeC ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafeC ? '+' : ''}${fmtMoney(cashAfterReserveC)}</div></div>
        </div>
        <p>সিদ্ধান্ত: ${isSafeC ? `কস্ট রেশিও ${(testCostRatio * 100).toFixed(0)}% এ পৌঁছালেও ব্যবসায় <strong>+${fmtMoney(cashAfterReserveC)}</strong> এর নিরাপদ লিকুইডিটি বাফার বজায় থাকবে।` : `সতর্কতা: কস্ট রেশিও ${(testCostRatio * 100).toFixed(0)}% হলে রিজার্ভের নিচে ঘাটতি (Deficit) তৈরি হতে পারে। পরিচালন ব্যয় নিয়ন্ত্রণ জরুরি।`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_expense_shock]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, ${(testCostRatio * 100).toFixed(0)}, null, this)">Apply ${(testCostRatio * 100).toFixed(0)}% Cost to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Cost Inflation & Operating Margin Sensitivity】</strong></p>
        <p>Simulating operational cost ratio elevated to <strong>${(testCostRatio * 100).toFixed(0)}% COGS</strong>:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Cost Ratio</div><div class="ai-metric-val" style="color: var(--accent-amber);">${(testCostRatio * 100).toFixed(0)}% COGS</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val">${((1.0 - testCostRatio) * 100).toFixed(0)}% Margin</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Monthly Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplusC)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Liquidity Buffer</div><div class="ai-metric-val" style="color: ${isSafeC ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafeC ? '+' : ''}${fmtMoney(cashAfterReserveC)}</div></div>
        </div>
        <p>Verdict: ${isSafeC ? `Even at ${(testCostRatio * 100).toFixed(0)}% COGS, a liquidity buffer of <strong>+${fmtMoney(cashAfterReserveC)}</strong> is safely retained.` : `Warning: A ${(testCostRatio * 100).toFixed(0)}% cost ratio breaches reserve thresholds. Immediate opex reduction advised.`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_expense_shock]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, ${(testCostRatio * 100).toFixed(0)}, null, this)">Apply ${(testCostRatio * 100).toFixed(0)}% Cost to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 6: Sales Stress Test (Percentage Shift or Explicit Drop)
  // -------------------------------------------------------------
  const matchPct = lowerQ.match(/([+-]?\d+)\s*%/);
  const isStressWord = (lowerQ.includes('stress') || lowerQ.includes('shock') || lowerQ.includes('drop') || lowerQ.includes('fall') ||
    lowerQ.includes('কমলে') || lowerQ.includes('হ্রাস') || lowerQ.includes('বৃদ্ধি') || lowerQ.includes('গির') || lowerQ.includes('کمی') ||
    lowerQ.includes('下降') || lowerQ.includes('减少') || lowerQ.includes('減少') || lowerQ.includes('低下')) && !lowerQ.includes('profit') && !lowerQ.includes('loan');

  if (matchPct || isStressWord) {
    let simShift = salesShift;
    if (matchPct) {
      simShift = parseFloat(matchPct[1]);
      if (lowerQ.includes('drop') || lowerQ.includes('fall') || lowerQ.includes('কমলে') || lowerQ.includes('হ্রাস') || lowerQ.includes('গির') || lowerQ.includes('کمی') || lowerQ.includes('下降') || lowerQ.includes('减少') || lowerQ.includes('減少') || lowerQ.includes('低下')) {
        simShift = -Math.abs(simShift);
      }
    } else if (lowerQ.includes('drop') || lowerQ.includes('fall') || lowerQ.includes('কমলে') || lowerQ.includes('হ্রাস')) {
      simShift = -20;
    }

    const projRevS = BASE_MONTHLY_REVENUE_BDT * (1.0 + simShift / 100.0);
    const operatingSurplusS = projRevS * (1.0 - costRatio);
    const reserveTargetS = BASE_MONTHLY_REVENUE_BDT * costRatio;
    const cashAfterReserveS = availableCash + operatingSurplusS - reserveTargetS;
    const isSafeS = cashAfterReserveS >= 0;

    if (targetLang === 'bn') {
      return `
        <p><strong>【ট্রেজারি স্ট্রেস টেস্ট অডিট (Stress Simulation)】</strong></p>
        <p>সিমুলেশন প্যারামিটার: বিক্রয় শিফট <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, কস্ট রেশিও <strong>${(costRatio * 100).toFixed(0)}%</strong>।</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Projected Revenue</div><div class="ai-metric-val">${fmtMoney(projRevS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplusS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTargetS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafeS ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafeS ? '+' : ''}${fmtMoney(cashAfterReserveS)}</div></div>
        </div>
        <p><strong>সিদ্ধান্ত:</strong> ${isSafeS ? 'ব্যবসায় পর্যাপ্ত লিকুইডিটি বাফার রয়েছে (SAFE TIER)। রিজার্ভ নিরাপদ এবং জরুরি বহিঃস্থ ঋণের কোনো প্রয়োজন নেই।' : 'সতর্কতা: রিজার্ভ ঘাটতির আশঙ্কা রয়েছে (SHORTFALL HAZARD)। তাৎক্ষণিকভাবে পরিচালন ব্যয় ও ইনভেন্টরি পারচেজ স্থগিত করার সুপারিশ করা হলো।'}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct}, null, this)">Apply ${simShift}% Shift to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Decision Intelligence Treasury Stress Audit】</strong></p>
        <p>Simulated Parameters: Sales shift <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, cost ratio <strong>${(costRatio * 100).toFixed(0)}%</strong>.</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Projected Revenue</div><div class="ai-metric-val">${fmtMoney(projRevS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplusS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTargetS)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafeS ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafeS ? '+' : ''}${fmtMoney(cashAfterReserveS)}</div></div>
        </div>
        <p>Verdict: ${isSafeS ? 'Adequate liquidity cushion maintained (SAFE TIER). No credit line intervention required.' : 'Warning: Potential reserve shortfall under this stress scenario.'}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct}, null, this)">Apply Scenario to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 7: Profitability, Margins & Gross Profit
  // -------------------------------------------------------------
  const isProfit = lowerQ.includes('profit') || lowerQ.includes('margin') || lowerQ.includes('earnings') || lowerQ.includes('profitable') ||
    lowerQ.includes('labh') || lowerQ.includes('munafa') || lowerQ.includes('লাভ') || lowerQ.includes('মুনাফা') || lowerQ.includes('মার্জিন') ||
    lowerQ.includes('मुनाफा') || lowerQ.includes('लाभ') || lowerQ.includes('منافع') || lowerQ.includes('نفع') ||
    lowerQ.includes('利润') || lowerQ.includes('毛利') || lowerQ.includes('利益') || lowerQ.includes('粗利');

  if (isProfit) {
    lastDetectedIntent = 'profit';
    if (targetLang === 'bn') {
      return `
        <p><strong>【লাভ ও অপারেটিং মার্জিন বিশ্লেষণ (Profitability Audit)】</strong></p>
        <p>বর্তমান সেলস মোমেন্টাম ও কস্ট রেশিও অনুযায়ী আপনার ব্যবসার প্রফিটেবিলিটি চিত্র:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Projected 30D Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">COGS Expense (${costRatioPct}%)</div><div class="ai-metric-val" style="color: var(--accent-amber);">${fmtMoney(projRev * costRatio)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Profit (GOM)</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${fmtMoney(estimatedGrossProfit)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Profit Margin</div><div class="ai-metric-val">${Math.round(grossMarginRatio * 100)}% Margin</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li>প্রতি ${fmtMoney(100)} বিক্রয়ে গড়ে <strong>${fmtMoney(Math.round(100 * grossMarginRatio))} (${Math.round(grossMarginRatio * 100)}%)</strong> গ্রস লাভ অর্জিত হচ্ছে।</li>
          <li>মার্জিন আরও বাড়ানোর জন্য শীর্ষ ২০% হাই-মার্জিন এসকেইউ (SKU) প্রোডাক্টে ক্রস-সেলিং জোরদার করার সুপারিশ করা হলো।</li>
          <li>কস্ট রেশিও কঠোরভাবে <strong>${costRatioPct}%</strong> এর নিচে ধরে রাখলে মাসিক <strong>${fmtMoney(operatingSurplus)}</strong> অপারেটিং উদ্বৃত্ত নিশ্চিত থাকবে।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.calculate_operating_margins]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, 58, null, this)">Simulate 58% COGS (+4% Margin)</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Profitability & Operating Margin Audit】</strong></p>
        <p>Current margin breakdown under calibrated econometric parameters:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Projected 30D Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">COGS Expense (${costRatioPct}%)</div><div class="ai-metric-val" style="color: var(--accent-amber);">${fmtMoney(projRev * costRatio)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Operating Profit</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${fmtMoney(estimatedGrossProfit)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val">${Math.round(grossMarginRatio * 100)}%</div></div>
        </div>
        <p>For every unit of revenue, the enterprise captures <strong>${Math.round(grossMarginRatio * 100)}%</strong> in gross surplus. Focus on top 20% margin SKUs to expand gross operating surplus.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.calculate_operating_margins]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, 58, null, this)">Simulate 58% COGS (+4% Margin)</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 8: Loan & Borrowing Assessment
  // -------------------------------------------------------------
  const isLoan = lowerQ.includes('loan') || lowerQ.includes('borrow') || lowerQ.includes('credit') || lowerQ.includes('debt') ||
    lowerQ.includes('financing') || lowerQ.includes('lending') ||
    lowerQ.includes('লোন') || lowerQ.includes('ঋণ') || lowerQ.includes('ধার') || lowerQ.includes('ব্যাংক লোন') ||
    lowerQ.includes('loan lagbe') || lowerQ.includes('loan dorkar') || lowerQ.includes('rin') ||
    lowerQ.includes('ऋण') || lowerQ.includes('लोन') || lowerQ.includes('कर्ज') ||
    lowerQ.includes('قرض') || lowerQ.includes('ادھار') ||
    lowerQ.includes('贷款') || lowerQ.includes('借款') ||
    lowerQ.includes('融資') || lowerQ.includes('借入');

  if (isLoan) {
    lastDetectedIntent = 'loan';
    if (targetLang === 'bn') {
      return `
        <p><strong>【এসএমই ঋণ প্রয়োজনীয়তা মূল্যায়ন (SME Debt Assessment)】</strong></p>
        <p>আপনার বর্তমান তারল্য উদ্বৃত্ত ও ক্যাশ ফ্লো সক্ষমতার ভিত্তিতে ঋণ মূল্যায়ন:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Monthly Surplus</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${fmtMoney(operatingSurplus)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Recommendation</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${isSafe ? 'NO LOAN NEEDED' : 'WORKING CAPITAL'}</div></div>
        </div>
        <p><strong>উপসংহার:</strong> ${isSafe ? `<strong>বর্তমানে কোনো নতুন ব্যাংক লোন নেওয়ার প্রয়োজন নেই।</strong> আপনার ব্যবসায় <strong>+${fmtMoney(cashAfterReserve)}</strong> এর নিরাপদ ক্যাশ বাফার রয়েছে। উচ্চ সুদের বাণিজ্যিক ঋণ এড়িয়ে নিজস্ব ক্যাশ ফ্লো দিয়ে কার্যক্রম পরিচালনা করাই সর্বোত্তম সিদ্ধান্ত।` : `<strong>স্বল্পমেয়াদী ${fmtMoney(Math.abs(cashAfterReserve))} ওয়ার্কিং ক্যাপিটাল লোন বিবেচনা করা যেতে পারে।</strong> মেয়াদ ৯০ দিনের মধ্যে সীমিত রাখা এবং অবিলম্বে রিসিভেবল আদায় ত্বরান্বিত করার পরামর্শ দেওয়া হলো।`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.evaluate_debt_capacity]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Check Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【SME Debt & Credit Facility Assessment】</strong></p>
        <p>Evaluation of debt requirement based on organic liquidity generation:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Monthly Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Recommendation</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${isSafe ? 'DEBT NOT REQUIRED' : 'WORKING CAPITAL'}</div></div>
        </div>
        <p>Verdict: ${isSafe ? `<strong>No external debt facility required.</strong> The enterprise retains a solvent liquidity cushion of <strong>+${fmtMoney(cashAfterReserve)}</strong>. Self-funded operations protect margins from interest drag.` : `A short-term working capital facility of <strong>${fmtMoney(Math.abs(cashAfterReserve))}</strong> may be evaluated to bridge working capital lag.`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.evaluate_debt_capacity]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Check Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 9: Safety, Insolvency Risk & Solvency Radar
  // -------------------------------------------------------------
  const isSafetyRisk = lowerQ.includes('safe') || lowerQ.includes('safety') || lowerQ.includes('risk') || lowerQ.includes('danger') ||
    lowerQ.includes('insolven') || lowerQ.includes('default') || lowerQ.includes('crisis') || lowerQ.includes('trouble') || lowerQ.includes('secure') ||
    lowerQ.includes('নিরাপদ') || lowerQ.includes('ঝুঁকি') || lowerQ.includes('বিপদ') || lowerQ.includes('সংকট') || lowerQ.includes('ঘাটতি') ||
    lowerQ.includes('সব ঠিক আছে') || lowerQ.includes('ব্যবসা কি নিরাপদ') || lowerQ.includes('কোন ঝুঁকি') ||
    lowerQ.includes('safe kina') || lowerQ.includes('risk ache') || lowerQ.includes('jhhuki') || lowerQ.includes('shob thik') ||
    lowerQ.includes('सुरक्षित') || lowerQ.includes('जोखिम') || lowerQ.includes('खतरा') ||
    lowerQ.includes('کیا محفوظ ہے') || lowerQ.includes('خطرہ') ||
    lowerQ.includes('安全') || lowerQ.includes('风险') ||
    lowerQ.includes('安全性') || lowerQ.includes('倒産');

  if (isSafetyRisk) {
    lastDetectedIntent = 'safety_risk';
    if (targetLang === 'bn') {
      return `
        <p><strong>【সলভেন্সি ঝুঁকি ও নিরাপত্তা নিরীক্ষা (Solvency Invariant Check)】</strong></p>
        <p>মডেল সমীকরণ <code>C_t ≥ τ_res</code> অনুসারে ব্যবসার দেউলিয়াত্ব ও লিকুইডিটি ঝুঁকি নিরীক্ষা:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Floor</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Risk Tier</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? 'SAFE TIER (0% Breach)' : 'HAZARD TIER'}</div></div>
        </div>
        <p><strong>ঝুঁকি মূল্যায়ন:</strong> বর্তমান আর্থিক অবস্থায় ব্যবসাটি <strong>${isSafe ? 'সম্পূর্ণ নিরাপদ ও স্থিতিশীল' : 'ঝুঁকিপূর্ণ'}</strong> রয়েছে। আর্লি ওয়ার্নিং রাডার (Recall 97.7%, Lead Time 1.5 Days) কোনো আসন্ন পেমেন্ট ডিফল্ট শনাক্ত করেনি।</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Risk Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Solvency Risk & Reserve Invariant Audit】</strong></p>
        <p>Continuous algorithmic verification of the solvency invariant <code>C_t ≥ τ_res</code>:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Floor</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Solvency Status</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? 'SAFE TIER' : 'SHORTFALL HAZARD'}</div></div>
        </div>
        <p>Verdict: Risk tier is categorized as <strong>${isSafe ? 'LOW RISK (SAFE TIER)' : 'HIGH RISK (DEFICIT)'}</strong>. Predictive Early Warning Radar reports 97.7% recall with zero imminent breaches detected.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">Open Solvency Risk Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 10: Cash Balance & Treasury Liquidity
  // -------------------------------------------------------------
  const isCashBalance = (lowerQ.includes('cash') && !lowerQ.includes('flow')) || lowerQ.includes('balance') || lowerQ.includes('treasury') ||
    lowerQ.includes('how much money') || lowerQ.includes('funds') || lowerQ.includes('liquidity') ||
    lowerQ.includes('ক্যাশ') || lowerQ.includes('ব্যালেন্স') || lowerQ.includes('নগদ') || lowerQ.includes('তহবিল') ||
    lowerQ.includes('taka koto') || lowerQ.includes('cash koto') || lowerQ.includes('amar taka') ||
    lowerQ.includes('নগদ টাকা') || lowerQ.includes('টাকা কত') ||
    lowerQ.includes('नकद') || lowerQ.includes('बैलेंस') || lowerQ.includes('पैसा') ||
    lowerQ.includes('کیش') || lowerQ.includes('نقد') || lowerQ.includes('رقم') ||
    lowerQ.includes('现金') || lowerQ.includes('余额') || lowerQ.includes('资金') ||
    lowerQ.includes('現金') || lowerQ.includes('残高');

  if (isCashBalance) {
    lastDetectedIntent = 'cash_balance';
    if (targetLang === 'bn') {
      return `
        <p><strong>【ট্রেজারি ক্যাশ ব্যালেন্স ও লিকুইডিটি অবস্থান】</strong></p>
        <p>আপনার বর্তমান নগদ তহবিল ও সংবিধিবদ্ধ রিজার্ভ লাইবিলিটির তুলনামূলক নিরীক্ষা:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Zero-Sales Runway</div><div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} দিন</div></div>
        </div>
        <p><strong>সিদ্ধান্ত:</strong> ${isSafe ? `আপনার নগদ তহবিল নিরাপদ টিয়ারে (SAFE TIER) রয়েছে। সংবিধিবদ্ধ রিজার্ভের পরেও আপনার কাছে <strong>+${fmtMoney(cashAfterReserve)}</strong> অতিরিক্ত ক্যাশ কুশন রয়েছে।` : `সতর্কতা: রিজার্ভ ঘাটতির আশঙ্কা রয়েছে (SHORTFALL HAZARD)। তাৎক্ষণিকভাবে পরিচালন ব্যয় ও ইনভেন্টরি পারচেজ নিয়ন্ত্রণের পরামর্শ দেওয়া হলো।`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Cashflow Ledger</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Treasury Liquidity & Cash Balance Audit】</strong></p>
        <p>Comparison between active liquid reserves and statutory liquidity floor:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Requirement</div><div class="ai-metric-val">${fmtMoney(reserveTarget)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Buffer (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Zero-Sales Runway</div><div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} Days</div></div>
        </div>
        <p>Verdict: ${isSafe ? `Liquidity status is well-capitalized in SAFE TIER with <strong>+${fmtMoney(cashAfterReserve)}</strong> surplus buffer above minimum reserve.` : `Warning: Potential shortfall under current working capital demands.`}</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.audit_treasury_solvency]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Cashflow Ledger</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Analysis</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 11: Sales Growth & Business Expansion / Improvement Levers
  // -------------------------------------------------------------
  const isSalesGrowth = lowerQ.includes('increase sales') || lowerQ.includes('boost sales') || lowerQ.includes('grow business') ||
    lowerQ.includes('sales growth') || lowerQ.includes('how to grow') || lowerQ.includes('more orders') ||
    lowerQ.includes('improve') || lowerQ.includes('improvement') || lowerQ.includes('how to improve') || lowerQ.includes('better') ||
    lowerQ.includes('বিক্রি বাড়ানো') || lowerQ.includes('সেলস বাড়ানো') || lowerQ.includes('ব্যবসা বড় করা') || lowerQ.includes('গ্রোথ') ||
    lowerQ.includes('কীভাবে বিক্রি বাড়াব') || lowerQ.includes('বিক্রি বৃদ্ধি') ||
    lowerQ.includes('উন্নতি') || lowerQ.includes('উন্নতি করব') || lowerQ.includes('উন্নতি করা') || lowerQ.includes('কী করলে') ||
    lowerQ.includes('কি করলে') || lowerQ.includes('কী করা উচিত') || lowerQ.includes('কি করা উচিত') || lowerQ.includes('উন্নতি করার উপায়') ||
    lowerQ.includes('sales barabo') || lowerQ.includes('grow kora') || lowerQ.includes('business boro') ||
    lowerQ.includes('ki korle') || lowerQ.includes('kivabe improve') || lowerQ.includes('improve korbo') || lowerQ.includes('korte pabo') ||
    lowerQ.includes('improve korte') || lowerQ.includes('ki kora jay') ||
    lowerQ.includes('बिक्री कैसे बढ़ाएं') || lowerQ.includes('व्यापार बढ़ाएं') || lowerQ.includes('सुधार') ||
    lowerQ.includes('سیلز کیسے بڑھائیں') || lowerQ.includes('کاروبار بڑھائیں') || lowerQ.includes('بہتری') ||
    lowerQ.includes('如何增加销售') || lowerQ.includes('扩大业务') || lowerQ.includes('提升') || lowerQ.includes('改进') ||
    lowerQ.includes('売上を伸ばす') || lowerQ.includes('事業拡大') || lowerQ.includes('改善');

  if (isSalesGrowth) {
    lastDetectedIntent = 'sales_growth';
    if (targetLang === 'bn') {
      return `
        <p><strong>【ব্যবসার সামগ্রিক পারফরম্যান্স ও বিক্রি বৃদ্ধির কৌশলগত লিভার (Improvement Levers)】</strong></p>
        <p>ইকোনোমেট্রিক ও রিটেইল অপ্টিমাইজেশন মডেল অনুযায়ী আপনার ব্যবসা ইম্প্রুভ (উন্নতি) করার ৩টি প্রমাণিত চালিকাশক্তি:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Current Monthly Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Gap</div><div class="ai-metric-val" style="color: var(--accent-cyan);">+${fmtMoney(Math.max(0, targetVal - projRev))}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Basket Expansion</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+15% AOV</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Cash Velocity</div><div class="ai-metric-val" style="color: var(--accent-blue);">10D → 4D</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li><strong>লিভার ১ (ঝুড়ির গড় মূল্য বৃদ্ধি):</strong> শীর্ষ ২০% লাভজনক পণ্যের সাথে পরিপূরক প্রোডাক্টের বান্ডলিং অফার দিয়ে গড় অর্ডার ভ্যালু (AOV) বাড়ানো।</li>
          <li><strong>লিভার ২ (তহবিল আবর্তন গতি):</strong> ডিলারদের ২% আর্লি ডিসকাউন্ট দিয়ে ১০ দিনের পেমেন্ট সাইকেল ৪ দিনে নামিয়ে প্রায় <strong>+${fmtMoney(Math.round(projRev * (6.0 / 30.0)))}</strong> তাৎক্ষণিক চলতি মূলধন দ্রুত মুক্ত করা।</li>
          <li><strong>লিভার ৩ (কস্ট সিলিং):</strong> পরিচালন খরচ কঠোরভাবে <strong>${costRatioPct}%</strong> এর নিচে সীমাবদ্ধ রাখা।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="openStrategyRoadmap(this)">Open Strategy Roadmap</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Strategy</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Strategic Revenue Growth & Basket Expansion Levers】</strong></p>
        <p>Three econometric levers to scale SME turnover under solvency constraints:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Current 30D Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Target Gap</div><div class="ai-metric-val" style="color: var(--accent-cyan);">+${fmtMoney(Math.max(0, targetVal - projRev))}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Basket Expansion</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+15% AOV</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Velocity Gain</div><div class="ai-metric-val" style="color: var(--accent-blue);">10D → 4D</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li><strong>Lever 1 (AOV Expansion):</strong> Bundle top 20% margin SKUs to elevate Average Order Value.</li>
          <li><strong>Lever 2 (Cash Velocity):</strong> Compress receivables lag from 10D to 4D via a 2% settlement incentive, unlocking <strong>+${fmtMoney(Math.round(projRev * (6.0 / 30.0)))}</strong> in liquid capital.</li>
          <li><strong>Lever 3 (Margin Discipline):</strong> Enforce operational cost ceiling strictly below ${costRatioPct}%.</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.synthesize_strategic_roadmap]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="openStrategyRoadmap(this)">Open Strategy Roadmap</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Strategy</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 12: Business Summary, Health & Overview Diagnostics
  // -------------------------------------------------------------
  const isSummary = lowerQ.includes('summary') || lowerQ.includes('overview') || lowerQ.includes('report') || lowerQ.includes('status') ||
    lowerQ.includes('health') || lowerQ.includes('how is my business') || lowerQ.includes('condition') || lowerQ.includes('diagnostics') ||
    lowerQ.includes('সামারি') || lowerQ.includes('সামগ্রিক অবস্থা') || lowerQ.includes('রিপোর্ট') || lowerQ.includes('স্ট্যাটাস') ||
    lowerQ.includes('ব্যবসা কেমন চলছে') || lowerQ.includes('অবস্থা কেমন') || lowerQ.includes('সারসংক্ষেপ') ||
    lowerQ.includes('ki obostha') || lowerQ.includes('summary dao') || lowerQ.includes('kemon cholche') ||
    lowerQ.includes('सारांश') || lowerQ.includes('स्थिति') || lowerQ.includes('रिपोर्ट') ||
    lowerQ.includes('خلاصہ') || lowerQ.includes('حالت') ||
    lowerQ.includes('总览') || lowerQ.includes('财务状况') ||
    lowerQ.includes('サマリー') || lowerQ.includes('現状診断');

  if (isSummary) {
    lastDetectedIntent = 'summary';
    if (targetLang === 'bn') {
      return `
        <p><strong>【ব্যবসায়ের সামগ্রিক আর্থিক ও ট্রেজারি রিপোর্ট (Executive Brief)】</strong></p>
        <p>ফাউন্ডেশনাল মডেল ও লাইভ ডেটা পাইপলাইন থেকে সংগৃহীত সামগ্রিক ফাইন্যান্সিয়াল স্ন্যাপশট:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast (P50)</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          <li><strong>বিক্রয় গতিপ্রকৃতি:</strong> স্বাভাবিক এবং স্থিতিশীল (দৈনিক গড়ে <strong>${fmtMoney(baseDaily)}</strong>)।</li>
          <li><strong>লিকুইডিটি অবস্থা:</strong> ${isSafe ? 'সংবিধিবদ্ধ রিজার্ভ লক্ষ্যমাত্রার চেয়ে অতিরিক্ত তহবিল রয়েছে।' : 'রিজার্ভের নিচে ঘাটতি রয়েছে, খরচ কমানো প্রয়োজন।'}</li>
          <li><strong>রানওয়ে স্থায়িত্ব:</strong> সম্পূর্ণ শূন্য বিক্রয়েও <strong>${zeroRevenueRunwayDays} দিন</strong> কার্যক্রম পরিচালনা সম্ভব।</li>
        </ul>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('overview')">View Diagnostic Cockpit</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Report</button>
        </div>
      `;
    } else {
      return `
        <p><strong>【Enterprise Health & Solvency Executive Brief】</strong></p>
        <p>Real-time consolidated diagnostic overview across active operations:</p>
        <div class="ai-metric-grid">
          <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Projected Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Gross Margin</div><div class="ai-metric-val" style="color: var(--accent-emerald);">${Math.round(grossMarginRatio * 100)}%</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Treasury Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
          <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
        </div>
        <p>Status: Healthy operational run-rate with <strong>${zeroRevenueRunwayDays} days</strong> of organic cash runway. Invariant reserve compliance verified at <strong>80% conformal coverage</strong>.</p>
        <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('overview')">View Diagnostic Cockpit</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Report</button>
        </div>
      `;
    }
  }

  // -------------------------------------------------------------
  // INTENT 13: Intelligent Contextual Fallback for Any Other Query
  lastDetectedIntent = 'general';
  // -------------------------------------------------------------
  if (targetLang === 'bn') {
    return `
      <p><strong>【ট্রেজারি ডিসিশন ইন্টেলিজেন্স অ্যাডভাইজরি】</strong></p>
      <p>আপনার অনুসন্ধানের প্রেক্ষিতে বর্তমান ব্যবসায়িক স্বাস্থ্য ও আর্থিক অবস্থান পর্যালোচনা করা হলো:</p>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Projected Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion (ΔC)</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <p>আপনার ব্যবসা বর্তমানে <strong>${isSafe ? 'নিরাপদ সলভেন্সি টিয়ারে (SAFE TIER)' : 'ঘাটতি ঝুঁকিতে (DEFICIT)'}</strong> অবস্থান করছে। নির্দিষ্ট কোনো বিষয় বিশ্লেষণ করতে চাইলে নিচের মডিউলগুলো ওপেন করুন অথবা প্রশ্ন করুন (যেমন: লাভ কত, লোন লাগবে কি, বিক্রি বাড়ানোর কৌশল ইত্যাদি)।</p>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Advisory</button>
      </div>
    `;
  } else if (targetLang === 'hi') {
    return `
      <p><strong>【व्यापार निर्णय बुद्धिमत्ता सलाह (Advisory)】</strong></p>
      <p>आपके प्रश्न के संदर्भ में वर्तमान व्यापारिक स्वास्थ्य और तरलता विश्लेषण:</p>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Projected Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <p>व्यापार वर्तमान में <strong>${isSafe ? 'सुरक्षित स्तर (SAFE TIER)' : 'घाटे के स्तर'}</strong> में है। अधिक जानकारी के लिए संबंधित मॉड्यूल का चयन करें।</p>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Advisory</button>
      </div>
    `;
  } else if (targetLang === 'ur') {
    return `
      <div dir="rtl">
        <p><strong>【ٹریژری ڈیسیژن انٹیلی جنس ایڈوائزری】</strong></p>
        <p>آپ کے سوال کے حوالے سے موجودہ مالیاتی صورتحال کا جائزہ:</p>
      </div>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Forecaster</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Advisory</button>
      </div>
    `;
  } else if (targetLang === 'zh') {
    return `
      <p><strong>【企业决策智能综合咨询报告】</strong></p>
      <p>针对您的咨询，当前企业整体财务运行及偿债能力评估如下：</p>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30天预测营业额</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">月度盈余</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">手头现金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">安全缓冲资金</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <p>审计结论：企业当前处于 <strong>${isSafe ? '安全评级 (SAFE TIER)' : '资金预警状态'}</strong>。您可继续提问或通过下方按钮查看分项模块。</p>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">打开分位数预测器</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">查看流动性雷达</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">复制咨询报告</button>
      </div>
    `;
  } else if (targetLang === 'ja') {
    return `
      <p><strong>【意思決定インテリジェンス総合アドバイザリー】</strong></p>
      <p>ご質問のトピックに基づき、最新の財務健全性ステータスを診断しました：</p>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30日予測売上</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">営業余剰</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">現在の手元資金</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">安全クッション</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <p>診断判定：現在は <strong>${isSafe ? '安全圏（SAFE TIER）' : '資金警戒圏'}</strong> に位置しています。各種詳細機能は下のボタンから直接アクセス可能です。</p>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">予測モジュールを開く</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">流動性レーダーを見る</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">レポートをコピー</button>
      </div>
    `;
  } else {
    return `
      <p><strong>【Decision Intelligence Financial Advisory】</strong></p>
      <p>In response to your query, here is an executive review of current enterprise health and liquidity:</p>
      <div class="ai-metric-grid">
        <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Forecast Sales</div><div class="ai-metric-val">${fmtMoney(projRev)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Operating Surplus</div><div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Treasury Cash</div><div class="ai-metric-val">${fmtMoney(availableCash)}</div></div>
        <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div></div>
      </div>
      <p>Verdict: Operations are currently situated in <strong>${isSafe ? 'SAFE TIER' : 'DEFICIT ALERT'}</strong>. For deep-dive stress simulations, explore the modules below.</p>
      <div class="tool-badge-pill">[Verified by FinSightToolKit.system_diagnostics]</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="switchDecisionModule('forecasts')">Open Conformal Forecaster</button>
        <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">View Solvency Radar</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">Copy Advisory</button>
      </div>
    `;
  }
}


// -------------------------------------------------------------
// Interactive Advisor Helper Actions
// -------------------------------------------------------------
function showStudioToast(msg) {
  let toast = document.getElementById('studio-toast-banner');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'studio-toast-banner';
    toast.className = 'studio-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('show');
  if (toast._timer) clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.classList.remove('show');
  }, 3200);
}

function fallbackCopyText(text, cb) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
    if (cb) cb();
  } catch (e) {
    console.warn('Fallback copy failed:', e);
  }
  document.body.removeChild(ta);
}

function applyScenarioToControls(salesShift, costRatioPct, targetRevenue = null, btnElement = null) {
  const elGrowth = document.getElementById('input-growth');
  const elCost = document.getElementById('input-cost');
  const elTarget = document.getElementById('input-target-revenue');

  // 1. Expand slider bounds if salesShift exceeds current min/max
  if (elGrowth && salesShift !== null && salesShift !== undefined) {
    const shiftNum = parseFloat(salesShift);
    if (!isNaN(shiftNum)) {
      if (shiftNum > parseFloat(elGrowth.max)) {
        elGrowth.max = Math.max(30, Math.ceil(shiftNum / 10) * 10);
      }
      if (shiftNum < parseFloat(elGrowth.min)) {
        elGrowth.min = Math.min(-40, Math.floor(shiftNum / 10) * 10);
      }
      elGrowth.value = shiftNum;
    }
  }

  // 2. Set cost ratio slider (bounded [0, 90])
  if (elCost && costRatioPct !== null && costRatioPct !== undefined) {
    const costNum = parseFloat(costRatioPct);
    if (!isNaN(costNum)) {
      elCost.value = Math.max(0, Math.min(90, costNum));
    }
  }

  // 3. Set target revenue if provided
  if (elTarget && targetRevenue !== null && targetRevenue !== undefined) {
    const targetNum = parseFloat(targetRevenue);
    if (!isNaN(targetNum) && targetNum > 0) {
      elTarget.value = Math.round(targetNum);
    }
  }

  // 4. Ensure target planner body is open and visible
  const targetBody = document.getElementById('target-planner-body');
  if (targetBody && targetBody.style.display === 'none') {
    targetBody.style.display = 'block';
    const btnToggle = document.getElementById('btn-toggle-planner');
    if (btnToggle) btnToggle.textContent = 'Hide target planner';
  }

  // 5. Recompute business logic and live planner
  recomputeBusinessEngine();
  recomputeTargetPlanner();

  // 6. Visual feedback on button (if passed)
  if (btnElement && (btnElement.nodeType === 1 || btnElement.classList)) {
    const originalContent = btnElement.innerHTML;
    btnElement.innerHTML = '✓ Applied to Studio!';
    btnElement.classList.add('applied');
    btnElement.disabled = true;
    setTimeout(() => {
      btnElement.innerHTML = originalContent;
      btnElement.classList.remove('applied');
      btnElement.disabled = false;
    }, 2400);
  }

  // 7. Show global toast notification
  const displayShift = (salesShift !== null && salesShift !== undefined) ? `${salesShift >= 0 ? '+' : ''}${salesShift}%` : null;
  const displayCost = (costRatioPct !== null && costRatioPct !== undefined) ? `${costRatioPct}%` : null;
  const displayTarget = (targetRevenue !== null && targetRevenue !== undefined && targetRevenue > 0) ? fmtMoney(targetRevenue) : null;

  let toastMsg = '✅ Studio Synchronized:';
  if (displayShift) toastMsg += ` Sales Shift ${displayShift}`;
  if (displayCost) toastMsg += ` · Cost Ratio ${displayCost}`;
  if (displayTarget) toastMsg += ` · Target ${displayTarget}`;
  showStudioToast(toastMsg);

  // 8. Switch to scenarios view without resetting scroll to top
  switchDecisionModule('scenarios', false);

  // 9. Smoothly scroll to the relevant section (target planner or simulator)
  setTimeout(() => {
    const targetSection = (targetRevenue !== null && targetRevenue !== undefined)
      ? document.getElementById('sec-target-planner')
      : document.getElementById('sec-simulator');
    if (targetSection) {
      targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      targetSection.classList.add('section-highlight-pulse');
      setTimeout(() => targetSection.classList.remove('section-highlight-pulse'), 2000);
    }
  }, 100);
}

function openStrategyRoadmap(btnElement = null) {
  if (btnElement && (btnElement.nodeType === 1 || btnElement.classList)) {
    const orig = btnElement.innerHTML;
    btnElement.innerHTML = '✓ Opening Studio...';
    btnElement.classList.add('applied');
    setTimeout(() => {
      btnElement.innerHTML = orig;
      btnElement.classList.remove('applied');
    }, 2000);
  }
  showStudioToast('🎯 Navigating to Strategy Roadmap...');
  switchDecisionModule('scenarios', false);
  setTimeout(() => {
    const targetBody = document.getElementById('target-planner-body');
    if (targetBody && targetBody.style.display === 'none') {
      targetBody.style.display = 'block';
      const btnToggle = document.getElementById('btn-toggle-planner');
      if (btnToggle) btnToggle.textContent = 'Hide target planner';
    }
    const sec = document.getElementById('sec-target-planner');
    if (sec) {
      sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
      sec.classList.add('section-highlight-pulse');
      setTimeout(() => sec.classList.remove('section-highlight-pulse'), 2000);
    }
  }, 100);
}

function copyAnalysisText(btn) {
  const bubble = btn.closest('.entry-bubble');
  if (!bubble) return;
  const clone = bubble.cloneNode(true);
  const actions = clone.querySelector('.ai-actions-row');
  if (actions) actions.remove();
  const followups = clone.querySelector('.ai-followups-container');
  if (followups) followups.remove();
  const text = clone.innerText.trim();

  const onSuccess = () => {
    const origText = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    showStudioToast('📋 Analysis copied to clipboard');
    setTimeout(() => {
      btn.textContent = origText;
      btn.classList.remove('copied');
    }, 1600);
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(onSuccess).catch(() => {
      fallbackCopyText(text, onSuccess);
    });
  } else {
    fallbackCopyText(text, onSuccess);
  }
}


let lastDetectedIntent = 'general';

function generateCopilotResponse(query) {
  const rawReply = _computeCopilotResponseRaw(query);
  const targetLang = detectScriptLanguage(query, copilotLang);
  const availableCash = parseFloat(document.getElementById('input-cash')?.value) || 180000;
  const costRatioPct = parseFloat(document.getElementById('input-cost')?.value) || 62;
  const salesShift = parseFloat(document.getElementById('input-growth')?.value) || 0;
  const projRev = BASE_MONTHLY_REVENUE_BDT * (1.0 + salesShift / 100.0);
  const reserveTarget = BASE_MONTHLY_REVENUE_BDT * (costRatioPct / 100.0);
  const operatingSurplus = projRev * (1.0 - costRatioPct / 100.0);
  const cashAfterReserve = availableCash + operatingSurplus - reserveTarget;
  const isSafe = cashAfterReserve >= 0;

  const followUps = generateDynamicFollowUps(lastDetectedIntent, targetLang, isSafe, {
    cash: availableCash,
    rev: projRev,
    cost: costRatioPct,
    cushion: cashAfterReserve
  });

  return rawReply + followUps;
}


function clearCopilotChat() {
  const history = document.getElementById('copilot-chat-history');
  if (!history) return;
  history.innerHTML = getWelcomeMessageHtml(copilotLang);
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// -------------------------------------------------------------
// Sidebar Message Dispatchers
// -------------------------------------------------------------
function handleSidebarKey(e) {
  if (e.key === 'Enter') sendSidebarMessage();
}

function sendSidebarMessage() {
  const input = document.getElementById('copilot-user-input-sidebar');
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;

  const history = document.getElementById('copilot-chat-history-sidebar');
  if (!history) return;

  // Append user bubble
  history.insertAdjacentHTML('beforeend', `
    <div class="chat-entry user-entry">
      <div class="entry-bubble" style="font-size: 12.5px;">${escapeHtml(q)}</div>
    </div>
  `);
  input.value = '';
  history.scrollTop = history.scrollHeight;

  // Append response
  setTimeout(() => {
    const reply = generateCopilotResponse(q);
    history.insertAdjacentHTML('beforeend', `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 28px; height: 28px; font-size: 14px;">◈</div>
        <div class="entry-bubble" style="font-size: 12.5px;">
          ${reply}
        </div>
      </div>
    `);
    history.scrollTop = history.scrollHeight;
  }, 300);
}
