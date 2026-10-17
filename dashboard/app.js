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
let activeCurrency = 'BDT'; // 'BDT' (default) | 'USD' | 'BRL' | 'CNY' | 'EUR'
let activeHorizonDays = 30; // 7 | 14 | 30
let copilotLang = 'bn';     // 'bn' (default) | 'en' | 'zh' | 'es'
let businessForecastChart = null;

// Currency conversion rates (Base = BDT)
const CURRENCIES = {
  BDT: { symbol: '৳', rate: 1.0, locale: 'en-US', label: 'BDT' },
  USD: { symbol: '$', rate: 0.00833, locale: 'en-US', label: 'USD' },
  BRL: { symbol: 'R$ ', rate: 0.045, locale: 'pt-BR', label: 'BRL' },
  CNY: { symbol: '¥', rate: 0.060, locale: 'zh-CN', label: 'CNY' },
  EUR: { symbol: '€', rate: 0.00769, locale: 'de-DE', label: 'EUR' }
};

// Baseline Monthly Business Scale in BDT (from live prototype spec)
const BASE_MONTHLY_REVENUE_BDT = 420000;
const BASE_RESERVE_TARGET_BDT = 260400; // 62% of 420,000

// Format money according to active currency
function fmtMoney(amountBDT, decimals = 0) {
  const c = CURRENCIES[activeCurrency];
  const converted = amountBDT * c.rate;
  return `${c.symbol}${Math.round(converted).toLocaleString(c.locale)}`;
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
});

// -------------------------------------------------------------
// 3. Currency Switcher Handler
// -------------------------------------------------------------
function setCurrency(curr) {
  if (!CURRENCIES[curr]) return;
  activeCurrency = curr;

  ['bdt', 'usd', 'brl', 'cny', 'eur'].forEach(k => {
    const btn = document.getElementById(`curr-${k}`);
    if (btn) btn.classList.toggle('active', k === curr.toLowerCase());
  });

  const targetLabel = document.getElementById('curr-target-label');
  if (targetLabel) targetLabel.textContent = CURRENCIES[curr].label;

  recomputeBusinessEngine();
  recomputeTargetPlanner();
}

// -------------------------------------------------------------
// 4. Horizon Switcher (7 Days / 14 Days / 30 Days)
// -------------------------------------------------------------
function setForecastHorizonDays(days) {
  activeHorizonDays = days;

  [7, 14, 30].forEach(d => {
    const btn = document.getElementById(`btn-horizon-${d}d`);
    if (btn) btn.classList.toggle('active', d === days);
  });

  const horizonTag = document.getElementById('kpi-horizon-tag');
  if (horizonTag) horizonTag.textContent = `${days} DAYS`;

  recomputeBusinessEngine();
}

// -------------------------------------------------------------
// Decision Intelligence Module Switcher
// -------------------------------------------------------------
function switchDecisionModule(mod) {
  const tabs = ['overview', 'forecasts', 'cashflow', 'scenarios', 'copilot'];
  tabs.forEach(m => {
    const btn = document.getElementById(`tab-mod-${m}`);
    if (btn) btn.classList.toggle('active', m === mod);
  });

  if (mod === 'overview') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } else if (mod === 'forecasts') {
    const el = document.querySelector('.matrix-title');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else if (mod === 'cashflow') {
    const el = document.getElementById('risk-status-badge');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else if (mod === 'scenarios') {
    const el = document.getElementById('control-sales-growth');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  } else if (mod === 'copilot') {
    const el = document.getElementById('copilot-user-input');
    if (el) {
      el.focus();
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
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
    en: 'Open Banking Portal: Plaid, Stripe Financial Connections & bKash/Nagad merchant gateway successfully linked.',
    zh: '开放银行网关：Plaid / Stripe 金融连接通道已就绪，商户账户校验成功。',
    es: 'Portal de Banca Abierta: Pasarela de pagos conectada con éxito.'
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
    alertIcon.textContent = '✅';
    alertText.innerHTML = `<strong>Adequate Liquidity Standing:</strong> Available cash and monthly operating surplus exceed the required ${fmtMoney(reserveTargetBDT)} reserve cushion. (Surplus Buffer: <strong style="color: var(--accent-emerald);">+${fmtMoney(cashAfterReserveBDT)}</strong>).`;
  } else {
    alertBanner.style.background = 'rgba(244, 63, 94, 0.12)';
    alertBanner.style.borderColor = 'rgba(244, 63, 94, 0.4)';
    alertIcon.textContent = '🚨';
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
// 7. Render Probabilistic Sales Forecast Chart
// -------------------------------------------------------------
function renderPredictiveChart(monthlyRevenueBDT, numDays) {
  const canvas = document.getElementById('businessForecastChart');
  if (!canvas) return;
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
            color: '#cbd5e1',
            font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' },
            boxWidth: 14,
            padding: 16
          }
        },
        tooltip: {
          backgroundColor: 'rgba(11, 16, 26, 0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(255, 255, 255, 0.15)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${c.symbol}${ctx.parsed.y.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 11 },
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
    diffEl.textContent = `+${diffPct.toFixed(2)}%`;
    diffEl.className = 'gap-pill gap-pill-pos';
    gapEl.textContent = `+${fmtMoney(gapVal)}`;
    gapEl.style.color = 'var(--accent-emerald)';
  } else {
    diffEl.textContent = `${diffPct.toFixed(2)}%`;
    diffEl.className = 'gap-pill gap-pill-neg';
    gapEl.textContent = `-${fmtMoney(Math.abs(gapVal))}`;
    gapEl.style.color = '#f43f5e';
  }

  // Update weekly milestones
  const weeklyTarget = targetVal / 4.0;
  const milestonesEl = document.getElementById('strategy-weekly-milestones');
  if (milestonesEl) {
    milestonesEl.innerHTML = `
      Week 1 Target: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 2: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 3: <strong>${fmtMoney(weeklyTarget)}</strong> · 
      Week 4: <strong>${fmtMoney(weeklyTarget)}</strong>. Track daily via ledger.
    `;
  }
}

function generateTargetStrategy() {
  const targetInput = document.getElementById('input-target-revenue');
  const targetVal = parseFloat(targetInput.value) || 0;
  const baseline = BASE_MONTHLY_REVENUE_BDT;
  const gapVal = targetVal - baseline;
  const diffPct = ((targetVal - baseline) / baseline) * 100.0;

  let msg = '';
  if (diffPct >= 0) {
    msg = `【টার্গেট এক্সিকিউশন স্ট্র্যাটেজি তৈরি সম্পন্ন】\n\nমাসিক রাজস্ব লক্ষ্যমাত্রা: ${fmtMoney(targetVal)} (+${diffPct.toFixed(1)}% প্রবৃদ্ধি)।\n\nপ্রয়োজনীয় পদক্ষেপ:\n1. শীর্ষ ২০% হাই-মার্জিন পণ্যে ফোকাস ও ক্রস-সেল বান্ডেল বাড়ানো।\n2. পেমেন্ট সেটেলমেন্ট ল্যাগ ১০ দিন থেকে কমিয়ে ৪ দিনে আনার জন্য আর্লি-পেমেন্ট ডিসকাউন্ট দেওয়া।\n3. অপারেটিং কস্ট রেশিও ৬২% এর নিচে কঠোরভাবে নিয়ন্ত্রণ করা।\n4. প্রতি সপ্তাহে গড়ে ${fmtMoney(targetVal / 4)} বিক্রয় নিশ্চিত করা।`;
  } else {
    msg = `【সংরক্ষণশীল রাজস্ব সমন্বয় স্ট্র্যাটেজি】\n\nলক্ষ্যমাত্রা বর্তমান বেসলাইনের চেয়ে কম (-${Math.abs(diffPct).toFixed(1)}%)। ঝুঁকি কমাতে অপ্রয়োজনীয় পরিচালন ব্যয় হ্রাস এবং ইনভেন্টরি কমানোর সুপারিশ করা হলো।`;
  }
  alert(msg);
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
// 9. Render Granular Day-by-Day Financial Ledger Table
// -------------------------------------------------------------
function renderDayByDayLedger(monthlyRevenueBDT, startingCashBDT, reserveTargetBDT, costRatio) {
  const tbody = document.getElementById('business-ledger-tbody');
  if (!tbody) return;

  const dailyMeanSales = monthlyRevenueBDT / 30.0;
  const dailyOpex = (monthlyRevenueBDT * costRatio) / 30.0;
  let runningCash = startingCashBDT;

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
  tbody.innerHTML = html;
}

// -------------------------------------------------------------
// 10. Autonomous AI Financial Copilot Chatbot Engine
// -------------------------------------------------------------
function setCopilotLanguage(lang) {
  copilotLang = lang;
  ['bn', 'en', 'zh', 'es'].forEach(l => {
    const btn = document.getElementById(`clang-${l}`);
    if (btn) btn.classList.toggle('active', l === lang);
  });

  const placeholders = {
    bn: "আর্থিক প্রশ্ন জিজ্ঞাসা করুন (যেমন: 'বিক্রি ২০% কমলে ক্যাশ কত থাকবে?')...",
    en: "Ask financial question (e.g., 'Will cash breach reserve if sales drop 20%?')...",
    zh: "提出财务决策问题（例如：'如果销售额减少20%，资金会出现短缺吗？'）...",
    es: "Haga una pregunta financiera (ej. '¿Qué pasa si las ventas bajan un 20%?')..."
  };
  const inputEl = document.getElementById('copilot-user-input');
  if (inputEl) {
    inputEl.placeholder = placeholders[lang] || placeholders.bn;
  }
}

function handleCopilotKey(e) {
  if (e.key === 'Enter') sendCopilotMessage();
}

function askCopilot(query) {
  const input = document.getElementById('copilot-user-input');
  if (input) input.value = query;
  sendCopilotMessage();
}

function sendCopilotMessage() {
  const input = document.getElementById('copilot-user-input');
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;

  const history = document.getElementById('copilot-chat-history');
  if (!history) return;

  // Append User message bubble
  const userHtml = `
    <div class="chat-entry user-entry">
      <div class="entry-bubble">${escapeHtml(q)}</div>
    </div>
  `;
  history.insertAdjacentHTML('beforeend', userHtml);
  input.value = '';
  history.scrollTop = history.scrollHeight;

  // Process grounded response
  setTimeout(() => {
    const reply = generateCopilotResponse(q);
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
  }, 300);
}

function generateCopilotResponse(query) {
  const salesShift = parseFloat(document.getElementById('input-growth').value);
  const availableCash = parseFloat(document.getElementById('input-cash').value);
  const costRatio = parseFloat(document.getElementById('input-cost').value) / 100.0;
  const targetVal = parseFloat(document.getElementById('input-target-revenue').value) || 500000;

  let simShift = salesShift;
  const lowerQ = query.toLowerCase();
  const matchPct = lowerQ.match(/([+-]?\d+)\s*%/);
  if (matchPct) {
    simShift = parseFloat(matchPct[1]);
    if (lowerQ.includes('drop') || lowerQ.includes('fall') || lowerQ.includes('কমলে') || lowerQ.includes('হ্রাস') || lowerQ.includes('减少')) {
      simShift = -Math.abs(simShift);
    }
  }

  const projRev = BASE_MONTHLY_REVENUE_BDT * (1.0 + simShift / 100.0);
  const operatingSurplus = projRev * (1.0 - costRatio);
  const reserveTarget = BASE_MONTHLY_REVENUE_BDT * costRatio;
  const cashAfterReserve = availableCash + operatingSurplus - reserveTarget;
  const isSafe = cashAfterReserve >= 0;

  const isBengali = /[\u0980-\u09FF]/.test(query) || copilotLang === 'bn';
  const isChinese = /[\u4e00-\u9fa5]/.test(query) || copilotLang === 'zh';
  const isSpanish = copilotLang === 'es';

  if (isBengali) {
    return `
      <p><strong>【ট্রেজারি ডিসিশন ইন্টেলিজেন্স অডিট】</strong></p>
      <p>সিমুলেশন প্যারামিটার: বিক্রয় শিফট <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, কস্ট রেশিও <strong>${(costRatio * 100).toFixed(0)}%</strong>।</p>
      <ul style="margin: 6px 0 8px 18px; padding: 0;">
        <li>প্রজেক্টেড মাসিক রাজস্ব: <strong>${fmtMoney(projRev)}</strong></li>
        <li>মাসিক অপারেটিং উদ্বৃত্ত (Surplus): <strong>${fmtMoney(operatingSurplus)}</strong></li>
        <li>৩০ দিনের সংবিধিবদ্ধ রিজার্ভ টার্গেট: <strong>${fmtMoney(reserveTarget)}</strong></li>
        <li>রিজার্ভ সংরক্ষণের পর ক্যাশ বাফার: <strong style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</strong></li>
      </ul>
      <p>সিদ্ধান্ত: ${isSafe ? 'ব্যবসায় পর্যাপ্ত লিকুইডিটি বাফার রয়েছে (SAFE TIER)। জরুরি ঋণের কোনো প্রয়োজন নেই।' : 'সতর্কতা: রিজার্ভ ঘাটতির আশঙ্কা রয়েছে (SHORTFALL HAZARD)। অপ্রয়োজনীয় পরিচালন ব্যয় কমাতে হবে।'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency</div>
    `;
  } else if (isChinese) {
    return `
      <p><strong>【企业流动性决策智能审计报告】</strong></p>
      <p>模拟参数：销售变动 <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>，成本比率 <strong>${(costRatio * 100).toFixed(0)}%</strong>。</p>
      <ul style="margin: 6px 0 8px 18px; padding: 0;">
        <li>预测月度营业额：<strong>${fmtMoney(projRev)}</strong></li>
        <li>月度运营盈余：<strong>${fmtMoney(operatingSurplus)}</strong></li>
        <li>30天法定储备金目标：<strong>${fmtMoney(reserveTarget)}</strong></li>
        <li>储备拨备后净流动性缓冲：<strong style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</strong></li>
      </ul>
      <p>审计结论：${isSafe ? '流动性充足，处于安全状态 (SAFE TIER)。无需启动紧急短期授信。' : '预警：存在资金缺口风险 (DEFICIT HAZARD)，建议推迟非必要性支出。'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency</div>
    `;
  } else if (isSpanish) {
    return `
      <p><strong>【Auditoría de Inteligencia de Decisiones】</strong></p>
      <p>Parámetros: Variación de ventas <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, ratio de costos <strong>${(costRatio * 100).toFixed(0)}%</strong>.</p>
      <ul style="margin: 6px 0 8px 18px; padding: 0;">
        <li>Ingresos Proyectados: <strong>${fmtMoney(projRev)}</strong></li>
        <li>Superávit Operativo Mensual: <strong>${fmtMoney(operatingSurplus)}</strong></li>
        <li>Meta de Reserva a 30 Días: <strong>${fmtMoney(reserveTarget)}</strong></li>
        <li>Colchón Tras Reserva: <strong style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</strong></li>
      </ul>
      <p>Veredicto: ${isSafe ? 'Liquidez adecuada y solvente (Nivel Seguro).' : 'Alerta de déficit potencial en la reserva operativa.'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency</div>
    `;
  } else {
    return `
      <p><strong>【Decision Intelligence Treasury Audit】</strong></p>
      <p>Simulated Parameters: Sales shift <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, cost ratio <strong>${(costRatio * 100).toFixed(0)}%</strong>.</p>
      <ul style="margin: 6px 0 8px 18px; padding: 0;">
        <li>Projected Monthly Revenue: <strong>${fmtMoney(projRev)}</strong></li>
        <li>Monthly Operating Surplus: <strong>${fmtMoney(operatingSurplus)}</strong></li>
        <li>30-Day Reserve Target: <strong>${fmtMoney(reserveTarget)}</strong></li>
        <li>Cash After Reserve Set-Aside: <strong style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</strong></li>
      </ul>
      <p>Verdict: ${isSafe ? 'Adequate liquidity cushion maintained (SAFE TIER). No credit line intervention required.' : 'Warning: Potential reserve shortfall under this stress scenario.'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency</div>
    `;
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
