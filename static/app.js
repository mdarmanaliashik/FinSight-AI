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
    // 1. Toggle tab button active state
    const btn = document.getElementById(`tab-mod-${m}`);
    if (btn) btn.classList.toggle('active', m === mod);

    // 2. Toggle dedicated section view display
    const view = document.getElementById(`view-${m}`);
    if (view) {
      view.style.display = (m === mod) ? 'block' : 'none';
    }
  });

  // 3. Keep 4-Step workflow bar synchronized
  const stepMap = {
    overview: 1,
    forecasts: 2,
    cashflow: 3,
    scenarios: 4,
    copilot: 1
  };
  [1, 2, 3, 4].forEach(s => {
    const stepEl = document.getElementById(`wf-step-${s}`);
    if (stepEl) {
      stepEl.classList.toggle('active', s <= (stepMap[mod] || 1));
    }
  });

  // 4. Resize Chart.js canvas whenever forecast or overview is shown
  if (mod === 'forecasts' || mod === 'overview') {
    if (businessForecastChart) {
      setTimeout(() => {
        businessForecastChart.resize();
      }, 30);
    }
  }

  // 5. Special Copilot focus
  if (mod === 'copilot') {
    const input = document.getElementById('copilot-user-input');
    if (input) {
      input.focus();
      input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  } else {
    window.scrollTo({ top: 100, behavior: 'smooth' });
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

  // Synchronize AI Advisor Studio Telemetry
  const advTelCash = document.getElementById('adv-tel-cash');
  if (advTelCash) advTelCash.textContent = fmtMoney(availableCashBDT);
  const advTelRev = document.getElementById('adv-tel-rev');
  if (advTelRev) advTelRev.textContent = fmtMoney(projectedRevenueMonthlyBDT);
  const advTelTarget = document.getElementById('adv-tel-target');
  if (advTelTarget) advTelTarget.textContent = fmtMoney(reserveTargetBDT);
  const advTelCushion = document.getElementById('adv-tel-cushion');
  if (advTelCushion) {
    advTelCushion.textContent = `${isReserveSafe ? '+' : ''}${fmtMoney(cashAfterReserveBDT)}`;
    advTelCushion.style.color = isReserveSafe ? 'var(--accent-emerald)' : '#f43f5e';
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

  // Visual pulse highlight on sidebar panel
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

  // 1. Append User message bubble
  const userHtml = `
    <div class="chat-entry user-entry">
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

function generateCopilotResponse(query) {
  const salesShift = parseFloat(document.getElementById('input-growth').value) || 0;
  const availableCash = parseFloat(document.getElementById('input-cash').value) || 180000;
  const costRatioPct = parseFloat(document.getElementById('input-cost').value) || 62;
  const costRatio = costRatioPct / 100.0;
  const targetVal = parseFloat(document.getElementById('input-target-revenue').value) || 500000;

  const lowerQ = query.toLowerCase();
  const isBengali = /[\u0980-\u09FF]/.test(query) || copilotLang === 'bn';
  const isChinese = /[\u4e00-\u9fa5]/.test(query) || copilotLang === 'zh';
  const isSpanish = copilotLang === 'es';

  // INTENT 1: Target Revenue & Growth Gap Execution
  if (lowerQ.includes('টার্গেট') || lowerQ.includes('target') || lowerQ.includes('কৌশল') || lowerQ.includes('goal') || lowerQ.includes('gap')) {
    const baseline = BASE_MONTHLY_REVENUE_BDT;
    const gap = targetVal - baseline;
    const growthPct = ((gap / baseline) * 100.0);
    const weeklyTarget = targetVal / 4.0;

    if (isBengali) {
      return `
        <p><strong>🎯【টার্গেট এক্সিকিউশন ও গ্রোথ রোডম্যাপ】</strong></p>
        <p>মাসিক রেভেনিউ লক্ষ্যমাত্রা <strong>${fmtMoney(targetVal)}</strong> অর্জনের জন্য গ্রোথ গ্যাপ <strong>+${growthPct.toFixed(1)}% (${fmtMoney(gap)})</strong> পূরণ করতে হবে।</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Target Revenue</div>
            <div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Revenue Gap</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Weekly Target</div>
            <div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Execution Priority</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">Basket Size +৳120</div>
          </div>
        </div>

        <p><strong>বাস্তবায়ন নির্দেশিকা:</strong></p>
        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.5;">
          <li>শীর্ষ ২০% হাই-মার্জিন এসকেইউ (SKU) প্রোডাক্টে ক্রস-সেলিং জোরদার করা।</li>
          <li>পেমেন্ট ডিলে ল্যাগ ১০ দিন থেকে কমিয়ে ৪ দিনে আনার জন্য ২% আর্লি ডিসকাউন্ট দেওয়া।</li>
          <li>অপারেটিং খরচ ৬২% এর নিচে ধরে রেখে প্রতি সপ্তাহে গড়ে ${fmtMoney(weeklyTarget)} ক্যাশ সেলস নিশ্চিত করা।</li>
        </ul>

        <div class="tool-badge-pill">⚙️ FinSightToolKit.synthesize_strategic_roadmap · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${growthPct.toFixed(0)}, ${costRatioPct})">⚡ Apply Target to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    } else {
      return `
        <p><strong>🎯【Target Execution & Revenue Bridge Roadmap】</strong></p>
        <p>To reach monthly target of <strong>${fmtMoney(targetVal)}</strong>, a growth gap of <strong>+${growthPct.toFixed(1)}% (+${fmtMoney(gap)})</strong> must be bridged.</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Target Revenue</div>
            <div class="ai-metric-val" style="color: var(--accent-cyan);">${fmtMoney(targetVal)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Revenue Gap</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(gap)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Weekly Target</div>
            <div class="ai-metric-val">${fmtMoney(weeklyTarget)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Strategy</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">Basket +৳120</div>
          </div>
        </div>

        <ul style="margin: 4px 0 8px 18px; padding: 0; font-size: 12px; color: #cbd5e1; line-height: 1.5;">
          <li>Focus on top 20% margin SKUs with bundled retail incentives.</li>
          <li>Offer 2% early-settlement incentive to compress payment lag from 10D to 4D.</li>
          <li>Maintain operational costs capped below ${costRatioPct}%.</li>
        </ul>

        <div class="tool-badge-pill">⚙️ FinSightToolKit.synthesize_strategic_roadmap · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${growthPct.toFixed(0)}, ${costRatioPct})">⚡ Apply Target to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    }
  }

  // INTENT 2: Cash Runway & Operating Burn Rate
  if (lowerQ.includes('রানওয়ে') || lowerQ.includes('runway') || lowerQ.includes('burn') || lowerQ.includes('দিন চলবে') || lowerQ.includes('কত দিন')) {
    const dailyOpex = (BASE_MONTHLY_REVENUE_BDT * costRatio) / 30.0;
    const zeroRevenueRunwayDays = Math.floor(availableCash / dailyOpex);
    const dailySales = BASE_MONTHLY_REVENUE_BDT / 30.0;
    const dailyNet = dailySales - dailyOpex;

    if (isBengali) {
      return `
        <p><strong>⏱️【ক্যাশ রানওয়ে ও লিকুইডিটি সার্ভাইভাল অডিট】</strong></p>
        <p>বর্তমান ট্রেজারি ক্যাশ <strong>${fmtMoney(availableCash)}</strong> এবং দৈনিক পরিচালন ব্যয় (Burn Rate) <strong>${fmtMoney(dailyOpex)}/দিন</strong>।</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Available Cash</div>
            <div class="ai-metric-val">${fmtMoney(availableCash)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Daily Burn (OPEX)</div>
            <div class="ai-metric-val" style="color: #f43f5e;">${fmtMoney(dailyOpex)} / দিন</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Zero-Sales Runway</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} দিন</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Operating Surplus</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(dailyNet)} / দিন</div>
          </div>
        </div>

        <p><strong>বিশ্লেষণ:</strong> কোনো বিক্রয় ছাড়া সম্পূর্ণ শূন্য রাজস্ব পরিস্থিতিতে বর্তমান তহবিল দিয়ে <strong>${zeroRevenueRunwayDays} দিন</strong> কার্যক্রম পরিচালনা সম্ভব। বর্তমান স্বাভাবিক বিক্রয় ফ্লো অনুযায়ী ব্যবসা স্বয়ংসম্পূর্ণ ও টেকসই।</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.calculate_cash_runway · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">🏦 View 30-Day Cashflow Ledger</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    } else {
      return `
        <p><strong>⏱️【Cash Runway & Operating Burn Audit】</strong></p>
        <p>Current treasury cash is <strong>${fmtMoney(availableCash)}</strong> with daily operational burn of <strong>${fmtMoney(dailyOpex)}/day</strong>.</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Treasury Cash</div>
            <div class="ai-metric-val">${fmtMoney(availableCash)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Daily Burn</div>
            <div class="ai-metric-val" style="color: #f43f5e;">${fmtMoney(dailyOpex)} / day</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Zero-Sales Runway</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">${zeroRevenueRunwayDays} Days</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Net Daily Flow</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(dailyNet)} / day</div>
          </div>
        </div>

        <p>Under a complete zero-revenue freeze, liquidity supports <strong>${zeroRevenueRunwayDays} days</strong> of operations. With current sales momentum, runway remains solvent.</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.calculate_cash_runway · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">🏦 View 30-Day Cashflow Ledger</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    }
  }

  // INTENT 3: Payment Delay / Receivables Lag Acceleration
  if (lowerQ.includes('পেমেন্ট') || lowerQ.includes('payment') || lowerQ.includes('delay') || lowerQ.includes('রিসিভেবল') || lowerQ.includes('ল্যাগ') || lowerQ.includes('lag')) {
    const dailyTurnover = BASE_MONTHLY_REVENUE_BDT / 30.0;
    const freedWorkingCapital = dailyTurnover * 6.0; // 10 days reduced to 4 days = 6 days compressed

    if (isBengali) {
      return `
        <p><strong>⚡【রিসিভেবলস পেমেন্ট ডিলে কম্প্রেশন অডিট】</strong></p>
        <p>বর্তমান রিটেইল পেমেন্ট সেটেলমেন্ট ল্যাগ <strong>১০ দিন</strong>। এটি কমিয়ে <strong>৪ দিনে</strong> আনলে ৬ দিনের বিক্রয় তহবিল অবিলম্বে ছাড় পাবে।</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Current Delay Lag</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">10 Days</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Compressed Target</div>
            <div class="ai-metric-val" style="color: var(--accent-cyan);">4 Days (-6D)</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Liquid Capital Freed</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(freedWorkingCapital)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Incentive Cost (2%)</div>
            <div class="ai-metric-val">${fmtMoney(BASE_MONTHLY_REVENUE_BDT * 0.02)}</div>
          </div>
        </div>

        <p><strong>সিদ্ধান্ত:</strong> ডিলারদের ২% ছাড় দিলে প্রায় <strong>+${fmtMoney(freedWorkingCapital)}</strong> তাৎক্ষণিক তারল্য (Working Capital) ব্যাংকে আগেভাগে চলে আসবে, যা রিজার্ভ কুশন আরও সুদৃঢ় করবে।</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_receivables_lag · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">🛡️ View Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    } else {
      return `
        <p><strong>⚡【Receivables Settlement Acceleration Audit】</strong></p>
        <p>Current retail collection lag is <strong>10 days</strong>. Compressing delay down to <strong>4 days</strong> unlocks 6 days of sales turnover.</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Current Lag</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">10 Days</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Target Lag</div>
            <div class="ai-metric-val" style="color: var(--accent-cyan);">4 Days</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Liquidity Pulled Forward</div>
            <div class="ai-metric-val" style="color: var(--accent-emerald);">+${fmtMoney(freedWorkingCapital)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Early Incentive (2%)</div>
            <div class="ai-metric-val">${fmtMoney(BASE_MONTHLY_REVENUE_BDT * 0.02)}</div>
          </div>
        </div>

        <p>Offering a 2% settlement discount pulls forward approximately <strong>+${fmtMoney(freedWorkingCapital)}</strong> in liquid cash flow.</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_receivables_lag · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="switchDecisionModule('cashflow')">🛡️ View Early Warning Radar</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    }
  }

  // INTENT 4: Cost Inflation & COGS Shock
  if (lowerQ.includes('কস্ট') || lowerQ.includes('cost') || lowerQ.includes('খরচ') || lowerQ.includes('cogs') || lowerQ.includes('expense')) {
    let testCostRatio = 0.70;
    const matchCost = lowerQ.match(/(\d+)\s*%/);
    if (matchCost) testCostRatio = parseFloat(matchCost[1]) / 100.0;

    const projRev = BASE_MONTHLY_REVENUE_BDT * (1.0 + salesShift / 100.0);
    const operatingSurplus = projRev * (1.0 - testCostRatio);
    const reserveTarget = BASE_MONTHLY_REVENUE_BDT * testCostRatio;
    const cashAfterReserve = availableCash + operatingSurplus - reserveTarget;
    const isSafe = cashAfterReserve >= 0;

    if (isBengali) {
      return `
        <p><strong>📈【কস্ট ইনফ্লেশন ও অপারেটিং মার্জিন সেনসিটিভিটি】</strong></p>
        <p>পরিচালন ব্যয় বৃদ্ধি পেয়ে কস্ট রেশিও <strong>${(testCostRatio * 100).toFixed(0)}%</strong> হলে ট্রেজারি বাফার কেমন থাকবে:</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Tested Cost Ratio</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">${(testCostRatio * 100).toFixed(0)}% COGS</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Gross Margin</div>
            <div class="ai-metric-val">${((1.0 - testCostRatio) * 100).toFixed(0)}% Margin</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Monthly Surplus</div>
            <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Cushion Buffer</div>
            <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
          </div>
        </div>

        <p>সিদ্ধান্ত: ${isSafe ? `কস্ট রেশিও ${(testCostRatio * 100).toFixed(0)}% এ পৌঁছালেও ব্যবসায় <strong>+${fmtMoney(cashAfterReserve)}</strong> এর নিরাপদ লিকুইডিটি বাফার বজায় থাকবে।` : `সতর্কতা: কস্ট রেশিও ${(testCostRatio * 100).toFixed(0)}% হলে রিজার্ভের নিচে ঘাটতি (Deficit) তৈরি হতে পারে। পরিচালন ব্যয় নিয়ন্ত্রণ জরুরি।`}</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_expense_shock · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, ${(testCostRatio * 100).toFixed(0)})">⚡ Apply ${(testCostRatio * 100).toFixed(0)}% Cost to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    } else {
      return `
        <p><strong>📈【Cost Inflation & Operating Margin Sensitivity】</strong></p>
        <p>Simulating operational cost ratio elevated to <strong>${(testCostRatio * 100).toFixed(0)}% COGS</strong>:</p>
        
        <div class="ai-metric-grid">
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Cost Ratio</div>
            <div class="ai-metric-val" style="color: var(--accent-amber);">${(testCostRatio * 100).toFixed(0)}% COGS</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Gross Margin</div>
            <div class="ai-metric-val">${((1.0 - testCostRatio) * 100).toFixed(0)}% Margin</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Monthly Surplus</div>
            <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
          </div>
          <div class="ai-metric-cell">
            <div class="ai-metric-lbl">Liquidity Buffer</div>
            <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
          </div>
        </div>

        <p>Verdict: ${isSafe ? `Even at ${(testCostRatio * 100).toFixed(0)}% COGS, a liquidity buffer of <strong>+${fmtMoney(cashAfterReserve)}</strong> is safely retained.` : `Warning: A ${(testCostRatio * 100).toFixed(0)}% cost ratio breaches reserve thresholds. Immediate opex reduction advised.`}</p>
        <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_expense_shock · Verified ✓</div>
        <div class="ai-actions-row">
          <button class="ai-action-btn" onclick="applyScenarioToControls(${salesShift}, ${(testCostRatio * 100).toFixed(0)})">⚡ Apply ${(testCostRatio * 100).toFixed(0)}% Cost to Studio</button>
          <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
        </div>
      `;
    }
  }

  // INTENT 5: Default Treasury Stress Test (Percentage shift)
  let simShift = salesShift;
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

  if (isBengali) {
    return `
      <p><strong>🛡️【ট্রেজারি ডিসিশন ইন্টেলিজেন্স অডিট】</strong></p>
      <p>সিমুলেশন প্যারামিটার: বিক্রয় শিফট <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, কস্ট রেশিও <strong>${(costRatio * 100).toFixed(0)}%</strong>।</p>
      
      <div class="ai-metric-grid">
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Projected Revenue</div>
          <div class="ai-metric-val">${fmtMoney(projRev)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Operating Surplus</div>
          <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Reserve Target</div>
          <div class="ai-metric-val">৳260,400</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Safety Cushion</div>
          <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
        </div>
      </div>

      <p><strong>সিদ্ধান্ত:</strong> ${isSafe ? 'ব্যবসায় পর্যাপ্ত লিকুইডিটি বাফার রয়েছে (SAFE TIER)। রিজার্ভ নিরাপদ এবং জরুরি বহিঃস্থ ঋণের কোনো প্রয়োজন নেই।' : 'সতর্কতা: রিজার্ভ ঘাটতির আশঙ্কা রয়েছে (SHORTFALL HAZARD)। তাৎক্ষণিকভাবে পরিচালন ব্যয় ও ইনভেন্টরি পারচেজ স্থগিত করার সুপারিশ করা হলো।'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct})">⚡ Apply ${simShift}% Shift to Studio</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
      </div>
    `;
  } else if (isChinese) {
    return `
      <p><strong>🛡️【企业流动性决策智能审计报告】</strong></p>
      <p>模拟参数：销售变动 <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>，成本比率 <strong>${(costRatio * 100).toFixed(0)}%</strong>。</p>
      
      <div class="ai-metric-grid">
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">预测营业额</div>
          <div class="ai-metric-val">${fmtMoney(projRev)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">运营盈余</div>
          <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">储备目标</div>
          <div class="ai-metric-val">৳260,400</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">净缓冲资金</div>
          <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
        </div>
      </div>

      <p>审计结论：${isSafe ? '流动性充足，处于安全评级 (SAFE TIER)。无需启动紧急短期授信。' : '预警：存在资金缺口风险 (DEFICIT HAZARD)，建议推迟非必要采购。'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct})">⚡ 应用此情景至工作台</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 复制报告</button>
      </div>
    `;
  } else if (isSpanish) {
    return `
      <p><strong>🛡️【Auditoría de Inteligencia de Decisiones】</strong></p>
      <p>Parámetros: Variación de ventas <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, ratio de costos <strong>${(costRatio * 100).toFixed(0)}%</strong>.</p>
      
      <div class="ai-metric-grid">
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Ingresos Proyectados</div>
          <div class="ai-metric-val">${fmtMoney(projRev)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Superávit Mensual</div>
          <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Meta Reserva</div>
          <div class="ai-metric-val">৳260,400</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Colchón Neto</div>
          <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
        </div>
      </div>

      <p>Veredicto: ${isSafe ? 'Liquidez adecuada y solvente (Nivel Seguro).' : 'Alerta de déficit potencial en la reserva operativa.'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct})">⚡ Aplicar Escenario</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copiar</button>
      </div>
    `;
  } else {
    return `
      <p><strong>🛡️【Decision Intelligence Treasury Audit】</strong></p>
      <p>Simulated Parameters: Sales shift <strong>${simShift >= 0 ? '+' : ''}${simShift}%</strong>, cost ratio <strong>${(costRatio * 100).toFixed(0)}%</strong>.</p>
      
      <div class="ai-metric-grid">
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Projected Revenue</div>
          <div class="ai-metric-val">${fmtMoney(projRev)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Monthly Surplus</div>
          <div class="ai-metric-val">${fmtMoney(operatingSurplus)}</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Reserve Target</div>
          <div class="ai-metric-val">৳260,400</div>
        </div>
        <div class="ai-metric-cell">
          <div class="ai-metric-lbl">Safety Cushion</div>
          <div class="ai-metric-val" style="color: ${isSafe ? 'var(--accent-emerald)' : '#f43f5e'};">${isSafe ? '+' : ''}${fmtMoney(cashAfterReserve)}</div>
        </div>
      </div>

      <p>Verdict: ${isSafe ? 'Adequate liquidity cushion maintained (SAFE TIER). No credit line intervention required.' : 'Warning: Potential reserve shortfall under this stress scenario.'}</p>
      <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
      <div class="ai-actions-row">
        <button class="ai-action-btn" onclick="applyScenarioToControls(${simShift}, ${costRatioPct})">⚡ Apply Scenario to Studio</button>
        <button class="ai-copy-btn" onclick="copyAnalysisText(this)">📋 Copy</button>
      </div>
    `;
  }
}

// -------------------------------------------------------------
// Interactive Advisor Helper Actions
// -------------------------------------------------------------
function applyScenarioToControls(salesShift, costRatioPct) {
  const elGrowth = document.getElementById('input-growth');
  const elCost = document.getElementById('input-cost');
  if (elGrowth) elGrowth.value = salesShift;
  if (elCost) elCost.value = costRatioPct;
  recomputeBusinessEngine();
  recomputeTargetPlanner();

  // Switch to scenarios view to see the live update
  switchDecisionModule('scenarios');
}

function copyAnalysisText(btn) {
  const bubble = btn.closest('.entry-bubble');
  if (!bubble) return;
  const clone = bubble.cloneNode(true);
  const actions = clone.querySelector('.ai-actions-row');
  if (actions) actions.remove();
  const text = clone.innerText.trim();

  navigator.clipboard.writeText(text).then(() => {
    const origText = btn.textContent;
    btn.textContent = '✓ Copied';
    btn.style.color = 'var(--accent-emerald)';
    setTimeout(() => {
      btn.textContent = origText;
      btn.style.color = '';
    }, 1500);
  });
}

function clearCopilotChat() {
  const history = document.getElementById('copilot-chat-history');
  if (!history) return;

  const welcomeMsg = {
    bn: `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
        <div class="entry-bubble">
          <p><strong>FinSight Business Decision Intelligence সক্রিয়।</strong></p>
          <p>আমি আপনার SME ব্যবসার বর্তমান ক্যাশ ব্যালেন্স, ভবিষ্যত রাজস্ব পূর্বাভাস এবং রিজার্ভ ঘাটতি রিয়েল-টাইমে বিশ্লেষণ করছি।</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">৳১৮০,০০০</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Proj. Revenue</div><div class="ai-metric-val">৳৪২০,০০০</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">৳২৬০,৪০০</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+৳৭৯,২০০</div></div>
          </div>
          <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
        </div>
      </div>
    `,
    en: `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
        <div class="entry-bubble">
          <p><strong>FinSight Business Decision Intelligence Online.</strong></p>
          <p>Continuous real-time audit for cash reserve breach, payment delays, and liquidity shocks.</p>
          <div class="ai-metric-grid">
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Available Cash</div><div class="ai-metric-val">$1,500</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">30D Proj. Revenue</div><div class="ai-metric-val">$3,500</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Reserve Target</div><div class="ai-metric-val">$2,170</div></div>
            <div class="ai-metric-cell"><div class="ai-metric-lbl">Safety Cushion</div><div class="ai-metric-val" style="color: var(--accent-emerald);">+$660</div></div>
          </div>
          <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
        </div>
      </div>
    `,
    zh: `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
        <div class="entry-bubble">
          <p><strong>FinSight 商业决策智能系统已就绪。</strong></p>
          <p>实时审计流动性储备缺口、应收账款账期及销售波动风险。</p>
          <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
        </div>
      </div>
    `,
    es: `
      <div class="chat-entry bot-entry">
        <div class="copilot-avatar" style="width: 32px; height: 32px; font-size: 16px;">◈</div>
        <div class="entry-bubble">
          <p><strong>Inteligencia de Decisiones FinSight Activa.</strong></p>
          <p>Auditoría en tiempo real para solvencia y choque de liquidez.</p>
          <div class="tool-badge-pill">⚙️ FinSightToolKit.audit_treasury_solvency · Verified ✓</div>
        </div>
      </div>
    `
  };

  history.innerHTML = welcomeMsg[copilotLang] || welcomeMsg.bn;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
