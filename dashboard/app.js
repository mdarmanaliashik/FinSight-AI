/**
 * FinSight AI — 2026 Shenzhen FinTechathon Executive Dashboard & Copilot
 * ======================================================================
 */

let appData = null;
let currentMarket = 'olist';
let salesChart = null;
let cashChart = null;

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('data.json');
    appData = await res.json();
    initCharts();
    runScenarioSimulation();
  } catch (err) {
    console.error('Failed to load data.json:', err);
  }
});

// -------------------------------------------------------------
// 1. Chart.js Initialization
// -------------------------------------------------------------
function initCharts() {
  const ts = appData.time_series;
  const cs = appData.cashflow_series;

  // Chart 1: Sales Forecasting with Multi-Quantile Risk Tunnel
  const ctxSales = document.getElementById('salesChart').getContext('2d');
  salesChart = new Chart(ctxSales, {
    type: 'line',
    data: {
      labels: ts.dates,
      datasets: [
        {
          label: 'P90 (Upside Surge)',
          data: ts.p90,
          borderColor: 'transparent',
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          fill: '+1', // Fill down to P10
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'P10 (Downside Stress)',
          data: ts.p10,
          borderColor: 'rgba(245, 158, 11, 0.4)',
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'Actual Sales',
          data: ts.actual,
          borderColor: '#38bdf8',
          backgroundColor: '#38bdf8',
          borderWidth: 2,
          pointRadius: 2.5,
          pointHoverRadius: 5,
          tension: 0.1
        },
        {
          label: 'Adaptive CatBoost',
          data: ts.adaptive,
          borderColor: '#10b981',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2
        },
        {
          label: 'P50 (Median)',
          data: ts.p50,
          borderColor: '#f59e0b',
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleFont: { family: 'Plus Jakarta Sans', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 11 },
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 12 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });

  // Chart 2: Cashflow Runway & Statutory Reserve Floor Line
  const ctxCash = document.getElementById('cashChart').getContext('2d');
  cashChart = new Chart(ctxCash, {
    type: 'line',
    data: {
      labels: cs.dates,
      datasets: [
        {
          label: 'Closing Cash Balance',
          data: cs.closing_cash,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          fill: true,
          borderWidth: 2.5,
          pointRadius: 1,
          tension: 0.15
        },
        {
          label: 'Cash Inflow',
          data: cs.inflow,
          borderColor: '#10b981',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1
        },
        {
          label: 'Statutory Reserve Limit (R)',
          data: cs.reserve_threshold,
          borderColor: '#ef4444',
          borderDash: [6, 4],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleFont: { family: 'Plus Jakarta Sans', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 11 },
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 12 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 2. Market Toggle: Olist (Brazil) vs JD.com (China)
// -------------------------------------------------------------
function setMarket(market) {
  currentMarket = market;
  document.getElementById('btn-olist').classList.toggle('active', market === 'olist');
  document.getElementById('btn-jd').classList.toggle('active', market === 'jd');

  if (market === 'jd') {
    // JD.com Mode (Chinese E-Commerce Retail Benchmark)
    document.getElementById('kpi-cum-wape').innerText = '5.90%';
    document.getElementById('kpi-daily-wape').innerText = '19.83%';
    document.getElementById('kpi-buffer').innerText = '+¥ 162,450';
    document.getElementById('kpi-tier').innerText = 'DIRECT TIER';
    
    // Add banner message to chat
    appendChatMessage('bot', `
      <p><strong>🇨🇳 Switched to JD.com Asian Retail Benchmark (Shenzhen / Beijing Direct):</strong></p>
      <p>• Delivery Lag: <strong>0.99 Days</strong> (Rapid Proprietary Fulfillment vs Olist 12.5 Days)</p>
      <p>• Zero-Sales Days: <strong>0.00%</strong> (High-Velocity Continuous Demand)</p>
      <p>• Top SKU Forecast WAPE: <strong>19.83%</strong> (3-Day Adaptive Smoothing)</p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.query_cross_market_benchmark(source='JD.com')</div>
    `);
  } else {
    // Olist Mode (Latin America SME Marketplace)
    document.getElementById('kpi-cum-wape').innerText = '0.93%';
    document.getElementById('kpi-daily-wape').innerText = '57.31%';
    document.getElementById('kpi-buffer').innerText = '+R$ 2,241.32';
    document.getElementById('kpi-tier').innerText = 'GREEN TIER';

    appendChatMessage('bot', `
      <p><strong>🇧🇷 Switched to Olist SME Marketplace Benchmark:</strong></p>
      <p>• Long Carrier Tail: <strong>12.5 Days Mean Lag</strong></p>
      <p>• Intermittency: <strong>39.53% Zero-Sales Days</strong></p>
      <p>• 7-Day Cumulative Inflow WAPE: <strong>0.93%</strong> (CFO Solvency Precision)</p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.query_cross_market_benchmark(source='Olist')</div>
    `);
  }
}

// -------------------------------------------------------------
// 3. Interactive Scenario Stress Lab (What-If Engine)
// -------------------------------------------------------------
function runScenarioSimulation() {
  const shock = parseFloat(document.getElementById('slider-shock').value);
  const lag = parseFloat(document.getElementById('slider-lag').value);
  const opex = parseFloat(document.getElementById('slider-opex').value);

  document.getElementById('val-shock').innerText = (shock > 0 ? '+' : '') + shock + '%';
  document.getElementById('val-lag').innerText = lag + ' Days';
  document.getElementById('val-opex').innerText = '+' + opex + '%';

  // Base parameters
  const baseCash = 51968.54;
  const reserveR = 2932.66;
  const base30Inflow = 8400.0 * (1 + shock / 100) * (10 / lag);
  const base30Outflow = 6200.0 * (1 + opex / 100);

  const projectedEndingCash = baseCash + base30Inflow - base30Outflow;
  const buffer = projectedEndingCash - reserveR;

  document.getElementById('sim-cash').innerText = 'R$ ' + projectedEndingCash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  document.getElementById('sim-buffer').innerText = (buffer > 0 ? '+R$ ' : '-R$ ') + Math.abs(buffer).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  const riskEl = document.getElementById('sim-risk');
  if (buffer > 5000) {
    riskEl.innerText = '0.0% (Safe Cushion)';
    riskEl.className = 'sim-value status-safe';
  } else if (buffer > 0) {
    riskEl.innerText = 'Low Risk (Amber Tier)';
    riskEl.className = 'sim-value';
    riskEl.style.color = '#f59e0b';
  } else {
    riskEl.innerText = 'BREACH ALERT (Deficit)';
    riskEl.className = 'sim-value status-danger';
  }
}

function resetStressLab() {
  document.getElementById('slider-shock').value = 0;
  document.getElementById('slider-lag').value = 10;
  document.getElementById('slider-opex').value = 0;
  runScenarioSimulation();
}

// -------------------------------------------------------------
// 4. Multilingual Tool-Grounded Copilot Chatbot Logic
// -------------------------------------------------------------
function handleChatKey(e) {
  if (e.key === 'Enter') sendUserMessage();
}

function askCopilot(query) {
  document.getElementById('chat-input').value = query;
  sendUserMessage();
}

function sendUserMessage() {
  const inputEl = document.getElementById('chat-input');
  const query = inputEl.value.trim();
  if (!query) return;

  appendChatMessage('user', query);
  inputEl.value = '';

  // Simulate tool dispatch & response
  setTimeout(() => {
    const response = generateCopilotResponse(query);
    appendChatMessage('bot', response);
  }, 400);
}

function appendChatMessage(sender, htmlContent) {
  const log = document.getElementById('chat-log');
  const msgDiv = document.createElement('div');
  msgDiv.className = `chat-msg ${sender}-msg`;

  if (sender === 'user') {
    msgDiv.innerHTML = `<div class="msg-bubble">${htmlContent}</div>`;
  } else {
    msgDiv.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble">${htmlContent}</div>
    `;
  }

  log.appendChild(msgDiv);
  log.scrollTop = log.scrollHeight;
}

function generateCopilotResponse(query) {
  const q = query.toLowerCase();

  const fmtCash = (v) => 'R$ ' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const baseCash = (appData && appData.summary && appData.summary.ending_cash) ? appData.summary.ending_cash : 51968.54;
  const reserveR = (appData && appData.summary && appData.summary.statutory_reserve_r) ? appData.summary.statutory_reserve_r : 2932.66;

  const isShock = /shock|drop|decline|breach|fall|decrease|down|less|loss|stress|what-if|যদি|কমে|হ্রাস|কমলে|ক্ষতি|গিরাউট|कमी|نقصان|کمی|下降|下滑|减少|跌|減少|下落/i.test(query);

  if (isShock) {
    let salesDropPct = 0.20;
    const pctMatch = query.match(/(\d+(?:\.\d+)?)\s*%/);
    if (pctMatch) {
      salesDropPct = parseFloat(pctMatch[1]) / 100.0;
    } else {
      const numMatch = query.match(/by\s+(\d+(?:\.\d+)?)/i);
      if (numMatch) salesDropPct = parseFloat(numMatch[1]) / 100.0;
    }

    let costIncreasePct = 0.0;
    const costMatch = query.match(/(?:cost|supplier|expense|খরচ|ব্যয়|लागत|成本|コスト)[^\d]*(\d+(?:\.\d+)?)\s*%/i);
    if (costMatch) {
      costIncreasePct = parseFloat(costMatch[1]) / 100.0;
    }

    let delayDays = 0;
    const dayMatch = query.match(/(\d+)\s*(?:day|days|দিন|दिन|天|日)/i);
    if (dayMatch && /delay|lag|late|দেরি|বিলম্ব|देरी|延迟|遅延/i.test(query)) {
      delayDays = parseInt(dayMatch[1]);
    }

    const baseInflowRate = 8400.0;
    const baseOutflowRate = 6200.0;
    const effectiveInflow = baseInflowRate * Math.max(0, 1.0 - salesDropPct) * (delayDays > 0 ? (10 / (10 + delayDays)) : 1.0);
    const effectiveOutflow = baseOutflowRate * (1.0 + costIncreasePct);

    const projectedEndingCash = baseCash + effectiveInflow - effectiveOutflow;
    const projectedMinCash = Math.min(baseCash, projectedEndingCash - (effectiveOutflow * 0.12));
    const netCushion = projectedMinCash - reserveR;
    const isBreach = netCushion < 0;
    const breachDays = isBreach ? Math.min(30, Math.max(1, Math.round(Math.abs(netCushion) / 300))) : 0;
    const breachRate = ((breachDays / 30) * 100).toFixed(1);
    const fmtPct = (salesDropPct * 100).toFixed(0);

    if (/[\u4E00-\u9FFF]/.test(query)) {
      return `
        <p><strong>【清华五道口 & WeBank 确定性金融工具测算】</strong></p>
        <p>已调度 <code>FinSightToolKit.simulate_scenario_shock(sales_shock=-${salesDropPct.toFixed(2)}, cost_shock=+${costIncreasePct.toFixed(2)})</code>：</p>
        <p>• <strong>模拟销售下滑：</strong> <code>-${fmtPct}%</code></p>
        <p>• <strong>法定偿付能力准备金 ($R$):</strong> ${fmtCash(reserveR)}</p>
        <p>• <strong>预测30天最低现金点:</strong> ${fmtCash(projectedMinCash)}</p>
        <p>• <strong>流动性净缓冲空间:</strong> <strong style="color: ${isBreach ? '#ef4444' : '#10b981'}">${isBreach ? '-' : '+'}${fmtCash(Math.abs(netCushion))}</strong></p>
        <p>• <strong>准备金击穿天数:</strong> <strong>${breachDays} 天 (${breachRate}%)</strong></p>
        <p><strong>结论：</strong> ${isBreach ? '<span style="color:#ef4444;">🚨 <strong>存在流动性违约风险（红色警报）</strong>。建议立即启动应收账款保理或短期周转信贷。</span>' : '<span style="color:#10b981;">✅ 商家保持 <strong>100% 偿付能力（绿色安全层级）</strong>。无需提前干预。</span>'}</p>
        <div class="tool-call-badge">⚙️ Contract: CashFlowEngine.audit_reserve_compliance(threshold_r=${reserveR.toFixed(2)})</div>
      `;
    }

    if (/[\u0980-\u09FF]/.test(query)) {
      return `
        <p><strong>【টুল-গ্রাউন্ডেড ক্যাশফ্লো সলভেন্সি অডিট】</strong></p>
        <p>এক্সিকিউটেড <code>FinSightToolKit.simulate_scenario_shock(sales_shock=-${salesDropPct.toFixed(2)}, cost_shock=+${costIncreasePct.toFixed(2)})</code>:</p>
        <p>• <strong>সিমুলেটেড সেলস পতন:</strong> <code>-${fmtPct}%</code></p>
        <p>• <strong>সংবিধিবদ্ধ রিজার্ভ সীমা ($R$):</strong> ${fmtCash(reserveR)}</p>
        <p>• <strong>প্রত্যাশিত ৩০ দিনের সর্বনিম্ন ক্যাশ:</strong> ${fmtCash(projectedMinCash)}</p>
        <p>• <strong>নিট লিকুইডিটি কুশন ($C_{\\min} - R$):</strong> <strong style="color: ${isBreach ? '#ef4444' : '#10b981'}">${isBreach ? '-' : '+'}${fmtCash(Math.abs(netCushion))}</strong></p>
        <p>• <strong>রিজার্ভ ব্রিচ সংখ্যা:</strong> <strong>${breachDays} দিন (${breachRate}%)</strong></p>
        <p><strong>সিদ্ধান্ত:</strong> ${isBreach ? '<span style="color:#ef4444;">🚨 <strong>লিকুইডিটি ঘাটতির ঝুঁকি রয়েছে (রেড অ্যালার্ট)</strong>। জরুরি ইনভয়েস ফ্যাক্টরিং বা RBF লোন সুপারিশ করা হচ্ছে।</span>' : '<span style="color:#10b981;">✅ প্রতিষ্ঠান <strong>১০০% সলভেন্ট (গ্রিন টিয়ার)</strong>। কোনো রিজার্ভ ব্রিচ ঘটবে না।</span>'}</p>
        <div class="tool-call-badge">⚙️ Contract: CashFlowEngine.audit_reserve_compliance(threshold_r=${reserveR.toFixed(2)})</div>
      `;
    }

    return `
      <p><strong>【Tool-Grounded Solvency Stress Audit】</strong></p>
      <p>Dispatched <code>FinSightToolKit.simulate_scenario_shock(sales_shock=-${salesDropPct.toFixed(2)}, cost_shock=+${costIncreasePct.toFixed(2)})</code>:</p>
      <p>• <strong>Simulated Sales Decline:</strong> <code>-${fmtPct}%</code></p>
      <p>• <strong>Statutory Reserve Threshold ($R$):</strong> ${fmtCash(reserveR)}</p>
      <p>• <strong>Projected 30-Day Minimum Cash:</strong> ${fmtCash(projectedMinCash)}</p>
      <p>• <strong>Net Liquidity Cushion ($C_{\\min} - R$):</strong> <strong style="color: ${isBreach ? '#ef4444' : '#10b981'}">${isBreach ? '-' : '+'}${fmtCash(Math.abs(netCushion))}</strong></p>
      <p>• <strong>Reserve Breach Count:</strong> <strong>${breachDays} Days (${breachRate}% Breach Rate)</strong></p>
      <p><strong>Verdict:</strong> ${isBreach ? '<span style="color:#ef4444;">🚨 <strong>LIQUIDITY BREACH WARNING (Red Tier)</strong>. Recommend immediate invoice factoring of pending platform receivables.</span>' : '<span style="color:#10b981;">The merchant remains <strong>100% SOLVENT (Green Tier)</strong>. No early warning intervention required.</span>'}</p>
      <div class="tool-call-badge">⚙️ Contract: CashFlowEngine.audit_reserve_compliance(threshold_r=${reserveR.toFixed(2)})</div>
    `;
  }

  if (/forecast|predict|sales|revenue|demand|বিক্রি|পূর্বাভাস|আয়|বিক্রি|पूर्वानुमान|فروخت|销售|预测|売上|予測/i.test(query)) {
    return `
      <p><strong>【Phase 3 Champion Forecasting Engine (Adaptive CatBoost)】</strong></p>
      <p>Dispatched <code>FinSightToolKit.get_sales_forecast(horizon=30, model='Adaptive CatBoost')</code>:</p>
      <p>• <strong>Holdout Test WAPE:</strong> <strong>59.80%</strong> (vs Naive 80.61%, +25.81% relative gain)</p>
      <p>• <strong>Directional Accuracy:</strong> <strong>71.1%</strong> ($p < 0.001$, Diebold-Mariano test)</p>
      <p>• <strong>Projected 30-Day Median Sales (P50):</strong> R$ 8,052.60 (R$ 268.42/day)</p>
      <p>• <strong>Calibrated Multi-Quantile Tunnel:</strong> P10 Downside: <strong>R$ 5,797.87</strong> | P90 Upside: <strong>R$ 10,468.38</strong></p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.get_sales_forecast(quantile='P10_P50_P90')</div>
    `;
  }

  if (/7-day|cumulative|weekly|thesis|wape|inflow/i.test(query)) {
    return `
      <p><strong>【The FinSight Dual-Resolution Thesis】</strong></p>
      <p>• <strong>Micro-Daily Sales (WAPE 59.80%):</strong> Single-SME daily transactions exhibit 39.53% zero-sales days and irreducible Poisson arrival variance. Predicting single-day sales 7 days ahead is inherently bounded by aleatoric noise.</p>
      <p>• <strong>Macro-Treasury Inflow (WAPE 0.93%, R² = 0.9994):</strong> SME solvency is an aggregation problem. Weekly cumulative rolling sums act as an ideal low-pass filter, cancelling out daily Poisson noise and enabling >99% precision for automated working capital credit underwriting!</p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.explain_dual_resolution_mechanics()</div>
    `;
  }

  if (/jd|china|brazil|olist|cross-market|transfer|চীন|ব্রাজিল/i.test(query)) {
    return `
      <p><strong>【Cross-Market Empirical Domain Transfer Audit】</strong></p>
      <p>Dispatched <code>FinSightToolKit.query_cross_market_benchmark()</code>:</p>
      <p>• <strong>Olist (Brazil):</strong> Carrier delivery lag = <strong>10.2 days</strong> | Intermittency = 39.53% zero-sales days | 7-Day Inflow WAPE = <strong>0.93%</strong>.</p>
      <p>• <strong>JD.com (China):</strong> Direct proprietary fulfillment = <strong>0.99 days</strong> | Zero-sales days = 0.00% | 7-Day Inflow WAPE = <strong>7.43%</strong> ($R^2 = 0.9966$).</p>
      <p>• <strong>Zero-Shot Transfer:</strong> Scale-invariant ratio space enables direct transfer from Brazil to China achieving <strong>WAPE 18.80%</strong> without local parameter tuning!</p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.query_cross_market_benchmark(source='Olist_to_JD')</div>
    `;
  }

  return `
    <p><strong>FinSight AI Financial Decision Copilot:</strong></p>
    <p>Current closing cash is <strong>${fmtCash(baseCash)}</strong> with a statutory reserve buffer of <strong style="color:#10b981;">+${fmtCash(baseCash - reserveR)}</strong> above threshold $R = ${fmtCash(reserveR)}$.</p>
    <p>You can test shock scenarios (e.g. <em>"What if sales drop by 15%?"</em> or <em>"If supplier costs rise by 25%?"</em>) or ask about cross-market liquidity comparisons.</p>
    <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.query_solvency_state(merchant_id='top_seller_1')</div>
  `;
}
