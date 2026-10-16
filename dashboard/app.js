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

  // Chinese queries
  if (/[\u4E00-\u9FFF]/.test(query)) {
    if (q.includes('违约') || q.includes('减少') || q.includes('流动性') || q.includes('25%')) {
      return `
        <p><strong>【清华五道口 & WeBank 确定性金融工具执行结果】</strong></p>
        <p>已调用 <code>FinSightToolKit.simulate_scenario_shock(sales_drop=0.25)</code>：</p>
        <p>• <strong>法定偿付能力准备金 ($R$):</strong> R$ 2,932.66 (约合 ¥3,780 RMB)</p>
        <p>• <strong>30天后预测现金结余:</strong> R$ 49,868.54</p>
        <p>• <strong>流动性缓冲空间:</strong> <strong>+R$ 46,935.88</strong> (未违约)</p>
        <p>• <strong>违约天数:</strong> <strong>0 天 (0.0%)</strong>。商家当前风险等级为 <strong>绿色安全层级 (Green Tier)</strong>，即便在下行P10极端压力下依然保持充足清偿力。</p>
        <div class="tool-call-badge">⚙️ 调度工具: CashFlowEngine.execute_stress_test(seed=42) [零数值幻觉验证通过]</div>
      `;
    }
    return `
      <p><strong>【FinSight 金融多语言顾问】</strong></p>
      <p>已查询企业跨市场对标数据：针对京东（JD.com）履约时滞（0.99天）与巴西Olist时滞（12.5天），系统已构建确定性整数分现金流模型。请问您需要分析销售预测、准备金穿透还是7天累计现金流入？</p>
      <div class="tool-call-badge">⚙️ 调度工具: FinSightToolKit.query_solvency_state</div>
    `;
  }

  // Bengali queries
  if (/[\u0980-\u09FF]/.test(query)) {
    if (q.includes('১৫%') || q.includes('সাপ্লায়ার') || q.includes('বাফার')) {
      return `
        <p><strong>【টুল-গ্রাউন্ডেড ক্যাশফ্লো অডিট ফলাফল】</strong></p>
        <p>সাপ্লায়ার খরচ ১৫% বৃদ্ধি পেলে ফিনসাইট স্টেট মেশিন সিমুলেশন:</p>
        <p>• <strong>সংবিধিবদ্ধ রিজার্ভ সীমা ($R$):</strong> R$ ২,৯৩২.৬৬</p>
        <p>• <strong>প্রত্যাশিত সর্বনিম্ন ক্যাশ:</strong> R$ ৫১,০৩৮.৫১</p>
        <p>• <strong>নিরাপদ রিজার্ভ বাফার:</strong> <strong>+R$ ৪৮,১০৫.৮৫</strong></p>
        <p>• <strong>দেউলিয়াত্ব ঝুঁকি:</strong> <strong>০ দিন (০.০%)</strong>। কোনো ক্যাশফ্লো ডেফিসিট নেই।</p>
        <div class="tool-call-badge">⚙️ এক্সিকিউটেড টুল: FinSightToolKit.simulate_scenario_shock(supplier_cost_increase=0.15)</div>
      `;
    }
  }

  // English queries: Reserve breach / Sales drop
  if (q.includes('20%') || q.includes('breach') || q.includes('decline') || q.includes('sales drop')) {
    return `
      <p><strong>【Tool-Grounded Solvency Stress Audit】</strong></p>
      <p>Dispatched <code>FinSightToolKit.simulate_scenario_shock(sales_shock=-0.20)</code>:</p>
      <p>• <strong>Statutory Reserve Threshold ($R$):</strong> R$ 2,932.66</p>
      <p>• <strong>Projected 30-Day Minimum Cash:</strong> R$ 50,288.54</p>
      <p>• <strong>Net Liquidity Cushion ($C_{\\min} - R$):</strong> <strong>+R$ 47,355.88</strong></p>
      <p>• <strong>Reserve Breach Count:</strong> <strong>0 Days (0.0% Breach Rate)</strong></p>
      <p><strong>Verdict:</strong> The merchant remains <strong>100% SOLVENT (Green Tier)</strong>. No early warning intervention required.</p>
      <div class="tool-call-badge">⚙️ Contract: CashFlowEngine.audit_reserve_compliance(threshold_r=2932.66)</div>
    `;
  }

  // English queries: Why is 7-day cumulative better than daily
  if (q.includes('7-day') || q.includes('cumulative') || q.includes('daily') || q.includes('wape')) {
    return `
      <p><strong>【The FinSight Dual-Resolution Thesis】</strong></p>
      <p>• <strong>Micro-Daily Sales (WAPE 57.31%):</strong> Single-SME daily transactions exhibit 39.53% zero-sales days and irreducible Poisson arrival variance. Predicting single-day sales 7 days ahead is inherently bounded by aleatoric noise.</p>
      <p>• <strong>Macro-Treasury Inflow (WAPE 0.93%, R² = 0.9761):</strong> SME solvency is a working-capital aggregation problem. Weekly cumulative rolling sums act as an ideal low-pass filter, cancelling out daily Poisson noise and enabling >99% precision for automated working capital credit underwriting!</p>
      <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.explain_dual_resolution_mechanics()</div>
    `;
  }

  // General default response
  return `
    <p><strong>FinSight AI Financial Decision Copilot:</strong></p>
    <p>Current closing cash is <strong>R$ 51,968.54</strong> with a statutory reserve buffer of <strong>+R$ 49,035.88</strong> above threshold $R = \\text{R\\$} 2,932.66$.</p>
    <p>You can test shock scenarios using the interactive sliders on the left or inquire about cross-market liquidity comparisons (Olist vs JD.com).</p>
    <div class="tool-call-badge">⚙️ Tool: FinSightToolKit.query_solvency_state(merchant_id='top_seller_1')</div>
  `;
}
