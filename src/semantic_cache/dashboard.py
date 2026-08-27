"""Dashboard -- self-contained HTML page with live-updating charts.

Replaces the need for Grafana. Opens in any browser, auto-refreshes
metrics every 5 seconds via fetch() to /v1/metrics/detailed.
"""


def render_dashboard_html() -> str:
    """Return the complete HTML dashboard as a string."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Semantic Cache Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
  h1 { font-size: 22px; color: #38bdf8; margin-bottom: 6px; }
  .subtitle { color: #94a3b8; font-size: 13px; margin-bottom: 24px; }
  .live-dot { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .card-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
  .card-value { font-size: 32px; font-weight: 700; margin-top: 4px; }
  .card-sub { font-size: 12px; color: #64748b; margin-top: 4px; }
  .green { color: #22c55e; } .blue { color: #38bdf8; } .yellow { color: #facc15; } .red { color: #f87171; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .chart-box { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .chart-title { font-size: 14px; color: #94a3b8; margin-bottom: 12px; }
  canvas { max-height: 260px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th { text-align: left; padding: 10px 12px; background: #334155; color: #94a3b8; font-size: 12px; text-transform: uppercase; }
  td { padding: 10px 12px; border-bottom: 1px solid #1e293b; font-size: 14px; }
  tr:hover td { background: #1e293b; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .badge-green { background: #166534; color: #86efac; }
  .badge-blue { background: #1e3a5f; color: #93c5fd; }
</style>
</head>
<body>

<h1><span class="live-dot"></span>Semantic Cache Dashboard</h1>
<p class="subtitle">Auto-refreshes every 5 seconds | Phase 4</p>

<div class="grid">
  <div class="card">
    <div class="card-label">Total Requests</div>
    <div class="card-value blue" id="totalReqs">0</div>
    <div class="card-sub" id="uptime">Uptime: 0s</div>
  </div>
  <div class="card">
    <div class="card-label">Hit Rate</div>
    <div class="card-value green" id="hitRate">0%</div>
    <div class="card-sub" id="hitMiss">0 hits / 0 misses</div>
  </div>
  <div class="card">
    <div class="card-label">Avg Latency Saved</div>
    <div class="card-value yellow" id="latSaved">0ms</div>
    <div class="card-sub" id="latDetail">Hit: 0ms | Miss: 0ms</div>
  </div>
  <div class="card">
    <div class="card-label">Cost Savings</div>
    <div class="card-value green" id="costSaved">$0.00</div>
    <div class="card-sub" id="tokensSaved">0 tokens saved</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box">
    <div class="chart-title">Cache Hit / Miss Ratio</div>
    <canvas id="hitMissChart"></canvas>
  </div>
  <div class="chart-box">
    <div class="chart-title">Latency Comparison (ms)</div>
    <canvas id="latencyChart"></canvas>
  </div>
</div>

<div class="card" style="margin-bottom: 24px;">
  <div class="chart-title">Per-Model Breakdown</div>
  <table>
    <thead><tr><th>Model</th><th>Hits</th><th>Misses</th><th>Hit Rate</th><th>Savings</th></tr></thead>
    <tbody id="modelTable"><tr><td colspan="5" style="color:#64748b;">No data yet</td></tr></tbody>
  </table>
</div>

<script>
const donutCtx = document.getElementById('hitMissChart').getContext('2d');
const donutChart = new Chart(donutCtx, {
  type: 'doughnut',
  data: {
    labels: ['Cache Hits', 'Cache Misses'],
    datasets: [{ data: [0, 0], backgroundColor: ['#22c55e', '#f87171'], borderWidth: 0 }]
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#94a3b8' } } },
    cutout: '65%'
  }
});

const barCtx = document.getElementById('latencyChart').getContext('2d');
const barChart = new Chart(barCtx, {
  type: 'bar',
  data: {
    labels: ['P50', 'P95', 'P99'],
    datasets: [
      { label: 'Hit', data: [0, 0, 0], backgroundColor: '#22c55e' },
      { label: 'Miss', data: [0, 0, 0], backgroundColor: '#f87171' }
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#94a3b8' } } },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
      y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' }, title: { display: true, text: 'ms', color: '#94a3b8' } }
    }
  }
});

function fmtTime(s) {
  if (s < 60) return Math.round(s) + 's';
  if (s < 3600) return Math.round(s/60) + 'm';
  return (s/3600).toFixed(1) + 'h';
}

async function refresh() {
  try {
    const r = await fetch('/v1/metrics/detailed');
    const d = await r.json();
    const req = d.requests || {};
    const lat = d.latency_ms || {};
    const cost = d.cost_savings || {};
    const hitL = lat.hit || {};
    const missL = lat.miss || {};

    document.getElementById('totalReqs').textContent = req.total || 0;
    document.getElementById('uptime').textContent = 'Uptime: ' + fmtTime(d.uptime_seconds || 0);
    document.getElementById('hitRate').textContent = req.hit_rate_pct || '0%';
    document.getElementById('hitMiss').textContent = (req.cache_hits||0) + ' hits / ' + (req.cache_misses||0) + ' misses';
    document.getElementById('latSaved').textContent = (lat.avg_savings_ms || 0).toFixed(1) + 'ms';
    document.getElementById('latDetail').textContent = 'Hit: ' + (hitL.avg||0).toFixed(1) + 'ms | Miss: ' + (missL.avg||0).toFixed(1) + 'ms';
    document.getElementById('costSaved').textContent = '$' + (cost.total_usd || 0).toFixed(4);
    document.getElementById('tokensSaved').textContent = (cost.total_tokens_saved || 0).toLocaleString() + ' tokens saved';

    donutChart.data.datasets[0].data = [req.cache_hits || 0, req.cache_misses || 0];
    donutChart.update();

    barChart.data.datasets[0].data = [hitL.p50||0, hitL.p95||0, hitL.p99||0];
    barChart.data.datasets[1].data = [missL.p50||0, missL.p95||0, missL.p99||0];
    barChart.update();

    const models = d.per_model || {};
    const tbody = document.getElementById('modelTable');
    if (Object.keys(models).length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:#64748b;">No data yet</td></tr>';
    } else {
      tbody.innerHTML = Object.entries(models).map(([m, s]) =>
        '<tr><td>' + m + '</td><td>' + s.hits + '</td><td>' + s.misses + '</td>' +
        '<td><span class="badge badge-green">' + (s.hit_rate * 100).toFixed(1) + '%</span></td>' +
        '<td>$' + s.savings_usd.toFixed(4) + '</td></tr>'
      ).join('');
    }
  } catch(e) { console.error('Refresh failed:', e); }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>'''