
// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════
const state = {
  clusters: [],
  logs: [],
  simJob: null,
  pollTimer: null,
  refreshTimer: null,
};

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  loadBenchmarkSamples();
  refreshAll();
  state.refreshTimer = setInterval(refreshAll, 5000);
});

async function refreshAll() {
  await Promise.all([fetchClusters(), fetchQueryLog()]);
  if (typeof renderTriggering === 'function') renderTriggering();
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK SAMPLES
// ═══════════════════════════════════════════════════════════════════
async function loadBenchmarkSamples() {
  try {
    const res = await fetch('/simulation/benchmark-samples?limit=50');
    const data = await res.json();
    const sel = document.getElementById('biasQuerySelect');
    sel.innerHTML = '<option value="">— pick a benchmark question —</option>';
    (data.samples || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.question;
      opt.textContent = s.question.length > 80 ? s.question.slice(0, 80) + '…' : s.question;
      opt.title = s.question;
      sel.appendChild(opt);
    });
  } catch(e) {
    console.warn('Could not load benchmark samples:', e);
  }
}

function onBiasSelect() {
  const val = document.getElementById('biasQuerySelect').value;
  if (val) document.getElementById('customQueryInput').value = '';
}

function onCustomQuery() {
  const val = document.getElementById('customQueryInput').value;
  if (val) document.getElementById('biasQuerySelect').value = '';
}

function clearCustomQuery() {
  document.getElementById('customQueryInput').value = '';
}

function getBiasedQuery() {
  return document.getElementById('customQueryInput').value.trim()
      || document.getElementById('biasQuerySelect').value
      || '';
}

// ═══════════════════════════════════════════════════════════════════
// SIMULATION
// ═══════════════════════════════════════════════════════════════════
async function startSimulation() {
  const biasedQuery = getBiasedQuery();
  if (!biasedQuery) { toast('Pick or type a biased query first.', 'error'); return; }

  const totalQueries = parseInt(document.getElementById('totalSlider').value);
  const biasPercent  = parseInt(document.getElementById('biasSlider').value);
  const fastMode     = document.getElementById('fastModeToggle').checked;

  setRunning(true);
  showProgress(0, 'Starting simulation…');

  try {
    const res = await fetch('/simulation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ total_queries: totalQueries, biased_query: biasedQuery, bias_percent: biasPercent, fast_mode: fastMode })
    });
    const { job_id } = await res.json();
    state.simJob = job_id;
    pollJob(job_id);
  } catch(e) {
    toast('Failed to start simulation: ' + e.message, 'error');
    setRunning(false);
    hideProgress();
  }
}

async function pollJob(jobId) {
  try {
    const res  = await fetch(`/simulation/status/${jobId}`);
    const job  = await res.json();
    const pct  = job.progress || 0;
    const done = job.results?.length || 0;
    showProgress(pct, `Running… ${done} / ${job.total} queries`);

    if (job.status === 'done') {
      clearInterval(state.pollTimer);
      setRunning(false);
      hideProgress();
      renderSimResults(job);
      await refreshAll();
      switchTabById('simresults');
      toast(`Simulation complete — ${done} queries processed. Synthesis: ${JSON.stringify(job.synthesis)}`, 'success');
    } else {
      state.pollTimer = setTimeout(() => pollJob(jobId), 800);
      await fetchClusters(); // live update clusters while running
    }
  } catch(e) {
    console.error('Poll error:', e);
    state.pollTimer = setTimeout(() => pollJob(jobId), 1500);
  }
}

// ═══════════════════════════════════════════════════════════════════
// CLUSTERS
// ═══════════════════════════════════════════════════════════════════
async function fetchClusters() {
  try {
    const res  = await fetch('/simulation/clusters');
    const data = await res.json();
    state.clusters = data.clusters || [];
    renderClusters();
    updateHeaderStats();
  } catch(e) { /* silent */ }
}

function clusterStatus(c) {
  if (c.synthesizing)     return 'syncing';
  if (c.synthesis_failed) return 'failed';
  if (c.synthesized)      return 'done';
  if (c.hit_count >= 10)  return 'ready';
  return 'idle';
}

function renderClusters() {
  const grid = document.getElementById('clusterGrid');
  document.getElementById('tabClusters').textContent = state.clusters.length;

  if (!state.clusters.length) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-icon">🔮</div>
      <div class="empty-title">No clusters yet</div>
      <div class="empty-sub">Run a simulation or send queries to the /query endpoint.</div>
    </div>`;
    return;
  }

  grid.innerHTML = state.clusters.map(c => {
    const status = clusterStatus(c);
    const statusLabel = { idle:'Idle', ready:'Ready', syncing:'Synthesizing…', done:'Synthesized', failed:'Failed' }[status];
    const cardClass   = { idle:'', ready:'is-ready', syncing:'is-syncing', done:'is-synth', failed:'is-failed' }[status];
    const badgeClass  = `badge-${status}`;
    const hitColor    = status === 'ready' ? 'var(--amber)' : status === 'done' ? 'var(--green)' : 'var(--cyan)';

    const chunks = c.chunk_ids || [];
    const chunkHtml = chunks.slice(0, 4).map(id =>
      `<span class="chunk-pill">${id}</span>`
    ).join('') + (chunks.length > 4 ? `<span class="chunk-more">+${chunks.length - 4} more</span>` : '');

    const lastSynced = c.last_synthesized_hit_count
      ? `${c.hit_count - c.last_synthesized_hit_count} since last synth` : 'Never synthesized';

    return `<div class="cluster-card ${cardClass}" id="cluster-${c.id}">
      <div class="card-header">
        <div class="card-hit-wrap">
          <div class="card-hit" style="color:${hitColor}">${c.hit_count}</div>
          <div class="card-hit-label">hits</div>
        </div>
        <div class="card-badges">
          <span class="badge ${badgeClass}">${statusLabel}</span>
          ${c.revision_count > 0 ? `<span class="badge badge-done">Rev ${c.revision_count}</span>` : ''}
        </div>
      </div>
      <div class="card-query" title="${esc(c.canonical_query)}">${esc(c.canonical_query)}</div>
      <div class="card-meta">
        <div class="meta-item">
          <span class="meta-key">First Seen</span>
          <span class="meta-val">${fmtTime(c.first_seen)}</span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Last Hit</span>
          <span class="meta-val">${fmtTime(c.last_hit)}</span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Cluster ID</span>
          <span class="meta-val">#${c.id}</span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Progress</span>
          <span class="meta-val">${lastSynced}</span>
        </div>
      </div>
      ${chunks.length ? `<div><div class="meta-key" style="margin-bottom:.3rem">Chunk IDs</div><div class="chunk-list">${chunkHtml}</div></div>` : ''}
      <div class="card-footer">
        ${c.history?.length ? `<button class="card-btn primary" onclick="showHistory(${c.id})">📖 View History (${c.revision_count})</button>` : `<button class="card-btn" disabled>No History Yet</button>`}
        <button class="card-btn" onclick="copyCluster(${c.id})">Copy JSON</button>
      </div>
    </div>`;
  }).join('');
}

function renderTriggering() {
  const tbody = document.getElementById('triggeringTableBody');
  if (!tbody) return;
  const synthesizedClusters = state.clusters.filter(c => c.synthesized && c.history && c.history.length > 0);
  document.getElementById('tabTriggering').textContent = synthesizedClusters.length;

  if (!synthesizedClusters.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:2rem;color:var(--muted)">No super nodes synthesized yet.</td></tr>';
    return;
  }

  tbody.innerHTML = synthesizedClusters.map(c => {
    const h = c.history[0]; // Latest revision
    const chunks = c.chunk_ids || [];
    
    // Find a sample asked query from logs
    const sampleLog = state.logs.find(l => l.cluster_id === c.id);
    const askedQuery = sampleLog ? sampleLog.raw_query : c.canonical_query;

    const factPct = (h.coverage_score * 100).toFixed(1) + '%';
    const relPct = ((1.0 - h.drift_margin) * 100).toFixed(1) + '%';
    const factCol = h.coverage_score >= 0.9 ? 'var(--green)' : 'var(--amber)';
    const relCol = h.drift_margin <= 0.08 ? 'var(--green)' : 'var(--amber)';

    return '<tr>' +
      '<td class="mono-sm">#' + c.id + '</td>' +
      '<td>' +
        '<div class="query-cell" style="color:var(--cyan);font-weight:600;margin-bottom:.3rem" title="' + esc(c.canonical_query) + '">' + esc(c.canonical_query) + '</div>' +
        '<div class="query-cell" style="color:var(--muted);font-size:.75rem" title="' + esc(askedQuery) + '">Asked: ' + esc(askedQuery) + '</div>' +
      '</td>' +
      '<td><button class="card-btn primary" style="padding: .4rem .8rem; width: auto;" onclick="openSynthesisModal(' + c.id + ')">▶ View Details</button></td>' +
      '<td style="color:' + factCol + ';font-weight:600">' + factPct + '</td>' +
      '<td style="color:' + relCol + ';font-weight:600">' + relPct + '</td>' +
      '<td class="mono-sm" style="color:var(--muted)">' + fmtTime(h.timestamp) + '</td>' +
    '</tr>';
  }).join('');
}

function openSynthesisModal(clusterId) {
  const c = state.clusters.find(x => x.id === clusterId);
  if (!c || !c.history || !c.history.length) return;
  const h = c.history[0];
  const chunks = c.chunk_ids || [];

  document.getElementById('synthesisModalTitle').textContent = `Synthesis Details — Cluster #${c.id}`;
  document.getElementById('synthesisModalSummary').textContent = h.summary;

  const chunksHtml = '<div class="chunk-list" style="display:flex; flex-direction:column; gap:.4rem;">' +
        chunks.map(ch => '<div style="display:flex; align-items:center; gap:.5rem;">' +
            '<button class="card-btn primary" style="padding:.2rem .4rem; flex:0 0 auto; min-width:unset; width:auto; line-height:1;" title="View Chunk Content" onclick="viewChunkContent(\'' + ch + '\')">▶</button>' +
            '<div class="chunk-pill" title="' + esc(ch) + '" style="max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;">' + esc(ch) + '</div>' +
          '</div>').join('') +
      '</div>' +
      '<div id="synthesisModalChunkContent" style="margin-top:1rem; padding:1rem; background:rgba(0,0,0,.3); border-radius:.5rem; border:1px solid var(--border); font-size:.8rem; white-space:pre-wrap; color:var(--text); max-height:250px; overflow-y:auto; display:none;"></div>';
  document.getElementById('synthesisModalChunks').innerHTML = chunksHtml;

  document.getElementById('synthesisModal').classList.add('open');
}

async function viewChunkContent(chunkId) {
  const box = document.getElementById('synthesisModalChunkContent');
  if (!box) return;
  box.style.display = 'block';
  box.innerHTML = '<span style="color:var(--muted)">Loading chunk...</span>';
  try {
    const res = await fetch('/simulation/chunk/' + encodeURIComponent(chunkId));
    const data = await res.json();
    box.innerHTML = '<div style="color:var(--cyan); font-family:var(--mono); font-size:.7rem; margin-bottom:.5rem;">' + esc(chunkId) + '</div>' + esc(data.text);
  } catch(e) {
    box.innerHTML = '<span style="color:var(--red)">Failed to load chunk: ' + esc(e.message) + '</span>';
  }
}

function closeSynthesisModal() {
  document.getElementById('synthesisModal').classList.remove('open');
}

function copyCluster(id) {
  const c = state.clusters.find(x => x.id === id);
  if (c) navigator.clipboard.writeText(JSON.stringify(c, null, 2));
  toast('Cluster JSON copied!', 'info');
}

// ═══════════════════════════════════════════════════════════════════
// QUERY LOG
// ═══════════════════════════════════════════════════════════════════
async function fetchQueryLog() {
  try {
    const res  = await fetch('/simulation/query-log?limit=150');
    const data = await res.json();
    state.logs = data.logs || [];
    renderQueryLog();
    updateHeaderStats();
  } catch(e) { /* silent */ }
}

const MAX_LAT = 3000; // ms cap for bar scaling

function renderQueryLog() {
  const tbody = document.getElementById('logTableBody');
  document.getElementById('tabLogs').textContent = state.logs.length;
  document.getElementById('hLogs').textContent   = state.logs.length + ' Queries';

  if (!state.logs.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--muted)">No query logs yet.</td></tr>';
    return;
  }

  tbody.innerHTML = state.logs.map((log, i) => {
    const chunks = log.retrieved_chunks || [];
    let chunksHtml = '<span class="mono-sm" style="color:var(--muted)">—</span>';
    if (chunks.length > 0) {
      chunksHtml = `<details class="chunk-details">
        <summary>${chunks.length} chunks</summary>
        <div class="chunk-list" style="margin-top:.4rem">
          ${chunks.map(c => `<div class="chunk-pill" title="${esc(c)}" style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c)}</div>`).join('')}
        </div>
      </details>`;
    }
    const matchHtml = log.matched_existing
      ? `<span class="match-yes">✓ Matched</span>`
      : `<span class="match-no">✦ New</span>`;

    // We don't store latency/similarity in DB; show from sim results if available
    const simEntry = getSimEntry(log.raw_query);
    const latMs    = simEntry?.latency_ms || 0;
    const simScore = simEntry?.similarity_score || 0;
    const latPct   = Math.min(latMs / MAX_LAT * 100, 100);
    const latColor = latMs < 500 ? 'var(--green)' : latMs < 1500 ? 'var(--amber)' : 'var(--red)';

    const isB = simEntry?.is_biased ? 'biased' : '';

    return `<tr class="${isB}">
      <td class="mono-sm">${state.logs.length - i}</td>
      <td><div class="query-cell" title="${esc(log.raw_query)}" onclick="alert('${esc(log.raw_query.replace(/'/g,"\\'"  ))}')">${esc(log.raw_query)}</div></td>
      <td class="mono-sm">${log.cluster_id ?? '—'}</td>
      <td class="mono-sm">${log.hit_count ?? '—'}</td>
      <td>${matchHtml}</td>
      <td>
        ${latMs ? `<div class="latency-bar">
          <div class="latency-bg"><div class="latency-fill" style="width:${latPct}%;background:${latColor}"></div></div>
          <span class="latency-val">${latMs.toFixed(0)}ms</span>
        </div>` : '<span class="mono-sm" style="color:var(--muted)">—</span>'}
      </td>
      <td class="mono-sm">${simScore ? (simScore*100).toFixed(1)+'%' : '—'}</td>
      <td>${chunksHtml}</td>
      <td class="mono-sm" style="color:var(--muted)">${fmtTime(log.timestamp)}</td>
    </tr>`;
  }).join('');
}

function getSimEntry(query) {
  if (!state.simJob) return null;
  const jobs = Object.values({});  // placeholder; results stored in simResults
  return state._simResults?.find(r => r.query === query) || null;
}

// ═══════════════════════════════════════════════════════════════════
// SIM RESULTS PANEL
// ═══════════════════════════════════════════════════════════════════
function renderSimResults(job) {
  state._simResults = (job.results || []).concat(state._simResults || []);
  const synth  = job.synthesis || {};
  const biased = (job.results || []).filter(r => r.is_biased).length;
  const errors = (job.results || []).filter(r => r.error).length;
  const avgLat = job.results?.length
    ? (job.results.reduce((a,b) => a + (b.latency_ms||0), 0) / job.results.length).toFixed(1)
    : 0;

  document.getElementById('tabResults').textContent = state._simResults.length;

  state._runCounter = (state._runCounter || 0) + 1;
  const runId = state._runCounter;

  const html = `
    <div class="sim-run-block" style="margin-bottom: 2rem;">
      <h3 style="margin-bottom: 1rem; color: var(--text); border-bottom: 1px dashed var(--border); padding-bottom: .5rem;">
        Simulation Run #${runId} <span style="font-size:.7rem;color:var(--muted);margin-left:.5rem;font-weight:normal">(${new Date().toLocaleTimeString()})</span>
      </h3>
      <div class="sim-summary">
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--cyan)">${job.total_queries}</span><span class="sim-stat-key">Total Queries</span></div>
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--amber)">${biased}</span><span class="sim-stat-key">Biased Queries</span></div>
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--green)">${synth.triggered ?? 0}</span><span class="sim-stat-key">Synthesis Triggered</span></div>
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--purple)">${synth.skipped ?? 0}</span><span class="sim-stat-key">Skipped (P4 stub)</span></div>
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--red)">${errors}</span><span class="sim-stat-key">Errors</span></div>
        <div class="sim-stat"><span class="sim-stat-val" style="color:var(--blue)">${avgLat}ms</span><span class="sim-stat-key">Avg Latency</span></div>
      </div>
      <div class="log-table-wrap" style="margin-top:1rem">
        <table class="log-table">
          <thead>
            <tr>
              <th>#</th><th>Query</th><th>Biased</th><th>Cluster</th>
              <th>Hit Count</th><th>Match</th><th>Latency</th><th>Similarity</th><th>Chunks</th><th>Time</th>
            </tr>
          </thead>
          <tbody>
            ${(job.results || []).map(r => {
              const latMs  = r.latency_ms || 0;
              const latPct = Math.min(latMs / MAX_LAT * 100, 100);
              const latCol = latMs < 500 ? 'var(--green)' : latMs < 1500 ? 'var(--amber)' : 'var(--red)';
              const chunksArr = r.chunk_ids || [];
              let chunksHtml = '<span class="mono-sm" style="color:var(--muted)">—</span>';
              if (chunksArr.length > 0) {
                chunksHtml = `<details class="chunk-details">
                  <summary>${chunksArr.length} chunks</summary>
                  <div class="chunk-list" style="margin-top:.4rem">
                    ${chunksArr.map(c => `<div class="chunk-pill" title="${esc(c)}" style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(c)}</div>`).join('')}
                  </div>
                </details>`;
              }
              return `<tr class="${r.is_biased ? 'biased' : ''}">
                <td class="mono-sm">${r.index}</td>
                <td><div class="query-cell" title="${esc(r.query)}">${esc(r.query)}</div></td>
                <td style="text-align:center">${r.is_biased ? '<span class="badge badge-ready">⭐ Biased</span>' : '—'}</td>
                <td class="mono-sm">${r.cluster_id ?? '—'}</td>
                <td class="mono-sm">${r.hit_count ?? '—'}</td>
                <td>${r.matched_existing ? '<span class="match-yes">✓</span>' : '<span class="match-no">✦</span>'}</td>
                <td><div class="latency-bar">
                  <div class="latency-bg"><div class="latency-fill" style="width:${latPct}%;background:${latCol}"></div></div>
                  <span class="latency-val">${latMs.toFixed(0)}ms</span>
                </div></td>
                <td class="mono-sm">${r.similarity_score ? (r.similarity_score*100).toFixed(1)+'%' : '—'}</td>
                <td>${chunksHtml}</td>
                <td class="mono-sm" style="color:var(--muted)">${r.timestamp ? r.timestamp.slice(11,19) : '—'}</td>
              </tr>`;
            }).join('')}
          </tbody>
        </table>
      </div>
    </div>`;

  const content = document.getElementById('simResultsContent');
  if (runId === 1) {
    content.innerHTML = html;
  } else {
    content.innerHTML = html + content.innerHTML;
  }
}

// ═══════════════════════════════════════════════════════════════════
// HISTORY MODAL
// ═══════════════════════════════════════════════════════════════════
function showHistory(clusterId) {
  const c = state.clusters.find(x => x.id === clusterId);
  if (!c) return;
  document.getElementById('modalTitle').textContent = `Cluster #${c.id} — Synthesis History`;
  const body = document.getElementById('modalBody');
  if (!c.history?.length) {
    body.innerHTML = '<div style="color:var(--muted);text-align:center;padding:1rem">No synthesis history yet.</div>';
  } else {
    body.innerHTML = c.history.map((h, i) => {
      const isLatest = i === 0;
      const isFirst  = i === c.history.length - 1;
      return `<div class="revision-card" style="${isLatest ? 'border-color:rgba(52,211,153,.35)' : ''}">
        <div class="rev-header">
          <span class="rev-num">${isLatest ? '✦ Latest — ' : ''}Revision ${h.revision}${isFirst && c.history.length > 1 ? ' (original)' : ''}</span>
          <div class="rev-scores">
            <span class="rev-score" style="color:var(--green)">Coverage: ${(h.coverage_score*100).toFixed(1)}%</span>
            <span class="rev-score" style="color:${h.drift_margin > 0.08 ? 'var(--amber)' : 'var(--muted)'}">Drift: ${h.drift_margin.toFixed(3)}</span>
          </div>
        </div>
        <div class="rev-text">${esc(h.summary || 'No summary stored.')}</div>
        <div class="rev-ts">${h.timestamp}</div>
      </div>`;
    }).join('');
  }
  document.getElementById('historyModal').classList.add('open');
}

function closeModal() {
  document.getElementById('historyModal').classList.remove('open');
}

// ═══════════════════════════════════════════════════════════════════
// SYNTHESIS TRIGGER
// ═══════════════════════════════════════════════════════════════════
async function triggerSynthesis() {
  toast('Firing synthesis trigger…', 'info');
  try {
    const res  = await fetch('/simulation/trigger-synthesis', { method: 'POST' });
    const data = await res.json();
    toast(`Synthesis done — triggered:${data.synthesis.triggered} skipped:${data.synthesis.skipped} failed:${data.synthesis.failed}`, 'success');
    await fetchClusters();
  } catch(e) {
    toast('Synthesis trigger failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// ADMIN LOGIN + BACKUP
// ═══════════════════════════════════════════════════════════════════
const ADMIN_USER = 'admin';
const ADMIN_PASS = 'sera@admin';

function openAdminLogin() {
  document.getElementById('adminUser').value = '';
  document.getElementById('adminPass').value = '';
  document.getElementById('adminErr').textContent = '';
  document.getElementById('adminLoginModal').classList.add('open');
  setTimeout(() => document.getElementById('adminUser').focus(), 100);
}

function closeAdminLogin() {
  document.getElementById('adminLoginModal').classList.remove('open');
}

function submitAdminLogin() {
  const user = document.getElementById('adminUser').value.trim();
  const pass = document.getElementById('adminPass').value;
  if (user === ADMIN_USER && pass === ADMIN_PASS) {
    closeAdminLogin();
    openBackupModal();
  } else {
    document.getElementById('adminErr').textContent = '✕ Invalid credentials. Please try again.';
    document.getElementById('adminPass').value = '';
    document.getElementById('adminPass').focus();
  }
}

function openBackupModal() {
  // Auto-suggest a version tag based on current date
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('backupVersion').value = today;
  document.getElementById('backupMessage').value = '';
  document.getElementById('backupErr').textContent = '';
  document.getElementById('backupPath').style.display = 'none';
  document.getElementById('backupSubmitBtn').disabled = false;
  document.getElementById('backupSubmitBtn').textContent = 'Create Backup';
  document.getElementById('backupModal').classList.add('open');
  setTimeout(() => document.getElementById('backupMessage').focus(), 100);
}

function closeBackupModal() {
  document.getElementById('backupModal').classList.remove('open');
}

async function executeBackup() {
  const version = document.getElementById('backupVersion').value.trim();
  const message = document.getElementById('backupMessage').value.trim();

  if (!version) {
    document.getElementById('backupErr').textContent = '✕ Version tag is required.';
    return;
  }
  if (!message) {
    document.getElementById('backupErr').textContent = '✕ Commit message is required.';
    return;
  }

  document.getElementById('backupErr').textContent = '';
  document.getElementById('backupSubmitBtn').disabled = true;
  document.getElementById('backupSubmitBtn').textContent = '⏳ Creating backup…';

  try {
    const res = await fetch('/admin/backup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version, message })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Backup failed.');
    }

    // Show success path
    const pathEl = document.getElementById('backupPath');
    pathEl.textContent = '✓ Saved to: ' + (data.backup_path || 'data/backups/' + version);
    pathEl.style.display = 'block';

    document.getElementById('backupSubmitBtn').textContent = '✓ Done';
    toast(`Backup "${version}" created successfully.`, 'success');

    // Auto-close after 2s
    setTimeout(closeBackupModal, 2000);

  } catch(e) {
    document.getElementById('backupErr').textContent = '✕ ' + e.message;
    document.getElementById('backupSubmitBtn').disabled = false;
    document.getElementById('backupSubmitBtn').textContent = 'Create Backup';
    toast('Backup failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// RESET
// ═══════════════════════════════════════════════════════════════════
async function confirmReset() {
  if (!confirm('⚠️ This will DELETE all clusters, query logs, and simulation jobs.\n\nbenchmark_qa data is preserved.\n\nAre you sure?')) return;
  try {
    await fetch('/simulation/reset', { method: 'POST' });
    state.clusters = []; state.logs = []; state._simResults = null;
    renderClusters(); renderQueryLog();
    document.getElementById('simResultsContent').innerHTML = 'Run a simulation to see per-query results here.';
    document.getElementById('tabResults').textContent = '—';
    updateHeaderStats();
    toast('All clusters and logs cleared.', 'success');
  } catch(e) {
    toast('Reset failed: ' + e.message, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// HEADER STATS
// ═══════════════════════════════════════════════════════════════════
function updateHeaderStats() {
  const total  = state.clusters.length;
  const ready  = state.clusters.filter(c => c.hit_count >= 10 && !c.synthesized && !c.synthesizing).length;
  const synth  = state.clusters.filter(c => c.synthesized).length;
  document.getElementById('hClusters').textContent   = total + ' Clusters';
  document.getElementById('hReady').textContent      = ready + ' Ready';
  document.getElementById('hSynthesized').textContent = synth + ' Synthesized';
}

// ═══════════════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════════════
function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}

function switchTabById(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${name}"]`)?.classList.add('active');
  document.getElementById('panel-' + name)?.classList.add('active');
}

// ═══════════════════════════════════════════════════════════════════
// PROGRESS UI
// ═══════════════════════════════════════════════════════════════════
function showProgress(pct, text) {
  document.getElementById('progressWrap').classList.add('visible');
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = text;
}
function hideProgress() {
  document.getElementById('progressWrap').classList.remove('visible');
}
function setRunning(running) {
  document.getElementById('runBtn').disabled = running;
  document.getElementById('runBtn').textContent = running ? '⏳ Running…' : '▶ Run Simulation';
}

// ═══════════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════════
function toast(msg, type = 'info') {
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  const colors = { success: 'var(--green)', error: 'var(--red)', info: 'var(--cyan)' };
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.innerHTML = `<span style="color:${colors[type]}">${icons[type]}</span> ${esc(msg)}`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ═══════════════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════════════
function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts.replace(' ', 'T') + (ts.includes('Z') ? '' : 'Z'));
    return d.toLocaleTimeString('en-GB', { hour12: false });
  } catch { return ts.slice(11, 19) || ts; }
}
