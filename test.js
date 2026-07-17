
window.onerror=function(m,s,l){
  const b=document.getElementById('js-error-banner');
  if(b){b.style.display='block';b.textContent='JS ERROR line '+l+': '+m}
};

/* ══ STATE ══ */
const S={
  clusters:[],logs:[],simJob:null,
  pollTimer:null,refreshTimer:null,
  _simResults:null,_runCounter:0,
  allSN:[],snFilter:'all',
  queryHistory:JSON.parse(sessionStorage.getItem('sera_hist')||'[]'),
};
let BASE_URL=localStorage.getItem('sera_base_url')||'http://localhost:8000';
const MAX_LAT=3000;
const ADMIN_USER='admin', ADMIN_PASS='sera@admin';

/* ══ INIT ══ */
window.addEventListener('DOMContentLoaded',()=>{
  loadBenchmarkSamples();
  refreshAll();
  S.refreshTimer=setInterval(refreshAll,5000);
  checkHealth();
  setInterval(checkHealth,5000);
});

async function refreshAll(){
  await Promise.all([fetchClusters(),fetchQueryLog()]);
  if(typeof renderTriggering==='function')renderTriggering();
}

/* ══ API helper ══ */
async function api(path,opts={}){
  const r=await fetch(BASE_URL+path,{headers:{'Content-Type':'application/json',...(opts.headers||{})},...opts});
  if(!r.ok){const e=await r.json().catch(()=>({detail:r.statusText}));throw new Error(e.detail||r.statusText)}
  return r.json();
}

/* ══ HEALTH ══ */
async function checkHealth(){
  const d=document.getElementById('healthDot'),l=document.getElementById('healthLabel');
  try{await api('/health');d.classList.add('online');l.textContent='Online'}
  catch{d.classList.remove('online');l.textContent='Offline'}
}

/* ══ SETTINGS ══ */
function openSettings(){document.getElementById('baseUrlInput').value=BASE_URL;document.getElementById('settingsModal').classList.add('open')}
function closeSettings(){document.getElementById('settingsModal').classList.remove('open')}
function saveSettings(){
  BASE_URL=document.getElementById('baseUrlInput').value.replace(/\/$/,'');
  localStorage.setItem('sera_base_url',BASE_URL);
  closeSettings();checkHealth();toast('API URL updated','info');
}

/* ══ BENCHMARK SAMPLES ══ */
async function loadBenchmarkSamples(){
  try{
    const r=await fetch(BASE_URL+'/simulation/benchmark-samples?limit=50');
    const d=await r.json();
    const sel=document.getElementById('biasQuerySelect');
    sel.innerHTML='<option value="">— pick a benchmark question —</option>';
    (d.samples||[]).forEach(s=>{
      const o=document.createElement('option');
      o.value=s.question;
      o.textContent=s.question.length>80?s.question.slice(0,80)+'…':s.question;
      o.title=s.question;sel.appendChild(o);
    });
  }catch(e){console.warn('benchmark samples:',e)}
}
function onBiasSelect(){const v=document.getElementById('biasQuerySelect').value;if(v)document.getElementById('customQueryInput').value=''}
function onCustomQuery(){const v=document.getElementById('customQueryInput').value;if(v)document.getElementById('biasQuerySelect').value=''}
function clearCustomQuery(){document.getElementById('customQueryInput').value=''}
function getBiasedQuery(){return document.getElementById('customQueryInput').value.trim()||document.getElementById('biasQuerySelect').value||''}

/* ══ SIMULATION ══ */
async function startSimulation(){
  const bq=getBiasedQuery();
  if(!bq){toast('Pick or type a biased query first.','error');return}
  const total=parseInt(document.getElementById('totalSlider').value);
  const bias=parseInt(document.getElementById('biasSlider').value);
  const fast=document.getElementById('fastModeToggle').checked;
  setRunning(true);showProgress(0,'Starting simulation…');
  try{
    const d=await api('/simulation/run',{method:'POST',body:JSON.stringify({total_queries:total,biased_query:bq,bias_percent:bias,fast_mode:fast})});
    S.simJob=d.job_id;pollJob(d.job_id);
  }catch(e){toast('Failed to start: '+e.message,'error');setRunning(false);hideProgress()}
}

async function pollJob(jobId){
  try{
    const job=await api('/simulation/status/'+jobId);
    const pct=job.progress||0,done=(job.results||[]).length;
    showProgress(pct,'Running… '+done+' / '+job.total+' queries');
    if(job.status==='done'){
      clearTimeout(S.pollTimer);setRunning(false);hideProgress();
      renderSimResults(job);await refreshAll();switchTabById('simresults');
      toast('Simulation complete — '+done+' queries. Synthesis: '+JSON.stringify(job.synthesis),'success');
    }else{
      S.pollTimer=setTimeout(()=>pollJob(jobId),800);
      fetchClusters();
    }
  }catch(e){console.error('poll:',e);S.pollTimer=setTimeout(()=>pollJob(jobId),1500)}
}

/* ══ CLUSTERS ══ */
async function fetchClusters(){
  try{const d=await api('/simulation/clusters');S.clusters=d.clusters||[];renderClusters();updateHeaderStats()}
  catch(e){/*silent*/}
}
function clusterStatus(c){
  if(c.synthesizing)return'syncing';if(c.synthesis_failed)return'failed';
  if(c.synthesized)return'done';if(c.hit_count>=10)return'ready';return'idle';
}
function renderClusters(){
  const g=document.getElementById('clusterGrid');
  document.getElementById('tabClusters').textContent=S.clusters.length;
  if(!S.clusters.length){
    g.innerHTML='<div class="empty-state"><div class="empty-icon">🔮</div><div class="empty-title">No clusters yet</div><div class="empty-sub">Run a simulation or send queries to /query to generate clusters.</div></div>';
    return;
  }
  g.innerHTML=S.clusters.map(c=>{
    const st=clusterStatus(c);
    const stLabel={idle:'Idle',ready:'Ready',syncing:'Synthesizing…',done:'Synthesized',failed:'Failed'}[st];
    const cardCls={idle:'',ready:'is-ready',syncing:'is-syncing',done:'is-synth',failed:'is-failed'}[st];
    const badgeCls='badge-'+st;
    const hitCol=st==='ready'?'var(--amber2)':st==='done'?'var(--green2)':'var(--cyan)';
    const chunks=c.chunk_ids||[];
    const chunkHtml=chunks.slice(0,4).map(id=>'<span class="chunk-pill">'+esc(id)+'</span>').join('')
      +(chunks.length>4?'<span class="chunk-more">+'+( chunks.length-4)+' more</span>':'');
    const lastSynced=c.last_synthesized_hit_count
      ?(c.hit_count-c.last_synthesized_hit_count)+' since last synth':'Never synthesized';
    return `<div class="cluster-card ${cardCls}" id="cluster-${c.id}">
      <div class="card-header">
        <div class="card-hit-wrap"><div class="card-hit" style="color:${hitCol}">${c.hit_count}</div><div class="card-hit-label">hits</div></div>
        <div class="card-badges">
          <span class="badge ${badgeCls}">${stLabel}</span>
          ${c.revision_count>0?'<span class="badge badge-done">Rev '+c.revision_count+'</span>':''}
        </div>
      </div>
      <div class="card-query" title="${esc(c.canonical_query)}">${esc(c.canonical_query)}</div>
      <div class="card-meta">
        <div class="meta-item"><span class="meta-key">First Seen</span><span class="meta-val">${fmtTime(c.first_seen)}</span></div>
        <div class="meta-item"><span class="meta-key">Last Hit</span><span class="meta-val">${fmtTime(c.last_hit)}</span></div>
        <div class="meta-item"><span class="meta-key">Cluster ID</span><span class="meta-val">#${c.id}</span></div>
        <div class="meta-item"><span class="meta-key">Progress</span><span class="meta-val">${lastSynced}</span></div>
      </div>
      ${chunks.length?'<div><div class="meta-key" style="margin-bottom:.3rem">Chunk IDs</div><div class="chunk-list">'+chunkHtml+'</div></div>':''}
      <div class="card-footer">
        ${c.history?.length?'<button class="card-btn primary" onclick="showHistory('+c.id+')">📖 History ('+c.revision_count+')</button>':'<button class="card-btn" disabled>No History</button>'}
        <button class="card-btn" onclick="copyCluster(${c.id})">Copy JSON</button>
      </div>
    </div>`;
  }).join('');
}

/* ══ TRIGGERING TAB ══ */
function renderTriggering(){
  const tbody=document.getElementById('triggeringTableBody');
  if(!tbody)return;
  const synth=S.clusters.filter(c=>c.synthesized&&c.history&&c.history.length>0);
  document.getElementById('tabTriggering').textContent=synth.length;
  if(!synth.length){tbody.innerHTML='<tr><td colspan="6" style="text-align:center;padding:2rem;color:var(--muted)">No super-nodes synthesized yet.</td></tr>';return}
  tbody.innerHTML=synth.map(c=>{
    const h=c.history[0];
    const sampleLog=S.logs.find(l=>l.cluster_id===c.id);
    const askedQ=sampleLog?sampleLog.raw_query:c.canonical_query;
    const factPct=(h.coverage_score*100).toFixed(1)+'%';
    const rawRel=typeof h.sim_summary_to_query==='number'?h.sim_summary_to_query:(1.0-h.drift_margin);
    const relNum=Math.min(100,Math.max(0,rawRel*100));
    const relPct=relNum.toFixed(1)+'%';
    const factCol=h.coverage_score>=0.9?'var(--green2)':'var(--amber2)';
    const relCol=relNum>=80?'var(--green2)':relNum>=60?'var(--amber)':'var(--red)';
    return '<tr>'
      +'<td class="mono-sm">#'+c.id+'</td>'
      +'<td><div class="query-cell" style="color:var(--cyan);font-weight:600;margin-bottom:.25rem" title="'+esc(c.canonical_query)+'">'+esc(c.canonical_query)+'</div>'
      +'<div class="query-cell" style="color:var(--muted);font-size:.73rem" title="'+esc(askedQ)+'">Asked: '+esc(askedQ)+'</div></td>'
      +'<td><button class="card-btn primary" style="padding:.35rem .7rem;width:auto;min-width:unset" onclick="openSynthesisModal('+c.id+')">▶ View Details</button></td>'
      +'<td style="color:'+factCol+';font-weight:600">'+factPct+'</td>'
      +'<td style="color:'+relCol+';font-weight:600">'+relPct+'</td>'
      +'<td class="mono-sm" style="color:var(--muted)">'+fmtTime(h.timestamp)+'</td>'
      +'</tr>';
  }).join('');
}

/* ══ SYNTHESIS MODAL ══ */
function openSynthesisModal(clusterId){
  const c=S.clusters.find(x=>x.id===clusterId);
  if(!c||!c.history?.length)return;
  const h=c.history[0];const chunks=c.chunk_ids||[];
  document.getElementById('synthesisModalTitle').textContent='Synthesis Details — Cluster #'+c.id;
  document.getElementById('synthesisModalSummary').textContent=h.summary||'';
  document.getElementById('synthesisModalChunks').innerHTML=
    '<div class="chunk-list" style="display:flex;flex-direction:column;gap:.35rem">'
    +chunks.map(ch=>'<div style="display:flex;align-items:center;gap:.4rem">'
      +'<button class="card-btn primary" style="padding:.2rem .4rem;flex:0 0 auto;min-width:unset;width:auto;line-height:1" onclick="viewChunkContent(\''+esc(ch)+'\')">▶</button>'
      +'<div class="chunk-pill" title="'+esc(ch)+'" style="max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">'+esc(ch)+'</div>'
      +'</div>').join('')
    +'</div>'
    +'<div class="synth-chunk-content" id="synthChunkContent"></div>'
    +'<div style="margin-top:1.2rem;padding-top:.9rem;border-top:1px solid var(--border)">'
    +'<button class="btn btn-primary" style="width:100%;justify-content:center;font-size:.83rem" onclick="openFullValidationModal('+c.id+')">⚖️ Run Full Validation Suite (Phase 4)</button>'
    +'</div>';
  document.getElementById('synthesisModal').classList.add('open');
}
async function viewChunkContent(chunkId){
  const box=document.getElementById('synthChunkContent');
  if(!box)return;
  box.style.display='block';
  box.innerHTML='<span style="color:var(--muted)">Loading chunk…</span>';
  try{
    const r=await fetch(BASE_URL+'/simulation/chunk/'+encodeURIComponent(chunkId));
    const d=await r.json();
    box.innerHTML='<div style="color:var(--cyan);font-family:var(--mono);font-size:.7rem;margin-bottom:.4rem">'+esc(chunkId)+'</div>'+esc(d.text);
  }catch(e){box.innerHTML='<span style="color:var(--red)">Failed: '+esc(e.message)+'</span>'}
}
function closeSynthesisModal(){document.getElementById('synthesisModal').classList.remove('open')}

/* ══ FULL VALIDATION MODAL ══ */
async function openFullValidationModal(clusterId){
  const c=S.clusters.find(x=>x.id===clusterId);
  if(!c||!c.history?.length)return;
  const h=c.history[0];const chunks=c.chunk_ids||[];
  document.getElementById('fullValidationModal').classList.add('open');
  document.getElementById('fullValLoading').style.display='block';
  document.getElementById('fullValContent').style.display='none';
  try{
    const data=await api('/simulation/validate-test',{method:'POST',body:JSON.stringify({chunk_ids:chunks,summary_text:h.summary,canonical_query:c.canonical_query})});
    const badge=document.getElementById('fValBadge');
    if(data.su2.passed){badge.textContent='PASSED';badge.style.color='var(--green)';badge.style.borderColor='rgba(16,185,129,.3)';badge.style.backgroundColor='rgba(16,185,129,.1)'}
    else{badge.textContent='FAILED';badge.style.color='var(--red)';badge.style.borderColor='rgba(239,68,68,.3)';badge.style.backgroundColor='rgba(239,68,68,.1)'}
    document.getElementById('fValScore').textContent=(data.su2.coverage_score*100).toFixed(1)+'%';
    document.getElementById('fValScore').style.color=data.su2.passed?'var(--green)':'var(--red)';
    const mf=document.getElementById('fValMissingFacts');
    mf.innerHTML=data.su2.missing_facts?.length?data.su2.missing_facts.map(f=>'<li style="margin-bottom:.2rem">'+esc(f)+'</li>').join(''):'<div style="color:var(--green);font-size:.8rem">None! All facts covered.</div>';
    const hal=document.getElementById('fValHallucinations');
    hal.innerHTML=data.hallucination.added_entities?.length?data.hallucination.added_entities.map(e=>'<li style="margin-bottom:.2rem;color:var(--amber)">'+esc(e)+'</li>').join(''):'<div style="color:var(--green);font-size:.8rem">No made-up entities detected.</div>';
    document.getElementById('fValSimQuery').textContent=data.su14.sim_summary_to_query.toFixed(3);
    document.getElementById('fValSimSource').textContent=data.su14.sim_summary_to_source.toFixed(3);
    document.getElementById('fValSimChunk').textContent=data.su14.sim_raw_to_query_max.toFixed(3);
    document.getElementById('fValDrift').textContent=data.su14.drift_margin.toFixed(3);
    document.getElementById('fValDriftWarning').style.display=(data.su14.suspected_overfit||data.su14.suspected_disconnect)?'block':'none';
    document.getElementById('fullValLoading').style.display='none';
    document.getElementById('fullValContent').style.display='block';
  }catch(e){document.getElementById('fullValLoading').innerHTML='<div style="color:var(--red)">Error: '+esc(e.message)+'</div>'}
}
function closeFullValidationModal(){document.getElementById('fullValidationModal').classList.remove('open')}

function copyCluster(id){const c=S.clusters.find(x=>x.id===id);if(c)navigator.clipboard.writeText(JSON.stringify(c,null,2));toast('Cluster JSON copied!','info')}

/* ══ QUERY LOG ══ */
async function fetchQueryLog(){
  try{const d=await api('/simulation/query-log?limit=150');S.logs=d.logs||[];renderQueryLog();updateHeaderStats()}
  catch(e){/*silent*/}
}
function renderQueryLog(){
  const tbody=document.getElementById('logTableBody');
  document.getElementById('tabLogs').textContent=S.logs.length;
  document.getElementById('hLogs').textContent=S.logs.length+' Queries';
  if(!S.logs.length){tbody.innerHTML='<tr><td colspan="9" style="text-align:center;padding:2rem;color:var(--muted)">No query logs yet.</td></tr>';return}
  tbody.innerHTML=S.logs.map((log,i)=>{
    const chunks=log.retrieved_chunks||[];
    let chunksHtml='<span class="mono-sm" style="color:var(--muted)">—</span>';
    if(chunks.length){chunksHtml=`<details class="chunk-details"><summary>${chunks.length} chunks</summary><div class="chunk-list" style="margin-top:.35rem">${chunks.map(c=>'<div class="chunk-pill" title="'+esc(c)+'" style="max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(c)+'</div>').join('')}</div></details>`}
    const matchHtml=log.matched_existing?'<span class="match-yes">✓ Matched</span>':'<span class="match-no">✦ New</span>';
    const simEntry=getSimEntry(log.raw_query);
    const latMs=simEntry?.latency_ms||0,simScore=simEntry?.similarity_score||0;
    const latPct=Math.min(latMs/MAX_LAT*100,100);
    const latCol=latMs<500?'var(--green2)':latMs<1500?'var(--amber2)':'var(--red2)';
    const isB=simEntry?.is_biased?'biased':'';
    return `<tr class="${isB}">
      <td class="mono-sm">${S.logs.length-i}</td>
      <td><div class="query-cell" title="${esc(log.raw_query)}" onclick="alert('${esc(log.raw_query.replace(/'/g,"\\'"))}')">${esc(log.raw_query)}</div></td>
      <td class="mono-sm">${log.cluster_id??'—'}</td>
      <td class="mono-sm">${log.hit_count??'—'}</td>
      <td>${matchHtml}</td>
      <td>${latMs?`<div class="latency-bar"><div class="latency-bg"><div class="latency-fill" style="width:${latPct}%;background:${latCol}"></div></div><span class="latency-val">${latMs.toFixed(0)}ms</span></div>`:'<span class="mono-sm" style="color:var(--muted)">—</span>'}</td>
      <td class="mono-sm">${simScore?(simScore*100).toFixed(1)+'%':'—'}</td>
      <td>${chunksHtml}</td>
      <td class="mono-sm" style="color:var(--muted)">${fmtTime(log.timestamp)}</td>
    </tr>`;
  }).join('');
}
function getSimEntry(query){return S._simResults?.find(r=>r.query===query)||null}

/* ══ SIM RESULTS ══ */
function renderSimResults(job){
  S._simResults=(job.results||[]).concat(S._simResults||[]);
  const synth=job.synthesis||{},biased=(job.results||[]).filter(r=>r.is_biased).length;
  const errors=(job.results||[]).filter(r=>r.error).length;
  const avgLat=job.results?.length?(job.results.reduce((a,b)=>a+(b.latency_ms||0),0)/job.results.length).toFixed(1):0;
  document.getElementById('tabResults').textContent=S._simResults.length;
  S._runCounter=(S._runCounter||0)+1;const runId=S._runCounter;
  const html=`<div class="sim-run-block" style="margin-bottom:2rem">
    <h3 style="margin-bottom:.85rem;color:var(--text);border-bottom:1px dashed var(--border);padding-bottom:.45rem">
      Simulation Run #${runId} <span style="font-size:.7rem;color:var(--muted);margin-left:.5rem;font-weight:400">(${new Date().toLocaleTimeString()})</span>
    </h3>
    <div class="sim-summary">
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--cyan)">${job.total||0}</span><span class="sim-stat-key">Total Queries</span></div>
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--amber)">${biased}</span><span class="sim-stat-key">Biased Queries</span></div>
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--green2)">${synth.triggered??0}</span><span class="sim-stat-key">Synthesis Triggered</span></div>
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--purple)">${synth.skipped??0}</span><span class="sim-stat-key">Skipped</span></div>
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--red)">${errors}</span><span class="sim-stat-key">Errors</span></div>
      <div class="sim-stat"><span class="sim-stat-val" style="color:var(--blue)">${avgLat}ms</span><span class="sim-stat-key">Avg Latency</span></div>
    </div>
    <div class="log-table-wrap" style="margin-top:.85rem">
      <table class="log-table">
        <thead><tr><th>#</th><th>Query</th><th>Biased</th><th>Cluster</th><th>Hit Count</th><th>Match</th><th>Latency</th><th>Similarity</th><th>Chunks</th><th>Time</th></tr></thead>
        <tbody>${(job.results||[]).map(r=>{
          const latMs=r.latency_ms||0,latPct=Math.min(latMs/MAX_LAT*100,100);
          const latCol=latMs<500?'var(--green2)':latMs<1500?'var(--amber2)':'var(--red2)';
          const chunksArr=r.chunk_ids||[];
          let cH='<span class="mono-sm" style="color:var(--muted)">—</span>';
          if(chunksArr.length)cH=`<details class="chunk-details"><summary>${chunksArr.length} chunks</summary><div class="chunk-list" style="margin-top:.35rem">${chunksArr.map(c=>'<div class="chunk-pill" title="'+esc(c)+'" style="max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc(c)+'</div>').join('')}</div></details>`;
          return `<tr class="${r.is_biased?'biased':''}">
            <td class="mono-sm">${r.index}</td>
            <td><div class="query-cell" title="${esc(r.query)}">${esc(r.query)}</div></td>
            <td style="text-align:center">${r.is_biased?'<span class="badge badge-ready">⭐</span>':'—'}</td>
            <td class="mono-sm">${r.cluster_id??'—'}</td>
            <td class="mono-sm">${r.hit_count??'—'}</td>
            <td>${r.matched_existing?'<span class="match-yes">✓</span>':'<span class="match-no">✦</span>'}</td>
            <td><div class="latency-bar"><div class="latency-bg"><div class="latency-fill" style="width:${latPct}%;background:${latCol}"></div></div><span class="latency-val">${latMs.toFixed(0)}ms</span></div></td>
            <td class="mono-sm">${r.similarity_score?(r.similarity_score*100).toFixed(1)+'%':'—'}</td>
            <td>${cH}</td>
            <td class="mono-sm" style="color:var(--muted)">${r.timestamp?r.timestamp.slice(11,19):'—'}</td>
          </tr>`;
        }).join('')}</tbody>
      </table>
    </div>
  </div>`;
  const content=document.getElementById('simResultsContent');
  content.innerHTML=runId===1?html:html+content.innerHTML;
}

/* ══ HISTORY MODAL ══ */
function showHistory(clusterId){
  const c=S.clusters.find(x=>x.id===clusterId);
  if(!c)return;
  document.getElementById('modalTitle').textContent='Cluster #'+c.id+' — Synthesis History';
  const body=document.getElementById('modalBody');
  if(!c.history?.length){body.innerHTML='<div style="color:var(--muted);text-align:center;padding:1rem">No synthesis history yet.</div>'}
  else{
    body.innerHTML=c.history.map((h,i)=>{
      const isLatest=i===0,isFirst=i===c.history.length-1;
      return `<div class="revision-card" style="${isLatest?'border-color:rgba(52,211,153,.35)':''}">
        <div class="rev-header">
          <span class="rev-num">${isLatest?'✦ Latest — ':''}Revision ${h.revision}${isFirst&&c.history.length>1?' (original)':''}</span>
          <div class="rev-scores">
            <span class="rev-score" style="color:var(--green2)">Coverage: ${(h.coverage_score*100).toFixed(1)}%</span>
            <span class="rev-score" style="color:${h.drift_margin>0.08?'var(--amber)':'var(--muted)'}">Drift: ${h.drift_margin.toFixed(3)}</span>
          </div>
        </div>
        <div class="rev-text">${esc(h.summary||'No summary stored.')}</div>
        <div class="rev-ts">${h.timestamp}</div>
      </div>`;
    }).join('');
  }
  document.getElementById('historyModal').classList.add('open');
}
function closeModal(){document.getElementById('historyModal').classList.remove('open')}

/* ══ SYNTHESIS TRIGGER ══ */
async function triggerSynthesis(){
  toast('Firing synthesis trigger…','info');
  try{
    const d=await api('/simulation/trigger-synthesis',{method:'POST'});
    toast('Synthesis — triggered:'+d.synthesis.triggered+' skipped:'+d.synthesis.skipped+' failed:'+d.synthesis.failed,'success');
    fetchClusters();
  }catch(e){toast('Trigger failed: '+e.message,'error')}
}

/* ══ SUPER-NODES TAB ══ */
async function loadSuperNodes(){
  try{
    const d=await api('/admin/super-nodes');
    S.allSN=d.nodes||[];
    const stale=S.allSN.filter(n=>n.is_stale).length;
    const flagged=S.allSN.filter(n=>n.fidelity_flagged).length;
    const avgCov=S.allSN.length?(S.allSN.reduce((s,n)=>s+(n.fact_coverage||0),0)/S.allSN.length*100).toFixed(1)+'%':'—';
    document.getElementById('sn-total').textContent=S.allSN.length;
    document.getElementById('sn-stale').textContent=stale;
    document.getElementById('sn-flagged').textContent=flagged;
    document.getElementById('sn-coverage').textContent=avgCov;
    document.getElementById('tabSN').textContent=S.allSN.length;
    filterSuperNodes();
  }catch(e){document.getElementById('snGrid').innerHTML='<div style="grid-column:1/-1;text-align:center;padding:2rem;color:var(--red)">'+esc(e.message)+'</div>'}
}
function setSnFilter(chip){document.querySelectorAll('.chip').forEach(c=>c.classList.remove('active'));chip.classList.add('active');S.snFilter=chip.dataset.filter;filterSuperNodes()}
function filterSuperNodes(){
  const q=(document.getElementById('snSearch').value||'').toLowerCase();
  let nodes=S.allSN;
  if(S.snFilter==='stale')nodes=nodes.filter(n=>n.is_stale);
  if(S.snFilter==='flagged')nodes=nodes.filter(n=>n.fidelity_flagged);
  if(S.snFilter==='ok')nodes=nodes.filter(n=>!n.is_stale&&!n.fidelity_flagged);
  if(q)nodes=nodes.filter(n=>(n.source_query||'').toLowerCase().includes(q));
  renderSuperNodes(nodes);
}
function renderSuperNodes(nodes){
  const grid=document.getElementById('snGrid');
  if(!nodes.length){grid.innerHTML='<div style="grid-column:1/-1;text-align:center;padding:2.5rem;color:var(--muted)">No super-nodes match the filter.</div>';return}
  grid.innerHTML=nodes.map(n=>{
    const cov=(n.fact_coverage||0)*100;
    const covCls=cov>=85?'green':cov>=70?'amber':'red';
    const staleBadge=n.is_stale?'<span class="badge badge-ready">⚠ Stale</span>':'';
    const flagBadge=n.fidelity_flagged?'<span class="badge badge-failed">⚠ Flagged</span>':'';
    const accessed=n.last_accessed?new Date(n.last_accessed).toLocaleDateString():'—';
    return `<div class="sn-card">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:7px">
        <div><div class="sn-id">${esc(n.id)}</div><div style="display:flex;gap:5px;margin-top:4px;flex-wrap:wrap">${staleBadge}${flagBadge}</div></div>
        <span class="badge badge-indigo" style="white-space:nowrap">Rev ${n.revision||1}</span>
      </div>
      <div style="font-size:.83rem;color:var(--muted2);line-height:1.5">${esc((n.source_query||'').slice(0,110))}${(n.source_query||'').length>110?'…':''}</div>
      <div class="sn-meta-grid">
        <div class="sn-meta-item"><span class="sn-meta-label">Coverage</span>
          <span class="sn-meta-val" style="color:var(--${covCls==='green'?'green2':covCls==='amber'?'amber2':'red2'})">${cov.toFixed(1)}%</span>
          <div class="prog-bar"><div class="prog-fill prog-${covCls}" style="width:${cov}%"></div></div>
        </div>
        <div class="sn-meta-item"><span class="sn-meta-label">Drift Margin</span><span class="sn-meta-val">${(n.drift_margin||0).toFixed(3)}</span></div>
        <div class="sn-meta-item"><span class="sn-meta-label">Hit Count</span><span class="sn-meta-val" style="color:var(--cyan)">🔥 ${n.hit_count||0}</span></div>
        <div class="sn-meta-item"><span class="sn-meta-label">Last Accessed</span><span class="sn-meta-val" style="font-size:.77rem;color:var(--muted2)">${accessed}</span></div>
      </div>
      <button class="card-btn primary" onclick="forceSynthAdmin(${n.cluster_id})">🔄 Re-Synthesize</button>
    </div>`;
  }).join('');
}
async function forceSynthAdmin(id){
  try{toast('Triggering synthesis for cluster #'+id+'…','info');
    const r=await api('/admin/synthesize/'+id,{method:'POST'});
    toast('Synthesized! Node: '+(r.super_node_id||'created'),'success');
    fetchClusters();loadSuperNodes();
  }catch(e){toast('Synthesis failed: '+e.message,'error')}
}



/* ══ ADMIN ══ */
function openAdminLogin(){document.getElementById('adminUser').value='';document.getElementById('adminPass').value='';document.getElementById('adminErr').textContent='';document.getElementById('adminLoginModal').classList.add('open');setTimeout(()=>document.getElementById('adminUser').focus(),100)}
function closeAdminLogin(){document.getElementById('adminLoginModal').classList.remove('open')}
function submitAdminLogin(){
  const u=document.getElementById('adminUser').value.trim(),p=document.getElementById('adminPass').value;
  if(u===ADMIN_USER&&p===ADMIN_PASS){closeAdminLogin();openBackupModal()}
  else{document.getElementById('adminErr').textContent='✕ Invalid credentials.';document.getElementById('adminPass').value='';document.getElementById('adminPass').focus()}
}
function openBackupModal(){
  const today=new Date().toISOString().slice(0,10);
  document.getElementById('backupVersion').value=today;
  document.getElementById('backupMessage').value='';
  document.getElementById('backupErr').textContent='';
  document.getElementById('backupPath').style.display='none';
  document.getElementById('backupSubmitBtn').disabled=false;
  document.getElementById('backupSubmitBtn').textContent='Create Backup';
  document.getElementById('backupModal').classList.add('open');
  setTimeout(()=>document.getElementById('backupMessage').focus(),100);
}
function closeBackupModal(){document.getElementById('backupModal').classList.remove('open')}
async function executeBackup(){
  const v=document.getElementById('backupVersion').value.trim();
  const m=document.getElementById('backupMessage').value.trim();
  if(!v){document.getElementById('backupErr').textContent='✕ Version tag required.';return}
  if(!m){document.getElementById('backupErr').textContent='✕ Commit message required.';return}
  document.getElementById('backupErr').textContent='';
  document.getElementById('backupSubmitBtn').disabled=true;
  document.getElementById('backupSubmitBtn').textContent='⏳ Creating…';
  try{
    const d=await api('/admin/backup',{method:'POST',body:JSON.stringify({version:v,message:m})});
    const pe=document.getElementById('backupPath');pe.textContent='✓ Saved to: '+(d.backup_path||'data/backups/'+v);pe.style.display='block';
    document.getElementById('backupSubmitBtn').textContent='✓ Done';
    toast('Backup "'+v+'" created successfully.','success');
    setTimeout(closeBackupModal,2000);
  }catch(e){document.getElementById('backupErr').textContent='✕ '+e.message;document.getElementById('backupSubmitBtn').disabled=false;document.getElementById('backupSubmitBtn').textContent='Create Backup';toast('Backup failed: '+e.message,'error')}
}

/* ══ RESET ══ */
async function confirmReset(){
  if(!confirm('⚠️ This will DELETE all clusters, query logs, and simulation jobs.\n\nbenchmark_qa data is preserved.\n\nAre you sure?'))return;
  try{
    await api('/simulation/reset',{method:'POST'});
    S.clusters=[];S.logs=[];S._simResults=null;
    renderClusters();renderQueryLog();
    document.getElementById('simResultsContent').innerHTML='Run a simulation to see per-query results here.';
    document.getElementById('tabResults').textContent='—';
    updateHeaderStats();
    toast('All clusters and logs cleared.','success');
  }catch(e){toast('Reset failed: '+e.message,'error')}
}

/* ══ HEADER STATS ══ */
function updateHeaderStats(){
  const total=S.clusters.length;
  const ready=S.clusters.filter(c=>c.hit_count>=10&&!c.synthesized&&!c.synthesizing).length;
  const synth=S.clusters.filter(c=>c.synthesized).length;
  document.getElementById('hClusters').textContent=total+' Clusters';
  document.getElementById('hReady').textContent=ready+' Ready';
  document.getElementById('hSynthesized').textContent=synth+' Synthesized';
}

/* ══ TABS ══ */
function switchTab(name,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-'+name).classList.add('active');
  if(name==='supernodes'&&!S.allSN.length)loadSuperNodes();
}
function switchTabById(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${name}"]`)?.classList.add('active');
  document.getElementById('panel-'+name)?.classList.add('active');
}

/* ══ PROGRESS ══ */
function showProgress(pct,text){document.getElementById('progressWrap').classList.add('visible');document.getElementById('progressFill').style.width=pct+'%';document.getElementById('progressText').textContent=text}
function hideProgress(){document.getElementById('progressWrap').classList.remove('visible')}
function setRunning(r){document.getElementById('runBtn').disabled=r;document.getElementById('runBtn').textContent=r?'⏳ Running…':'▶ Run Simulation'}

/* ══ TOAST ══ */
function toast(msg,type='info'){
  const icons={success:'✓',error:'✕',info:'ℹ'};const colors={success:'var(--green2)',error:'var(--red2)',info:'var(--cyan)'};
  const el=document.createElement('div');el.className='toast toast-'+type;
  el.innerHTML='<span style="color:'+colors[type]+'">'+icons[type]+'</span> '+esc(msg);
  document.getElementById('toast-container').appendChild(el);setTimeout(()=>el.remove(),5000);
}

/* ══ UTILS ══ */
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function fmtTime(ts){
  if(!ts)return'—';
  try{const d=new Date(ts.replace(' ','T')+(ts.includes('Z')?'':'Z'));return d.toLocaleTimeString('en-GB',{hour12:false})}
  catch{return ts.slice(11,19)||ts}
}

/* shimmer keyframe for skeleton */
const shimmerStyle=document.createElement('style');
shimmerStyle.textContent='@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}';
document.head.appendChild(shimmerStyle);
