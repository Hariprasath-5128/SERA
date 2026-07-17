import re

with open('frontend/dashboard.html', 'r', encoding='utf-8') as f:
    d = f.read()

# Add Chat link to nav
old_nav = '<div class="health-dot" id="healthDot"></div>\n    <span style="font-size:.75rem;color:var(--muted)" id="healthLabel">Checking…</span>\n    <button class="icon-btn" onclick="openSettings()" title="Settings">⚙</button>'
new_nav = '<div class="health-dot" id="healthDot"></div>\n    <span style="font-size:.75rem;color:var(--muted)" id="healthLabel">Checking…</span>\n    <a href="home.html" class="sb-btn sb-btn-primary" style="padding:4px 12px; font-size:0.75rem; border-radius: 6px; text-decoration:none; margin-left: 8px;">💬 Chat</a>\n    <button class="icon-btn" onclick="openSettings()" title="Settings">⚙</button>'
d = d.replace(old_nav, new_nav)

# Remove mini sidebar chat
mini_start = d.find('<hr class="divider">\n    <div class="sb-title">💬 Ask SERA</div>')
mini_end = d.find('<div class="refresh-hint"', mini_start)
if mini_start != -1 and mini_end != -1:
    d = d[:mini_start] + '\n    ' + d[mini_end:]

# Remove chat tab
d = d.replace('<div class="tab" onclick="switchTab(\'chat\',this)">💬 Chat</div>', '')

# Remove chat panel
chat_start = d.find('<!-- CHAT -->')
chat_end = d.find('</main>', chat_start)
if chat_start != -1 and chat_end != -1:
    d = d[:chat_start] + d[chat_end:]

# Remove renderHistory call
d = d.replace('renderHistory();\n});', '});')

# Remove chat JS
chat_js_start = d.find('/* ══ CHAT ══ */')
chat_js_end = d.find('/* ══ ADMIN ══ */')
if chat_js_start != -1 and chat_js_end != -1:
    d = d[:chat_js_start] + d[chat_js_end:]

with open('frontend/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(d)


with open('frontend/home.html', 'r', encoding='utf-8') as f:
    h = f.read()

css_add = """
.nav-left{display:flex;align-items:center;gap:20px}
.hist-btn{display:inline-flex;align-items:center;gap:7px;background:var(--surface2);border:1px solid var(--border);border-radius:999px;padding:6px 14px;color:var(--text);font-size:.85rem;font-weight:600;cursor:pointer;transition:all var(--trans);font-family:inherit}
.hist-btn:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.2);color:#fff}
.history-panel{position:fixed;top:58px;bottom:0;left:-300px;width:300px;background:rgba(11,15,26,.95);backdrop-filter:blur(16px);border-right:1px solid var(--border);transition:left var(--trans);z-index:90;display:flex;flex-direction:column;padding:20px}
.history-panel.open{left:0}
.hist-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;color:var(--text)}
.hist-list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.hist-item{padding:12px 14px;border-radius:10px;background:var(--surface);border:1px solid transparent;font-size:.85rem;color:var(--muted2);cursor:pointer;transition:all var(--trans);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hist-item:hover{background:rgba(255,255,255,.08);border-color:var(--indigo);color:var(--text)}
.hist-del-btn{background:none;border:none;color:var(--muted);font-size:.9rem;cursor:pointer;padding:4px 6px;border-radius:6px;transition:all var(--trans);margin-left:8px}
.hist-del-btn:hover{background:rgba(239,68,68,.2);color:var(--red)}
"""
h = h.replace('</style>', css_add + '</style>')

history_html = """
<!-- History Sidebar -->
<div class="history-panel" id="historyPanel">
  <div class="hist-header">
    <span style="font-weight:600;font-size:0.9rem">Recent Queries</span>
    <button class="nav-btn" onclick="toggleHistory()" style="background:none;border:none;color:var(--muted);font-size:1.1rem;padding:0;cursor:pointer">✕</button>
  </div>
  <div id="historyList" class="hist-list"></div>
</div>
"""
h = h.replace('<body>\n<div class="mesh"></div>', '<body>\n' + history_html + '\n<div class="mesh"></div>')

nav_old = """<nav>
  <a class="brand" href="index.html">
    <div class="brand-icon">🧠</div>
    <span class="brand-text">SERA</span>
    <span class="brand-sub">v5.0</span>
  </a>
  <div class="nav-right">"""
nav_new = """<nav>
  <div class="nav-left">
    <button class="hist-btn" onclick="toggleHistory()">🕒 History</button>
    <a class="brand" href="home.html">
      <div class="brand-icon">🧠</div>
      <span class="brand-text">SERA</span>
      <span class="brand-sub">v5.0</span>
    </a>
  </div>
  <div class="nav-right">"""
h = h.replace(nav_old, nav_new)

feat_start = h.find('<div class="features">')
feat_end = h.find('</section>', feat_start)
if feat_start != -1 and feat_end != -1:
    h = h[:feat_start] + h[feat_end:]

h = h.replace('<span class="btn-icon" id="btnIcon">⚡</span>', '<span class="btn-icon" id="btnIcon"></span>')
h = h.replace("icon.textContent='⚡'", "icon.textContent=''")

js_old = """    renderAnswer(await r.json());
  }catch(e){const ec=document.getElementById('errorCard');ec.textContent='✕  '+e.message;ec.classList.add('show')}"""
js_new = """    renderAnswer(await r.json());
    const hist = JSON.parse(sessionStorage.getItem('sera_hist')||'[]');
    hist.unshift({q:q,t:new Date().toLocaleTimeString()});
    if(hist.length>20) hist.pop();
    sessionStorage.setItem('sera_hist',JSON.stringify(hist));
    if(document.getElementById('historyPanel').classList.contains('open')) renderHistory();
  }catch(e){const ec=document.getElementById('errorCard');ec.textContent='✕  '+e.message;ec.classList.add('show')}"""
h = h.replace(js_old, js_new)

js_add = """
function toggleHistory(){
  const p=document.getElementById('historyPanel');
  p.classList.toggle('open');
  if(p.classList.contains('open')) renderHistory();
}
function renderHistory(){
  const list=document.getElementById('historyList');
  const hist = JSON.parse(sessionStorage.getItem('sera_hist')||'[]');
  if(!hist.length){
    list.innerHTML='<div style="color:var(--muted);font-size:.85rem;text-align:center;margin-top:20px">No recent queries.</div>';
    return;
  }
  list.innerHTML=hist.map((h,i)=>`<div class="hist-item" style="display:flex;justify-content:space-between;align-items:center;padding-right:8px">
    <div style="flex:1;overflow:hidden;text-overflow:ellipsis" onclick="loadHistoryQuery('${h.q.replace(/'/g, "\\\\'")}')">${h.q}</div>
    <button class="hist-del-btn" onclick="event.stopPropagation(); deleteHistoryQuery(${i}, '${h.q.replace(/'/g, "\\\\'")}')" title="Delete">✕</button>
  </div>`).join('');
}
async function deleteHistoryQuery(idx, q){
  if(confirm("Are you sure you want to permanently delete this query from your history and the real-time database?")){
    const hist = JSON.parse(sessionStorage.getItem('sera_hist')||'[]');
    hist.splice(idx, 1);
    sessionStorage.setItem('sera_hist', JSON.stringify(hist));
    renderHistory();
    try{
      await fetch(BASE_URL+'/query/', {
        method: 'DELETE',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: q})
      });
    } catch(e) { console.error('Failed to delete from DB:', e); }
  }
}
function loadHistoryQuery(q){
  document.getElementById('queryInput').value=q;
  toggleHistory();
  askQuery();
}
"""
h = h.replace('function renderAnswer(data){', js_add + '\nfunction renderAnswer(data){')

with open('frontend/home.html', 'w', encoding='utf-8') as f:
    f.write(h)
