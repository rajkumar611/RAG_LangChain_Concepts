const BASE = "http://localhost:8080";

function toggleSidebar() {
  document.querySelector(".sidebar").classList.toggle("collapsed");
}

function setActiveToc(id) {
  document.querySelectorAll(".toc-link").forEach(l => l.classList.remove("active"));
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
}

const _pageCache = {};
async function navigate(page, tocId) {
  const content = document.getElementById('content');
  if (!_pageCache[page]) {
    const r = await fetch(`/pages/${page}.html`, { cache: 'no-cache' });
    _pageCache[page] = await r.text();
  }
  content.innerHTML = _pageCache[page];
  setActiveToc(tocId);
  if (page === 'langchain') {
    document.getElementById('lc-overview').style.display = 'block';
    document.getElementById('lc-demo-container').style.display = 'none';
  }
}

function showHome()        { navigate('home', 'toc-home'); }
function showNaiveRAG()    { navigate('naive-rag', 'toc-naive'); }
function showAdvancedRAG() { navigate('advanced-rag', 'toc-advanced'); }
function showAgenticRAG()  { navigate('agentic-rag', 'toc-agentic'); }
function showHybridRAG()   { navigate('hybrid-rag', 'toc-hybrid'); }
function showGraphRAG()    { navigate('graph-rag', 'toc-graph'); }
function showSummary()     { navigate('summary', 'toc-summary'); }
function showEval()        { navigate('rag-evaluation', 'toc-eval'); }

async function runEval() {
  const question    = document.getElementById('ev-question').value.trim();
  const answer      = document.getElementById('ev-answer').value.trim();
  const ctxRaw      = document.getElementById('ev-contexts').value.trim();
  const groundTruth = document.getElementById('ev-groundtruth').value.trim();
  if (!question || !answer || !ctxRaw) { alert('Question, Answer, and Contexts are required.'); return; }
  const contexts = ctxRaw.split('\n').map(l => l.trim()).filter(Boolean);
  const resultEl = document.getElementById('eval-result');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<div style="color:var(--muted);font-size:13px;padding:16px 0">Evaluating with Claude as judge…</div>';
  try {
    const resp = await fetch(`${BASE}/rag/evaluate`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question, answer, contexts, ground_truth: groundTruth}),
    });
    const data = await resp.json();
    const scoreColor = s => s >= 0.8 ? 'var(--green)' : s >= 0.5 ? 'var(--amber)' : 'var(--red)';
    const scoreBar = s => {
      const pct = Math.round(s * 100);
      return `<div style="background:var(--bg);border-radius:4px;height:6px;width:100%;margin-top:4px">
        <div style="height:6px;border-radius:4px;background:${scoreColor(s)};width:${pct}%"></div>
      </div>`;
    };
    const metricCard = (label, color, key) => {
      const s = data.scores[key] ?? 0;
      const d = data.details[key] ?? {};
      return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px">
        <div style="font-size:11px;font-weight:700;color:${color};text-transform:uppercase;margin-bottom:4px">${label}</div>
        <div style="font-size:22px;font-weight:800;color:${scoreColor(s)}">${(s * 100).toFixed(0)}<span style="font-size:13px;font-weight:400">%</span></div>
        ${scoreBar(s)}
        <div style="font-size:11px;color:var(--muted);margin-top:8px;line-height:1.5">${esc(d.reasoning || '')}</div>
      </div>`;
    };
    const overall = data.overall ?? 0;
    resultEl.innerHTML = `
      <div style="background:var(--surface);border:1px solid ${scoreColor(overall)};border-radius:12px;padding:16px 20px;margin-bottom:16px;display:flex;align-items:center;gap:16px">
        <div>
          <div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase">Overall Score</div>
          <div style="font-size:32px;font-weight:800;color:${scoreColor(overall)}">${(overall * 100).toFixed(0)}%</div>
        </div>
        <div style="flex:1;font-size:12px;color:var(--muted);line-height:1.7">
          Based on ${Object.keys(data.scores).length} metrics across ${data.inputs.context_count} context chunk(s).
          ${data.inputs.has_ground_truth ? 'Correctness scored against provided ground truth.' : ''}
        </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px">
        ${metricCard('Faithfulness','var(--blue)','faithfulness')}
        ${metricCard('Answer Relevancy','var(--purple)','answer_relevancy')}
        ${metricCard('Context Utilization','var(--green)','context_utilization')}
        ${data.scores.correctness !== undefined ? metricCard('Correctness','var(--amber)','correctness') : ''}
      </div>
    `;
  } catch(e) {
    resultEl.innerHTML = `<div style="color:var(--red);font-size:13px">Error: ${e.message}</div>`;
  }
}

function showGovernance() { navigate('governance', 'toc-governance').then(loadGovernance); }
function showLangchain()  { navigate('langchain', 'toc-langchain'); }

async function loadGovernance() {
  const el = document.getElementById('gov-content');
  if (!el) return;
  el.innerHTML = '<div style="color:var(--muted);font-size:13px;padding:20px 0">Loading policy…</div>';
  try {
    const data = await fetch(`${BASE}/governance`).then(r => r.json());
    const scoreColor = s => s >= 0.8 ? 'var(--green)' : s >= 0.5 ? 'var(--amber)' : 'var(--red)';

    function policyCard(title, color, rows) {
      const rowHtml = rows.map(([k, v]) => `
        <div style="display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">
          <div style="min-width:180px;color:var(--muted);flex-shrink:0">${k}</div>
          <div style="color:var(--text);word-break:break-word">${v}</div>
        </div>`).join('');
      return `<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;margin-bottom:16px">
        <div style="font-size:11px;font-weight:700;color:${color};text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px">${title}</div>
        ${rowHtml}
      </div>`;
    }

    const sf = data.security_filters;
    const grd = data.guardrails?.faithfulness ?? {};
    const hitl = data.human_in_the_loop ?? {};
    const mdl = data.models ?? {};
    const lim = data.limits ?? {};
    const patterns = (sf?.prompt_injection?.patterns ?? []).map(p => `<code style="background:var(--bg);padding:1px 6px;border-radius:4px;font-size:11px;margin-right:4px">${esc(p)}</code>`).join(' ');

    el.innerHTML =
      policyCard('Security Filters — Input Side', 'var(--red)', [
        ['Prompt injection', `${sf.prompt_injection.enabled ? 'Enabled' : 'Disabled'} · ${sf.prompt_injection.action}`],
        ['Detected patterns', `<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px">${patterns}</div>`],
        ['Indirect injection', `${sf.indirect_injection.enabled ? 'Enabled' : 'Disabled'} · ${sf.indirect_injection.action}`],
        ['Scope', sf.indirect_injection.scope],
      ]) +
      policyCard('Guardrail — Output Side', 'var(--amber)', [
        ['Faithfulness check', `${grd.enabled ? 'Enabled' : 'Disabled'} · runs on every RAG answer`],
        ['Threshold', `<span style="color:${scoreColor(grd.threshold)};font-weight:700">${grd.threshold}</span> — below this triggers amber warning`],
        ['Judge model', grd.model],
        ['Action', grd.action],
      ]) +
      policyCard('Human in the Loop', 'var(--blue)', [
        ['Trigger', esc(hitl.trigger ?? '')],
        ['Response', esc(hitl.response ?? '')],
      ]) +
      policyCard('Models', 'var(--purple)', [
        ['RAG answering', mdl.rag],
        ['RAGAS evaluation', mdl.evaluation],
        ['Embeddings', mdl.embeddings],
      ]) +
      policyCard('Limits', 'var(--muted)', [
        ['Max query length', `${lim.max_query_length} characters`],
        ['Max chunks per upload', lim.max_chunks_per_upload],
        ['Max chars per chunk', lim.max_chunk_chars],
        ['Max agentic search rounds', lim.max_agentic_search_rounds],
      ]);
  } catch (e) {
    el.innerHTML = `<div style="color:var(--red);font-size:13px">Error loading policy: ${esc(e.message)}</div>`;
  }
}

// ── Run query ──────────────────────────────────────────────────────────────
async function runQuery(id, endpoint, showSources, showGraph) {
  const query = document.getElementById(`q-${id}`).value.trim();
  if (!query) {
    const ta = document.getElementById(`q-${id}`);
    ta.placeholder = "⚠ Please type a question first, then click Run.";
    ta.focus();
    return;
  }

  const btn = document.getElementById(`btn-${id}`);
  btn.disabled = true;
  btn.innerHTML = `<div class="spinner"></div> Running…`;

  document.getElementById(`steps-${id}`).innerHTML = `<div class="empty">Running…</div>`;
  document.getElementById(`docs-${id}`).innerHTML  = `<div class="empty">—</div>`;
  document.getElementById(`answer-${id}`).textContent = "—";

  try {
    const res  = await fetch(BASE + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) {
      const msg = data?.detail || `HTTP ${res.status}`;
      document.getElementById(`steps-${id}`).innerHTML =
        `<div class="empty" style="color:var(--red)">Server error: ${esc(String(msg))}</div>`;
      document.getElementById(`docs-${id}`).innerHTML = `<div class="empty">—</div>`;
      document.getElementById(`answer-${id}`).textContent = `Error: ${msg}`;
      return;
    }
    renderResults(id, data, showSources, showGraph);
  } catch (e) {
    document.getElementById(`steps-${id}`).innerHTML =
      `<div class="empty" style="color:var(--red)">Error: ${e.message} — is the server running?</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run";
  }
}

function renderResults(id, data, showSources, showGraph) {
  // Steps
  const stepsEl = document.getElementById(`steps-${id}`);
  if (data.steps?.length) {
    let html = "";
    if (data.rewritten_query) {
      html += `<div class="rewrite-pill">Rewritten: <strong>${esc(data.rewritten_query)}</strong></div>`;
    }
    data.steps.forEach((s, i) => {
      html += `
        <div class="step-item">
          <div class="step-num">${i+1}</div>
          <div>
            <div class="step-name">${esc(s.step)}</div>
            <div class="step-detail">${esc(s.detail)}</div>
          </div>
        </div>`;
    });
    stepsEl.innerHTML = html;
  } else {
    stepsEl.innerHTML = `<div class="empty">No steps returned.</div>`;
  }

  // Docs
  const docsEl = document.getElementById(`docs-${id}`);
  if (data.docs?.length) {
    docsEl.innerHTML = data.docs.map((d, i) => {
      let badges = "";
      if (showSources && d.sources) {
        if (d.sources.includes("vector")) badges += `<span class="badge-sm badge-vec">vector</span>`;
        if (d.sources.includes("bm25"))   badges += `<span class="badge-sm badge-bm25">bm25</span>`;
      }
      const preview = d.text.length > 180 ? d.text.slice(0, 180) + "…" : d.text;
      return `
        <div class="doc-card">
          <div class="doc-num">d${d.id}</div>
          <div class="doc-body">
            <div class="doc-text">${esc(preview)}</div>
            <div class="doc-meta">${badges}<span class="score">score ${d.score}</span></div>
          </div>
        </div>`;
    }).join("");
  }

  // Graph node viz
  if (showGraph && data.visited_nodes) {
    const top = new Set((data.docs||[]).map(d => `d${d.id}`));
    const viz = data.visited_nodes
      .map(n => `<span class="node-pill ${top.has(n)?"seed":"visited"}">${n}</span>`)
      .join("");
    docsEl.innerHTML += `
      <div style="margin-top:14px">
        <div class="card-title">Graph Nodes Visited</div>
        <div class="graph-viz">${viz}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:8px">
          <span style="color:var(--green)">●</span> top-ranked &nbsp;
          <span style="color:var(--muted)">●</span> BFS-expanded
        </div>
      </div>`;
  }

  // Answer
  const ansEl = document.getElementById(`answer-${id}`);
  ansEl.style.color = "";
  ansEl.style.fontSize = "";
  ansEl.textContent = data.answer ?? "No answer returned.";
}

function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function _renderGuardrail(elId, guardrail) {
  const el = document.getElementById(elId);
  if (!el || !guardrail) return;
  const passed = guardrail.passed;
  const score  = guardrail.faithfulness_score ?? 0;
  const pct    = Math.round(score * 100);
  const color      = passed ? 'var(--green)' : 'var(--amber)';
  const borderClr  = passed ? '#14532d55'    : '#78350f55';
  const bg         = passed ? '#0a2e0a'      : '#2a1a00';
  const label      = passed ? 'Grounded in document' : 'May contain LLM inference';
  const dot        = passed ? '●' : '▲';
  const subtext    = passed
    ? `Faithfulness guardrail: ${pct}% — every claim verified against the retrieved context`
    : `Faithfulness guardrail: ${pct}% — answer may include content not present in the uploaded document`;
  const hitl = passed ? '' : `
    <div style="margin-top:8px;padding:8px 10px;background:#1a1000;border:1px solid #78350f33;border-radius:6px">
      <div style="font-size:11px;font-weight:700;color:var(--amber);margin-bottom:4px">What you can do</div>
      <ul style="margin:0;padding-left:14px;font-size:11px;color:var(--muted);line-height:1.8">
        <li>Rephrase your query to be more specific to the document content</li>
        <li>Upload a document that more directly addresses your question</li>
        <li>Treat this answer as a starting point and verify the claims manually</li>
      </ul>
    </div>`;
  el.innerHTML = `<div style="margin-top:10px;padding:10px 14px;background:${bg};border:1px solid ${borderClr};border-radius:8px">
    <div style="display:flex;align-items:flex-start;gap:12px">
      <span style="color:${color};font-size:16px;flex-shrink:0;line-height:1.2">${dot}</span>
      <div style="flex:1">
        <div style="font-size:12px;font-weight:700;color:${color}">${label}</div>
        <div style="font-size:11px;color:var(--muted);margin-top:3px;line-height:1.5">${subtext}</div>
      </div>
    </div>${hitl}
  </div>`;
}

// ═══════════════════════════════════════════════════════════════════════════
// LANGCHAIN DEMOS  (fixed: lcRunBtn now uses number n, not hyphenated id)
// ═══════════════════════════════════════════════════════════════════════════

let _lcSession = null;
function _getLCSession() {
  if (!_lcSession) _lcSession = Math.random().toString(36).slice(2);
  return _lcSession;
}

// ── shared helpers ──────────────────────────────────────────────────────────
function lcCard(title, body, accentColor) {
  const c = accentColor || 'var(--border)';
  return `<div class="card" style="border-color:${c}">${title ? `<div class="card-title">${esc(title)}</div>` : ''}${body}</div>`;
}
function lcPre(text) {
  return `<pre style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;font:12px/1.6 'Cascadia Code',Consolas,monospace;color:var(--text);white-space:pre-wrap;word-break:break-word;margin:0">${esc(String(text))}</pre>`;
}
function lcStep(label, content) {
  return `<div style="display:flex;gap:10px;align-items:flex-start">
    <div style="min-width:20px;height:20px;background:var(--blue);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex-shrink:0;margin-top:2px">${esc(label)}</div>
    <div style="flex:1">${content}</div>
  </div>`;
}
function lcRunBtn(n, label) {
  return `<button id="lc${n}-btn" class="run-btn" style="margin-top:4px" onclick="lcRun(${n})">${label || 'Run'}</button>`;
}
function lcTextarea(id, placeholder, value, rows) {
  return `<textarea id="${id}" rows="${rows||3}" placeholder="${placeholder||''}"
    style="width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:10px 12px;font:13px inherit;resize:vertical;outline:none"
    onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='var(--border)'"
  >${value||''}</textarea>`;
}
function lcInput(id, placeholder, value) {
  return `<input id="${id}" type="text" placeholder="${placeholder||''}" value="${value||''}"
    style="width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:9px 12px;font:13px inherit;outline:none"
    onfocus="this.style.borderColor='#3b82f6'" onblur="this.style.borderColor='var(--border)'"
  />`;
}
function lcBusy(btnId) {
  const btn = document.getElementById(btnId);
  if (btn) { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Running…'; }
}
function lcIdle(btnId, label) {
  const btn = document.getElementById(btnId);
  if (btn) { btn.disabled = false; btn.innerHTML = label || 'Run'; }
}
async function lcPost(endpoint, body) {
  const res  = await fetch(BASE + endpoint, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body),
  });
  if (!res.ok) { const e = await res.json().catch(()=>({})); throw new Error(e.detail || res.statusText); }
  return res.json();
}
function lcError(msg) {
  document.getElementById('lc-right').innerHTML =
    `<div class="card" style="border-color:var(--red)"><div style="color:var(--red);font-size:13px">${esc(msg)}</div></div>`;
}

// ── demo configs ────────────────────────────────────────────────────────────
const LC_CONFIGS = {
  1: {
    title: '1 · Prompt Management',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">When you call Claude via API, your prompt is usually a hardcoded string — rewritten every time something changes. In a real app you have dozens of prompts for different roles, languages, and tasks, all scattered across your codebase.<br><br>
          <strong style="color:var(--text)">LangChain's ChatPromptTemplate treats prompts as reusable templates with named variables.</strong> You define the structure once — <code style="color:var(--green)">"You are a {role}. Answer: {question}"</code> — and fill in the blanks at runtime. Change the wording in one place and it updates everywhere. It also handles system messages, few-shot examples, and chat history injection automatically.</p>
        </div>
        <div class="card">
          <div class="card-title">What is Prompt Management?</div>
          <p class="desc">A prompt template is like a form with blanks. You write it once:<br>
          <code style="color:var(--green);font-size:11px">"You are a {role}. Answer: {question}"</code><br><br>
          LangChain fills in <code>{role}</code> and <code>{question}</code> at runtime. Change the role or question anywhere in your app without touching the template.</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px">
            <strong style="color:var(--blue)">①</strong> The raw template code (with {role} and {question} still as placeholders)<br>
            <strong style="color:var(--purple)">②</strong> The filled-in prompt that actually gets sent to Claude<br>
            <strong style="color:var(--green)">③</strong> Claude's answer, speaking in the role you chose
          </p>
        </div>
        <div class="card">
          <div class="card-title">Role — who should Claude pretend to be?</div>
          <select id="lc1-role" style="width:100%;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--text);padding:9px 12px;font:13px inherit">
            <option>pirate</option><option>scientist</option><option>poet</option>
            <option>teacher</option><option>chef</option><option>medieval knight</option>
          </select>
        </div>
        <div class="card">
          <div class="card-title">Question — what to ask Claude</div>
          ${lcTextarea('lc1-q','e.g. What is machine learning?','What is machine learning?',2)}
        </div>
        ${lcRunBtn(1,'Run Template')}`;
    },
  },
  2: {
    title: '2 · LLM Chaining',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Claude does one thing well per call. If you ask it to translate, summarise, and format as JSON in a single prompt, quality drops — it tries to do too much at once.<br><br>
          <strong style="color:var(--text)">The solution is chaining: break the task into steps, where each Claude call does one thing, and the output flows automatically to the next step.</strong><br><br>
          Without LangChain, you write every API call by hand and manually pass outputs between steps — repetitive and error-prone. LangChain's <code style="color:var(--green)">|</code> operator connects steps in one line: the output of Step 1 becomes the input of Step 2 with no wiring code.</p>
        </div>
        <div class="card">
          <div class="card-title">What is LLM Chaining?</div>
          <p class="desc">This demo has 3 steps:<br><br>
          <strong style="color:var(--blue)">Step 1 — Translate</strong><br>Claude translates your text to English<br><br>
          <strong style="color:var(--purple)">Step 2 — Summarise</strong><br>Claude summarises the English text to one sentence<br><br>
          <strong style="color:var(--green)">Step 3 — Format as JSON</strong><br>Claude wraps the summary in <code>{"summary": "..."}</code></p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px">Three coloured output boxes, one per step — the English translation, then the one-sentence summary, then the JSON wrapper.</p>
        </div>
        <div class="card">
          <div class="card-title">Input text — paste in any language</div>
          ${lcTextarea('lc2-text','Paste text in any language…','La inteligencia artificial está transformando el mundo moderno de maneras increíbles, desde la medicina hasta el transporte.',4)}
          <div style="margin-top:6px;font-size:11px;color:var(--muted)">The sample above is Spanish — Step 1 will translate it first.</div>
        </div>
        ${lcRunBtn(2,'Run Chain')}`;
    },
  },
  3: {
    title: '3 · RAG',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Claude was trained up to a certain date and knows nothing about your internal documents — HR policies, product manuals, last quarter's reports. If you ask about them, it either guesses (hallucinates) or says it doesn't know.<br><br>
          <strong style="color:var(--text)">RAG solves this without retraining the model.</strong> Before Claude answers, you search your own documents for the most relevant sections and include them in the prompt as context. Claude then answers using those sections — not its training memory.<br><br>
          This is how enterprise chatbots read internal PDFs and company wikis. LangChain provides the entire pipeline — loader, chunker, embedder, vector store, retriever — as ready-made components.</p>
        </div>
        <div class="card">
          <div class="card-title">What is RAG?</div>
          <p class="desc"><strong style="color:var(--text)">RAG = Retrieval Augmented Generation.</strong> Three steps:<br><br>
          <strong style="color:var(--blue)">Step 1</strong> — Search a knowledge base for facts relevant to the question<br>
          <strong style="color:var(--green)">Step 2</strong> — Give those facts to Claude as extra context<br>
          <strong style="color:var(--amber)">Step 3</strong> — Claude answers using the facts, not its training memory</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">Knowledge base loaded for this demo</div>
          <p class="desc" style="font-size:11px;margin-bottom:6px">7 facts pre-loaded — ask about any of these topics:</p>
          ${['Python','LangChain','Vector databases','RAG','FAISS','Embeddings','Chunking'].map(f=>`<div style="font-size:11px;color:var(--muted);padding:2px 0">• ${f}</div>`).join('')}
          <p class="desc" style="font-size:11px;margin-top:8px">You will see which 2 chunks were retrieved, and Claude's answer based only on those chunks.</p>
        </div>
        <div class="card">
          <div class="card-title">Your question</div>
          ${lcTextarea('lc3-q','e.g. What is RAG? What is FAISS?','What is FAISS used for?',2)}
        </div>
        ${lcRunBtn(3,'Ask')}`;
    },
  },
  4: {
    title: '4 · Memory',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">When you use Claude Chat on claude.ai, conversation history is automatically maintained — Claude remembers everything said earlier in the session. But when you access Claude via API directly, <strong style="color:var(--text)">each call is completely stateless — Claude remembers nothing from previous calls.</strong><br><br>
          This is where LangChain Memory helps — it automatically collects and passes the full conversation history on every API call, simulating memory.<br><br>
          So LangChain Memory is not needed for chat interfaces, but is <strong style="color:var(--text)">essential when building conversational apps via API</strong> — chatbots, assistants, support agents.</p>
        </div>
        <div class="card">
          <div class="card-title">What is Memory?</div>
          <p class="desc">LangChain keeps a growing list of every message exchanged. On each new call, it injects the full history into the prompt via <code>MessagesPlaceholder</code> — so Claude sees everything said so far, plus your new message, every single time.</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">Try this sequence to see it work</div>
          <p class="desc" style="font-size:11px">
            1. <em>"Hi! My name is Raj and I love cricket."</em><br>
            2. <em>"I work as a software engineer in Singapore."</em><br>
            3. <em>"What sport do I love?"</em> — Claude will remember<br>
            4. <em>"Where do I work?"</em> — Claude will remember this too
          </p>
        </div>
        <div class="card">
          <div class="card-title">Your message</div>
          ${lcTextarea('lc4-msg','Type a message…','Hi! My name is Raj and I love cricket.',2)}
        </div>
        ${lcRunBtn(4,'Send')}
        <button onclick="lcMemoryClear()" style="background:none;border:1px solid var(--border);border-radius:6px;color:var(--muted);font:12px inherit;padding:5px 12px;cursor:pointer;margin-top:4px;transition:all .15s"
          onmouseover="this.style.borderColor='var(--red)';this.style.color='var(--red)'"
          onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)'">Clear session &amp; start over</button>`;
    },
  },
  5: {
    title: '5 · Tools & Function Calling',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Claude is a language model — it can reason and write, but it cannot check today's weather, run guaranteed-accurate calculations, look up live data, or call your company's internal APIs. Without tools, Claude simply guesses or makes things up when it needs real-world information.<br><br>
          <strong style="color:var(--text)">Tools bridge that gap.</strong> You write Python functions and hand them to Claude. When Claude needs external data to answer your question, it requests a tool call — your code runs the function and hands the result back — and Claude builds the final answer from real data.<br><br>
          LangChain's <code>@tool</code> decorator does the heavy lifting: it reads your function name and docstring, auto-generates the schema Claude needs, and manages the multi-round loop so Claude can call multiple tools in sequence.</p>
        </div>
        <div class="card">
          <div class="card-title">What are Tools?</div>
          <p class="desc">You write Python functions. LangChain tells Claude they exist. When Claude needs one:<br><br>
          <strong style="color:var(--amber)">①</strong> Claude decides which tool to call and what to pass in<br>
          <strong style="color:var(--amber)">②</strong> LangChain runs the function and returns the result<br>
          <strong style="color:var(--amber)">③</strong> Claude reads the result and builds its final answer<br><br>
          <strong style="color:var(--text)">Claude never runs code itself</strong> — it only decides what to call.</p>
        </div>
        <div class="card" style="border-color:var(--amber)">
          <div class="card-title" style="color:var(--amber)">How it works — step by step</div>
          <div style="display:flex;flex-direction:column;gap:6px;margin-bottom:12px">
            ${[
              ['1','Developer writes a Python function with the <code>@tool</code> decorator.'],
              ['2','LangChain reads the function name and docstring and auto-generates a <strong>Schema</strong>.'],
              ['3','LangChain sends all Schemas to the LLM so it knows what tools are available.'],
              ['4','The user asks a question.'],
              ['5','The LLM reads the Schemas, reasons, and decides which tool to call and what inputs to pass.'],
              ['6','LangChain runs the corresponding Python function.'],
              ['7','The function returns real data to LangChain.'],
              ['8','LangChain passes the result back to the LLM.'],
              ['9','The LLM builds the final answer using the real data and responds to the user.'],
            ].map(([n,t])=>`
              <div style="display:flex;gap:8px;align-items:flex-start">
                <span style="background:#78350f44;color:var(--amber);font-size:10px;font-weight:700;border-radius:50%;width:18px;height:18px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px">${n}</span>
                <span style="font-size:11px;color:var(--muted);line-height:1.6">${t}</span>
              </div>`).join('')}
          </div>
          <div class="card-title" style="margin-bottom:8px">Key concepts</div>
          <div style="display:flex;flex-direction:column;gap:5px">
            ${[
              ['Function',    'var(--text)',   'The Python code the developer writes to fetch real-world data.'],
              ['Tool',        'var(--amber)',  'What the LLM sees and can request — generated from the Function via <code>@tool</code>.'],
              ['Schema',      'var(--blue)',   'The auto-generated structured description that tells the LLM a tool\'s name, purpose, and expected inputs.'],
              ['Who calls?',  'var(--green)',  'LangChain runs the tool. The LLM only decides which one to call.'],
              ['How chosen?', 'var(--purple)', 'The LLM reads each Schema and reasons which tool matches the user\'s question.'],
              ['Many tools',  'var(--amber)',  'The LLM reasons and picks the best match based on Schemas.'],
              ['Similar tools','var(--red)',   'Becomes ambiguous — always write clear, distinct docstrings.'],
              ['LangChain\'s role','var(--blue)', 'The bridge between the LLM and your Python functions — generates Schemas, manages execution, handles the multi-round loop.'],
            ].map(([k,c,v])=>`
              <div style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:7px 10px;display:grid;grid-template-columns:90px 1fr;gap:6px;align-items:start">
                <span style="font-size:10px;font-weight:700;color:${c}">${k}</span>
                <span style="font-size:11px;color:var(--muted);line-height:1.5">${v}</span>
              </div>`).join('')}
          </div>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">3 tools available in this demo</div>
          ${[['🧮','calculator(expression)','Evaluates any maths. e.g. 144/12 → 12'],['🌤','get_weather(city)','Mock weather. Try: London, Singapore, New York, Sydney'],['🔢','word_count(text)','Counts words in any text']].map(([i,n,d])=>`
            <div style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:6px">
              <div style="font-size:12px;font-weight:700;color:var(--amber)">${i} ${n}</div>
              <div style="font-size:11px;color:var(--muted);margin-top:2px">${d}</div>
            </div>`).join('')}
          <p class="desc" style="font-size:11px;margin-top:4px">You will see which tools Claude chose, the exact input it sent, and what came back.</p>
        </div>
        <div class="card">
          <div class="card-title">Question — try asking something that needs all 3 tools</div>
          ${lcTextarea('lc5-q','Ask something requiring tools…',"What is 144 / 12? Also, what's the weather in Singapore? And how many words are in 'LangChain makes tool calling easy'?",4)}
        </div>
        ${lcRunBtn(5,'Run')}`;
    },
  },
  6: {
    title: '6 · Document Processing',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      const sample = `Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. Unlike traditional software where every rule is hand-coded, ML algorithms discover patterns automatically from training examples and apply them to new, unseen situations.\n\nDeep learning uses multi-layered neural networks to process images, speech, and text.\n\nNatural language processing gives computers the ability to read, understand, and generate human language. It powers chatbots, real-time translation engines, document summarisation, and sentiment analysis tools. Modern NLP relies on transformer models pre-trained on billions of words from the internet, books, and code.\n\nVector databases store text embeddings for fast semantic search in AI pipelines.`;
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Claude has a context window — a hard limit on how much text it can read in one call. A 200-page PDF is far too large to send all at once. Even if it fit, sending the entire document every time would be slow and expensive.<br><br>
          <strong style="color:var(--text)">The solution is chunking: split the document into small, overlapping pieces so only the relevant pieces get sent to Claude.</strong><br><br>
          But naive splitting — every 500 characters — cuts mid-sentence and loses meaning at boundaries. LangChain's splitters are smarter: they try to cut on paragraph breaks first, then sentence breaks, then word boundaries, so each chunk stays coherent. Without LangChain, you write and maintain this logic yourself for every file format you support.</p>
        </div>
        <div class="card">
          <div class="card-title">What is Document Processing?</div>
          <p class="desc">LangChain has two splitters that cut text differently:</p>
          <div style="margin-top:8px;display:flex;flex-direction:column;gap:8px">
            <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px">
              <div style="font-size:12px;font-weight:700;color:var(--blue);margin-bottom:3px">CharacterTextSplitter</div>
              <div style="font-size:11px;color:var(--muted)">Cuts only on <code>\\n\\n</code> (paragraph breaks). Simple, but chunks can be uneven if paragraphs are long.</div>
            </div>
            <div style="background:var(--bg);border:1px solid var(--green);border-radius:8px;padding:10px 12px">
              <div style="font-size:12px;font-weight:700;color:var(--green);margin-bottom:3px">RecursiveCharacterTextSplitter ✓ recommended</div>
              <div style="font-size:11px;color:var(--muted)">Smarter — tries <code>\\n\\n</code> first, then <code>\\n</code>, then spaces. Produces more even chunks. This is what most production apps use.</div>
            </div>
          </div>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px">Both splitters run with <strong>chunk_size = 200</strong>, overlap = 20.<br><br>
          The sample text has 4 paragraphs separated by blank lines. Paragraphs 1 and 3 are intentionally long — over 200 chars each.<br><br>
          <strong style="color:var(--red)">CharacterTextSplitter</strong> can only cut on <code>\\n\\n</code>, so it keeps each paragraph whole — Chunks 1 and 3 will far exceed the 200-char limit.<br><br>
          <strong style="color:var(--green)">RecursiveCharacterTextSplitter</strong> also tries spaces when paragraphs are too long, so all chunks stay within 200 chars — more chunks, but all the right size.</p>
        </div>
        <div class="card">
          <div class="card-title">Text to split (edit freely)</div>
          ${lcTextarea('lc6-text','Paste any long text…',sample,7)}
        </div>
        ${lcRunBtn(6,'Split Document')}`;
    },
  },
  7: {
    title: '7 · Output Parsers',
    badge: 'No Agent', badgeColor: 'var(--green)', badgeBg: '#14532d22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Claude always replies in plain text — even when you ask for structured data. Ask for a list of 5 items and you might get them numbered, bulleted, or comma-separated depending on Claude's phrasing. Ask for JSON and the response might be wrapped in <code>&#96;&#96;&#96;json</code> fences with extra explanation. Your code then has to handle all those variations manually.<br><br>
          <strong style="color:var(--text)">Output Parsers solve this in two steps:</strong> they automatically add format instructions to your prompt so Claude knows exactly what structure to produce, and then they parse Claude's response into the correct Python type — a <code>str</code>, a <code>list</code>, a <code>dict</code> — ready to use directly in your code, with automatic retries if the format is wrong.</p>
        </div>
        <div class="card">
          <div class="card-title">What are Output Parsers?</div>
          <p class="desc">Output Parsers do two things automatically:<br><br>
          <strong style="color:var(--text)">① Instruct Claude</strong> to format its response in a specific way<br>
          <strong style="color:var(--text)">② Parse the response</strong> into the right Python type<br><br>
          So instead of writing <code style="color:var(--muted)">answer.split(",")</code> yourself, LangChain handles it.</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px">The same topic, three times, three output types:<br><br>
          <strong style="color:var(--text)">StrOutputParser</strong> → raw string (one sentence)<br>
          <strong style="color:var(--blue)">CommaSeparatedListOutputParser</strong> → a real Python list of 5 items<br>
          <strong style="color:var(--green)">JsonOutputParser</strong> → a JSON object with name, description, key_facts</p>
        </div>
        <div class="card">
          <div class="card-title">Topic</div>
          ${lcInput('lc7-topic','e.g. machine learning, cricket, Singapore…','machine learning')}
        </div>
        ${lcRunBtn(7,'Parse')}`;
    },
  },
  8: {
    title: '8 · Single Agent (ReAct)',
    badge: 'Agent', badgeColor: 'var(--amber)', badgeBg: '#78350f22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">In Demos 1–7, <em>you</em> define every step in advance. But what if the right steps depend on what you find along the way? A financial question might need a currency conversion first, then a calculation on the result — you don't know the intermediate value until step 1 completes.<br><br>
          <strong style="color:var(--text)">A Single Agent lets Claude decide its own steps.</strong> You give it tools and a question. Claude reasons about what's needed, calls a tool, reads the result, reasons again, calls another tool if needed, and answers when done. This loop is called <strong style="color:var(--amber)">ReAct (Reason + Act)</strong>.<br><br>
          Without LangChain, you write this entire loop yourself — schemas, dispatch logic, multi-round conversation management — every time. LangChain's <code>AgentExecutor</code> handles it in two lines.</p>
        </div>
        <div class="card">
          <div class="card-title">What is a Single Agent?</div>
          <p class="desc">Claude runs a <strong style="color:var(--amber)">ReAct loop</strong> until it has a complete answer:<br><br>
          <strong>Think</strong> — "What do I need to answer this?"<br>
          <strong>Act</strong> — Call a tool to get information<br>
          <strong>Observe</strong> — Read what the tool returned<br>
          <strong>Think again</strong> — "Is this enough, or do I need more?"<br>
          <strong>Answer</strong> — when satisfied</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">3 tools available</div>
          ${[['💱','get_exchange_rate','USD_SGD, EUR_USD, GBP_USD, INR_USD'],['🧮','calculator','any maths expression'],['🌍','get_country_info','Singapore, India, USA']].map(([i,n,d])=>`<div style="font-size:11px;color:var(--muted);padding:3px 0"><span style="color:var(--amber);font-weight:700">${i} ${n}</span> — ${d}</div>`).join('')}
          <p class="desc" style="font-size:11px;margin-top:8px">You will see each tool call in order, then the final answer built from all results.</p>
        </div>
        <div class="card">
          <div class="card-title">Question — try something that needs multiple steps</div>
          ${lcTextarea('lc8-q','Ask a multi-step question…','I have $1000 USD. How much is that in SGD? If I add 18% tax and a flat fee of $50 SGD, what is my total?',4)}
        </div>
        ${lcRunBtn(8,'Run Agent')}`;
    },
  },
  9: {
    title: '9 · Multi-Agent (Simple)',
    badge: 'Agent', badgeColor: 'var(--amber)', badgeBg: '#78350f22',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">A single agent trying to research, write, review, and format all at once produces mediocre results — like one person being asked to do every job simultaneously. Each role gets a vague, unfocused system prompt and the quality suffers.<br><br>
          <strong style="color:var(--text)">Specialist agents work better: each agent has one narrow, focused job.</strong> The Research Agent only finds facts. The Writer Agent only writes. Each gets a tight system prompt it can follow precisely.<br><br>
          This is the simplest form of multi-agent: a fixed linear pipeline — Research → Write. The orchestrator calls them in sequence, passing the output of one as the input of the next. Use this pattern when the steps are predictable and always run in the same order.</p>
        </div>
        <div class="card">
          <div class="card-title">What is Multi-Agent (Simple)?</div>
          <p class="desc">Two specialist agents wired in sequence:<br><br>
          <strong style="color:var(--blue)">Research Agent</strong> — <em>"Return 5 bullet-point facts."</em> Focuses only on facts.<br><br>
          <strong style="color:var(--green)">Writer Agent</strong> — <em>"Turn these notes into a blog post."</em> Focuses only on writing.<br><br>
          The orchestrator passes Research output → Writer input.</p>
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px"><strong style="color:var(--blue)">Research Agent's output</strong> — 5 bullet-point facts<br><strong style="color:var(--green)">Writer Agent's output</strong> — a blog post written from those exact facts. Two separate Claude calls, same topic, different roles.</p>
        </div>
        <div class="card">
          <div class="card-title">Topic</div>
          ${lcInput('lc9-topic','e.g. The impact of AI on jobs…','The impact of AI on software engineering jobs')}
        </div>
        ${lcRunBtn(9,'Run Pipeline')}`;
    },
  },
  10: {
    title: '10 · LangGraph Multi-Agent',
    badge: 'LangGraph', badgeColor: 'var(--purple)', badgeBg: '#581c8722',
    buildLeft() {
      return `
        <div class="card" style="border-color:var(--blue)">
          <div class="card-title" style="color:var(--blue)">Why do you need this?</div>
          <p class="desc">Demo 9's pipeline always runs in one direction: Research → Write → Done. But real workflows need loops: Write → Review → Revise → Review again → Publish. You can't express that as a simple sequence.<br><br>
          <strong style="color:var(--text)">LangGraph lets you define those loops as a graph.</strong> Agents are nodes. Connections between them are edges with rules — "if APPROVED, end; if REVISE, go back to the Writer." Those conditional rules are impossible in plain chaining.<br><br>
          This is how you build production-grade AI workflows with feedback loops, retries, parallel branches, and human-in-the-loop approval steps. Without LangGraph, you write all the state management, loop guards, and routing logic yourself — which quickly becomes hundreds of lines of fragile code.</p>
        </div>
        <div class="card">
          <div class="card-title">What is LangGraph?</div>
          <p class="desc">Agents are <strong style="color:var(--purple)">nodes</strong> in a graph. Connections are <strong style="color:var(--purple)">edges with rules</strong>: "go here if approved, go back if not." This demo has a feedback loop the Reviewer can trigger — impossible with simple chaining.</p>
        </div>
        <div class="card" style="border-color:#581c87">
          <div class="card-title">The 4 graph nodes</div>
          ${[['Manager','Plans the pipeline. Kicks off the process.'],['Research Agent','Gathers 5 bullet-point facts about the topic.'],['Writer Agent','Writes a 3-paragraph blog post. Can be sent back here by the Reviewer.'],['Reviewer Agent','Reads the draft. Says APPROVED → done, or REVISE: feedback → loops back to Writer.']].map(([n,d])=>`
            <div style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin-bottom:6px">
              <div style="font-size:12px;font-weight:700;color:var(--purple);margin-bottom:2px">${n}</div>
              <div style="font-size:11px;color:var(--muted);line-height:1.5">${d}</div>
            </div>`).join('')}
        </div>
        <div class="card" style="border-color:#1e3a5f">
          <div class="card-title">What you will see after clicking Run</div>
          <p class="desc" style="font-size:11px">A live log of each node's decision as the graph runs, and the final approved blog post. If the Reviewer requested a revision you will see the Writer node appear twice.</p>
        </div>
        <div class="card">
          <div class="card-title">Topic</div>
          ${lcInput('lc10-topic','e.g. How LangGraph improves AI systems…','How LangGraph improves multi-agent AI systems')}
        </div>
        ${lcRunBtn(10,'Run Graph')}`;
    },
  },
};

function showLCOverview() {
  document.getElementById('lc-overview').style.display = 'block';
  document.getElementById('lc-demo-container').style.display = 'none';
  _lcSession = null;
}

function showLCDemo(n) {
  const cfg = LC_CONFIGS[n];
  if (!cfg) return;
  _lcCurrentDemo = n;
  document.getElementById('lc-overview').style.display = 'none';
  const dc = document.getElementById('lc-demo-container');
  dc.style.display = 'flex';

  document.getElementById('lc-demo-heading').textContent = cfg.title;
  const badge = document.getElementById('lc-demo-badge');
  badge.textContent  = cfg.badge;
  badge.style.color  = cfg.badgeColor;
  badge.style.background = cfg.badgeBg;
  badge.style.border = `1px solid ${cfg.badgeColor}`;

  document.getElementById('lc-left').innerHTML  = cfg.buildLeft();
  document.getElementById('lc-right').innerHTML = `<div class="card"><div class="empty">Fill in the inputs on the left and click Run.</div></div>`;
}

async function lcRun(n) {
  const btnId = `lc${n}-btn`;
  lcBusy(btnId);
  try {
    await _lcRunners[n]();
  } catch(e) {
    lcError(e.message);
  }
  lcIdle(btnId, _lcBtnLabels[n] || 'Run');
}

const _lcBtnLabels = {1:'Run Template',2:'Run Chain',3:'Ask',4:'Send',5:'Run',6:'Split',7:'Parse',8:'Run Agent',9:'Run Pipeline',10:'Run Graph'};

const _lcRunners = {
  // ── 1. Prompt ──────────────────────────────────────────────────────────────
  1: async () => {
    const role = document.getElementById('lc1-role').value;
    const q    = document.getElementById('lc1-q').value.trim();
    if (!q) throw new Error('Enter a question.');
    const data = await lcPost('/langchain/prompt', {role, question: q});
    document.getElementById('lc-right').innerHTML = `
      ${lcCard('Template', lcPre('ChatPromptTemplate.from_messages([\n  ("system", "You are a {role}. Answer in under 2 sentences."),\n  ("human", "{question}"),\n])'))}
      ${lcCard('Rendered Prompt', lcPre(data.rendered))}
      ${lcCard('LLM Response', `<div style="font-size:14px;line-height:1.8;color:var(--text)">${esc(data.answer)}</div>`, 'var(--blue)')}
    `;
  },

  // ── 2. Chaining ────────────────────────────────────────────────────────────
  2: async () => {
    const text = document.getElementById('lc2-text').value.trim();
    if (!text) throw new Error('Enter some text.');
    const data = await lcPost('/langchain/chaining', {text});
    const colors = ['var(--blue)', 'var(--purple)', 'var(--green)'];
    document.getElementById('lc-right').innerHTML = data.steps.map((s, i) =>
      lcCard(s.label, `<div style="font-size:14px;line-height:1.7;color:var(--text)">${esc(s.output)}</div>`, colors[i])
    ).join('');
  },

  // ── 3. RAG ─────────────────────────────────────────────────────────────────
  3: async () => {
    const q = document.getElementById('lc3-q').value.trim();
    if (!q) throw new Error('Enter a question.');
    const data = await lcPost('/langchain/rag', {question: q});
    document.getElementById('lc-right').innerHTML = `
      ${lcCard('Retrieved Chunks (FAISS top-2)', data.chunks.map((c,i)=>`
        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:8px;display:flex;gap:10px">
          <div style="min-width:24px;height:24px;background:var(--surface);border:1px solid var(--border);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:var(--muted);flex-shrink:0">${i+1}</div>
          <div style="font-size:14px;color:var(--text)">${esc(c)}</div>
        </div>`).join(''))}
      ${lcCard('Answer', `<div style="font-size:14px;line-height:1.8;color:var(--text)">${esc(data.answer)}</div>`, 'var(--blue)')}
    `;
  },

  // ── 4. Memory ──────────────────────────────────────────────────────────────
  4: async () => {
    const msg = document.getElementById('lc4-msg').value.trim();
    if (!msg) throw new Error('Type a message.');
    const data = await lcPost('/langchain/memory', {message: msg, session_id: _getLCSession()});
    document.getElementById('lc4-msg').value = '';
    const chatHtml = data.history.map(m => {
      const isUser = m.role === 'user';
      return `<div style="display:flex;justify-content:${isUser?'flex-end':'flex-start'};margin-bottom:8px">
        <div style="max-width:80%;background:${isUser?'#1e3a5f':'var(--surface)'};border:1px solid ${isUser?'var(--blue)':'var(--border)'};border-radius:10px;padding:8px 12px;font-size:13px;line-height:1.6;color:var(--text)">${esc(m.text)}</div>
      </div>`;
    }).join('');
    document.getElementById('lc-right').innerHTML = `
      ${lcCard(`Conversation (${data.history.length} messages)`, `<div>${chatHtml}</div>`)}
    `;
  },

  // ── 5. Tools ───────────────────────────────────────────────────────────────
  5: async () => {
    const q = document.getElementById('lc5-q').value.trim();
    if (!q) throw new Error('Enter a question.');
    const data = await lcPost('/langchain/tools', {question: q});
    const toolsHtml = data.tool_calls.length
      ? data.tool_calls.map(tc => `
          <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:8px">
            <div style="font-size:11px;font-weight:700;color:var(--amber);margin-bottom:6px">${esc(tc.tool)}()</div>
            <div style="display:grid;grid-template-columns:60px 1fr;gap:4px;font-size:12px">
              <span style="color:var(--muted)">Input</span><span style="color:var(--text);font-family:monospace">${esc(JSON.stringify(tc.input))}</span>
              <span style="color:var(--muted)">Result</span><span style="color:var(--green)">${esc(tc.result)}</span>
            </div>
          </div>`)
        .join('')
      : '<div style="color:var(--muted);font-size:13px">LLM answered directly (no tools needed)</div>';
    document.getElementById('lc-right').innerHTML = `
      ${lcCard('Tool Calls', toolsHtml)}
      ${lcCard('Final Answer', `<div style="font-size:14px;line-height:1.8">${esc(data.answer)}</div>`, 'var(--blue)')}
    `;
  },

  // ── 6. Document Processing ─────────────────────────────────────────────────
  6: async () => {
    const text = document.getElementById('lc6-text').value.trim();
    if (!text) throw new Error('Enter some text.');
    const data = await lcPost('/langchain/documents', {text});
    function chunksHtml(arr) {
      return arr.map(c => `
        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px 12px;margin-bottom:6px">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:11px;font-weight:700;color:var(--blue)">Chunk ${c.i}</span>
            <span style="font-size:11px;color:var(--muted)">${c.chars} chars</span>
          </div>
          <div style="font-size:12px;color:var(--text);line-height:1.5">${esc(c.text)}</div>
        </div>`).join('');
    }
    document.getElementById('lc-right').innerHTML = `
      <div style="font-size:12px;color:var(--muted);margin-bottom:4px">Total: ${data.total_chars} chars</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        <div>${lcCard(`CharacterTextSplitter (${data.char_chunks.length} chunks)`, chunksHtml(data.char_chunks))}</div>
        <div>${lcCard(`RecursiveCharacterTextSplitter (${data.rec_chunks.length} chunks)`, chunksHtml(data.rec_chunks), 'var(--green)')}</div>
      </div>`;
  },

  // ── 7. Output Parsers ──────────────────────────────────────────────────────
  7: async () => {
    const topic = document.getElementById('lc7-topic').value.trim();
    if (!topic) throw new Error('Enter a topic.');
    const data = await lcPost('/langchain/parsers', {topic});
    const listHtml = (data.list_output || []).map(i=>`<div style="display:flex;gap:6px;align-items:center;margin-bottom:4px"><span style="color:var(--blue)">•</span><span style="font-size:13px;color:var(--text)">${esc(i.trim())}</span></div>`).join('');
    const jsonStr  = JSON.stringify(data.json_output, null, 2);
    document.getElementById('lc-right').innerHTML = `
      ${lcCard('StrOutputParser — plain text', `<div style="font-size:14px;line-height:1.8;color:var(--text)">${esc(data.string_output)}</div>`)}
      ${lcCard('CommaSeparatedListOutputParser — Python list', `<div style="margin-bottom:6px;font-size:11px;color:var(--muted)">Type: list[str] · ${(data.list_output||[]).length} items</div>${listHtml}`, 'var(--blue)')}
      ${lcCard('JsonOutputParser — typed JSON', lcPre(jsonStr), 'var(--green)')}
    `;
  },

  // ── 8. Single Agent ────────────────────────────────────────────────────────
  8: async () => {
    const q = document.getElementById('lc8-q').value.trim();
    if (!q) throw new Error('Enter a question.');
    const data = await lcPost('/langchain/agent', {question: q});
    const stepsHtml = (data.steps || []).map((s, i) => `
      <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:8px">
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
          <div style="min-width:20px;height:20px;background:var(--amber);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700">${i+1}</div>
          <span style="font-size:11px;font-weight:700;color:var(--amber)">${esc(s.tool)}()</span>
        </div>
        <div style="display:grid;grid-template-columns:60px 1fr;gap:4px;font-size:12px">
          <span style="color:var(--muted)">Input</span><span style="color:var(--text);font-family:monospace">${esc(JSON.stringify(s.input))}</span>
          <span style="color:var(--muted)">Result</span><span style="color:var(--green)">${esc(s.observation)}</span>
        </div>
      </div>`).join('') || '<div style="color:var(--muted);font-size:13px">No tool calls — LLM answered directly.</div>';
    document.getElementById('lc-right').innerHTML = `
      ${lcCard(`ReAct Steps (${(data.steps||[]).length} tool calls)`, stepsHtml)}
      ${lcCard('Final Answer', `<div style="font-size:14px;line-height:1.8">${esc(data.answer)}</div>`, 'var(--blue)')}
    `;
  },

  // ── 9. Multi-Agent Simple ──────────────────────────────────────────────────
  9: async () => {
    const topic = document.getElementById('lc9-topic').value.trim();
    if (!topic) throw new Error('Enter a topic.');
    const data = await lcPost('/langchain/multiagent', {topic});
    document.getElementById('lc-right').innerHTML = `
      ${lcCard('Research Agent — bullet facts', `<div style="font-size:14px;line-height:1.9;color:var(--text);white-space:pre-wrap">${esc(data.research)}</div>`, 'var(--blue)')}
      ${lcCard('Writer Agent — blog post', `<div style="font-size:14px;line-height:1.9;color:var(--text)">${esc(data.blog)}</div>`, 'var(--green)')}
    `;
  },

  // ── 10. LangGraph ──────────────────────────────────────────────────────────
  10: async () => {
    const topic = document.getElementById('lc10-topic').value.trim();
    if (!topic) throw new Error('Enter a topic.');
    const data = await lcPost('/langchain/langgraph', {topic});
    const nodeColors = {Manager:'var(--blue)',Research:'var(--blue)','Research Agent':'var(--blue)',Writer:'var(--green)','Writer Agent':'var(--green)',Reviewer:'var(--amber)','Reviewer Agent':'var(--amber)'};
    const statusIcon = {ok:'✓', approved:'✅', revise:'↩'};
    const logHtml = (data.log || []).map(l => {
      const col = nodeColors[l.node] || 'var(--muted)';
      const icon = statusIcon[l.status] || '→';
      return `<div style="display:flex;gap:10px;margin-bottom:10px;align-items:flex-start">
        <div style="min-width:90px;padding:3px 8px;background:${col}22;border:1px solid ${col};border-radius:999px;font-size:10px;font-weight:700;color:${col};text-align:center;flex-shrink:0">${esc(l.node)}</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.6;flex:1">${icon} ${esc(l.detail)}</div>
      </div>`;
    }).join('');
    document.getElementById('lc-right').innerHTML = `
      ${lcCard(`Graph Execution Log · ${data.revisions} revision(s)`, logHtml)}
      ${lcCard('Final Blog Post', `<div style="font-size:13px;line-height:1.9;color:var(--text)">${esc(data.final)}</div>`, 'var(--purple)')}
    `;
  },
};

// ── Code modal ──────────────────────────────────────────────────────────────
let _lcCurrentDemo = 1;

const LC_CODE = {
  1: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
parser = StrOutputParser()

# ── Define the template once with {variables} ──────────────────────────────
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in under 2 sentences."),
    ("human",  "{question}"),
])

# ── Build the chain with the | pipe operator ───────────────────────────────
#    template → LLM → parse output to plain string
chain = template | llm | parser

# ── Invoke: fill in variables at runtime — no hardcoded strings ───────────
result = chain.invoke({
    "role":     "pirate",
    "question": "What is machine learning?",
})
print(result)
# → Arrr, machine learning be when computers learn patterns from data...`,

  2: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
parser = StrOutputParser()

# ── Each step is its own mini-chain ──────────────────────────────────────
step1 = ChatPromptTemplate.from_template(
    "Translate to English. Return ONLY the translation.\\n\\n{text}"
) | llm | parser

step2 = ChatPromptTemplate.from_template(
    "Summarise in exactly one sentence:\\n\\n{t}"
) | llm | parser

step3 = ChatPromptTemplate.from_template(
    'Wrap in JSON: {{"summary": "..."}}\\nSummary: {s}'
) | llm | parser

# ── Run in sequence — output of each step feeds the next ─────────────────
spanish    = "La inteligencia artificial está transformando el mundo..."

translated = step1.invoke({"text": spanish})
summary    = step2.invoke({"t": translated})
json_out   = step3.invoke({"s": summary})

print("Step 1 — Translate:", translated)
print("Step 2 — Summarise:", summary)
print("Step 3 — JSON:     ", json_out)`,

  3: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── 1. Build a knowledge base ─────────────────────────────────────────────
docs = [
    Document(page_content="RAG = Retrieval Augmented Generation."),
    Document(page_content="FAISS is Meta AI's library for fast similarity search."),
    Document(page_content="Embeddings convert text to numerical vectors."),
    Document(page_content="Chunking splits large documents into smaller pieces."),
    # ... add as many documents as you need
]

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})

# ── 2. Retrieve: embed the question, find 2 closest chunks ───────────────
question = "What is FAISS used for?"
chunks   = retriever.invoke(question)
context  = "\\n".join(f"- {d.page_content}" for d in chunks)

# ── 3. Generate: answer ONLY from retrieved context, not training memory ──
llm    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
prompt = ChatPromptTemplate.from_template(
    "Use ONLY the context below to answer.\\n\\n"
    "Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
)
answer = (prompt | llm | StrOutputParser()).invoke({
    "context":  context,
    "question": question,
})
print(answer)`,

  4: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

llm    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)
parser = StrOutputParser()

# ── MessagesPlaceholder injects the full history into every prompt ────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Remember everything."),
    MessagesPlaceholder(variable_name="history"),   # ← history goes here
    ("human", "{input}"),
])

chain = prompt | llm | parser

# ── History = a plain Python list we manage ourselves ────────────────────
history = []

def chat(user_input: str) -> str:
    # Pass the full history + new message to Claude
    answer = chain.invoke({"history": history, "input": user_input})
    # Append both sides to history for next turn
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    return answer

# ── Each call includes the full conversation history ─────────────────────
print(chat("Hi! My name is Raj and I love cricket."))
print(chat("I work as a software engineer in Singapore."))
print(chat("What sport do I love?"))    # Claude remembers: cricket
print(chat("Where do I work?"))         # Claude remembers: Singapore
print(f"Messages in history: {len(history)}")`,

  5: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── 1. Define tools with the @tool decorator ──────────────────────────────
#    The docstring is what Claude reads to understand what the tool does.
@tool
def calculator(expression: str) -> str:
    """Evaluate a maths expression, e.g. '25 * 4 + 10'."""
    return str(eval(expression, {"__builtins__": {}}, {}))

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city (mock data)."""
    data = {
        "london":    "15°C, Cloudy",
        "singapore": "32°C, Humid",
        "new york":  "22°C, Sunny",
    }
    return data.get(city.lower(), "No data for that city.")

@tool
def word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    return str(len(text.split()))

# ── 2. Bind tools to the LLM — Claude now knows they exist ───────────────
tools         = [calculator, get_weather, word_count]
tools_map     = {t.name: t for t in tools}
lc            = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
lc_with_tools = lc.bind_tools(tools)

# ── 3. Claude decides which tools to call ────────────────────────────────
question = "What is 144/12? Also, what is the weather in Singapore?"
response = lc_with_tools.invoke(question)

# ── 4. Execute each tool Claude requested, show results ──────────────────
for block in response.content:
    if hasattr(block, "type") and block.type == "tool_use":
        result = tools_map[block.name].invoke(block.input)
        print(f"Tool called : {block.name}")
        print(f"Input sent  : {block.input}")
        print(f"Result back : {result}\\n")`,

  6: `from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

text = """Artificial Intelligence is the simulation of human intelligence.
It covers machine learning, deep learning, and natural language processing.

Machine Learning is a subset of AI where models learn patterns from data.
Supervised, unsupervised, and reinforcement learning are the three paradigms.

Deep Learning uses neural networks with many layers to learn representations.
It powers image recognition, speech recognition, and large language models."""

doc = Document(page_content=text)

# ── CharacterTextSplitter: cuts ONLY on \\n\\n ─────────────────────────────
char_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\\n\\n",       # cuts on paragraph breaks only
)
char_chunks = char_splitter.split_documents([doc])
print(f"CharacterTextSplitter → {len(char_chunks)} chunks")
for i, c in enumerate(char_chunks):
    print(f"  Chunk {i+1} ({len(c.page_content)} chars)")

# ── RecursiveCharacterTextSplitter: tries \\n\\n → \\n → space ──────────────
rec_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    # default separators: ["\\n\\n", "\\n", " ", ""]
)
rec_chunks = rec_splitter.split_documents([doc])
print(f"\\nRecursiveCharacterTextSplitter → {len(rec_chunks)} chunks")
for i, c in enumerate(rec_chunks):
    print(f"  Chunk {i+1} ({len(c.page_content)} chars)")

# In production: replace Document(...) with PyPDFLoader, WebBaseLoader, etc.`,

  7: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
)

lc    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
topic = "machine learning"

# ── 1. StrOutputParser — plain string (default) ───────────────────────────
str_chain = (
    ChatPromptTemplate.from_template("Write one sentence explaining '{t}'.")
    | lc | StrOutputParser()
)
result = str_chain.invoke({"t": topic})
print(type(result))   # <class 'str'>
print(result)

# ── 2. CommaSeparatedListOutputParser — Python list ───────────────────────
list_parser = CommaSeparatedListOutputParser()
list_chain  = (
    ChatPromptTemplate.from_template(
        "List five items related to '{t}'. {fi}"
    ).partial(format_instructions=list_parser.get_format_instructions())
    | lc | list_parser
)
result = list_chain.invoke({"t": topic})
print(type(result))   # <class 'list'>
print(result)         # ['Neural networks', 'Python', 'Scikit-learn', ...]

# ── 3. JsonOutputParser — Python dict ─────────────────────────────────────
json_parser = JsonOutputParser()
json_chain  = (
    ChatPromptTemplate.from_template(
        "Return JSON about '{t}' with keys: name, description, key_facts. {fi}"
    ).partial(format_instructions=json_parser.get_format_instructions())
    | lc | json_parser
)
result = json_chain.invoke({"t": topic})
print(type(result))   # <class 'dict'>
print(result)         # {"name": "Machine Learning", "description": "...", ...}`,

  8: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ── 1. Define tools ───────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a maths expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

@tool
def get_exchange_rate(currency_pair: str) -> str:
    """Get exchange rate. Format: 'USD_SGD'."""
    rates = {"usd_sgd": 1.35, "eur_usd": 1.08, "gbp_usd": 1.27}
    key   = currency_pair.lower().replace("/", "_")
    rate  = rates.get(key)
    parts = key.split("_")
    return f"1 {parts[0].upper()} = {rate} {parts[1].upper()}"

tools = [calculator, get_exchange_rate]

# ── 2. Build the agent ────────────────────────────────────────────────────
lc     = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed."),
    ("human",  "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # agent writes its thoughts here
])

agent    = create_tool_calling_agent(lc, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                    # prints the full ReAct loop to console
    return_intermediate_steps=True,  # gives us each tool call + result
)

# ── 3. Run — the agent loops until it is satisfied ───────────────────────
result = executor.invoke({
    "input": "I have $1000 USD. Convert to SGD, then add 18% tax and $50 flat fee."
})
print("Final answer:", result["output"])

# result["intermediate_steps"] contains each (tool_call, observation) pair
for action, observation in result["intermediate_steps"]:
    print(f"  Called: {action.tool}({action.tool_input})")
    print(f"  Got:    {observation}")`,

  9: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

# ── Helper: create a specialist agent with a focused system prompt ────────
def make_agent(system_prompt: str) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    return AgentExecutor(
        agent=create_tool_calling_agent(lc, [], prompt),
        tools=[], verbose=False,
    )

# ── Agent 1: Research — only job is to find facts ────────────────────────
research_agent = make_agent(
    "You are a research assistant. Given a topic, "
    "return 5 key bullet-point facts. Be concise and factual."
)

# ── Agent 2: Writer — only job is to write a blog post ───────────────────
writer_agent = make_agent(
    "You are a blog writer. Given research notes, write a short "
    "3-paragraph blog post (max 150 words). Use a friendly tone."
)

# ── Orchestrate: call Research, feed its output directly to Writer ────────
topic    = "The impact of AI on software engineering jobs"
research = research_agent.invoke({"input": topic})["output"]
blog     = writer_agent.invoke({"input": f"Research notes:\\n{research}"})["output"]

print("=== Research Agent ===")
print(research)
print("\\n=== Writer Agent ===")
print(blog)`,

  10: `from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

lc = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
p  = StrOutputParser()

# ── Shared state — passed between ALL nodes ───────────────────────────────
class BlogState(TypedDict):
    topic:     str
    research:  str
    draft:     str
    feedback:  str
    final:     str
    revisions: int

# ── Nodes — plain Python functions that receive and return state ──────────
def manager_node(state: BlogState) -> BlogState:
    print(f"[Manager] Starting pipeline for: {state['topic']}")
    return state   # passes topic unchanged to the next node

def research_node(state: BlogState) -> BlogState:
    out = (ChatPromptTemplate.from_template(
        "Research '{t}'. Return 5 concise bullet-point facts."
    ) | lc | p).invoke({"t": state["topic"]})
    return {"research": out}

def writer_node(state: BlogState) -> BlogState:
    fb  = f"\\n\\nRevision feedback:\\n{state['feedback']}" if state.get("feedback") else ""
    out = (ChatPromptTemplate.from_template(
        "Write a 3-paragraph blog post (max 120 words).\\n\\nResearch:\\n{r}{fb}"
    ) | lc | p).invoke({"r": state["research"], "fb": fb})
    return {"draft": out, "revisions": state.get("revisions", 0)}

def reviewer_node(state: BlogState) -> BlogState:
    review = (ChatPromptTemplate.from_template(
        "Review this draft. Reply APPROVED or REVISE: <feedback>\\n\\nDraft:\\n{d}"
    ) | lc | p).invoke({"d": state["draft"]})
    if review.strip().upper().startswith("APPROVED"):
        return {"final": state["draft"], "feedback": ""}
    return {"feedback": review, "revisions": state.get("revisions", 0) + 1}

# ── Conditional edge: loop back to writer, or end? ────────────────────────
def should_revise(state: BlogState) -> Literal["writer", "end"]:
    if state.get("final") or state.get("revisions", 0) >= 2:
        return "end"
    return "writer"   # ← this is what creates the feedback loop

# ── Build the graph ───────────────────────────────────────────────────────
graph = StateGraph(BlogState)
graph.add_node("manager",  manager_node)
graph.add_node("research", research_node)
graph.add_node("writer",   writer_node)
graph.add_node("reviewer", reviewer_node)

graph.set_entry_point("manager")
graph.add_edge("manager",  "research")
graph.add_edge("research", "writer")
graph.add_edge("writer",   "reviewer")
graph.add_conditional_edges(
    "reviewer", should_revise,
    {"writer": "writer", "end": END}   # ← reviewer → writer is the loop
)

# ── Compile and run ───────────────────────────────────────────────────────
app    = graph.compile()
result = app.invoke({
    "topic": "How LangGraph improves multi-agent AI systems",
    "research": "", "draft": "", "feedback": "", "final": "", "revisions": 0,
})
print(result.get("final") or result.get("draft"))`,
};

const _lcFilenames = {
  1:'01_prompt_management.py', 2:'02_llm_chaining.py', 3:'03_rag.py',
  4:'04_memory.py', 5:'05_tools_function_calling.py', 6:'06_document_processing.py',
  7:'07_output_parsers.py', 8:'08_single_agent.py',
  9:'09_multi_agent_simple.py', 10:'10_langgraph_multiagent.py',
};

// ── Without-LangChain code: realistic production versions showing the manual work ─
// Each demo uses IDENTICAL inputs/steps as the "With LangChain" version above.
const LC_CODE_PLAIN = {
  1: `# ── WITHOUT LangChain: Prompt Management ────────────────────────────────────
# Same inputs as the LangChain version: role="pirate", question="What is machine learning?"
# Without LangChain, every template is a raw string you manage yourself.

import anthropic, re, time

client = anthropic.Anthropic()

# ── Templates defined as plain strings (no structure, no registry) ────────
SYSTEM_TEMPLATE = "You are a {role}. Answer in under 2 sentences."
HUMAN_TEMPLATE  = "{question}"

def fill_template(template: str, variables: dict) -> str:
    """No built-in validation — you must detect missing variables yourself."""
    missing = [k for k in re.findall(r"\\{(\\w+)\\}", template) if k not in variables]
    if missing:
        raise ValueError(f"Missing template variables: {missing}")
    return template.format(**variables)

def call_claude(system: str, human: str, retries: int = 3) -> str:
    """Retry wrapper — must be written from scratch in every single project."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": human}]
            )
            return resp.content[0].text
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt)   # exponential backoff — you write this too

# ── Same inputs as the LangChain version ─────────────────────────────────
role     = "pirate"
question = "What is machine learning?"

system_msg = fill_template(SYSTEM_TEMPLATE, {"role": role})
human_msg  = fill_template(HUMAN_TEMPLATE,  {"question": question})
result     = call_claude(system_msg, human_msg)
print(result)
# → Arrr, machine learning be when computers learn patterns from data...

# ✗ No few-shot example support — add it yourself to every template
# ✗ No chat-history injection — wire it manually each time
# ✗ No provider switching — changing to OpenAI means rewriting all API calls
# ✗ Template validation, retry logic, logging all rewritten per project
#
# Compare: With LangChain — same result in 3 lines, no boilerplate.
#   chain  = ChatPromptTemplate.from_messages([("system","You are a {role}..."),("human","{question}")]) | llm | StrOutputParser()
#   result = chain.invoke({"role": "pirate", "question": "What is machine learning?"})`,

  2: `# ── WITHOUT LangChain: LLM Chaining ────────────────────────────────────────
# Same 3 steps as the LangChain version: Translate → Summarise → JSON.
# Without LangChain, each step is a separate API call you must write,
# wire, validate, and error-check entirely by hand.

import anthropic, json, time

client = anthropic.Anthropic()

def call_claude(user_prompt: str, retries: int = 3) -> str:
    """Retry wrapper — must be copy-pasted into every project that needs chaining."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                messages=[{"role": "user", "content": user_prompt}]
            )
            output = resp.content[0].text.strip()
            if not output:
                raise ValueError("Empty response from LLM")  # ← you check this
            return output
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt)   # exponential backoff — you write this too

text = "La inteligencia artificial está transformando el mundo moderno de maneras increíbles."

# Step 1 — Translate to English
# (you write a full API call for every single step)
print("Step 1: Translating…")
translated = call_claude(f"Translate to English. Return ONLY the translation.\\n\\n{text}")

# Step 2 — Summarise
# (you manually pass Step 1's output as Step 2's input — easy to wire wrong)
print("Step 2: Summarising…")
summary = call_claude(f"Summarise in exactly one sentence:\\n\\n{translated}")

# Step 3 — Wrap in JSON
# (adding one more step = another full block of boilerplate)
print("Step 3: Wrapping in JSON…")
raw_json = call_claude(f'Wrap this in JSON as: {{"summary": "..."}}.\\nSummary: {summary}')

# Strip markdown fences if LLM added them — you handle this edge case yourself
if raw_json.startswith("\`\`\`"):
    raw_json = raw_json.split("\`\`\`")[1].lstrip("json").strip()

json_out = json.loads(raw_json)   # crashes if LLM ignored the format — no retry

print("\\nStep 1 — Translate:", translated)
print("Step 2 — Summarise:", summary)
print("Step 3 — JSON:     ", json.dumps(json_out))

# ✗ 3 steps = 3 separate API calls written by hand — 40+ lines of boilerplate
# ✗ Adding a 4th step = copy-paste another full block
# ✗ No streaming — requires full rewrite to use stream=True
# ✗ Intermediate values not logged unless you add print() everywhere
# ✗ Retry logic duplicated across every file that chains LLM calls
# ✗ If Step 2 output is empty, Step 3 silently runs on bad input
#
# Compare: With LangChain — same 3 steps in 5 lines, retry & logging built in.
#   chain = (translate_prompt | llm | parser
#          | summarise_prompt | llm | parser
#          | json_prompt      | llm | JsonOutputParser())
#   result = chain.invoke({"text": text})`,

  3: `# ── WITHOUT LangChain: RAG (Retrieval-Augmented Generation) ─────────────────
# Same 4 documents and same question as the LangChain version.
# Same 3 steps: Build KB → Retrieve → Generate.
# Without LangChain, you must wire every piece of the pipeline yourself.

import anthropic, numpy as np

client = anthropic.Anthropic()

# ── Step 1: Build knowledge base — same 4 docs as the LangChain version ──
docs = [
    "RAG = Retrieval Augmented Generation.",
    "FAISS is Meta AI's library for fast similarity search.",
    "Embeddings convert text to numerical vectors.",
    "Chunking splits large documents into smaller pieces.",
]

# Anthropic has no embeddings API — must call a SECOND provider (OpenAI)
# LangChain hides this with HuggingFaceEmbeddings (runs locally, no extra API)
from openai import OpenAI
embedder = OpenAI()

def embed(text: str) -> list[float]:
    """Extra API provider required — LangChain can use local models instead."""
    resp = embedder.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding   # 1536-dim vector

print("Embedding knowledge base (4 API calls)…")
doc_vecs = [embed(d) for d in docs]   # 4 separate API calls — slow, costs money

# ── Step 2: Retrieve — cosine similarity written from scratch ─────────────
def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, k: int = 2) -> list[str]:
    qv     = embed(query)   # another API call just to embed the query
    scores = [(cosine_sim(qv, dv), doc) for dv, doc in zip(doc_vecs, docs)]
    return [doc for _, doc in sorted(scores, reverse=True)[:k]]

# ── Step 3: Generate — same prompt as the LangChain version ──────────────
question = "What is FAISS used for?"
context  = "\\n".join(f"- {c}" for c in retrieve(question))

resp = client.messages.create(
    model="claude-haiku-4-5-20251001", max_tokens=256,
    messages=[{"role": "user", "content":
        f"Use ONLY the context below to answer.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"}]
)
print(resp.content[0].text)

# ✗ Requires TWO API providers (Anthropic + OpenAI) — extra cost and dependency
# ✗ 5 embedding API calls just to answer one question
# ✗ Index rebuilt from scratch on every restart — no persistence
# ✗ No metadata per chunk (page, source file, date)
# ✗ PDF/CSV/HTML each need a separate loader library
#
# Compare: With LangChain — same 3 steps in 8 lines, one provider, persists to disk.
#   embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # runs locally
#   vectorstore = FAISS.from_documents(docs, embeddings)
#   chunks      = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(question)
#   answer      = (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": question})`,

  4: `# ── WITHOUT LangChain: Memory / Conversation History ────────────────────────
# Same 4 turns as the LangChain version: Raj, cricket, Singapore, recall questions.
# Without LangChain, you manage the message list, token budget, and trimming yourself.

import anthropic, time

client = anthropic.Anthropic()

# ── History is just a plain Python list you manage yourself ──────────────
history: list[dict] = []

def estimate_tokens(messages: list) -> int:
    """Rough estimate — real accuracy needs the Anthropic token-counting API."""
    return sum(len(str(m.get("content","")).split()) * 1.4 for m in messages)

def trim_history(history: list) -> list:
    """Drop oldest pairs when context grows too large — you write this yourself.
    WARNING: old context is LOST, not summarised like LangChain's SummaryMemory."""
    while estimate_tokens(history) > 2000 and len(history) >= 2:
        history = history[2:]   # remove oldest user+assistant pair
    return history

def call_claude(user_input: str, retries: int = 3) -> str:
    """Must write retry logic from scratch in every project."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=256,
                system="You are a friendly assistant. Remember everything.",
                messages=history + [{"role": "user", "content": user_input}]
            )
            return resp.content[0].text
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt)

def chat(user_input: str) -> str:
    global history
    answer = call_claude(user_input)
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": answer})
    history = trim_history(history)   # must call manually every single turn
    return answer

# ── Same 4 turns as the LangChain version ────────────────────────────────
print(chat("Hi! My name is Raj and I love cricket."))
print(chat("I work as a software engineer in Singapore."))
print(chat("What sport do I love?"))    # should recall: cricket
print(chat("Where do I work?"))         # should recall: Singapore
print(f"Messages in history: {len(history)}")

# ✗ Trimming drops oldest turns entirely — memory is LOST, not summarised
# ✗ Token counting is approximate — real accuracy requires another API call
# ✗ No DB persistence — restart the process and history is gone
# ✗ For long conversations, Claude gradually forgets the beginning
#
# Compare: With LangChain — MessagesPlaceholder handles injection in one line.
#   prompt = ChatPromptTemplate.from_messages([("system","..."), MessagesPlaceholder("history"), ("human","{input}")])
#   chain  = prompt | llm | parser
#   answer = chain.invoke({"history": history, "input": user_input})`,

  5: `# ── WITHOUT LangChain: Tools & Function Calling ─────────────────────────────
# Same 3 tools and same question as the LangChain version.
# Without LangChain, you write schemas, dispatch loop, and multi-round
# conversation management entirely by hand — for every tool you add.

import anthropic

client = anthropic.Anthropic()

# ── Same 3 tool implementations as the LangChain version ─────────────────
def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

def get_weather(city: str) -> str:
    data = {"london": "15°C, Cloudy", "singapore": "32°C, Humid", "new york": "22°C, Sunny"}
    return data.get(city.lower(), "No data for that city.")

def word_count(text: str) -> str:
    return str(len(text.split()))

# ── Must write JSON schema by hand for EVERY tool — @tool does this for you ──
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a maths expression, e.g. '25 * 4 + 10'.",
        "input_schema": {"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}
    },
    {
        "name": "get_weather",
        "description": "Return current weather for a city (mock data).",
        "input_schema": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    },
    {
        "name": "word_count",
        "description": "Count the number of words in a piece of text.",
        "input_schema": {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
    }
]

# ── Dispatcher — must update this block manually for every new tool ───────
def dispatch(name: str, inputs: dict) -> str:
    if name == "calculator": return calculator(**inputs)
    if name == "get_weather": return get_weather(**inputs)
    if name == "word_count":  return word_count(**inputs)
    return f"Error: unknown tool '{name}'"

# ── Same question as the LangChain version ────────────────────────────────
question = "What is 144/12? Also, what is the weather in Singapore?"
messages  = [{"role": "user", "content": question}]

# Multi-round loop — you write this boilerplate for every project with tools
for _ in range(10):   # safety cap against infinite loops — your responsibility
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=512,
        tools=TOOLS, messages=messages
    )
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        break

    tool_results = []
    for block in resp.content:
        if block.type == "tool_use":
            result = dispatch(block.name, block.input)
            print(f"Tool called : {block.name}")
            print(f"Input sent  : {block.input}")
            print(f"Result back : {result}\\n")
            tool_results.append({"type":"tool_result","tool_use_id":block.id,"content":result})
    messages.append({"role": "user", "content": tool_results})

# ✗ 3 tools = 3 schema blocks + dispatch() case + tool function — all manual
# ✗ Add a 4th tool = update schemas AND dispatch() in two separate places
# ✗ JSON schema typos cause silent failures (wrong key name → tool never called)
# ✗ No retries, no tracing, no callback hooks built in
#
# Compare: With LangChain — add a tool with just a decorator, zero schema writing.
#   @tool
#   def calculator(expression: str) -> str:
#       "Evaluate a maths expression."       ← docstring IS the schema description
#       return str(eval(expression, {"__builtins__":{}}, {}))
#   lc_with_tools = ChatAnthropic(...).bind_tools([calculator, get_weather, word_count])`,

  6: `# ── WITHOUT LangChain: Document Processing ──────────────────────────────────
# Same text and same two splitting strategies as the LangChain version:
#   Strategy 1 — Character split (cuts ONLY on \\n\\n)
#   Strategy 2 — Recursive split (tries \\n\\n → \\n → space in order)
# Without LangChain, you implement both algorithms from scratch.

# ── Same input text as the LangChain version ─────────────────────────────
text = """Artificial Intelligence is the simulation of human intelligence.
It covers machine learning, deep learning, and natural language processing.

Machine Learning is a subset of AI where models learn patterns from data.
Supervised, unsupervised, and reinforcement learning are the three paradigms.

Deep Learning uses neural networks with many layers to learn representations.
It powers image recognition, speech recognition, and large language models."""

# ── Strategy 1: Character splitter — cuts ONLY on the chosen separator ───
def character_split(text: str, chunk_size: int = 200, sep: str = "\\n\\n") -> list[str]:
    """Splits only on sep. If a paragraph exceeds chunk_size, it overflows as-is."""
    parts  = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current: chunks.append(current)
            current = part
    if current: chunks.append(current)
    return chunks

char_chunks = character_split(text, chunk_size=200, sep="\\n\\n")
print(f"Character split → {len(char_chunks)} chunks")
for i, c in enumerate(char_chunks):
    print(f"  Chunk {i+1} ({len(c)} chars)")

# ── Strategy 2: Recursive splitter — falls back \\n\\n → \\n → space → char ──
def recursive_split(text: str, chunk_size: int = 200, seps=None) -> list[str]:
    """Tries each separator in order; recurses on pieces still too large."""
    if seps is None:
        seps = ["\\n\\n", "\\n", " "]
    if not seps or len(text) <= chunk_size:
        return [text] if text.strip() else []
    sep   = seps[0]
    parts = text.split(sep)
    chunks, current = [], ""
    for part in parts:
        candidate = (current + sep + part).lstrip(sep) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current: chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(recursive_split(part, chunk_size, seps[1:]))  # recurse
                current = ""
            else:
                current = part
    if current: chunks.append(current)
    return chunks

rec_chunks = recursive_split(text, chunk_size=200)
print(f"\\nRecursive split → {len(rec_chunks)} chunks")
for i, c in enumerate(rec_chunks):
    print(f"  Chunk {i+1} ({len(c)} chars)")

# ✗ Strategy 1 is ~15 lines just to split on a separator — LangChain: 3 lines
# ✗ Strategy 2 is ~20 lines and still misses edge cases LangChain handles
# ✗ No metadata per chunk (source, page number, document ID)
# ✗ Loading a PDF/CSV/HTML still needs a separate library per format
#
# Compare: With LangChain — same two strategies in 3 lines each.
#   CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\\n\\n").split_documents([doc])
#   RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents([doc])`,

  7: `# ── WITHOUT LangChain: Output Parsers ───────────────────────────────────────
# Same 3 output types and same topic as the LangChain version: "machine learning"
#   1. Plain string  (StrOutputParser equivalent)
#   2. Python list   (CommaSeparatedListOutputParser equivalent)
#   3. Python dict   (JsonOutputParser equivalent)
# Without LangChain, you write format instructions, parsing, and retries by hand.

import anthropic, json, re, time

client = anthropic.Anthropic()
topic = "machine learning"

def call_claude(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt)

def strip_fences(text: str) -> str:
    """LLMs often wrap output in \`\`\`json … \`\`\` even when told not to."""
    if text.startswith("\`\`\`"):
        text = re.sub(r"^\`\`\`[\\w]*\\n?", "", text)
        text = re.sub(r"\\n?\`\`\`$", "", text)
    return text.strip()

# ── 1. StrOutputParser — raw text, no parsing needed ─────────────────────
raw = call_claude(f"Write one sentence explaining '{topic}'.")
result_str = raw   # already a string — this part is fine without LangChain
print(type(result_str))   # <class 'str'>
print(result_str)

# ── 2. CommaSeparatedListOutputParser — must write format instruction yourself ──
raw = call_claude(
    f"List five items related to '{topic}'. "
    "Respond with items separated by commas only. No numbers, no bullets."
)
raw = re.sub(r"^\\d+\\.\\s*", "", raw, flags=re.MULTILINE)  # strip numbered list if LLM disobeyed
result_list = [item.strip() for item in raw.split(",") if item.strip()]
print(type(result_list))   # <class 'list'>
print(result_list)         # ['Neural networks', 'Python', 'Scikit-learn', ...]

# ── 3. JsonOutputParser — full retry loop required ────────────────────────
FORMAT_HINT = (
    f"Return JSON about '{topic}' with keys: name, description, key_facts. "
    'Example: {"name":"...","description":"...","key_facts":["...","..."]}. '
    "No markdown, no explanation — just the JSON object."
)
MAX_RETRIES = 3
for attempt in range(MAX_RETRIES):
    raw = strip_fences(call_claude(FORMAT_HINT))
    try:
        result_dict = json.loads(raw)
        required = {"name", "description", "key_facts"}
        if not required.issubset(result_dict.keys()):
            raise ValueError(f"Missing keys: {required - result_dict.keys()}")
        break
    except (json.JSONDecodeError, ValueError) as e:
        if attempt == MAX_RETRIES - 1:
            raise RuntimeError(f"Failed after {MAX_RETRIES} tries: {e}")
        FORMAT_HINT += f"\\n\\nPrevious response was invalid ({e}). Try again."

print(type(result_dict))   # <class 'dict'>
print(result_dict)         # {"name":"Machine Learning","description":"...","key_facts":[...]}

# ✗ StrOutputParser: trivial — no difference
# ✗ List parser: 3 lines to write the format instruction + strip + split — easy to break
# ✗ JSON parser: 15 lines with retry loop — crashes if LLM ignores the format even once
# ✗ Type coercion not enforced — "year": "1991" (string) passes even if you wanted int
#
# Compare: With LangChain — format instructions injected automatically, retries built in.
#   CommaSeparatedListOutputParser()              # → Python list, auto format instructions
#   JsonOutputParser()                            # → Python dict, auto retry on bad JSON
#   chain = (prompt.partial(fi=parser.get_format_instructions()) | lc | parser)`,

  8: `# ── WITHOUT LangChain: Single Agent (ReAct loop) ─────────────────────────────
# Same 2 tools and same question as the LangChain version.
# Without LangChain, you build the entire reason → act → observe loop yourself.

import anthropic

client = anthropic.Anthropic()

# ── Same 2 tool implementations as the LangChain version ─────────────────
def calculator(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))

def get_exchange_rate(currency_pair: str) -> str:
    rates = {"usd_sgd": 1.35, "eur_usd": 1.08, "gbp_usd": 1.27}
    key   = currency_pair.lower().replace("/", "_")
    rate  = rates.get(key, "N/A")
    parts = key.split("_")
    return f"1 {parts[0].upper()} = {rate} {parts[1].upper()}"

# ── Must write JSON schema by hand for every tool ─────────────────────────
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a maths expression.",
        "input_schema": {"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}
    },
    {
        "name": "get_exchange_rate",
        "description": "Get exchange rate. Format: 'USD_SGD'.",
        "input_schema": {"type":"object","properties":{"currency_pair":{"type":"string"}},"required":["currency_pair"]}
    }
]

def dispatch(name: str, inputs: dict) -> str:
    if name == "calculator":      return calculator(**inputs)
    if name == "get_exchange_rate": return get_exchange_rate(**inputs)
    return f"Error: unknown tool '{name}'"

# ── Same question as the LangChain version ────────────────────────────────
question = "I have $1000 USD. Convert to SGD, then add 18% tax and $50 flat fee."
messages = [{"role": "user", "content": question}]
intermediate_steps = []

# ReAct loop: reason → act → observe → reason again — written entirely by hand
for _ in range(10):   # safety cap against infinite loops — your responsibility
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=1024,
        tools=TOOLS, messages=messages
    )
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason != "tool_use":
        break   # agent is satisfied — exit loop

    tool_results = []
    for block in resp.content:
        if block.type == "tool_use":
            result = dispatch(block.name, block.input)
            intermediate_steps.append((block.name, block.input, result))
            tool_results.append({"type":"tool_result","tool_use_id":block.id,"content":result})
    messages.append({"role": "user", "content": tool_results})

final = next((b.text for b in resp.content if hasattr(b, "text")), "No answer")
print("Final answer:", final)
for tool, inp, obs in intermediate_steps:
    print(f"  Called: {tool}({inp})")
    print(f"  Got:    {obs}")

# ✗ 55 lines for a 2-tool agent — grows linearly with every new tool
# ✗ dispatch() must be updated manually every time you add a tool
# ✗ No built-in verbose tracing — add print() statements to debug
# ✗ No streaming, no callbacks, no return_intermediate_steps flag
#
# Compare: With LangChain — @tool decorator + AgentExecutor does all the above.
#   executor = AgentExecutor(agent=agent, tools=[calculator, get_exchange_rate],
#                            verbose=True, return_intermediate_steps=True)
#   result = executor.invoke({"input": question})
#   # result["intermediate_steps"] → each (tool_call, observation) pair, free`,

  9: `# ── WITHOUT LangChain: Multi-Agent (Simple) ──────────────────────────────────
# Two agents wired together manually. Add a third agent = rewrite the orchestrator.
# No shared tracing, no callbacks, no retry coordination between agents.

import anthropic, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
client = anthropic.Anthropic()

def call_claude(system: str, user: str, max_tokens: int = 512, retries: int = 3) -> str:
    """Shared retry wrapper — must exist in every file that calls the API."""
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            result = resp.content[0].text.strip()
            if not result:
                raise ValueError("Empty response")
            return result
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise RuntimeError(f"Agent call failed after {retries} retries: {e}")
            time.sleep(2 ** attempt)

# ── Agent 1: Research specialist
def research_agent(topic: str) -> str:
    logging.info(f"[Research Agent] Starting on: {topic}")
    result = call_claude(
        system="You are a research assistant. Return exactly 5 concise bullet-point facts. No fluff.",
        user=f"Research this topic thoroughly: {topic}"
    )
    logging.info(f"[Research Agent] Done. ({len(result)} chars)")
    return result

# ── Agent 2: Blog writer specialist
def writer_agent(research_notes: str, topic: str) -> str:
    logging.info("[Writer Agent] Writing blog post…")
    result = call_claude(
        system="You are a skilled blog writer. Write a friendly, engaging 3-paragraph post (max 150 words).",
        user=f"Topic: {topic}\\n\\nResearch notes:\\n{research_notes}"
    )
    logging.info(f"[Writer Agent] Done. ({len(result)} chars)")
    return result

# ── Orchestrator: manually wires the two agents in sequence
# Adding a third agent (e.g. Reviewer) means rewriting this block
def run_pipeline(topic: str) -> str:
    logging.info(f"[Orchestrator] Pipeline started for: {topic}")

    # Step 1: Research
    facts = research_agent(topic)

    # Step 2: Write (manually pass output of step 1 to step 2)
    blog = writer_agent(facts, topic)

    logging.info("[Orchestrator] Pipeline complete.")
    return blog

topic  = "The impact of AI on software engineering jobs"
result = run_pipeline(topic)
print(result)

# ✗ Adding Agent 3 (Reviewer) means expanding the orchestrator manually
# ✗ No shared trace ID — impossible to correlate logs across agents
# ✗ No way to call agents in parallel without rewriting with asyncio
# ✗ If Writer fails, Research re-runs too — no checkpointing
#
# Compare: With LangChain — agents call each other as @tool functions.
#   @tool
#   def research_on_topic(topic: str) -> str:
#       "Research a topic and return key facts."
#       return research_executor.invoke({"input": topic})["output"]
#   orchestrator = AgentExecutor(agent=..., tools=[research_on_topic, write_blog_post], verbose=True)
#   orchestrator.invoke({"input": f"Write a blog post about: {topic}"})`,

  10: `# ── WITHOUT LangChain/LangGraph: Multi-Agent with State & Loops ─────────────
# Replicating LangGraph's StateGraph manually: you write every node,
# every edge, every loop guard, and all state transitions yourself.

import anthropic, time, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
client = anthropic.Anthropic()

def call_claude(system: str, user: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=512,
                system=system, messages=[{"role":"user","content":user}]
            )
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt == retries - 1: raise
            time.sleep(2 ** attempt)

# ── Shared state — plain dict (no type safety, no validation)
state = {
    "topic":     "How LangGraph improves multi-agent AI systems",
    "research":  "",
    "draft":     "",
    "feedback":  "",
    "final":     "",
    "revisions": 0
}

# ── Node 1: Manager (just logs; real apps add logic here)
def manager_node(state: dict) -> dict:
    logging.info("[Manager] Planning pipeline…")
    return state

# ── Node 2: Research
def research_node(state: dict) -> dict:
    logging.info("[Research] Gathering facts…")
    state["research"] = call_claude(
        "You are a researcher. Return 5 concise bullet-point facts.",
        f"Research: {state['topic']}"
    )
    return state

# ── Node 3: Writer (must handle feedback branch manually)
def writer_node(state: dict) -> dict:
    logging.info(f"[Writer] Writing draft (revision {state['revisions']})…")
    feedback_block = f"\\n\\nFeedback to address:\\n{state['feedback']}" if state["feedback"] else ""
    state["draft"] = call_claude(
        "You are a blog writer. Write a 3-paragraph post (max 120 words).",
        f"Research:\\n{state['research']}{feedback_block}"
    )
    return state

# ── Node 4: Reviewer
def reviewer_node(state: dict) -> dict:
    logging.info("[Reviewer] Reviewing draft…")
    review = call_claude(
        "Review this blog draft. Reply APPROVED or REVISE: <reason>.",
        f"Draft:\\n{state['draft']}"
    )
    logging.info(f"[Reviewer] Decision: {review[:60]}…")
    if review.strip().upper().startswith("APPROVED"):
        state["final"] = state["draft"]
        state["feedback"] = ""
    else:
        state["feedback"]  = review
        state["revisions"] += 1
    return state

# ── Conditional routing — you code every branch of the graph yourself
def should_revise(state: dict) -> bool:
    return not state["final"] and state["revisions"] < 2   # cap: 2 revision loops

# ── Manual graph execution (replaces LangGraph's StateGraph.compile().invoke())
state = manager_node(state)
state = research_node(state)
state = writer_node(state)

while should_revise(state):
    state = reviewer_node(state)
    if should_revise(state):   # still needs revision
        state = writer_node(state)

# Final approval pass if loop exited due to revision cap
if not state["final"]:
    state = reviewer_node(state)

print("\\n=== FINAL POST ===")
print(state["final"] or state["draft"])
print(f"Revisions: {state['revisions']}")

# ✗ 80 lines to replicate what LangGraph does declaratively in 15
# ✗ Graph topology is hidden in imperative code — no visualisation
# ✗ No checkpointing: if Reviewer crashes, entire pipeline re-runs
# ✗ Parallel branches (research + fact-check at the same time) need asyncio rewrite
# ✗ Human-in-the-loop pausing requires a complete architecture redesign
#
# Compare: With LangGraph — declare the graph, not the execution.
#   graph.add_node("manager",  manager_node)
#   graph.add_node("research", research_node)
#   graph.add_node("writer",   writer_node)
#   graph.add_node("reviewer", reviewer_node)
#   graph.add_conditional_edges("reviewer", should_revise, {"writer":"writer","end":END})
#   app = graph.compile()           # ← handles execution, checkpointing, tracing
#   result = app.invoke(initial_state)`,
};

function showLCCode(type) {
  const n   = _lcCurrentDemo;
  const cfg = LC_CONFIGS[n];
  const isLangChain = (type === 'with');
  const code = isLangChain
    ? (LC_CODE[n]       || '# Code not available')
    : (LC_CODE_PLAIN[n] || '# Code not available');

  document.getElementById('lc-code-modal-title').textContent = cfg ? cfg.title : 'Code';

  const badge = document.getElementById('lc-code-variant-badge');
  if (isLangChain) {
    badge.textContent        = 'With LangChain';
    badge.style.color        = '#a855f7';
    badge.style.background   = '#581c8722';
    badge.style.border       = '1px solid #a855f7';
    document.getElementById('lc-code-icon').style.stroke = '#a855f7';
    document.getElementById('lc-code-dot').style.background = '#a855f7';
    document.getElementById('lc-code-filename').textContent = 'langchain_demos/' + (_lcFilenames[n] || 'demo.py');
  } else {
    badge.textContent        = 'Without LangChain';
    badge.style.color        = '#f59e0b';
    badge.style.background   = '#78350f22';
    badge.style.border       = '1px solid #f59e0b';
    document.getElementById('lc-code-icon').style.stroke = '#f59e0b';
    document.getElementById('lc-code-dot').style.background = '#f59e0b';
    document.getElementById('lc-code-filename').textContent = 'Pure Python — Anthropic SDK only (no LangChain)';
  }

  document.getElementById('lc-code-content').textContent = code;
  document.getElementById('lc-copy-btn').textContent     = '  Copy';
  document.getElementById('lc-code-modal').style.display = 'flex';
}

function hideLCCode() {
  document.getElementById('lc-code-modal').style.display = 'none';
}

async function lcCopyCode() {
  const code = document.getElementById('lc-code-content').textContent;
  try {
    await navigator.clipboard.writeText(code);
    const btn = document.getElementById('lc-copy-btn');
    btn.textContent = '✓ Copied!';
    btn.style.color = 'var(--green)';
    btn.style.borderColor = 'var(--green)';
    setTimeout(() => {
      btn.textContent = '  Copy';
      btn.style.color = '';
      btn.style.borderColor = '';
    }, 2000);
  } catch(e) {
    alert('Copy failed — please select and copy manually.');
  }
}

async function lcMemoryClear() {
  const sid = _getLCSession();
  await fetch(`${BASE}/langchain/memory/${sid}`, {method:'DELETE'}).catch(()=>{});
  _lcSession = null;
  document.getElementById('lc-right').innerHTML = `<div class="card"><div style="color:var(--green);font-size:13px">Session cleared. Start a new conversation.</div></div>`;
}

async function advancedUploadFile(input) {
  const file = input.files[0];
  if (!file) return;
  const statusEl = document.getElementById('advanced-upload-status');
  statusEl.style.color = 'var(--muted)';
  statusEl.textContent = 'Uploading…';
  const fd = new FormData();
  fd.append('files', file);
  try {
    const r = await fetch(`${BASE}/upload`, {method:'POST', body: fd});
    const d = await r.json();
    if (r.ok) {
      statusEl.style.color = 'var(--green)';
      statusEl.textContent = `✓ ${file.name} — ${d.total_chunks} chunks indexed`;
      const sharedStatus = document.getElementById('uploadStatus');
      if (sharedStatus) { sharedStatus.textContent = `✓ ${file.name} — ${d.total_chunks} chunks`; sharedStatus.className = 'upload-status ok'; }
    } else {
      statusEl.style.color = 'var(--red)';
      statusEl.textContent = d.detail || 'Upload failed';
    }
  } catch {
    statusEl.style.color = 'var(--red)';
    statusEl.textContent = 'Upload error';
  }
}

async function runAdvancedRAG() {
  const query = document.getElementById('advanced-query').value.trim();
  if (!query) return;
  const resultEl = document.getElementById('advanced-result');
  const answerEl = document.getElementById('advanced-answer');
  const stepsEl  = document.getElementById('advanced-steps');
  resultEl.style.display = 'block';
  answerEl.textContent = 'Running…';
  stepsEl.innerHTML = '';
  try {
    const r = await fetch(`${BASE}/rag/advanced`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query})
    });
    const d = await r.json();
    answerEl.textContent = d.answer || d.detail || JSON.stringify(d);
    _renderGuardrail('advanced-guardrail', d.guardrail);
    if (d.steps && d.steps.length) {
      stepsEl.innerHTML = `<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Pipeline Steps</div>` +
        d.steps.map(s => `<div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:12px;color:var(--muted);margin-bottom:6px"><span style="color:var(--amber);font-weight:700">${s.step}</span>  ${s.detail}</div>`).join('');
    }
    if (d.docs && d.docs.length) {
      stepsEl.innerHTML += `<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:12px;margin-bottom:8px">Retrieved Context Chunks (${d.docs.length})</div>` +
        d.docs.map((c, i) => `<div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;color:var(--muted);line-height:1.7;margin-bottom:6px"><span style="color:var(--blue);font-weight:700">Chunk ${i+1}</span><span style="color:var(--muted);font-size:11px;margin-left:8px">score: ${c.score ?? ''}</span><br>${c.text ?? c}</div>`).join('');
    }
  } catch (e) {
    answerEl.textContent = 'Error: ' + e.message;
  }
}

async function naiveUploadFile(input) {
  const file = input.files[0];
  if (!file) return;
  const statusEl = document.getElementById('naive-upload-status');
  statusEl.style.color = 'var(--muted)';
  statusEl.textContent = 'Uploading…';
  const fd = new FormData();
  fd.append('files', file);
  try {
    const r = await fetch(`${BASE}/upload`, {method:'POST', body: fd});
    const d = await r.json();
    if (r.ok) {
      statusEl.style.color = 'var(--green)';
      statusEl.textContent = `✓ ${file.name} — ${d.total_chunks} chunks indexed`;
      const sharedStatus = document.getElementById('uploadStatus');
      if (sharedStatus) { sharedStatus.textContent = `✓ ${file.name} — ${d.total_chunks} chunks`; sharedStatus.className = 'upload-status ok'; }
    } else {
      statusEl.style.color = 'var(--red)';
      statusEl.textContent = d.detail || 'Upload failed';
    }
  } catch {
    statusEl.style.color = 'var(--red)';
    statusEl.textContent = 'Upload error';
  }
}

async function runNaiveRAG() {
  const query = document.getElementById('naive-query').value.trim();
  if (!query) return;
  const resultEl = document.getElementById('naive-result');
  const answerEl = document.getElementById('naive-answer');
  const chunksEl = document.getElementById('naive-chunks');
  resultEl.style.display = 'block';
  answerEl.textContent = 'Running…';
  chunksEl.innerHTML = '';
  try {
    const r = await fetch(`${BASE}/rag/naive`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query})
    });
    const d = await r.json();
    answerEl.textContent = d.answer || d.detail || JSON.stringify(d);
    _renderGuardrail('naive-guardrail', d.guardrail);
    if (d.docs && d.docs.length) {
      chunksEl.innerHTML = `<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Retrieved Context Chunks (${d.docs.length})</div>` +
        d.docs.map((c, i) => `<div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;color:var(--muted);line-height:1.7;margin-bottom:6px"><span style="color:var(--blue);font-weight:700">Chunk ${i+1}</span><span style="color:var(--muted);font-size:11px;margin-left:8px">score: ${c.score ?? ''}</span><br>${c.text ?? c}</div>`).join('');
    }
  } catch (e) {
    answerEl.textContent = 'Error: ' + e.message;
  }
}

// ── Upload/run helpers for Agentic, Hybrid, Graph pages ──────────────────────
async function _sharedUpload(input, statusId) {
  if (!input.files.length) return;
  const s = document.getElementById(statusId);
  s.style.color = 'var(--muted)'; s.textContent = 'Uploading…';
  const fd = new FormData();
  for (const f of input.files) fd.append('files', f);
  try {
    const r = await fetch(`${BASE}/upload`, {method:'POST', body: fd});
    const d = await r.json();
    if (r.ok) {
      s.style.color = 'var(--green)';
      const label = d.files.length === 1
        ? `✓ ${d.files[0].filename} — ${d.total_chunks} chunks indexed`
        : `✓ ${d.files.length} files — ${d.total_chunks} chunks indexed`;
      s.textContent = label;
    } else { s.style.color='var(--red)'; s.textContent=d.detail||'Upload failed'; }
  } catch { s.style.color='var(--red)'; s.textContent='Upload error'; }
}
async function _sharedRun(queryId, resultId, answerId, stepsId, endpoint, guardrailId) {
  const query = document.getElementById(queryId).value.trim(); if (!query) return;
  const resultEl = document.getElementById(resultId);
  const answerEl = document.getElementById(answerId);
  resultEl.style.display='block'; answerEl.textContent='Running…';
  if (stepsId) document.getElementById(stepsId).innerHTML='';
  try {
    const r = await fetch(`${BASE}${endpoint}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})});
    const d = await r.json();
    answerEl.textContent = d.answer||d.detail||JSON.stringify(d);
    if (guardrailId) _renderGuardrail(guardrailId, d.guardrail);
    if (stepsId && d.steps?.length) {
      document.getElementById(stepsId).innerHTML = '<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">Pipeline Steps</div>' +
        d.steps.map(s=>`<div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px 12px;font-size:12px;color:var(--muted);margin-bottom:6px"><span style="color:var(--amber);font-weight:700">${s.step}</span>  ${s.detail}</div>`).join('');
    }
    if (stepsId && d.docs?.length) {
      document.getElementById(stepsId).innerHTML += '<div style="font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:12px;margin-bottom:8px">Retrieved Context Chunks (' + d.docs.length + ')</div>' +
        d.docs.map((c,i)=>`<div style="background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:10px 12px;font-size:12px;color:var(--muted);line-height:1.7;margin-bottom:6px"><span style="color:var(--blue);font-weight:700">Chunk ${i+1}</span><span style="color:var(--muted);font-size:11px;margin-left:8px">score: ${c.score??''}</span><br>${c.text??c}</div>`).join('');
    }
  } catch(e) { answerEl.textContent='Error: '+e.message; }
}

function agenticUploadFile(i)  { _sharedUpload(i,'agentic-upload-status'); }
function hybridUploadFile(i)   { _sharedUpload(i,'hybrid-upload-status'); }
function graphUploadFile(i)    { _sharedUpload(i,'graph-upload-status'); }
function runAgenticRAG()       { _sharedRun('agentic-query','agentic-result','agentic-answer','agentic-steps','/rag/agentic','agentic-guardrail'); }
function runHybridRAG()        { _sharedRun('hybrid-query','hybrid-result','hybrid-answer','hybrid-chunks','/rag/hybrid','hybrid-guardrail'); }
function runGraphRAG()         { _sharedRun('graph-query','graph-result','graph-answer','graph-nodes','/rag/graph','graph-guardrail'); }

document.addEventListener('DOMContentLoaded', () => navigate('home', 'toc-home'));
