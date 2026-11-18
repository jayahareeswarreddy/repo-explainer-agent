"""
Repo Explainer Agent
--------------------
Paste a public GitHub repo URL, then ask questions about the code. A Groq LLM
drives a plan-act-observe agent loop over three tools (search_code, read_file,
list_files) and answers with file/line citations.

Lightweight by design: embeddings are called from the Jina API (not run
in-process), with in-memory cosine search, so the web app's memory footprint
stays tiny and fits a free Render instance. No GPU, no torch, no local model.
"""

import io
import os
import re
import json
import time
import zipfile
from collections import defaultdict

import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq

# ----------------------------- config -------------------------------------
GROQ_MODEL = "openai/gpt-oss-20b"
JINA_API_KEY = os.environ.get("JINA_API_KEY") or os.environ.get("JINA_API_TOKEN", "")
JINA_MODEL = "jina-embeddings-v3"
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rb", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".kt", ".swift", ".scala",
    ".md", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".sql",
}
SKIP_DIRS = {"node_modules", ".git", "dist", "build", "vendor", ".next",
             "__pycache__", "venv", ".venv", "site-packages"}
MAX_FILES = 400
MAX_FILE_BYTES = 100_000
MAX_CHUNKS = 1500
CHUNK_LINES = 60
CHUNK_OVERLAP = 10

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GH_HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    GH_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

app = FastAPI(title="Repo Explainer Agent")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def embed(texts):
    """Embed a list of strings via the Jina API. Returns an (n, dim) float32 array.
    Batches are small and paced, with 429 back-off, to respect the free tier's
    per-minute token limit on large repos."""
    vectors = []
    for i in range(0, len(texts), 48):
        batch = texts[i:i + 48]
        for attempt in range(6):
            r = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {JINA_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": JINA_MODEL, "input": batch},
                timeout=60,
            )
            if r.status_code == 200:
                break
            if r.status_code == 429:
                time.sleep(5)  # per-minute token limit — back off and retry
                continue
            raise ValueError(f"Jina embeddings error {r.status_code}: {r.text[:300]}")
        else:
            raise ValueError("Jina rate limit hit repeatedly — wait a minute, or try a smaller repo.")
        items = sorted(r.json()["data"], key=lambda d: d["index"])
        vectors.extend(d["embedding"] for d in items)
        time.sleep(0.3)  # pace requests to stay under the per-minute token cap
    return np.array(vectors, dtype=np.float32)

# in-memory index for the currently loaded repo (single-session demo)
INDEX = {"repo": None, "branch": "main", "files": {}, "chunks": [], "matrix": None, "norms": None}
CONVO = []  # recent (question, answer) pairs, for follow-up questions

# ----- simple in-memory per-IP rate limiting (protects free Groq/Jina quota) --
_HITS = defaultdict(list)


def rate_ok(ip, bucket, limit, window=600):
    now = time.time()
    q = _HITS[(ip, bucket)]
    while q and q[0] < now - window:
        q.pop(0)
    if len(q) >= limit:
        return False
    q.append(now)
    return True

# ----------------------------- ingestion ----------------------------------
def parse_repo(url: str):
    m = re.search(r"github\.com[/:]([^/]+)/([^/#?]+?)(?:\.git)?(?:[/#?].*)?$", url.strip())
    if not m:
        raise ValueError("Not a valid GitHub repo URL.")
    return m.group(1), m.group(2)


def fetch_repo_files(owner: str, repo: str):
    meta = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                        headers=GH_HEADERS, timeout=30)
    if meta.status_code == 403 and meta.headers.get("X-RateLimit-Remaining") == "0":
        raise ValueError("GitHub API rate limit reached (unauthenticated = 60/hr). "
                         "Set a GITHUB_TOKEN env var to raise it to 5000/hr.")
    if meta.status_code == 404:
        raise ValueError("Repo not found — check it's public and the URL is correct.")
    if meta.status_code != 200:
        raise ValueError(f"GitHub API error ({meta.status_code}).")
    branch = meta.json().get("default_branch", "main")
    zurl = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    z = requests.get(zurl, headers=GH_HEADERS, timeout=60)
    z.raise_for_status()

    files = {}
    with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            rel = name.split("/", 1)[1] if "/" in name else name
            parts = set(rel.split("/"))
            if parts & SKIP_DIRS:
                continue
            ext = os.path.splitext(rel)[1].lower()
            if ext not in CODE_EXTS:
                continue
            info = zf.getinfo(name)
            if info.file_size > MAX_FILE_BYTES:
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            files[rel] = text
            if len(files) >= MAX_FILES:
                break
    return files, branch


def chunk_file(path: str, text: str):
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        seg = lines[i:i + CHUNK_LINES]
        out.append({
            "path": path,
            "start": i + 1,
            "end": min(i + CHUNK_LINES, len(lines)),
            "text": "\n".join(seg),
        })
        if i + CHUNK_LINES >= len(lines):
            break
        i += CHUNK_LINES - CHUNK_OVERLAP
    return out


def build_index(repo_url: str):
    owner, repo = parse_repo(repo_url)
    files, branch = fetch_repo_files(owner, repo)
    if not files:
        raise ValueError("No readable code files found in this repo.")

    chunks = []
    for path, text in files.items():
        chunks.extend(chunk_file(path, text))
        if len(chunks) >= MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
            break

    vectors = embed([c["text"] for c in chunks])
    norms = np.linalg.norm(vectors, axis=1) + 1e-8

    INDEX.update(repo=f"{owner}/{repo}", branch=branch, files=files, chunks=chunks,
                 matrix=vectors, norms=norms)
    CONVO.clear()  # new repo => fresh conversation
    return {"repo": f"{owner}/{repo}", "branch": branch,
            "num_files": len(files), "num_chunks": len(chunks)}

# ----------------------------- tools --------------------------------------
def tool_search_code(query: str, k: int = 6):
    if INDEX["matrix"] is None:
        return "No repo loaded."
    qv = embed([query])[0]
    sims = (INDEX["matrix"] @ qv) / (INDEX["norms"] * (np.linalg.norm(qv) + 1e-8))
    top = sims.argsort()[-k:][::-1]
    results = []
    for i in top:
        c = INDEX["chunks"][int(i)]
        results.append(f"{c['path']}:{c['start']}-{c['end']}\n{c['text']}")
    return "\n\n---\n\n".join(results)


def tool_read_file(path: str, max_lines: int = 200):
    text = INDEX["files"].get(path)
    if text is None:
        # tolerate near-misses
        for p in INDEX["files"]:
            if p.endswith(path):
                text = INDEX["files"][p]
                path = p
                break
    if text is None:
        return f"File not found: {path}"
    lines = text.split("\n")[:max_lines]
    return f"{path}\n" + "\n".join(lines)


def tool_list_files():
    return "\n".join(sorted(INDEX["files"].keys())) or "No files."


TOOLS = [
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Semantic search over the repo. Use for 'how/where/why' questions. Returns code chunks with path:line ranges.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Return the contents of a specific file path in the repo.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List all indexed file paths in the repo.",
        "parameters": {"type": "object", "properties": {}}}},
]

DISPATCH = {
    "search_code": lambda a: tool_search_code(a.get("query", "")),
    "read_file": lambda a: tool_read_file(a.get("path", "")),
    "list_files": lambda a: tool_list_files(),
}

SYSTEM = (
    "You are a code assistant answering questions about a specific GitHub repo. "
    "Use the tools to gather evidence before answering: search_code for conceptual "
    "questions, read_file to inspect a file, list_files to see the structure. "
    "Always ground your answer in retrieved code and cite file paths with line "
    "ranges like `path/to/file.py:40-70`. If the answer isn't in the repo, say so "
    "plainly instead of guessing."
)

# ----------------------------- agent loop ---------------------------------
def _complete_with_tools(messages):
    return groq_client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        temperature=0.2, extra_body={"reasoning_effort": "low"},
    )


def _synthesize(question, context, history=None):
    """Final answer from gathered code, as a clean tool-free call."""
    excerpts = "\n\n".join(context)[:14000]
    msgs = [
        {"role": "system", "content":
         "You are a code assistant. Answer the question using ONLY the code excerpts "
         "provided. Cite sources as `path:line-range`. If the excerpts don't contain "
         "the answer, say so plainly."},
    ]
    if history:
        prior = "\n".join(f"Q: {q}\nA: {a[:400]}" for q, a in history)
        msgs.append({"role": "system",
                     "content": f"Earlier in this conversation (for follow-up context):\n{prior}"})
    msgs.append({"role": "user", "content": f"Question: {question}\n\nCode excerpts:\n{excerpts}"})
    t = time.time()
    print(f"[synthesize] model={GROQ_MODEL} prompt_chars={len(excerpts)} -> calling Groq...", flush=True)
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=msgs, temperature=0.2,
            extra_body={"reasoning_effort": "low"})
        print(f"[synthesize] Groq returned in {time.time()-t:.1f}s  usage={getattr(resp, 'usage', None)}", flush=True)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[synthesize] ERROR after {time.time()-t:.1f}s: {e}", flush=True)
        # last-resort fallback: hand back the most relevant excerpt
        return "Here are the most relevant code sections:\n\n" + (context[0] if context else "(none)")


def run_agent(question: str, deep: bool = False):
    """Default: fast single-pass (retrieve top code, one LLM call to answer).
    deep=True runs the full tool-calling agent loop (slower, for hard questions)."""
    t = time.time()
    # For follow-ups, fold the previous question into the search so vague refs
    # like "and how is that tested?" still retrieve the right code.
    search_q = f"{CONVO[-1][0]} {question}" if CONVO else question
    seed = tool_search_code(search_q, k=5)
    print(f"[agent] deep={deep} seed_retrieval={time.time()-t:.2f}s seed_chars={len(seed)}", flush=True)
    cited = set(re.findall(r"[\w./-]+:\d+-\d+", seed))

    if not deep:
        answer = _synthesize(question, [seed], history=CONVO[-2:])
        CONVO.append((question, answer))
        print(f"[agent] total /ask time {time.time()-t:.1f}s", flush=True)
        return {"answer": answer, "citations": sorted(cited)}

    # --- deep mode: let the model call tools to gather more context ---
    context = [seed]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Repo: {INDEX['repo']}\n\nQuestion: {question}\n\n"
         f"Relevant code from an initial search:\n{seed}\n\n"
         "Answer using this code. Call a tool only if you genuinely need more."},
    ]
    for _ in range(3):
        msg = _complete_with_tools(messages).choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "citations": sorted(cited)}
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = str(DISPATCH.get(tc.function.name, lambda a: "Unknown tool")(args))
            context.append(result)
            for m in re.findall(r"[\w./-]+:\d+-\d+", result):
                cited.add(m)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:4000]})
    return {"answer": _synthesize(question, context), "citations": sorted(cited)}

# ----------------------------- API ----------------------------------------
class IngestReq(BaseModel):
    repo_url: str


class AskReq(BaseModel):
    question: str
    deep: bool = False


def _client_ip(request: Request):
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@app.post("/ingest")
def ingest(req: IngestReq, request: Request):
    if not rate_ok(_client_ip(request), "ingest", limit=20):
        return JSONResponse({"error": "Rate limit: too many repo loads. Try again in a few minutes."},
                            status_code=429)
    try:
        return build_index(req.repo_url)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ask")
def ask(req: AskReq, request: Request):
    if INDEX["matrix"] is None:
        return JSONResponse({"error": "Load a repo first."}, status_code=400)
    if not rate_ok(_client_ip(request), "ask", limit=40):
        return JSONResponse({"error": "Rate limit: too many questions. Try again in a few minutes."},
                            status_code=429)
    try:
        return run_agent(req.question, deep=req.deep)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Repo Explainer Agent</title>
<meta name="description" content="Load any public GitHub repo and ask questions about its code, answered with file-line citations.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>
 :root{
   --bg:#0F172A; --card:#1B2336; --muted:#272F42; --border:#334155;
   --fg:#F8FAFC; --mfg:#94A3B8; --accent:#22C55E; --accent-fg:#0F172A;
   --ring:#38BDF8; --danger:#F87171;
   --sans:'IBM Plex Sans',system-ui,sans-serif; --mono:'JetBrains Mono',ui-monospace,monospace;
   color-scheme:dark;
 }
 *{box-sizing:border-box}
 body{font-family:var(--sans);max-width:860px;margin:0 auto;padding:2rem 1rem 4rem;background:var(--bg);color:var(--fg);line-height:1.55}
 a{color:var(--ring)}
 .brand{display:flex;align-items:center;gap:.6rem;margin-bottom:.2rem}
 .brand svg{width:30px;height:30px;flex:none}
 h1{font-family:var(--mono);font-size:1.45rem;font-weight:700;margin:0;letter-spacing:-.5px}
 .tag{color:var(--mfg);margin:.1rem 0 1rem;font-size:.95rem}
 input,textarea,button{font:inherit;color:var(--fg)}
 input,textarea{width:100%;background:var(--muted);border:1px solid var(--border);border-radius:10px;padding:.7rem .8rem}
 input:focus,textarea:focus{outline:2px solid var(--ring);outline-offset:1px;border-color:var(--ring)}
 .btn{background:var(--accent);color:var(--accent-fg);border:0;border-radius:10px;padding:.65rem 1.1rem;font-weight:600;cursor:pointer;transition:filter .18s,opacity .18s}
 .btn:hover{filter:brightness(1.08)} .btn:disabled{opacity:.5;cursor:not-allowed}
 .btn.ghost{background:transparent;color:var(--fg);border:1px solid var(--border)}
 .btn.ghost:hover{background:var(--muted);filter:none}
 .row{display:flex;gap:.55rem;flex-wrap:wrap;align-items:center;margin-top:.5rem}
 .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1rem}
 .muted{color:var(--mfg);font-size:.9rem}
 .chip{background:var(--muted);border:1px solid var(--border);color:var(--fg);border-radius:999px;padding:.3rem .7rem;font-size:.82rem;cursor:pointer;transition:background .18s,border-color .18s}
 .chip:hover{border-color:var(--accent)}
 .chips{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;margin-top:.6rem}
 .status{margin-top:.6rem;font-size:.9rem;min-height:1.2em}
 .status.err{color:var(--danger)} .status.ok{color:var(--accent)}
 .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--mfg);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:.4rem}
 @keyframes spin{to{transform:rotate(360deg)}}
 #tools{display:none;gap:.5rem;flex-wrap:wrap;align-items:center;margin:1.2rem 0 .2rem}
 .toggle{display:inline-flex;align-items:center;gap:.4rem;color:var(--mfg);font-size:.88rem;cursor:pointer;user-select:none}
 .toggle input{width:auto}
 .repo-sep{margin:1.6rem 0 .2rem;padding:.35rem .7rem;border-left:3px solid var(--accent);color:var(--mfg);font-size:.88rem;font-family:var(--mono)}
 .cell{margin-top:1.3rem;padding-top:1.1rem;border-top:1px solid var(--border)}
 .cell:first-of-type{border-top:0}
 .cell .ask{margin-top:.5rem}
 .cits{display:flex;gap:.35rem;flex-wrap:wrap;margin:.6rem 0 .3rem}
 .cit{display:inline-block;background:#1e3a2a;border:1px solid var(--accent);color:#bbf7d0;border-radius:6px;padding:.12rem .45rem;font-size:.8rem;font-family:var(--mono);text-decoration:none;transition:filter .15s}
 a.cit:hover{filter:brightness(1.2)}
 .ans{overflow-x:auto;margin-top:.3rem}
 .ans.loading{color:var(--mfg)}
 .ans-foot{margin-top:.4rem}
 .copy{background:transparent;border:1px solid var(--border);color:var(--mfg);border-radius:6px;padding:.2rem .5rem;font-size:.78rem;cursor:pointer}
 .copy:hover{border-color:var(--ring);color:var(--fg)}
 .md table{border-collapse:collapse;margin:.5rem 0;font-size:.9rem;display:block;overflow-x:auto}
 .md th,.md td{border:1px solid var(--border);padding:.3rem .55rem;text-align:left}
 .md th{background:var(--muted)}
 .md code{background:var(--muted);padding:.1rem .35rem;border-radius:4px;font-size:.88em;font-family:var(--mono)}
 .md pre{background:#0b1220;padding:.8rem;border-radius:8px;overflow-x:auto;border:1px solid var(--border)}
 .md pre code{background:none;padding:0}
 .md h1,.md h2,.md h3{font-family:var(--mono);font-size:1.02rem;margin:.8rem 0 .3rem}
 .md p{margin:.45rem 0} .md ul,.md ol{margin:.3rem 0 .3rem 1.25rem}
 .mermaid{background:#0b1220;border:1px solid var(--border);border-radius:8px;padding:.8rem;margin:.6rem 0;text-align:center}
 details.about{margin:.3rem 0 1rem;border:1px solid var(--border);border-radius:12px;background:var(--card)}
 details.about summary{cursor:pointer;padding:.6rem .9rem;font-weight:600;list-style:none}
 details.about summary::-webkit-details-marker{display:none}
 details.about .about-body{padding:0 .9rem .85rem;color:var(--mfg);font-size:.92rem}
 #overlay{position:fixed;inset:0;background:#0009;opacity:0;pointer-events:none;transition:opacity .2s;z-index:40}
 #overlay.open{opacity:1;pointer-events:auto}
 #drawer{position:fixed;top:0;right:0;height:100%;width:480px;max-width:92vw;background:var(--card);border-left:1px solid var(--border);transform:translateX(100%);transition:transform .25s ease;z-index:50;display:flex;flex-direction:column}
 #drawer.open{transform:translateX(0)}
 .drawer-head{display:flex;align-items:center;justify-content:space-between;padding:.8rem 1rem;border-bottom:1px solid var(--border);font-family:var(--mono);font-weight:600}
 .drawer-body{padding:1rem;overflow:auto}
 .x{background:transparent;border:0;color:var(--mfg);font-size:1.5rem;line-height:1;cursor:pointer;padding:0 .3rem}
 .x:hover{color:var(--fg)}
 @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head>
<body>
 <div class="brand">
   <svg viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
   <h1>Repo Explainer Agent</h1>
 </div>
 <p class="tag">Load a public GitHub repo, then ask questions about its code — answered with file:line citations.</p>

 <details class="about"><summary>About this project &amp; how it works ▾</summary>
   <div class="about-body">
     <b style="color:var(--fg)">Repo Explainer Agent</b> loads any public GitHub repository and answers natural-language
     questions about its code. It retrieves the most relevant code with semantic search and a Groq LLM answers
     with <b style="color:var(--fg)">file:line citations</b> (click them to jump to GitHub), so you can verify
     every claim. Ask multiple questions and follow-ups once a repo is loaded.<br><br>
     <b style="color:var(--fg)">Loading a repo may take a little time</b> — it downloads the repo, splits it into
     chunks and embeds them (bigger repos take longer). The first request after the demo has been idle also needs
     ~30–50s to wake up (free hosting).<br><br>
     Built with FastAPI · Groq (gpt-oss-20b) · Jina embeddings · in-memory vector search.
   </div>
 </details>

 <div class="card">
   <div class="row">
     <input id="repo" placeholder="https://github.com/owner/repo" style="flex:1 1 260px" onkeydown="if(event.key==='Enter')ingest()">
     <button class="btn" id="load" onclick="ingest()">Load repo</button>
   </div>
   <div class="status" id="status" aria-live="polite"></div>
 </div>

 <div id="tools">
   <button class="btn ghost" onclick="preset('overview')">⌗ Architecture overview</button>
   <button class="btn ghost" onclick="preset('diagram')">◇ Diagram</button>
   <label class="toggle"><input type="checkbox" id="deep"> Thorough mode (slower, uses tools)</label>
 </div>
 <div id="cells"></div>

 <div id="overlay" onclick="closeDrawer()"></div>
 <aside id="drawer" aria-hidden="true" aria-label="Result panel">
   <div class="drawer-head"><span id="drawerTitle"></span><button class="x" onclick="closeDrawer()" aria-label="Close panel">&times;</button></div>
   <div class="drawer-body md" id="drawerBody"></div>
 </aside>

<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.1/mermaid.min.js"></script>
<script>
let REPO=null, BRANCH='main', loadTimer=null;
if(window.mermaid) mermaid.initialize({startOnLoad:false, theme:'dark'});
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function useExample(url){ document.getElementById('repo').value=url; ingest(); }

async function ingest(){
  const repoUrl=document.getElementById('repo').value.trim(); if(!repoUrl) return;
  const b=document.getElementById('load'), st=document.getElementById('status');
  b.disabled=true; st.className='status';
  const t0=Date.now();
  const tick=()=>{ st.innerHTML=`<span class="spin"></span> Loading <b>${esc(repoUrl.replace('https://github.com/',''))}</b> — downloading &amp; embedding… ${((Date.now()-t0)/1000|0)}s`; };
  tick(); clearInterval(loadTimer); loadTimer=setInterval(tick,1000);
  try{
    const r=await fetch('/ingest',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({repo_url:repoUrl})});
    const d=await r.json();
    clearInterval(loadTimer); b.disabled=false;
    if(d.error){ st.className='status err'; st.textContent='⚠ '+d.error; return; }
    REPO=d.repo; BRANCH=d.branch||'main';
    st.className='status ok'; st.textContent=`✓ ${d.repo} — ${d.num_files} files, ${d.num_chunks} chunks indexed`;
    document.getElementById('tools').style.display='flex';
    const cells=document.getElementById('cells');
    const sep=document.createElement('div'); sep.className='repo-sep'; sep.textContent='📁 '+d.repo;
    cells.appendChild(sep);   // keep earlier Q&A; just mark the new repo
    addCell();
  }catch(e){ clearInterval(loadTimer); b.disabled=false; st.className='status err'; st.textContent='⚠ '+e; }
}

function ghLink(ref){
  const m=ref.match(/^(.*):(\\d+)-(\\d+)$/);
  if(m && REPO) return `https://github.com/${REPO}/blob/${BRANCH}/${m[1]}#L${m[2]}-L${m[3]}`;
  return null;
}
function renderCits(cits){
  return (cits||[]).map(c=>{ const u=ghLink(c);
    return u ? `<a class="cit" href="${u}" target="_blank" rel="noopener">${esc(c)}</a>`
             : `<span class="cit">${esc(c)}</span>`; }).join('');
}
function renderMermaid(root){
  if(!window.mermaid) return;
  root.querySelectorAll('code.language-mermaid').forEach(code=>{
    const div=document.createElement('div'); div.className='mermaid'; div.textContent=code.textContent;
    (code.closest('pre')||code).replaceWith(div);
  });
  const nodes=root.querySelectorAll('.mermaid:not([data-processed])');
  if(nodes.length){ try{ mermaid.run({nodes}); }catch(e){} }
}

async function runQuery(cell, question, slow){
  const ans=cell.querySelector('.ans'), cits=cell.querySelector('.cits');
  ans.className='ans md loading';
  ans.innerHTML=`<span class="spin"></span>${slow?'Working on it — this may take a bit…':'Thinking…'}`;
  cits.innerHTML='';
  const deep=document.getElementById('deep') && document.getElementById('deep').checked;
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, deep})});
    const d=await r.json();
    ans.className='ans md';
    if(d.error){ ans.textContent='⚠ '+d.error; return false; }
    ans.innerHTML=window.marked?marked.parse(d.answer):esc(d.answer);
    renderMermaid(ans);
    cits.innerHTML=renderCits(d.citations);
    const foot=document.createElement('div'); foot.className='ans-foot';
    const cp=document.createElement('button'); cp.className='copy'; cp.textContent='Copy';
    cp.onclick=()=>{ navigator.clipboard.writeText(d.answer); cp.textContent='Copied ✓'; setTimeout(()=>cp.textContent='Copy',1500); };
    foot.appendChild(cp); ans.after(foot);
    return true;
  }catch(e){ ans.className='ans md'; ans.textContent='⚠ '+e; return false; }
}

function appendCell(interactive){
  const cell=document.createElement('div'); cell.className='cell';
  cell.innerHTML=`<textarea class="q" rows="2" placeholder="Ask about the repo…  (Ctrl/⌘+Enter to send)"></textarea>`
    +`<div><button class="btn ask">Ask</button></div><div class="cits"></div><div class="ans md"></div>`;
  document.getElementById('cells').appendChild(cell);
  const ta=cell.querySelector('.q'), btn=cell.querySelector('.ask');
  if(interactive){
    btn.onclick=async()=>{ const q=ta.value.trim(); if(!q) return; ta.disabled=true; btn.disabled=true;
      const ok=await runQuery(cell,q,false); if(ok) appendCell(true); };
    ta.addEventListener('keydown',e=>{ if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){ e.preventDefault(); btn.click(); } });
    ta.focus();
  }
  cell.scrollIntoView({behavior:'smooth',block:'center'});
  return cell;
}
function addCell(){ return appendCell(true); }

function openDrawer(title){
  document.getElementById('drawerTitle').textContent=title;
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawer').setAttribute('aria-hidden','false');
  document.getElementById('overlay').classList.add('open');
}
function closeDrawer(){
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').setAttribute('aria-hidden','true');
  document.getElementById('overlay').classList.remove('open');
}
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeDrawer(); });

async function preset(kind){
  const q = kind==='overview'
    ? "Give a concise architecture overview of this repository: what it does, its main components, and how data and control flow between them. Use short markdown headings and bullet points, and cite files as path:line-range."
    : "Draw a Mermaid flowchart of the main components/modules of this repository and how they connect. Respond with ONLY a ```mermaid code block (use 'flowchart TD'), no prose.";
  openDrawer(kind==='overview' ? 'Architecture overview' : 'Component diagram');
  const body=document.getElementById('drawerBody'); body.className='drawer-body md';
  body.innerHTML=`<span class="spin"></span> Working on it — this may take a bit…`;
  const deep=document.getElementById('deep') && document.getElementById('deep').checked;
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q, deep})});
    const d=await r.json();
    if(d.error){ body.textContent='⚠ '+d.error; return; }
    body.innerHTML=window.marked?marked.parse(d.answer):esc(d.answer);
    renderMermaid(body);
    if(d.citations && d.citations.length){
      const c=document.createElement('div'); c.className='cits'; c.innerHTML=renderCits(d.citations); body.appendChild(c);
    }
  }catch(e){ body.textContent='⚠ '+e; }
}
</script>
</body></html>"""
