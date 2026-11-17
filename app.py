"""
Repo Explainer Agent
--------------------
Paste a public GitHub repo URL, then ask questions about the code. A Groq LLM
drives a plan-act-observe agent loop over three tools (search_code, read_file,
list_files) and answers with file/line citations.

Lightweight by design (ONNX embeddings via fastembed, in-memory cosine search)
so it fits a free Render instance. No GPU, no torch.
"""

import io
import os
import re
import json
import zipfile

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from groq import Groq

# ----------------------------- config -------------------------------------
GROQ_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
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
embedder = TextEmbedding(model_name=EMBED_MODEL)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# in-memory index for the currently loaded repo (single-session demo)
INDEX = {"repo": None, "files": {}, "chunks": [], "matrix": None, "norms": None}

# ----------------------------- ingestion ----------------------------------
def parse_repo(url: str):
    m = re.search(r"github\.com[/:]([^/]+)/([^/#?]+?)(?:\.git)?(?:[/#?].*)?$", url.strip())
    if not m:
        raise ValueError("Not a valid GitHub repo URL.")
    return m.group(1), m.group(2)


def fetch_repo_files(owner: str, repo: str):
    meta = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                        headers=GH_HEADERS, timeout=30)
    if meta.status_code != 200:
        raise ValueError(f"GitHub API error ({meta.status_code}). Repo private or not found?")
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
    return files


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
    files = fetch_repo_files(owner, repo)
    if not files:
        raise ValueError("No readable code files found in this repo.")

    chunks = []
    for path, text in files.items():
        chunks.extend(chunk_file(path, text))
        if len(chunks) >= MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS]
            break

    vectors = np.array(list(embedder.embed([c["text"] for c in chunks])), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1) + 1e-8

    INDEX.update(repo=f"{owner}/{repo}", files=files, chunks=chunks,
                 matrix=vectors, norms=norms)
    return {"repo": f"{owner}/{repo}", "num_files": len(files), "num_chunks": len(chunks)}

# ----------------------------- tools --------------------------------------
def tool_search_code(query: str, k: int = 6):
    if INDEX["matrix"] is None:
        return "No repo loaded."
    qv = np.array(next(embedder.embed([query])), dtype=np.float32)
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


def _synthesize(question, context):
    """Final answer from gathered code, as a clean tool-free call."""
    excerpts = "\n\n".join(context)[:14000]
    msgs = [
        {"role": "system", "content":
         "You are a code assistant. Answer the question using ONLY the code excerpts "
         "provided. Cite sources as `path:line-range`. If the excerpts don't contain "
         "the answer, say so plainly."},
        {"role": "user", "content": f"Question: {question}\n\nCode excerpts:\n{excerpts}"},
    ]
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL, messages=msgs, temperature=0.2,
            extra_body={"reasoning_effort": "low"})
        return resp.choices[0].message.content or ""
    except Exception:
        # last-resort fallback: hand back the most relevant excerpt
        return "Here are the most relevant code sections:\n\n" + (context[0] if context else "(none)")


def run_agent(question: str, max_steps: int = 2):
    # Pre-retrieve relevant code (instant, no LLM) so the model can usually answer
    # in a single call instead of discovering files over several round-trips.
    seed = tool_search_code(question, k=6)
    cited = {m.split(":")[0] for m in re.findall(r"[\w./-]+:\d+-\d+", seed)}
    context = [seed]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
         f"Repo: {INDEX['repo']}\n\nQuestion: {question}\n\n"
         f"Relevant code from an initial search:\n{seed}\n\n"
         "Answer using this code. Only call a tool if you genuinely need more context."},
    ]
    for _ in range(max_steps):
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
                cited.add(m.split(":")[0])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:4000]})

    # Used the step budget on tool calls — synthesize a final answer separately.
    return {"answer": _synthesize(question, context), "citations": sorted(cited)}

# ----------------------------- API ----------------------------------------
class IngestReq(BaseModel):
    repo_url: str


class AskReq(BaseModel):
    question: str


@app.post("/ingest")
def ingest(req: IngestReq):
    try:
        return build_index(req.repo_url)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/ask")
def ask(req: AskReq):
    if INDEX["matrix"] is None:
        return JSONResponse({"error": "Load a repo first."}, status_code=400)
    try:
        return run_agent(req.question)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Repo Explainer Agent</title>
<style>
 :root{color-scheme:dark}
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;background:#0d1117;color:#e6edf3}
 h1{font-size:1.4rem} input,textarea,button{font:inherit}
 input,textarea{width:100%;box-sizing:border-box;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:.6rem;margin:.3rem 0}
 button{background:#238636;color:#fff;border:0;border-radius:8px;padding:.6rem 1rem;cursor:pointer}
 button:disabled{opacity:.5}
 .muted{color:#8b949e;font-size:.9rem} pre{white-space:pre-wrap;background:#161b22;padding:1rem;border-radius:8px;border:1px solid #30363d}
 .cit{display:inline-block;background:#1f6feb33;border:1px solid #1f6feb;border-radius:6px;padding:.1rem .4rem;margin:.15rem;font-size:.85rem}
 .qa-card{margin-top:1rem;padding:.8rem 1rem;border:1px solid #30363d;border-radius:10px;background:#0f141b}
 .qa-card .q{font-weight:600;margin-bottom:.4rem}
 .qa-card .q::before{content:"🧑 ";}
</style></head>
<body>
 <h1>🔍 Repo Explainer Agent</h1>
 <p class="muted">Load a public GitHub repo, then ask questions about its code. Answers cite the files they came from.</p>
 <input id="repo" placeholder="https://github.com/owner/repo">
 <button id="load" onclick="ingest()">Load repo</button>
 <span id="status" class="muted"></span>
 <div id="qa" style="display:none;margin-top:1.2rem">
   <textarea id="q" rows="2" placeholder="Ask a question about the repo… (you can ask as many as you like)"></textarea>
   <button id="ask" onclick="ask()">Ask</button>
   <div id="log"></div>
 </div>
<script>
async function ingest(){
  const b=document.getElementById('load'); b.disabled=true;
  document.getElementById('status').textContent=' loading… (first load builds the index)';
  const r=await fetch('/ingest',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({repo_url:document.getElementById('repo').value})});
  const d=await r.json(); b.disabled=false;
  if(d.error){document.getElementById('status').textContent=' ⚠ '+d.error;return;}
  document.getElementById('status').textContent=` ✓ ${d.repo}: ${d.num_files} files, ${d.num_chunks} chunks`;
  document.getElementById('qa').style.display='block';
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function ask(){
  const qEl=document.getElementById('q'); const q=qEl.value.trim(); if(!q) return;
  const b=document.getElementById('ask'); b.disabled=true;
  const card=document.createElement('div'); card.className='qa-card';
  card.innerHTML=`<div class="q">${esc(q)}</div><div class="cits"></div><pre class="ans">Thinking…</pre>`;
  document.getElementById('log').prepend(card);
  qEl.value='';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})});
    const d=await r.json();
    if(d.error){card.querySelector('.ans').textContent='⚠ '+d.error;}
    else{
      card.querySelector('.ans').textContent=d.answer;
      card.querySelector('.cits').innerHTML=(d.citations||[]).map(c=>`<span class="cit">${esc(c)}</span>`).join('');
    }
  }catch(e){card.querySelector('.ans').textContent='⚠ '+e;}
  b.disabled=false; qEl.focus();
}
</script>
</body></html>"""
