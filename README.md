# 🔍 Repo Explainer Agent

Ask natural-language questions about any **public GitHub repository** and get
answers grounded in the actual code, **with file/line citations**. A Groq LLM
drives a plan-act-observe **agent loop** over three tools, deciding when to
search, read a file, or browse the structure before it answers.

**Links:** [Live demo](#) · Built with FastAPI, Groq (Llama 3.3), fastembed (ONNX)

## What makes it an agent (not just RAG)

A plain RAG app does one retrieve-then-answer step. This one runs a loop:

```
question → LLM decides which tool to call → tool runs → LLM observes
        → (search again / read a file / browse) → ... → grounded answer + citations
```

**Tools the agent can call**
- `search_code(query)` — semantic search over the repo (ONNX embeddings + cosine)
- `read_file(path)` — inspect a specific file
- `list_files()` — browse the repo structure

The system prompt forces it to cite `path/to/file.py:40-70` and to say "not found"
rather than hallucinate.

## Design notes

- **Lightweight on purpose** — embeddings run on CPU via `fastembed` (ONNX, no
  PyTorch) and the vector search is in-memory NumPy cosine, so it fits a free
  Render instance.
- Repos are fetched as a zip via the GitHub API, filtered to code/text files,
  and chunked by line windows (60 lines, 10 overlap), capped for memory safety.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env     # add your GROQ_API_KEY
# load env vars, then:
uvicorn app:app --reload
```
Open http://127.0.0.1:8000 , paste a repo URL, click **Load repo**, and ask.

Get a free Groq key at https://console.groq.com .

## Deploy to Render

1. Push this repo to GitHub.
2. On Render: **New → Blueprint**, point it at this repo (`render.yaml` is included).
3. Set the `GROQ_API_KEY` environment variable (and optionally `GITHUB_TOKEN`).
4. Deploy — Render gives you a public URL.

## Environment variables

| Var | Required | Purpose |
|-----|----------|---------|
| `GROQ_API_KEY` | yes | LLM inference (Groq) |
| `GITHUB_TOKEN` | no | Raises GitHub API rate limit (read-only token) |

## Limitations

- Public repos only; large repos are capped (≤400 files, ≤1500 chunks) to stay
  within free-tier memory.
- Single in-memory index (one repo at a time) — fine for a demo.

## License
MIT
