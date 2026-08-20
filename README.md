# HR Assistant — RAG-Powered HR Document Q&A

A retrieval-augmented generation (RAG) app that lets HR teams and employees ask natural-language questions about CVs, HR policies, and onboarding documents — and get grounded, cited answers in an HR-appropriate tone, instead of digging through PDFs.

Built as a durable, observable pipeline: every ingestion and query runs as a tracked, retryable workflow rather than a fire-and-forget script.

---

## Features

- **Upload any PDF** (CVs, policy docs, onboarding guides) and have it automatically chunked, embedded, and indexed
- **Ask questions in plain language** and get answers grounded strictly in the uploaded documents — the assistant explicitly says when it doesn't know, rather than guessing
- **Source citations** on every answer, so you can trace a claim back to the exact document it came from
- **HR-appropriate tone by design** — the system prompt encodes real prompting practices (structured XML instructions, explicit grounding rules, sensitive-topic handling) rather than a one-line "be professional" instruction
- **Durable workflows** — ingestion and querying run as Inngest functions, meaning a network blip or API timeout retries the failed step instead of losing the whole job

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Claude Haiku 4.5 (Anthropic) | Fast, low-cost — chosen deliberately to keep a public demo affordable |
| Embeddings | Voyage AI (`voyage-3-large`, 1024-dim) | Anthropic's recommended embedding partner; no native Anthropic embedding model exists |
| Vector DB | Qdrant Cloud | Free-tier persistent vector storage, decoupled from the app's own disk |
| Orchestration | Inngest | Durable step functions — retries, observability, and replay for free instead of hand-rolled job queues |
| PDF parsing / chunking | LlamaIndex (`PDFReader`, `SentenceSplitter`) | Battle-tested chunking with configurable overlap |
| Backend | FastAPI + Inngest's `fast_api` serve adapter | Lightweight HTTP layer for Inngest to call into |
| Frontend | Streamlit | Fast to build, free to host, good enough UX for a focused internal tool |

---

## Architecture

```
 PDF upload (Streamlit)
        │
        ▼
 rag/ingest_pdf event ──▶ Inngest ──▶ load & chunk (LlamaIndex)
                                      │
                                      ▼
                                embed chunks (Voyage AI)
                                      │
                                      ▼
                                upsert into Qdrant

 Question (Streamlit)
        │
        ▼
 rag/query_pdf_ai event ──▶ Inngest ──▶ embed query (Voyage AI)
                                        │
                                        ▼
                                  search Qdrant (top-k chunks)
                                        │
                                        ▼
                              build context + system prompt
                                        │
                                        ▼
                            Claude (via step.ai.infer) ──▶ answer + sources
```

---

## Getting Started (Local)

### 1. Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- [Inngest CLI](https://www.inngest.com/docs/getting-started/nodejs-quick-start) for the local dev server

### 2. Clone and install

```bash
git clone <your-repo-url>
cd HR_RAG_APP
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt   # or: uv sync, if using pyproject.toml
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key
VOYAGE_API_KEY=your_voyage_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_key
INNGEST_DEV=1
```

### 4. Run it — three processes, three terminals

```bash
# Terminal 1 — Inngest local dev server (orchestration + dashboard)
inngest dev

# Terminal 2 — backend (FastAPI + Inngest functions)
uvicorn main:app --reload

# Terminal 3 — frontend
streamlit run app.py
```

Open `http://localhost:8288` to watch runs live in the Inngest dashboard, and `http://localhost:8501` for the app itself.

---

## Project Structure

```
├── main.py            # FastAPI app + Inngest functions (ingest_pdf, query_pdf_ai)
├── app.py              # Streamlit frontend — chat UI, document upload, run polling
├── vector_db.py         # QdrantStorage class — collection setup, upsert, search
├── data_loader.py        # PDF loading, chunking, and embedding helpers
├── custom_types.py       # Pydantic models for typed step outputs
├── requirements.txt / pyproject.toml
└── .env                # Not committed — see Environment Variables above
```

---

## Deploying

- **Frontend** → Streamlit Community Cloud (free)
- **Backend** → Render free web service
- **Vector DB** → Qdrant Cloud free tier (1GB)
- **Orchestration** → Inngest Cloud free tier

Set `is_production=True` on both `Inngest()` client instantiations, add `INNGEST_EVENT_KEY` and `INNGEST_SIGNING_KEY` as environment variables in your hosting provider, deploy, then sync the app's URL (`https://your-app.onrender.com/api/inngest`) from the Inngest dashboard under **Apps → Sync New App**.

---

## What I Learned Building This

This project ended up being less about "call an LLM API" and more about building a small distributed system correctly — most of the real learning came from debugging, not the happy path.

- **RAG is a pipeline, not a single call.** Chunking strategy, embedding model choice, and retrieval `top_k` all directly shape answer quality — a well-written prompt can't compensate for retrieval that returns nothing.
- **Durable execution changes how you think about failure.** Wrapping ingestion and querying as Inngest steps meant a Voyage API timeout or a transient DNS failure retried automatically instead of losing an entire job — this matters more than it sounds once you've had a multi-minute ingestion job die on the last chunk.
- **API contracts drift, and dependency pinning matters.** Hit this directly: `qdrant-client` deprecated `.search()` in favor of `.query_points()` mid-project, silently changing the return shape from a flat list to a `.points`-wrapped response — a bug that failed *silently* (empty results, no exception) rather than loudly, which was a good lesson in why "it didn't crash" isn't the same as "it worked."
- **Embedding dimension mismatches are a quiet failure mode.** Voyage's `output_dimension` options are fixed (256/512/1024/2048) — mismatching that against a Qdrant collection's configured vector size is an easy, easy-to-miss bug if the two are set in different files without a single source of truth.
- **Local-only state doesn't survive a deploy.** Relative file paths (`./qdrant_storage`) and local dev servers work fine on one laptop and quietly break the moment a working directory changes or you push to a host with ephemeral disk — this is what pushed the move to Qdrant Cloud and Inngest Cloud rather than trying to keep local-mode components alive in production.
- **Prompt engineering for a specific domain is a design exercise, not a one-liner.** The system prompt went through several passes — structuring it with XML tags (`<tone>`, `<grounding_rules>`, `<citation_rules>`, `<sensitive_topics>`) rather than a paragraph, and deliberately deciding what the assistant should *refuse* to speculate about (a named employee's performance, legal interpretation) — was as much a product decision as an engineering one.
- **Cost-consciousness is part of the design, not an afterthought.** Choosing Haiku over a larger model for a public portfolio demo, and thinking through rate limiting before sharing a link, was a deliberate tradeoff between demo quality and not waking up to a surprise bill.

---

## Notes for Employers / Reviewers

This project was built to demonstrate:

- **End-to-end RAG system design** — from PDF ingestion through chunking, embedding, vector retrieval, and grounded generation, not just a wrapper around a chat API
- **Durable/distributed workflow orchestration** with Inngest — designing for retries, observability, and step-level failure isolation
- **Debugging discipline** — most bugs here were silent-failure modes (empty results, wrong working directory, dimension mismatches) rather than loud stack traces, which required tracing data through the pipeline rather than just reading an error message
- **Deliberate, tone-aware prompt engineering** for a specific professional domain, including explicit guardrails around sensitive HR topics
- **Practical deployment tradeoffs** — building on a fully free-tier stack (Streamlit Cloud, Render, Qdrant Cloud, Inngest Cloud) while keeping the architecture cloud-portable rather than laptop-bound

Feel free to reach out with questions about any design decision in this repo — happy to walk through the reasoning.