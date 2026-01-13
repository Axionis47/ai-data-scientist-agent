# AI Data Scientist Agent

[![CI](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

A Python agent that does EDA and causal inference on your data. It won't give you a causal estimate unless the data passes sanity checks first.

Built with LangGraph + FastAPI. Runs on Cloud Run.

## What it does

- **Upload context docs** (.docx) — chunks and embeds them for RAG retrieval
- **Upload datasets** (.csv) — profiles columns, caches metadata
- **Answer questions** — routes to EDA playbooks or causal analysis
- **Block bad estimates** — runs 6 diagnostic checks before any causal estimate
- **Compare LLMs** — shadow mode lets you test new models without affecting users

## Quick start

```bash
git clone https://github.com/Axionis47/ai-data-scientist-agent.git
cd ai-data-scientist-agent
pip install -r services/api/requirements-dev.txt

# Run locally (uses fake LLM - no GCP needed)
APP_ENV=dev uvicorn services.api.main:app --reload

# Run tests
pytest services/api/tests/ -v
```

Or with Docker:

```bash
docker build -t ai-data-scientist-agent .
docker run -p 8080:8080 -e APP_ENV=dev ai-data-scientist-agent
```

## How it's built

```
FastAPI → LangGraph (11 nodes) → Vertex AI / Fake clients
                ↓
    ┌───────────┼───────────┐
    │           │           │
  Router     RAG        Causal Gate
    │      Retrieval    (diagnostics)
    │           │           │
    └───────────┴───────────┘
                ↓
           Response
```

The agent uses a LangGraph StateGraph with conditional edges. A router node looks at the question and decides where to send it:

- **ANALYSIS** → retrieve context, pick a playbook, run EDA tools, call LLM
- **CAUSAL** → run diagnostics, check assumptions, maybe estimate
- **REPORTING** → (placeholder for future)

## The causal gate

This is the main safety feature. Before any causal estimate, it runs:

| Check | What it looks for | Blocks if |
|-------|-------------------|-----------|
| Treatment type | Is treatment binary? | >10 unique values |
| Time ordering | Does treatment come before outcome? | Can't parse dates |
| Missingness | How much data is missing? | >20% missing |
| Leakage | Post-treatment variables in covariates? | Suspicious columns |
| Positivity | Overlap in propensity scores? | >10% extreme scores |
| Balance | Are groups comparable? | SMD > 0.25 |

If any check fails, you get a diagnostic report instead of an estimate. If checks warn, you have to confirm assumptions before proceeding.

## Project layout

```
packages/
  agent/
    graph.py              # LangGraph state machine
    tools_causal.py       # Diagnostic checks
    tools_causal_estimation.py  # IPW, regression adjustment
    tools_eda.py          # Dataset profiling, groupby, trends
    retrieval.py          # Embed + retrieve chunks
    llm_provider.py       # Pick Vertex vs fake, shadow mode
    fake_clients.py       # Deterministic fakes for tests
    vertex_clients.py     # Real Gemini + embeddings
  contracts/              # Pydantic models

services/api/
  main.py                 # FastAPI app
  tests/                  # 53 tests

.github/workflows/
  ci.yml                  # Lint + test + docker build
  deploy_staging.yml      # Deploy to Cloud Run
```

## API

| Endpoint | What it does |
|----------|--------------|
| `GET /health` | Returns `{"status": "healthy"}` |
| `GET /version` | Git SHA and build time |
| `POST /upload_context_doc` | Upload .docx, get doc_id back |
| `POST /upload_dataset` | Upload .csv, get dataset_id + profile |
| `POST /ask` | Ask a question, get answer + artifacts |
| `GET /debug/config` | Show LLM config (dev only) |

## Config

Set these environment variables:

| Variable | What it does | Default |
|----------|--------------|---------|
| `APP_ENV` | `dev`, `test`, `staging`, or `prod` | `dev` |
| `GCP_PROJECT` | Your GCP project (needed for Vertex) | — |
| `GCP_LOCATION` | GCP region | `us-central1` |
| `VERTEX_LLM_MODEL` | Which Gemini model | `gemini-1.5-flash` |
| `SHADOW_MODE_ENABLED` | Run a second LLM in parallel | `false` |
| `SHADOW_MODE_SAMPLE_RATE` | What fraction to shadow | `0.0` |

In dev/test, it uses fake clients so you don't need GCP credentials.


## Why these choices?

### LangGraph instead of autonomous agents

I wanted explicit control flow, not an LLM deciding what to do next. With LangGraph:

- The router uses pattern matching, not LLM calls
- Each node is a regular Python function you can unit test
- Conditional edges let the causal gate block bad requests
- State is typed (`AgentState` dataclass), so you catch bugs early

### File-based embeddings instead of a vector DB

For now, embeddings live in JSON files:

```
storage/contexts/{doc_id}/embeddings.json
```

This works because:
- Most context docs are <50 chunks
- Cosine similarity on 256-dim vectors is fast enough in numpy
- No Pinecone/Qdrant to set up and pay for
- Tests are deterministic (no external state)

When to switch: if you're indexing >1000 chunks, add Vertex AI Vector Search or Pinecone.

### Fake clients in dev/test

The `FakeLLMClient` returns canned responses. The `FakeEmbeddingsClient` returns deterministic vectors. This means:

- Zero network calls in CI
- Tests run in ~2 seconds
- No GCP credentials needed locally
- Same outputs every time

The switch happens automatically:

```python
if is_ci_environment() or APP_ENV in ("dev", "test"):
    return FakeLLMClient()
else:
    return VertexLLMClient()
```

### Why block estimates behind a gate?

Causal inference is easy to get wrong. A naive estimate can be wildly misleading if:
- Treatment isn't really binary
- There's no overlap between groups
- You're conditioning on post-treatment variables

So the agent refuses to estimate until checks pass. Users see exactly what failed and what to fix.

### Shadow mode for safe comparisons

Want to try gemini-1.5-pro instead of flash? Turn on shadow mode. It runs both models, logs the diff, but only returns the primary response. You can compare quality without risking prod.

## Extending this

The codebase is set up to add features without touching core logic.

**Add a playbook** — put triggers in `planner.py`, tools in `tools_eda.py`, wire in `graph.py`.

**Add an estimator** — write the function in `tools_causal_estimation.py`, add it to recommended list, call it from `run_estimation_node`.

**Add a diagnostic** — write a check in `tools_causal.py`, call it from `run_causal_diagnostics()`.

**Add an LLM provider** — implement the `LLMClient` protocol, add selection logic in `llm_provider.py`.

**Add auth** — middleware in `main.py`, user context in `AgentState`.

**Add a vector DB** — new client class, swap out file reads in `retrieval.py`.

## Tests

53 tests across 5 files. Run them with:

```bash
pytest services/api/tests/ -v
```

The CI runs ruff + pytest + docker build on every PR.

## Deploying

Push to `dev` → CI runs → merge to `staging` → deploys to Cloud Run → smoke checks hit `/health` and `/version`.

Cloud Run config:
- us-central1
- 512Mi memory
- 0-10 instances (staging)
- Secrets via Secret Manager

## What's next

Done:
- API spine
- LangGraph + RAG
- EDA playbooks
- Causal gate
- IPW + regression adjustment
- Shadow mode

Not done yet:
- Chat UI
- Multi-turn memory
- More estimators (matching, diff-in-diff)
- PDF reports

## License

MIT

## Contributing

1. Fork it
2. Branch from `dev`
3. Write tests
4. Make sure CI passes
5. Open a PR