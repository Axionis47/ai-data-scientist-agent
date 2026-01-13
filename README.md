# AI Data Scientist Agent

[![CI](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Axionis47/ai-data-scientist-agent/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

An AI agent that analyzes datasets and performs causal inference. It refuses to give you a causal estimate until your data passes 6 diagnostic checks.

**Staging**: https://sdlc-api-staging-du6qod3mja-uc.a.run.app

## What This Does

You upload a context document (.docx) that describes your dataset, then upload the dataset itself (.csv). Then you ask questions. The agent figures out what kind of question you're asking and handles it appropriately:

- **"Give me an overview"** → Runs EDA tools, returns statistics
- **"What's the correlation between X and Y?"** → Computes correlation, returns table
- **"What is the effect of treatment on outcome?"** → Runs diagnostic checks, only estimates if data is ready

The key feature is the **causal gate**. Most AI tools will happily give you a causal estimate even when the data doesn't support one. This agent won't. It checks for treatment balance, positivity violations, missing data, and more before running any estimation.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Axionis47/ai-data-scientist-agent.git
cd ai-data-scientist-agent
pip install -r services/api/requirements.txt

# Run locally (no GCP needed - uses fake LLM)
APP_ENV=dev uvicorn services.api.main:app --reload --port 8080

# In another terminal, test it
curl http://localhost:8080/health
# {"status":"ok"}
```

Or with Docker:
```bash
docker build -t ai-data-scientist-agent .
docker run -p 8080:8080 -e APP_ENV=dev ai-data-scientist-agent
```

---

## How It Works

### The Request Flow

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                        ROUTER                                │
│  Looks at the question text and picks a route:              │
│  - Contains "effect", "causal", "impact" → CAUSAL           │
│  - Contains "report", "summary" → REPORTING                 │
│  - Everything else → ANALYSIS                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
      ANALYSIS        CAUSAL         REPORTING
          │               │               │
          ▼               ▼               ▼
    Run EDA tools    Run diagnostics   (future)
    Call LLM         Check gate
    Return answer    Maybe estimate
```

### The Analysis Route

For regular questions like "describe this dataset" or "what's the average age by region":

1. **Retrieve context**: Embed the question, find relevant chunks from your context doc
2. **Pick a playbook**: Match question to OVERVIEW, UNIVARIATE, GROUPBY, TREND, or CORRELATION
3. **Run tools**: Execute the relevant pandas operations
4. **Call LLM**: Generate a natural language answer using the tool results
5. **Return**: Answer + artifacts (tables, charts data, etc.)

### The Causal Route

For questions like "what is the effect of treatment on outcome":

1. **Retrieve causal context**: Pull relevant chunks, prioritizing "Known Caveats" and "Causal Assumptions" sections
2. **Run diagnostics**: 6 checks to validate the data is ready
3. **Gate decision**:
   - All pass → Ready to estimate
   - Any warn → Ask user to confirm assumptions
   - Any fail → Block estimation, explain why
4. **If confirmed**: Run IPW or regression adjustment, return estimate with confidence interval

---

## Key Decisions Explained

### Why LangGraph Instead of Autonomous Agents?

Most AI agent frameworks let the LLM decide what to do next. That's scary for a data science tool because:

- The LLM might skip important checks
- You can't unit test non-deterministic behavior
- The LLM might hallucinate a causal estimate

LangGraph gives explicit control flow. The router uses **pattern matching** (regex), not LLM calls. Each node is a regular Python function. Conditional edges let the causal gate block bad requests. The result: you can test every path.

```python
# Router uses patterns, not LLM
CAUSAL_PATTERNS = [r"\beffect\b", r"\bimpact\b", r"\bcausal\b", ...]

def route_node(state):
    question = state["question"].lower()
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, question):
            return "CAUSAL"
    return "ANALYSIS"
```

### Why Section-Aware Chunking?

Naive chunking (just split every 500 characters) loses context. A chunk that starts mid-sentence from "Known Caveats" looks like random text. Our chunking:

- Detects document sections from headings (Dataset Overview, Data Dictionary, etc.)
- Chunks within sections to preserve context
- Tries to break at sentence boundaries, not mid-word
- Tags each chunk with its source section

This isn't fancy ML—it's just parsing the document structure and being careful about where we split.

### Why Boost Certain Sections?

Different questions need different parts of the context document:

- "What columns are available?" → boost Data Dictionary
- "What could go wrong with causal claims?" → boost Known Caveats
- "What is this dataset about?" → boost Dataset Overview

We detect keywords in the query and apply a 1.2x score boost to chunks from relevant sections. Simple but effective.

### Why File-Based Storage Instead of a Vector DB?

Context documents are small (usually <50 chunks). Cosine similarity on 256-dimension vectors is fast in numpy. This means:

- No Pinecone/Qdrant to set up and pay for
- No external dependencies in tests
- Fully deterministic behavior
- Works on a laptop

**When to switch**: If you're indexing >1000 chunks, add Vertex AI Vector Search or Pinecone. The `retrieval.py` module is designed to be swapped out.

### Why Fake Clients for Dev/Test?

The `FakeLLMClient` returns canned responses. The `FakeEmbeddingsClient` returns deterministic vectors based on text hash. This means:

- **Zero network calls in CI** - Tests don't hit GCP
- **Tests run in ~2 seconds** - No API latency
- **No credentials needed locally** - Just clone and run
- **Reproducible** - Same inputs → same outputs

The switch is automatic:

```python
def get_llm_client():
    if is_ci_environment() or APP_ENV in ("dev", "test"):
        return FakeLLMClient()
    else:
        return VertexLLMClient()
```

### Why Block Estimates Behind a Gate?

Causal inference is easy to mess up. A naive estimate can be wildly misleading if:

- Treatment isn't really binary (you're comparing apples to oranges)
- There's no overlap between groups (positivity violation)
- You're conditioning on post-treatment variables (bias)
- One group is tiny (variance explosion)

The causal gate runs 6 checks before any estimate:

| Check | What It Looks For | Blocks If |
|-------|-------------------|-----------|
| Treatment type | Is treatment binary? | >10 unique values |
| Time ordering | Treatment before outcome? | Can't verify |
| Missingness | How much data is missing? | >20% missing |
| Leakage | Post-treatment vars in covariates? | Suspicious columns |
| Positivity | Overlap in propensity scores? | >10% extreme |
| Balance | Are groups comparable? | SMD > 0.25 |

If any check fails, you get a detailed diagnostic report, not an estimate. If checks warn, you must confirm assumptions first.

### Why Session Memory?

Users often ask follow-up questions. Without memory, every question starts fresh. With memory:

- "What is the dataset about?" → Agent answers
- "Tell me more about the age column" → Agent remembers context

Memory is stored in `storage/sessions/{session_id}.json`. Pass `session_id` in your request to enable it.

### Why Shadow Mode?

Want to try a new model without risking production? Shadow mode runs both models in parallel, logs the comparison, but only returns the primary response.

```bash
# Enable shadow mode (only works in staging/prod)
SHADOW_MODE_ENABLED=true
SHADOW_MODE_SAMPLE_RATE=0.1  # 10% of requests
SHADOW_VERTEX_LLM_MODEL=gemini-1.5-pro  # Compare flash vs pro
```

You get trace events showing latency and output diff without affecting users.

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status": "ok"}` |
| `/version` | GET | Returns git SHA, build time, environment |
| `/upload_context_doc` | POST | Upload .docx, returns doc_id |
| `/upload_dataset` | POST | Upload .csv, returns dataset_id + profile |
| `/ask` | POST | Ask a question, get answer + artifacts |
| `/debug/config` | GET | Show LLM config (dev only) |

### Upload Context Document

Context docs must be .docx files with these required headings:
- Dataset Overview
- Target Use / Primary Questions
- Data Dictionary
- Known Caveats

```bash
curl -X POST http://localhost:8080/upload_context_doc \
  -F "file=@my_context.docx"
```

Response:
```json
{
  "doc_id": "abc123",
  "num_chunks": 12,
  "status": "indexed"
}
```

### Upload Dataset

```bash
curl -X POST http://localhost:8080/upload_dataset \
  -F "file=@my_data.csv"
```

Response:
```json
{
  "dataset_id": "xyz789",
  "n_rows": 1000,
  "n_cols": 8,
  "column_names": ["id", "treatment", "age", "outcome"],
  "inferred_types": {"id": "integer", "treatment": "integer", ...}
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the effect of treatment on outcome?",
    "doc_id": "abc123",
    "dataset_id": "xyz789"
  }'
```

For causal questions, you can provide specification and confirmations:

```json
{
  "question": "What is the effect of treatment on outcome?",
  "doc_id": "abc123",
  "dataset_id": "xyz789",
  "causal_spec_override": {
    "treatment": "treatment",
    "outcome": "outcome",
    "confounders": ["age", "income"]
  },
  "causal_confirmations": {
    "assignment_mechanism": "randomized",
    "interference_assumption": "no_interference",
    "missing_data_policy": "listwise_delete",
    "ok_to_estimate": true
  }
}
```

---

## Project Structure

```
packages/
  agent/
    graph.py              # LangGraph state machine (11 nodes)
    tools_causal.py       # 6 diagnostic checks
    tools_causal_estimation.py  # IPW, regression adjustment
    tools_eda.py          # overview, groupby, correlation, trends
    causal_gate.py        # Gate logic, readiness report
    retrieval.py          # Embed + cosine similarity search
    planner.py            # Match question → playbook
    llm_provider.py       # Pick Vertex vs fake, shadow mode
    memory.py             # Session store protocol + file backend
    fake_clients.py       # Deterministic fakes for tests
    vertex_clients.py     # Real Gemini + embeddings
    versioning.py         # Track prompt/segment versions
  contracts/
    models.py             # All Pydantic models

services/api/
  main.py                 # FastAPI endpoints
  storage/                # Contexts, datasets, traces, sessions
  tests/                  # 69 tests

evals/
  scenarios.json          # 15 golden test scenarios
  balanced_causal.csv     # Test dataset for causal scenarios

scripts/
  run_evals.py            # Eval runner for regression testing
  demo_staging.sh         # Demo script for staging

.github/workflows/
  ci.yml                  # Lint + test + evals + docker build
  deploy_staging.yml      # Deploy to Cloud Run on staging push
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | `dev`, `test`, `staging`, or `prod` | `dev` |
| `GCP_PROJECT` | GCP project ID (required for Vertex AI) | — |
| `GCP_LOCATION` | GCP region | `us-central1` |
| `VERTEX_LLM_MODEL` | LLM model name | `gemini-1.5-flash` |
| `VERTEX_EMBED_MODEL` | Embeddings model name | `text-embedding-005` |
| `SHADOW_MODE_ENABLED` | Enable shadow mode | `false` |
| `SHADOW_MODE_SAMPLE_RATE` | Fraction of requests to shadow | `0.0` |
| `STORAGE_DIR` | Override storage location | `./storage` |

### When Each Client Is Used

| Environment | LLM Client | Embeddings Client |
|-------------|------------|-------------------|
| `APP_ENV=dev` | FakeLLMClient | FakeEmbeddingsClient |
| `APP_ENV=test` | FakeLLMClient | FakeEmbeddingsClient |
| CI (GitHub Actions) | FakeLLMClient | FakeEmbeddingsClient |
| `APP_ENV=staging` | VertexLLMClient | VertexEmbeddingsClient |
| `APP_ENV=prod` | VertexLLMClient | VertexEmbeddingsClient |

---

## Testing

### Run Tests

```bash
# All tests
pytest services/api/tests/ -v

# Specific test file
pytest services/api/tests/test_causal_gate.py -v

# With coverage
pytest services/api/tests/ --cov=packages --cov=services
```

### Run Evals

Evals are golden scenario tests that check routing, playbook selection, and causal gate behavior:

```bash
# Run all 15 scenarios
python scripts/run_evals.py

# Run specific scenario
python scripts/run_evals.py --scenario route_causal_effect

# Verbose output
python scripts/run_evals.py --verbose
```

### Current Test Count

- **69 unit tests** across 5 test files
- **15 eval scenarios** for regression testing
- All tests run in CI on every push

---

## Deployment

### Branch Strategy

| Branch | Purpose | Auto-Deploy |
|--------|---------|-------------|
| `dev` | Development work | No |
| `staging` | Pre-production testing | Yes → Cloud Run |
| `main` | Production-ready | Manual |

### Deploy to Staging

```bash
# Push to dev, CI runs
git push origin dev

# Merge to staging to trigger deployment
git checkout staging
git merge dev
git push origin staging
# → Deploys to Cloud Run automatically
```

### Cloud Run Configuration

- Region: us-central1
- Memory: 512Mi
- Instances: 0-10 (auto-scaling)
- Authentication: GCP Workload Identity Federation
- Secrets: Via Secret Manager

---

## Extending the System

### Add a New Playbook

1. Add triggers in `planner.py`:
   ```python
   if "cohort" in question_lower:
       return "COHORT", {"time_col": "date"}, 0.8
   ```

2. Add tool function in `tools_eda.py`:
   ```python
   def cohort_analysis(df, time_col, group_col):
       # Your analysis logic
       return result_dict
   ```

3. Wire it up in `graph.py` execute_playbook node

### Add a New Diagnostic Check

1. Add check function in `tools_causal.py`:
   ```python
   def my_new_check(df, treatment, outcome, confounders):
       # Check logic
       return {
           "name": "my_check",
           "status": "PASS",  # or "WARN" or "FAIL"
           "details": {...},
           "recommendations": [...]
       }
   ```

2. Call it from `run_causal_diagnostics()` in the same file

### Add a New Estimator

1. Add function in `tools_causal_estimation.py`:
   ```python
   def run_matching_ate(df, treatment, outcome, confounders, ...):
       # Matching logic
       return CausalEstimateArtifact(...)
   ```

2. Add to `select_estimator()` recommended list

3. Add case in `run_causal_estimation()`

### Add a New LLM Provider

1. Implement the `LLMClient` protocol:
   ```python
   class MyLLMClient:
       def generate(self, prompt: str) -> str:
           # Your logic
           return response
   ```

2. Add selection logic in `llm_provider.py`

---

## What's Done

- ✅ FastAPI spine with health/version endpoints
- ✅ Document upload with section-aware chunking
- ✅ Dataset upload with profiling
- ✅ LangGraph orchestration (12 nodes)
- ✅ RAG retrieval with section boosting and dynamic k
- ✅ 5 EDA playbooks (overview, univariate, groupby, trend, correlation)
- ✅ Causal gate with 6 diagnostic checks
- ✅ IPW and regression adjustment estimators
- ✅ Shadow mode for LLM comparison
- ✅ Session memory for multi-turn conversations
- ✅ Segment versioning for reproducibility
- ✅ 15 golden eval scenarios
- ✅ CI/CD with GitHub Actions
- ✅ Cloud Run deployment

## What's Not Done Yet

- ❌ Chat UI (API only for now)
- ❌ More estimators (matching, diff-in-diff)
- ❌ Vector database (file-based for now)
- ❌ PDF report export
- ❌ User authentication

---

## License

MIT

## Contributing

1. Fork the repo
2. Branch from `dev`
3. Write tests for your changes
4. Make sure CI passes
5. Open a PR