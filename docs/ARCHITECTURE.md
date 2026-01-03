# SDLC API Architecture

## Overview

This is a deterministic, testable foundation for an intelligent SDLC assistant.

- **Phase 0**: API spine, contracts, validation logic
- **Phase 1**: LangGraph orchestration, RAG retrieval, Vertex AI integration
- **Phase 2**: EDA playbooks, dataset upload, deterministic analysis tools
- **Phase 3**: Causal Safety Gate with structured diagnostics and readiness checks

## Design Principles

1. **Contracts First**: All API interactions use strongly-typed Pydantic models
2. **Deterministic**: All operations are reproducible and testable
3. **Validation-Heavy**: Strict validation of inputs before processing
4. **Observable**: Trace IDs and structured logging throughout
5. **Cloud-Native**: Designed for Cloud Run deployment

## Architecture Layers

### 1. Contracts Layer (`packages/contracts/`)

Defines all Pydantic models for:
- Request/response schemas
- Router decisions
- Artifacts (text, table, checklist)
- Trace events

This layer has zero dependencies on implementation details.

### 2. API Layer (`services/api/`)

FastAPI service exposing:
- `GET /health` - Health check
- `POST /upload_context_doc` - Upload and validate Word (.docx) context documents
- `POST /upload_dataset` - Upload datasets (stub)
- `POST /ask` - Ask questions (stub deterministic response)

#### Context Document Processing

1. **Validation**:
   - File format: Must be .docx
   - Required headings (case-sensitive):
     - "Dataset Overview"
     - "Target Use / Primary Questions"
     - "Data Dictionary"
     - "Known Caveats"
   - Minimum content: 800 characters

2. **Processing**:
   - Extract text via python-docx
   - Normalize whitespace
   - Compute SHA-256 hash
   - Deterministic chunking (500 chars, 100 overlap)

3. **Storage**:
   - `storage/contexts/{doc_id}/raw.txt` - Full extracted text
   - `storage/contexts/{doc_id}/chunks.json` - Chunked text with indices

### 3. Agent Layer (`packages/agent/`)

LangGraph-based orchestration with RAG retrieval:

#### Key Components

- `interfaces.py` - Protocol definitions for `EmbeddingsClient` and `LLMClient`
- `fake_clients.py` - Deterministic fake clients for CI/testing (no network)
- `vertex_clients.py` - Vertex AI clients for production (Gemini + embeddings)
- `retrieval.py` - Embedding storage and cosine similarity retrieval
- `graph.py` - LangGraph StateGraph implementation

#### Phase 1 Graph Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Question                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTE NODE                               │
│  - Pattern matching for CAUSAL, REPORTING keywords         │
│  - Default: ANALYSIS                                        │
│  - CAUSAL → NEEDS_CLARIFICATION (Phase 1 placeholder)      │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌─────────────────────────────┐
│ NEEDS_CLARIFY / │             │     RETRIEVE_CONTEXT        │
│ REPORTING       │             │  - Embed query              │
│ (skip RAG)      │             │  - Cosine similarity search │
└────────┬────────┘             │  - Return top-k chunks      │
         │                      └──────────────┬──────────────┘
         │                                     │
         │                                     ▼
         │                      ┌─────────────────────────────┐
         │                      │      COMPOSE_PROMPT         │
         │                      │  - System instructions      │
         │                      │  - Insert retrieved chunks  │
         │                      │  - Format user question     │
         │                      └──────────────┬──────────────┘
         │                                     │
         │                                     ▼
         │                      ┌─────────────────────────────┐
         │                      │        CALL_LLM             │
         │                      │  - Send prompt to LLM       │
         │                      │  - Get response             │
         │                      └──────────────┬──────────────┘
         │                                     │
         └──────────────────┬──────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUILD_RESPONSE                            │
│  - Create TextArtifact with retrieved chunk info           │
│  - Assemble AskQuestionResponse                            │
└─────────────────────────────────────────────────────────────┘
```

## RAG Retrieval System

### Why No Vector Database (Phase 1)

For Phase 1, we use a simple file-based approach:
- Context documents are small (typically <50 chunks)
- Cosine similarity over 256-dim vectors is fast in pure Python/numpy
- No external dependencies to manage
- Fully deterministic and testable

### Embeddings Storage

After document upload, embeddings are stored at:
```
storage/contexts/{doc_id}/embeddings.json
```

Format:
```json
[
  {
    "chunk_index": 0,
    "text": "This dataset contains...",
    "section": null,
    "embedding": [0.123, -0.456, ...]
  },
  ...
]
```

### Retrieval Process

1. Embed the user's query using the same embeddings client
2. Compute cosine similarity against all stored chunk embeddings
3. Return top-k chunks (default k=4) sorted by similarity score

## Environment Configuration

### Vertex AI Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (dev/test/staging/prod) | dev |
| `GCP_PROJECT` | GCP project ID | (required for staging/prod) |
| `GCP_LOCATION` | GCP region | us-central1 |
| `VERTEX_LLM_MODEL` | LLM model name | gemini-1.5-flash |
| `VERTEX_EMBED_MODEL` | Embeddings model name | text-embedding-005 |

### Client Selection

- **dev/test**: Uses `FakeEmbeddingsClient` and `FakeLLMClient` (deterministic, no network)
- **staging/prod**: Uses `VertexEmbeddingsClient` and `VertexLLMClient` (requires GCP credentials)

## Deployment Strategy

### Development
- Local: `uvicorn services.api.main:app --reload`
- Docker: `docker run -p 8080:8080 sdlc-api`

### Staging
- Cloud Run service
- Environment: `STAGING`
- Auto-scaling: 0-10 instances
- Secrets: Vertex AI credentials via Secret Manager

### Production
- Cloud Run service
- Environment: `PROD`
- Auto-scaling: 0-100 instances
- Monitoring: Cloud Logging + Trace
- Alerts: Error rate, latency p95

## Phase 3: Causal Safety Gate

### Overview

The Causal Safety Gate ensures causal analysis never proceeds without proper validation. It runs deterministic diagnostic checks and produces a structured `CausalReadinessReport`.

**Key Principle**: No causal estimate (ATE) is ever returned unless readiness status is PASS. In Phase 3, even PASS status returns "Ready for estimation" message - actual estimation is deferred to Phase 4.

### Causal Gate Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Causal Question                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTER NODE                              │
│  - Matches causal keywords (effect, impact, causal, ATE)   │
│  - Routes to CAUSAL                                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  CAUSAL_GATE NODE                           │
│  1. Validate dataset exists                                 │
│  2. Infer or validate treatment/outcome                    │
│  3. Run diagnostic checks (see below)                      │
│  4. Build CausalReadinessReport                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              CAUSAL_DECIDE_NEXT NODE                        │
│  - FAIL → NEEDS_CLARIFICATION + followup questions         │
│  - WARN → NEEDS_CLARIFICATION + targeted questions         │
│  - PASS → "Ready for estimation" + recommended estimators  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUILD_RESPONSE                            │
│  - Include all diagnostic artifacts                        │
│  - Include CausalSpecArtifact                              │
│  - Include CausalReadinessReport                           │
└─────────────────────────────────────────────────────────────┘
```

### Diagnostic Checks

Each check returns a `DiagnosticArtifact` with status (PASS/WARN/FAIL), details, and recommendations:

| Check | What It Does | Thresholds |
|-------|--------------|------------|
| **treatment_type_check** | Verifies treatment is binary | PASS: 2 values, WARN: 3-10 values, FAIL: continuous |
| **time_ordering_check** | Verifies temporal precedence | PASS: parseable time column, WARN: no time column |
| **missingness_check** | Checks missing data rates | PASS: ≤5%, WARN: 5-20%, FAIL: >20% |
| **leakage_check** | Detects post-treatment variables | WARN: suspicious column names detected |
| **positivity_check** | Propensity score distribution | FAIL: >10% extreme propensities, WARN: >20% borderline |
| **balance_check** | Standardized Mean Differences | PASS: SMD ≤0.1, WARN: 0.1-0.25, FAIL: >0.25 |

### Readiness Criteria

1. **FAIL** (blocks estimation):
   - No dataset provided
   - Treatment or outcome cannot be inferred
   - Treatment is continuous (>10 unique values)
   - Missingness >20%
   - Positivity violation (>10% extreme propensities)

2. **WARN** (requires user confirmation):
   - Treatment is multi-class (3-10 values)
   - No time column to verify ordering
   - Moderate missingness (5-20%)
   - Moderate imbalance (SMD 0.1-0.25)

3. **PASS** (ready for estimation):
   - All critical checks pass
   - May still have assumption questions for user

### Assumption Questions (Always Asked)

These are included in followup_questions even when readiness is PASS:
- "What is the assignment mechanism? (randomized, policy rule, self-selection)"
- "Is there potential for interference between units (SUTVA violation)?"
- "Are there any unmeasured confounders you are aware of?"
- "What is your preferred approach for handling missing data?"

### Artifacts

Phase 3 adds these artifact types:

```python
class CausalSpecArtifact:
    treatment: str | None
    outcome: str | None
    unit: str | None
    time_col: str | None
    horizon: str | None
    confounders_selected: list[str]
    confounders_missing: list[str]  # Known unmeasured confounders
    assumptions: list[str]
    questions: list[str]

class DiagnosticArtifact:
    name: str  # e.g., "positivity_check"
    status: Literal["PASS", "WARN", "FAIL"]
    details: dict
    recommendations: list[str]

class CausalReadinessReport:
    readiness_status: Literal["PASS", "WARN", "FAIL"]
    spec: CausalSpecArtifact
    diagnostics: list[DiagnosticArtifact]
    followup_questions: list[str]
    ready_for_estimation: bool
    recommended_estimators: list[str]  # Phase 4
```

### API Changes

`AskQuestionRequest` now accepts an optional `causal_spec_override`:

```json
{
  "question": "What is the effect of treatment on outcome?",
  "doc_id": "abc123",
  "dataset_id": "xyz789",
  "causal_spec_override": {
    "treatment": "treatment_col",
    "outcome": "outcome_col",
    "confounders": ["age", "income"]
  }
}
```

## Deferred Items (Phase 4+)

1. **Causal Effect Estimation** (Phase 4):
   - Inverse Probability Weighting (IPW)
   - Augmented IPW (Doubly Robust)
   - Propensity Score Matching
   - Sensitivity Analysis (Rosenbaum bounds)

2. **Vector Database**:
   - Migrate from file-based embeddings to Vertex AI Vector Search or Pinecone
   - Required for larger documents (>1000 chunks)

3. **Reporting/Export**:
   - Implement REPORTING playbook
   - Generate PDF/HTML reports
   - Export data summaries

4. **Authentication**:
   - User auth via Firebase/Auth0
   - Session management
   - Rate limiting

## Testing Strategy

- **Unit Tests**: pytest for all endpoints and utilities
- **Integration Tests**: Full workflow tests with real .docx files
- **Contract Tests**: Validate Pydantic models
- **Docker Tests**: Build and run container
- **CI/CD**: GitHub Actions on every PR

## Security Considerations

- Input validation (file types, sizes)
- Content sanitization
- Rate limiting (future)
- Authentication (future)
- Secrets management via Secret Manager

