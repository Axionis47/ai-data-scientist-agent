"""
FastAPI service for SDLC API.
"""

import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from docx import Document
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.agent import (
    FakeEmbeddingsClient,
    FakeLLMClient,
    embed_and_store_chunks,
    run_agent,
)
from packages.contracts import (
    AskQuestionRequest,
    AskQuestionResponse,
    ChecklistArtifact,
    RouterDecision,
    TextArtifact,
    UploadContextDocResponse,
    UploadDatasetResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment
APP_ENV = os.getenv("APP_ENV", "dev")

app = FastAPI(title="SDLC API", version="0.1.0")

# Storage directories
STORAGE_DIR = Path(__file__).parent / "storage" / "contexts"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = Path(__file__).parent / "storage" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Required headings (case-sensitive)
REQUIRED_HEADINGS = [
    "Dataset Overview",
    "Target Use / Primary Questions",
    "Data Dictionary",
    "Known Caveats",
]

# Minimum content length
MIN_CONTENT_LENGTH = 800

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def get_embeddings_client():
    """Get the appropriate embeddings client based on environment."""
    if APP_ENV in ("staging", "prod"):
        try:
            from packages.agent.vertex_clients import VertexEmbeddingsClient
            return VertexEmbeddingsClient()
        except ImportError:
            logger.warning("Vertex AI not available, falling back to fake client")
    return FakeEmbeddingsClient()


def get_llm_client():
    """Get the appropriate LLM client based on environment."""
    if APP_ENV in ("staging", "prod"):
        try:
            from packages.agent.vertex_clients import VertexLLMClient
            return VertexLLMClient()
        except ImportError:
            logger.warning("Vertex AI not available, falling back to fake client")
    return FakeLLMClient()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a .docx file."""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    # Normalize whitespace
    text = "\n".join(paragraphs)
    # Normalize multiple spaces/newlines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return text


def validate_required_headings(text: str) -> list[str]:
    """Validate that all required headings are present. Returns list of missing headings."""
    missing = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            missing.append(heading)
    return missing


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Deterministically chunk text with overlap."""
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
        })

        chunk_index += 1
        start = end - overlap

        # Prevent infinite loop
        if start >= len(text):
            break

    return chunks


@app.post("/upload_context_doc", response_model=UploadContextDocResponse)
async def upload_context_doc(file: UploadFile = File(...)):
    """
    Upload and validate a context document (.docx).

    Requirements:
    - Must be .docx format
    - Must contain all required headings
    - Must have at least 800 characters
    """
    # Check file extension
    if not file.filename or not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=415,
            detail={"error": "File must be a .docx (Word document)"}
        )

    # Save temporary file
    doc_id = str(uuid.uuid4())
    doc_dir = STORAGE_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    temp_path = doc_dir / "uploaded.docx"

    try:
        # Write uploaded file
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Extract text
        text = extract_text_from_docx(temp_path)

        # Validate required headings
        missing_headings = validate_required_headings(text)
        if missing_headings:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Missing required headings",
                    "missing_headings": missing_headings
                }
            )

        # Validate minimum content length
        num_chars = len(text)
        if num_chars < MIN_CONTENT_LENGTH:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"Content too short. Minimum {MIN_CONTENT_LENGTH} characters required.",
                    "actual_chars": num_chars
                }
            )

        # Compute hash
        doc_hash = hashlib.sha256(text.encode()).hexdigest()

        # Chunk text
        chunks = chunk_text(text)

        # Save raw text
        raw_path = doc_dir / "raw.txt"
        with open(raw_path, "w") as f:
            f.write(text)

        # Save chunks
        chunks_path = doc_dir / "chunks.json"
        with open(chunks_path, "w") as f:
            json.dump(chunks, f, indent=2)

        # Generate and store embeddings
        embeddings_client = get_embeddings_client()
        embed_and_store_chunks(
            doc_id=doc_id,
            chunks=chunks,
            embeddings_client=embeddings_client,
            storage_dir=STORAGE_DIR,
        )
        logger.info(f"Stored embeddings for doc_id={doc_id}, num_chunks={len(chunks)}")

        # Clean up temp file
        temp_path.unlink()

        return UploadContextDocResponse(
            doc_id=doc_id,
            doc_hash=doc_hash,
            num_chars=num_chars,
            num_chunks=len(chunks),
            status="indexed",
            errors=None
        )

    except HTTPException:
        # Clean up on validation error
        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)
        raise
    except Exception as e:
        # Clean up on any error
        if doc_dir.exists():
            import shutil
            shutil.rmtree(doc_dir)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to process document: {e!s}"}
        )


def infer_column_type(series: pd.Series) -> str:
    """Infer a human-readable type for a pandas column."""
    dtype_str = str(series.dtype)
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    elif pd.api.types.is_float_dtype(series):
        return "float"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif dtype_str == "object":
        # Check if it looks like a date string
        sample = series.dropna().head(10)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise")
                return "datetime_string"
            except (ValueError, TypeError):
                pass
        return "string"
    else:
        return dtype_str


def compute_profile(df: pd.DataFrame) -> dict:
    """Compute a deterministic profile of the dataset."""
    profile = {"columns": {}}

    for col in sorted(df.columns):
        series = df[col]
        col_profile = {
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_profile["stats"] = {
                "count": int(desc.get("count", 0)),
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": round(float(desc.get("min", 0)), 4),
                "25%": round(float(desc.get("25%", 0)), 4),
                "50%": round(float(desc.get("50%", 0)), 4),
                "75%": round(float(desc.get("75%", 0)), 4),
                "max": round(float(desc.get("max", 0)), 4),
            }
        else:
            # Categorical: top 10 values
            value_counts = series.value_counts().head(10)
            col_profile["top_values"] = [
                {"value": str(val), "count": int(cnt)}
                for val, cnt in value_counts.items()
            ]

        profile["columns"][col] = col_profile

    return profile


@app.post("/upload_dataset", response_model=UploadDatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and profile a CSV dataset.

    - Computes sha256 hash
    - Stores data.csv, metadata.json, profile.json
    - Returns dataset info including column types
    """
    # Check file extension
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=415,
            detail={"error": "File must be a .csv (CSV file)"}
        )

    dataset_id = str(uuid.uuid4())
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Read file content
        content = await file.read()

        # Compute hash
        dataset_hash = hashlib.sha256(content).hexdigest()

        # Save raw CSV
        csv_path = dataset_dir / "data.csv"
        with open(csv_path, "wb") as f:
            f.write(content)

        # Parse with pandas
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail={"error": f"Failed to parse CSV: {e!s}"}
            )

        n_rows, n_cols = df.shape
        column_names = list(df.columns)

        # Infer types
        inferred_types = {col: infer_column_type(df[col]) for col in column_names}

        # Create metadata
        metadata = {
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "column_names": column_names,
            "inferred_types": inferred_types,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Save metadata
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Compute and save profile
        profile = compute_profile(df)
        profile_path = dataset_dir / "profile.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Uploaded dataset {dataset_id}: {n_rows} rows, {n_cols} cols")

        return UploadDatasetResponse(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            n_rows=n_rows,
            n_cols=n_cols,
            column_names=column_names,
            inferred_types=inferred_types,
            status="profiled",
            errors=None,
        )

    except HTTPException:
        # Clean up on validation error
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)
        raise
    except Exception as e:
        # Clean up on any error
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to process dataset: {e!s}"}
        )


@app.post("/ask", response_model=AskQuestionResponse)
async def ask_question(request: AskQuestionRequest):
    """
    Ask a question using the RAG agent.

    The agent will:
    1. Route the question (ANALYSIS, CAUSAL, REPORTING, etc.)
    2. Retrieve relevant context chunks
    3. Generate an answer using the LLM
    """
    # Validate doc_id exists
    doc_dir = STORAGE_DIR / request.doc_id
    if not doc_dir.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": f"Document not found: {request.doc_id}"}
        )

    embeddings_path = doc_dir / "embeddings.json"
    if not embeddings_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": f"Embeddings not found for document: {request.doc_id}"}
        )

    # Get clients
    embeddings_client = get_embeddings_client()
    llm_client = get_llm_client()

    logger.info(f"Processing question for doc_id={request.doc_id}: {request.question[:50]}...")

    # Run the agent
    try:
        final_state = run_agent(
            question=request.question,
            doc_id=request.doc_id,
            embeddings_client=embeddings_client,
            llm_client=llm_client,
            storage_dir=STORAGE_DIR,
            dataset_id=request.dataset_id,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Agent error: {e}"}
        )

    # Log trace events
    for event in final_state.get("trace_events", []):
        logger.info(f"TraceEvent: {event['event_type']} - {event['payload']}")

    # Convert artifacts to Pydantic models
    artifacts = []
    for art in final_state.get("artifacts", []):
        if art.get("type") == "text":
            artifacts.append(TextArtifact(content=art["content"]))
        elif art.get("type") == "checklist":
            artifacts.append(ChecklistArtifact(items=art["items"]))

    return AskQuestionResponse(
        answer_text=final_state.get("llm_response") or "No response generated.",
        router_decision=RouterDecision(
            route=final_state.get("route", "ANALYSIS"),
            confidence=final_state.get("route_confidence", 0.0),
            reasons=final_state.get("route_reasons", []),
        ),
        artifacts=artifacts,
        trace_id=final_state.get("trace_id", ""),
    )

