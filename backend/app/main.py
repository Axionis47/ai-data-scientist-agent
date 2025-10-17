import json
import os
import re
import shutil
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Env for OpenAI optional
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# --- Logging moved to core.logs ---
from .core.logs import (
    log,
    setup_json_logging,
)
# Initialize JSON structured logging once
setup_json_logging()

# --- Config moved to core.config ---
from .core.config import (
    JOBS_DIR,
    MAX_UPLOAD_MB,
    ALLOWED_EXTS,
)

# Modularized imports

# Lazy-import run_modeling within _run_pipeline to avoid import-time dependency failures
from .eda.eda import (
    infer_format,
    detect_delimiter,
)
from .core.clarify import apply_clarification

# EDA sampling constant
SAMPLE_TARGET_ROWS = 100_000

# --- App ---
app = FastAPI(title="AI Data Scientist Agent", version="0.1.0")

# CORS configuration - allow localhost and Cloud Run frontend
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles

app.mount("/static/jobs", StaticFiles(directory=str(JOBS_DIR)), name="jobs-static")

# Guard to prevent serving original/ files by default
from fastapi import Request
from fastapi.responses import Response
from .core.config import STATIC_EXPOSE_ORIGINAL


@app.middleware("http")
async def block_original_static(request: Request, call_next):
    try:
        url = request.url.path or ""
        if (
            (not STATIC_EXPOSE_ORIGINAL)
            and url.startswith("/static/jobs/")
            and "/original/" in url
        ):
            return Response("Forbidden", status_code=403)
    except Exception:
        pass
    return await call_next(request)


# Expose _is_large_file for tests (wrapper around pipeline helper)
from .pipeline.run import _is_large_file as _pipeline_is_large
from pathlib import Path as _Path


def _is_large_file(p: _Path) -> bool:  # pragma: no cover - thin wrapper for tests
    try:
        return _pipeline_is_large(p)
    except Exception:
        return False


# --- Models ---
class UploadResponse(BaseModel):
    job_id: str
    dataset_path: str
    file_format: str
    sheet_names: Optional[List[str]] = None


class AnalyzeRequest(BaseModel):
    job_id: Optional[str] = None
    dataset_path: str
    file_format: Optional[str] = None
    nl_description: str = Field(..., description="Business / domain context")
    question: str = Field(..., description="User question / task")
    sheet_name: Optional[str] = None
    delimiter: Optional[str] = None
    profile: Optional[str] = Field(default=None, description="Run profile: 'lean' or 'full'")


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)


class ClarifyRequest(BaseModel):
    job_id: str
    message: str


# Minimal EDA schema for validation (fields optional to be resilient)
class EDAOutput(BaseModel):
    schema_version: Optional[str] = None
    shape: Dict[str, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing: Dict[str, Dict[str, float]]
    nunique: Optional[Dict[str, int]] = None
    id_candidates: Optional[List[str]] = None
    constant_columns: Optional[List[str]] = None
    numeric_stats: Optional[Dict[str, Dict[str, float]]] = None
    skew: Optional[Dict[str, float]] = None
    kurtosis: Optional[Dict[str, float]] = None
    outlier_share: Optional[Dict[str, float]] = None
    top_correlations: Optional[List[Dict[str, float]]] = None
    categorical_topk: Optional[Dict[str, List[Dict[str, Any]]]] = None
    rare_levels: Optional[Dict[str, List[Dict[str, Any]]]] = None
    time_columns: Optional[List[str]] = None
    time_like_candidates: Optional[List[str]] = None
    text_stats: Optional[Dict[str, Dict[str, float]]] = None
    memory_usage_bytes: Optional[int] = None
    head: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    timeseries: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, Any]] = None
    target_relations: Optional[Dict[str, Any]] = None


# --- Job store (thread-safe, replaceable) ---
from .platform.jobstore import get_job_store

JOB_STORE = get_job_store()


# --- Utils ---
class EDAValidationError(Exception):
    pass


def _safe_filename(name: str) -> str:
    base = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Data loading & EDA helpers moved to app.eda.eda ---

# --- Transformers moved to app.modeling.transformers ---
# All EDA, router, and reporting helpers have been modularized into app.eda.eda, app.agent.router, and app.reporting.report.
# The inlined duplicates are removed in favor of imports at the top of this file.

# --- Clarify helpers ---
# Use centralized implementation in app.core.clarify (imported above)

# --- Pipeline ---
from .pipeline.run import run_pipeline as _run_pipeline


# --- Core endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    # Create job
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    orig_dir = job_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)

    # Validate extension before save
    safe_name = _safe_filename(file.filename or "dataset")
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, detail=f"Unsupported file type: {ext}")

    # Save file safely
    dst = orig_dir / safe_name
    try:
        with dst.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save file: {e}")

    # Enforce size limit after write (covers missing content-length)
    try:
        size = dst.stat().st_size
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            dst.unlink(missing_ok=True)
            raise HTTPException(413, detail="File too large")
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"Could not stat uploaded file: {e}")

    file_format = infer_format(dst)

    sheet_names: Optional[List[str]] = None
    if file_format == "excel":
        try:
            xl = pd.ExcelFile(dst)
            sheet_names = xl.sheet_names
        except Exception:
            sheet_names = None

    # Build minimal job record
    JOB_STORE.create(
        job_id,
        {
            "status": "UPLOADED",
            "progress": 5,
            "stage": "ingest",
            "messages": [{"role": "system", "content": "File uploaded."}],
            "dataset_path": str(dst),
            "file_format": file_format,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return UploadResponse(
        job_id=job_id,
        dataset_path=str(dst),
        file_format=file_format,
        sheet_names=sheet_names,
    )


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    # Determine job
    job_id = body.job_id or uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Initialize job record if new
    job = JOB_STORE.get(job_id)
    if not job:
        JOB_STORE.create(
            job_id,
            {
                "status": "CREATED",
                "progress": 0,
                "stage": "ingest",
                "messages": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        job = JOB_STORE.get(job_id) or {}

    # Persist manifest
    manifest = {
        "job_id": job_id,
        "question": body.question,
        "context": body.nl_description,
        "dataset_path": body.dataset_path,
        "file_format": body.file_format,
        "sheet_name": body.sheet_name,
        "delimiter": body.delimiter,
        "profile": (body.profile or None),
        "created_at": job.get("created_at"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Enrich manifest with file info (restrict path to jobs directory)
    dpath = Path(body.dataset_path)
    try:
        dpath_resolved = dpath.resolve()
        jobs_resolved = JOBS_DIR.resolve()
        if not str(dpath_resolved).startswith(str(jobs_resolved)):
            raise HTTPException(400, detail="dataset_path must be under jobs directory")
    except Exception:
        # If resolution fails, treat as invalid
        raise HTTPException(400, detail="invalid dataset_path")
    if not dpath.exists():
        raise HTTPException(400, detail="dataset_path does not exist")
    try:
        sha256 = _sha256_of_file(dpath)
    except Exception:
        sha256 = None
    manifest.update(
        {
            "source_file": os.path.basename(str(dpath)),
            "sha256": sha256,
            "size_bytes": dpath.stat().st_size,
        }
    )
    # Hint for large-file path; allows tests to monkeypatch _is_large_file via main
    try:
        manifest["force_large"] = bool(_is_large_file(dpath))
    except Exception:
        manifest["force_large"] = False

    # Quick preview for schema inference
    preview: Dict[str, Any] = {}
    try:
        fmt = body.file_format or infer_format(dpath)
        if fmt == "excel":
            # Limit rows for preview to avoid heavy loads
            df = pd.read_excel(dpath, sheet_name=body.sheet_name, nrows=5000)
        else:
            if fmt == "tsv":
                sep = "\t"
            elif fmt == "csv":
                with dpath.open("r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(5000)
                sep = body.delimiter or detect_delimiter(sample)
            else:
                sep = ","
            df = pd.read_csv(dpath, sep=sep, nrows=5000)
        preview = {
            "columns": [
                {"name": c, "dtype": str(t)} for c, t in zip(df.columns, df.dtypes)
            ],
            "head": df.head(5).to_dict(orient="records"),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
    except Exception as e:
        job.setdefault("messages", []).append(
            {"role": "system", "content": f"Preview failed: {e}"}
        )

    manifest["preview"] = preview

    # Save manifest
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Kick off pipeline via queue runner and return immediately
    JOB_STORE.update(job_id, {"status": "QUEUED", "stage": "eda", "progress": 10})

    # Use background queue with concurrency limits
    from .platform.queue_runner import get_default_queue_runner

    def handler(payload: Dict[str, Any]) -> None:
        _run_pipeline(payload["job_id"], payload["manifest"], None, JOB_STORE)

    qr = get_default_queue_runner(handler)
    qr.enqueue({"job_id": job_id, "manifest": manifest})
    return {"job_id": job_id}


@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    JOB_STORE.update(job_id, {"cancel": True})
    return {"ok": True}


@app.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    return StatusResponse(
        job_id=job_id,
        status=job.get("status", "UNKNOWN"),
        progress=int(job.get("progress", 0)),
        stage=job.get("stage"),
        messages=job.get("messages", []),
    )


@app.get("/health")
def health():
    return {"ok": True}



@app.get("/openai-smoke")
def openai_smoke(live: bool = False):
    """Lightweight OpenAI readiness probe.

    - live=false (default): only checks that the SDK is importable and a key is present.
    - live=true: performs a tiny chat completion against gpt-4o-mini to confirm network access.
    """
    import os
    try:
        from openai import OpenAI  # type: ignore
        sdk_installed = True
    except Exception:
        OpenAI = None  # type: ignore
        sdk_installed = False

    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if not live:
        return {"ok": bool(sdk_installed and has_key), "sdk_installed": sdk_installed, "has_key": has_key, "live": False}

    if not sdk_installed:
        return {"ok": False, "error": "sdk_not_installed", "live": True}
    if not has_key:
        return {"ok": False, "error": "no_key", "live": True}

    try:
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            temperature=0.0,
            max_tokens=3,
        )
        txt = (r.choices[0].message.content or "").strip()
        return {"ok": txt.upper().startswith("OK"), "reply": txt, "live": True}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "live": True}


@app.post("/sample")
def sample():
    """Create a job with a small built-in sample CSV and start analysis.
    Useful for demos; avoids unsafe arbitrary local paths.
    """
    import pandas as pd

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    orig_dir = job_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)
    # Build a tiny Titanic-like dataset
    df = pd.DataFrame(
        {
            "Pclass": [3, 1, 3, 1, 3, 2, 3, 1, 2, 3],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "female",
                "male",
                "female",
                "male",
            ],
            "Age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 20],
            "Fare": [
                7.25,
                71.2833,
                7.925,
                53.1,
                8.05,
                51.8625,
                21.075,
                11.1333,
                30.0708,
                8.4583,
            ],
            "Survived": [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    sample_path = orig_dir / "titanic_tiny.csv"
    df.to_csv(sample_path, index=False)
    # Initialize job record
    JOB_STORE.create(
        job_id,
        {
            "status": "UPLOADED",
            "progress": 5,
            "stage": "ingest",
            "messages": [{"role": "system", "content": "Sample dataset prepared."}],
            "dataset_path": str(sample_path),
            "file_format": "csv",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # Enqueue analysis
    manifest = {
        "job_id": job_id,
        "dataset_path": str(sample_path),
        "file_format": "csv",
        "nl_description": "Demo Titanic tiny sample",
        "question": "Classify which passengers survived. target=Survived",
        "sheet_name": None,
        "delimiter": ",",
    }
    # Persist manifest
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (job_dir / "manifest.done").write_text("ok")

    def handler(payload: Dict[str, Any]) -> None:
        _run_pipeline(payload["job_id"], payload["manifest"], None, JOB_STORE)

    from .platform.queue_runner import get_default_queue_runner

    qr = get_default_queue_runner(handler)
    qr.enqueue({"job_id": job_id, "manifest": manifest})
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def result(job_id: str):
    job_dir = JOBS_DIR / job_id
    p = job_dir / "result.json"
    if not p.exists():
        raise HTTPException(404, detail="result not ready")

    def _json_safe(x):
        import math
        try:
            import numpy as np  # type: ignore
        except Exception:  # pragma: no cover
            np = None  # type: ignore
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        if np is not None and isinstance(x, (np.floating, np.integer)):
            x = x.item()
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        return x

    try:
        data = json.loads(p.read_text())
        # Optional validation at API boundary (non-fatal)
        try:
            from .core.models import ResultPayload
            _ = ResultPayload(**data)
        except Exception:
            pass
        return JSONResponse(content=_json_safe(data))
    except Exception as e:
        raise HTTPException(500, detail=f"failed to load result: {e}")


@app.post("/clarify")
def clarify(body: ClarifyRequest):
    job = JOB_STORE.get(body.job_id)
    if not job:
        raise HTTPException(404, detail="job not found")
    job.setdefault("messages", []).append({"role": "user", "content": body.message})
    # Load manifest
    job_dir = JOBS_DIR / body.job_id
    manifest_p = job_dir / "manifest.json"
    if manifest_p.exists():
        manifest = json.loads(manifest_p.read_text())
    else:
        manifest = {}
    apply_clarification(body.job_id, body.message, manifest)
    # If pipeline was waiting on clarify, resume from modeling stage
    if job.get("stage") == "clarify" and job.get("status") == "RUNNING":
        job.setdefault("messages", []).append(
            {"role": "assistant", "content": "Thanks, resuming."}
        )
        from .platform.queue_runner import get_default_queue_runner

        def handler(payload: Dict[str, Any]) -> None:
            _run_pipeline(payload["job_id"], payload["manifest"], "modeling", JOB_STORE)

        qr = get_default_queue_runner(handler)
        qr.enqueue({"job_id": body.job_id, "manifest": manifest})
    else:
        job.setdefault("messages", []).append(
            {"role": "assistant", "content": "Clarification noted."}
        )
    return {"ok": True}


# --- Entrypoint ---
def run():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
