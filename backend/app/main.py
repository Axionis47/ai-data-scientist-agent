import json
import os
import re
import shutil
import hashlib
import time
import uuid
import logging
import threading
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
from .core.logs import log, eda_logger, model_logger, eda_decision, model_decision

# --- Config moved to core.config ---
from .core.config import (
    ROOT, DATA_DIR, JOBS_DIR, MAX_UPLOAD_MB, ALLOWED_EXTS, LARGE_FILE_MB,
    BLEND_DELTA, EARLY_STOP_SAMPLE, HGB_MIN_ROWS, CALIBRATE_ENABLED, CV_FOLDS,
    PDP_TOP_NUM, SEARCH_TIME_BUDGET, SHAP_ENABLED, SHAP_MAX_ROWS,
    REPORT_PRIMARY, REPORT_ACCENT, REPORT_BG, REPORT_SURFACE, REPORT_TEXT, REPORT_MUTED,
    REPORT_OK, REPORT_WARN, REPORT_ERROR, REPORT_FONT_FAMILY, REPORT_LOGO_URL
)

# Modularized imports
from .agent.router import build_context_pack, plan_with_router
from .reporting.report import reporting_expert
# Lazy-import run_modeling within _run_pipeline to avoid import-time dependency failures
from .eda.eda import (
    compute_eda, compute_target_relations, compute_timeseries_hints,
    load_dataframe, load_sampled_chunked_csv, infer_format, detect_delimiter
)
from .core.clarify import apply_clarification

# EDA sampling constant
SAMPLE_TARGET_ROWS = 100_000

# --- App ---
app = FastAPI(title="AI Data Scientist Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.staticfiles import StaticFiles
app.mount("/static/jobs", StaticFiles(directory=str(JOBS_DIR)), name="jobs-static")

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

# --- In-memory job store (replaceable) ---
JOBS: Dict[str, Dict[str, Any]] = {}

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
from pandas.api import types as ptypes
from collections import Counter

# --- Transformers moved to app.modeling.transformers ---
# All EDA, router, and reporting helpers have been modularized into app.eda.eda, app.agent.router, and app.reporting.report.
# The inlined duplicates are removed in favor of imports at the top of this file.

# --- Clarify helpers ---
_target_re = re.compile(r"target\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_metric_re = re.compile(r"metric\s*=\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)


def apply_clarification(job_id: str, text: str, manifest: Dict[str, Any]):
    target_m = _target_re.search(text or "")
    if target_m:
        manifest.setdefault("framing", {})["target"] = target_m.group(1)
    metric_m = _metric_re.search(text or "")
    if metric_m:
        manifest.setdefault("framing", {})["metric"] = metric_m.group(1)
    # persist manifest change
    job_dir = JOBS_DIR / job_id
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

# --- Pipeline ---
def _run_pipeline(job_id: str, manifest: Dict[str, Any], resume_from: Optional[str] = None):
    job = JOBS[job_id]
    job.update({"status": "RUNNING"})
    # Determine starting stage
    if resume_from == "modeling":
        job.update({"stage": "modeling", "progress": max(int(job.get("progress", 0)), 60)})
    else:
        job.update({"stage": "eda", "progress": 10})
        job.setdefault("messages", []).append({"role": "assistant", "content": "Starting EDA."})
        eda_decision(job_id, "Starting EDA phase")
    # Prepare dirs and attempt to reuse prior EDA if present
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    eda_path = job_dir / "eda.json"

    # Load a sample (or full if small) for fast EDA
    dpath = Path(manifest["dataset_path"])  # assume exists
    file_format = manifest.get("file_format")
    sheet_name = manifest.get("sheet_name")
    delimiter = manifest.get("delimiter")

    eda = None
    df = None
    if eda_path.exists():
        try:
            eda = json.loads(eda_path.read_text())
            job.setdefault("messages", []).append({"role": "system", "content": "Reusing previous EDA."})
            eda_decision(job_id, "Reusing previous EDA artifact")
        except Exception:
            eda = None

    if eda is None:
        t0 = time.time()
        try:
            # If large CSV/TSV: chunked sampling for missingness and sample
            from .eda.eda import infer_format, is_large_file, load_sampled_chunked_csv, load_dataframe
            if (file_format or infer_format(dpath)) in {"csv","tsv"} and is_large_file(dpath, LARGE_FILE_MB):
                sep = "\t" if (file_format or infer_format(dpath)) == "tsv" else (delimiter or ",")
                chunked = load_sampled_chunked_csv(dpath, sep, sample_target=SAMPLE_TARGET_ROWS)
                df = chunked["df"]
                eda = compute_eda(df)
                # overwrite missing with accurate chunked missingness
                eda["missing"] = chunked["missing"]
                eda.setdefault("meta", {})["total_rows_est"] = chunked["total_rows"]
                eda.setdefault("meta", {})["sampled_rows"] = int(len(df))
            else:
                # Up to 50k rows for EDA to keep it responsive
                df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=50000)
                eda = compute_eda(df)
                eda.setdefault("meta", {})["sampled_rows"] = int(len(df))
            eda.setdefault("meta", {})["profile_ms"] = int((time.time() - t0) * 1000)
        except Exception as e:
            eda = {"error": f"EDA failed: {e}"}
            job.setdefault("messages", []).append({"role": "system", "content": str(e)})

        # Generate plots (sample to 5k rows for speed)
        plots_dir = job_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            if df is None:
                # Load small sample for plots if needed
                df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=10000)
            sdf = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df
            # Use shared plotting helper
            try:
                from .eda.eda import generate_basic_plots as _gen_plots
                plots_meta = _gen_plots(job_id, sdf, plots_dir)
                eda["plots"] = plots_meta
            except Exception:
                # fallback inline plots if needed (skip to keep modular)
                pass
        except Exception as e:
            job.setdefault("messages", []).append({"role": "system", "content": f"Plotting failed: {e}"})

        # Target-aware relations (if target present)
        try:
            from .eda.eda import compute_target_relations as _ctr, compute_timeseries_hints as _cth
            framing = manifest.get("framing", {})
            tcol = framing.get("target")
            if tcol and (df is not None) and tcol in df.columns:
                eda["target_relations"] = _ctr(df, tcol)
        except Exception as e:
            job.setdefault("messages", []).append({"role": "system", "content": f"target relations failed: {e}"})
        # Time-series hints (if time column present)
        try:
            time_col = None
            if eda.get("time_columns"):
                time_col = eda["time_columns"][0]
            elif eda.get("time_like_candidates"):
                time_col = eda["time_like_candidates"][0]
            if time_col:
                # choose metric: first numeric column
                metric_col = None
                for c in df.columns if df is not None else []:
                    if c != time_col and ptypes.is_numeric_dtype(df[c]):
                        metric_col = c; break
                eda.setdefault("timeseries", {}).update(_cth(df, time_col, metric_col))
        except Exception as e:
            job.setdefault("messages", []).append({"role": "system", "content": f"timeseries hints failed: {e}"})

        # Save EDA artifact
        eda_path.write_text(json.dumps(eda, indent=2))
    job.update({"progress": 48})

    # Clarify if needed (target/metric suggestions)
    framing = manifest.get("framing", {})
    target = framing.get("target")
    # Heuristic: ask for target if question implies classification/regression and target missing
    q = (manifest.get("question") or "").lower()
    implies_modeling = any(k in q for k in ["classify", "predict", "regression", "model"])
    if target and df is None:
        # Ensure df exists for balance check
        try:
            df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=20000)
        except Exception:
            df = None
    if target and (df is not None) and target in df.columns:
        vc = df[target].value_counts(dropna=False)
        total = int(vc.sum()) or 1
        ratio = max(vc) / total
        if ratio < 0.1 or ratio > 0.9:
            job.setdefault("messages", []).append({"role": "assistant", "content": "Severe class imbalance detected. Consider metric=F1 and class_weight=balanced. Reply 'metric=f1' to confirm."})
    if implies_modeling and not target:
        # Suggest likely targets (binary-like or known names)
        likely = [c for c in (eda.get("columns") or []) if str(c).lower() in {"target","label","survived","churn","defaulted","clicked"}]
        job.setdefault("messages", []).append({"role": "assistant", "content": f"I need a target column to model. Candidates: {', '.join(likely) or 'please specify'}. Reply 'target=<column>'"})
        job.update({"stage": "clarify"})
        return  # Gate progression until /clarify

    # Modeling delegated to modular pipeline (lazy import to avoid import-time failures)
    job.update({"stage": "modeling", "progress": 60})
    eda_decision(job_id, "Starting modeling via modular pipeline")
    try:
        from .modeling.pipeline import run_modeling  # lazy import
        modeling_result = run_modeling(job_id, df, eda, manifest.get("framing", {}) or {})
        modeling = modeling_result
        explain = modeling_result.get("explain", {})
    except Exception as e:
        modeling = {"error": f"Modeling failed: {e}"}
        job.setdefault("messages", []).append({"role": "system", "content": f"Modeling error: {e}"})


    job.update({"stage": "report", "progress": 85})
    time.sleep(0.1)
    job.update({"stage": "qa", "progress": 95})
    time.sleep(0.1)

    # Validate and build a result payload including EDA + modeling
    validation_error = None
    try:
        _ = EDAOutput(**eda)
    except Exception as ve:
        validation_error = str(ve)
        model_decision(job_id, f"EDA schema validation failed: {ve}")
    result = {
        "eda": eda,
        "modeling": modeling if isinstance(locals().get("modeling"), dict) else {"note": "skipped"},
        "explain": explain if isinstance(locals().get("explain"), dict) else {},
        "qa": {"issues": []},
    }
    if validation_error:
        result.setdefault("error", {})["eda"] = f"schema validation failed: {validation_error}"
    # Generate report HTML (OpenAI or fallback)
    try:
        report_html = reporting_expert(job_id, eda, modeling, explain)
        result["report_html"] = report_html
    except Exception as e:
        model_decision(job_id, f"Reporting generation failed: {e}")
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))
    # Final status: mark as FAILED if validation_error, else COMPLETED
    if validation_error:
        job.update({"status": "FAILED", "progress": 100, "stage": "error"})
    else:
        job.update({"status": "COMPLETED", "progress": 100, "stage": "done"})

# --- Core endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    # Create job
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    orig_dir = job_dir / "original"
    orig_dir.mkdir(parents=True, exist_ok=True)

    # Save file safely
    safe_name = _safe_filename(file.filename or "dataset")
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
    JOBS[job_id] = {
        "status": "UPLOADED",
        "progress": 5,
        "stage": "ingest",
        "messages": [{"role": "system", "content": "File uploaded."}],
        "dataset_path": str(dst),
        "file_format": file_format,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return UploadResponse(job_id=job_id, dataset_path=str(dst), file_format=file_format, sheet_names=sheet_names)

@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    # Determine job
    job_id = body.job_id or uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Initialize job record if new
    job = JOBS.setdefault(job_id, {
        "status": "CREATED",
        "progress": 0,
        "stage": "ingest",
        "messages": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    # Persist manifest
    manifest = {
        "job_id": job_id,
        "question": body.question,
        "context": body.nl_description,
        "dataset_path": body.dataset_path,
        "file_format": body.file_format,
        "sheet_name": body.sheet_name,
        "delimiter": body.delimiter,
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
    manifest.update({
        "source_file": os.path.basename(str(dpath)),
        "sha256": sha256,
        "size_bytes": dpath.stat().st_size,
    })

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
            "columns": [{"name": c, "dtype": str(t)} for c, t in zip(df.columns, df.dtypes)],
            "head": df.head(5).to_dict(orient="records"),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
    except Exception as e:
        job.setdefault("messages", []).append({"role": "system", "content": f"Preview failed: {e}"})

    manifest["preview"] = preview

    # Save manifest
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Kick off pipeline in background and return immediately
    job.update({"status": "QUEUED", "stage": "eda", "progress": 10})
    t = threading.Thread(target=_run_pipeline, args=(job_id, manifest), daemon=True)
    t.start()

    return {"job_id": job_id}

@app.get("/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):
    job = JOBS.get(job_id)
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


@app.get("/result/{job_id}")
def result(job_id: str):
    job_dir = JOBS_DIR / job_id
    p = job_dir / "result.json"
    if not p.exists():
        raise HTTPException(404, detail="result not ready")
    try:
        return JSONResponse(content=json.loads(p.read_text()))
    except Exception as e:
        raise HTTPException(500, detail=f"failed to load result: {e}")

@app.post("/clarify")
def clarify(body: ClarifyRequest):
    job = JOBS.get(body.job_id)
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
        job.setdefault("messages", []).append({"role": "assistant", "content": "Thanks, resuming."})
        threading.Thread(target=_run_pipeline, args=(body.job_id, manifest, "modeling"), daemon=True).start()
    else:
        job.setdefault("messages", []).append({"role": "assistant", "content": "Clarification noted."})
    return {"ok": True}

# --- Entrypoint ---
def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()

