from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from pandas.api import types as ptypes

from ..core.logs import eda_decision, model_decision
from ..core.config import JOBS_DIR, LARGE_FILE_MB
from ..eda.eda import (
    compute_eda,
    compute_target_relations,
    compute_timeseries_hints,
    load_dataframe,
    load_sampled_chunked_csv,
    infer_format,
)
from ..reporting.report import reporting_expert

# EDA sampling constant (keep consistent with main)
SAMPLE_TARGET_ROWS = 100_000


def run_pipeline(JOBS: Dict[str, Dict[str, Any]], job_id: str, manifest: Dict[str, Any], resume_from: Optional[str] = None) -> None:
    """Run the E2E pipeline. Mutates JOBS[job_id] and writes artifacts under the job dir.
    This function mirrors the previous _run_pipeline logic from main.py.
    """
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

    eda: Dict[str, Any] | None = None
    df: pd.DataFrame | None = None
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
            if (file_format or infer_format(dpath)) in {"csv", "tsv"} and _is_large_file(dpath):
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
                from ..eda.eda import generate_basic_plots as _gen_plots

                plots_meta = _gen_plots(job_id, sdf, plots_dir)
                eda["plots"] = plots_meta
            except Exception:
                # fallback inline plots if needed (skip to keep modular)
                pass
        except Exception as e:
            job.setdefault("messages", []).append({"role": "system", "content": f"Plotting failed: {e}"})

        # Target-aware relations (if target present)
        try:
            framing = manifest.get("framing", {})
            tcol = framing.get("target")
            if tcol and (df is not None) and tcol in df.columns:
                eda["target_relations"] = compute_target_relations(df, tcol)
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
                        metric_col = c
                        break
                eda.setdefault("timeseries", {}).update(compute_timeseries_hints(df, time_col, metric_col))
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
            job.setdefault("messages", []).append({
                "role": "assistant",
                "content": "Severe class imbalance detected. Consider metric=F1 and class_weight=balanced. Reply 'metric=f1' to confirm.",
            })
    if implies_modeling and not target:
        # Suggest likely targets (binary-like or known names)
        likely = [c for c in (eda.get("columns") or []) if str(c).lower() in {"target", "label", "survived", "churn", "defaulted", "clicked"}]
        job.setdefault("messages", []).append({
            "role": "assistant",
            "content": f"I need a target column to model. Candidates: {', '.join(likely) or 'please specify'}. Reply 'target=<column>'",
        })
        job.update({"stage": "clarify"})
        return  # Gate progression until /clarify

    # Modeling delegated to modular pipeline (lazy import to avoid import-time failures)
    job.update({"stage": "modeling", "progress": 60})
    eda_decision(job_id, "Starting modeling via modular pipeline")
    try:
        from ..modeling.pipeline import run_modeling  # lazy import

        modeling_result = run_modeling(job_id, df, eda, manifest.get("framing", {}) or {})
        modeling = modeling_result
        explain = modeling_result.get("explain", {})
    except Exception as e:
        modeling = {"error": f"Modeling failed: {e}"}
        job.setdefault("messages", []).append({"role": "system", "content": f"Modeling error: {e}"})
        explain = {}

    job.update({"stage": "report", "progress": 85})
    time.sleep(0.1)
    job.update({"stage": "qa", "progress": 95})
    time.sleep(0.1)

    # Validate and build a result payload including EDA + modeling
    validation_error = None
    try:
        # Minimal EDA schema validation (same as main's EDAOutput)
        # Avoid importing Pydantic model here to keep module decoupled
        assert isinstance(eda, dict) and "shape" in eda and "missing" in eda
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


def _is_large_file(path: Path) -> bool:
    try:
        return path.stat().st_size > LARGE_FILE_MB * 1024 * 1024
    except Exception:
        return False

