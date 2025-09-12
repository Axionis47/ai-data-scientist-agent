from __future__ import annotations
import json
import time
import re

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
    infer_format,
)
from ..reporting.report import reporting_expert
from ..platform.jobstore import JobStore  # type: ignore

# EDA sampling constant (keep consistent with main)
SAMPLE_TARGET_ROWS = 100_000


def run_pipeline(job_id: str, manifest: Dict[str, Any], resume_from: Optional[str] = None, job_store: Optional[JobStore] = None) -> None:
    """Run the E2E pipeline. Writes updates directly to JobStore.
    """
    if job_store is None:
        from ..platform.jobstore import get_job_store
        job_store = get_job_store()
    job = job_store.get(job_id) or {}

    def _update(patch: Dict[str, Any]) -> None:
        nonlocal job
        job.update(patch)
        try:
            job_store.update(job_id, patch)
        except Exception:
            pass

    def _hb() -> None:
        try:
            _update({"heartbeat_ts": time.time()})
        except Exception:
            pass

    _update({"status": "RUNNING"})
    _hb()

    # Determine starting stage
    from ..platform.statemachine import transition_stage
    if resume_from == "modeling":
        _update({"progress": max(int(job.get("progress", 0)), 60)})
        try:
            if job_store is not None:
                transition_stage(job_store, job_id, "modeling")
        except Exception:
            _update({"stage": "modeling"})
    else:
        _update({"progress": 10})
        try:
            if job_store is not None:
                transition_stage(job_store, job_id, "eda")
        except Exception:
            _update({"stage": "eda"})
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
    eda_done_path = job_dir / "eda.done"
    if eda_path.exists() and eda_done_path.exists():
        try:
            # Validate minimal EDA structure before skipping work
            eda_candidate = json.loads(eda_path.read_text())
            if isinstance(eda_candidate, dict) and "shape" in eda_candidate and "missing" in eda_candidate:
                eda = eda_candidate
                job.setdefault("messages", []).append({"role": "system", "content": "Reusing previous EDA."})
                eda_decision(job_id, "Reusing previous EDA artifact")
            else:
                eda = None
        except Exception:
            eda = None

    if eda is None:
        eda_start = time.time()
        # Write a marker once we begin EDA so future restarts can detect progress
        try:
            eda_done_path.touch(exist_ok=True)
        except Exception:
            pass

        t0 = time.time()
        try:
            # EDA timeout enforcement
            from ..core.config import EDA_TIMEOUT_S
            deadline = t0 + (EDA_TIMEOUT_S or 1e12)

            # If large CSV/TSV: chunked sampling for missingness and sample
            is_large = _is_large_file(dpath)
            if manifest.get("force_large") is True:
                is_large = True
            if (file_format or infer_format(dpath)) in {"csv", "tsv"} and is_large:
                sep = "\t" if (file_format or infer_format(dpath)) == "tsv" else (delimiter or ",")
                from ..eda.eda import load_sampled_chunked_csv  # local import to avoid cycles
                chunked = load_sampled_chunked_csv(dpath, sep, sample_target=SAMPLE_TARGET_ROWS)
                df = chunked["df"]
                eda = compute_eda(df)
                eda["missing"] = chunked["missing"]
                eda.setdefault("meta", {})["total_rows_est"] = chunked["total_rows"]
                eda.setdefault("meta", {})["sampled_rows"] = int(len(df))
            else:
                df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=50000)
                eda = compute_eda(df)
                eda.setdefault("meta", {})["sampled_rows"] = int(len(df))

            if time.time() > deadline:
                raise TimeoutError(f"EDA exceeded timeout {EDA_TIMEOUT_S}s")

            eda.setdefault("meta", {})["profile_ms"] = int((time.time() - t0) * 1000)
            try:
                from ..platform.statemachine import transition_stage
                dur_ms = int((time.time() - eda_start) * 1000)
                job.setdefault("timeline", {})
                job.setdefault("durations_ms", {})["eda"] = dur_ms
                job_store.update(job_id, {"durations_ms": job.get("durations_ms")})
            except Exception:
                pass
        except Exception as e:
            eda = {"error": f"EDA failed: {e}"}
            job.setdefault("messages", []).append({"role": "system", "content": str(e)})
        # Mark EDA done
        try:
            (job_dir / "eda.done").write_text("ok")
        except Exception:
            pass


        # Cancellation check before plotting
        cur = job_store.get(job_id) or {} if job_store is not None else {}
        if cur.get("cancel"):
            try:
                from ..platform.statemachine import transition_stage
                transition_stage(job_store, job_id, "cancelled", {"status": "CANCELLED", "progress": 100})
            except Exception:
                _update({"stage": "cancelled", "status": "CANCELLED", "progress": 100})
            return

        # Generate plots (sample to 5k rows for speed)
        try:
            if df is None:
                # Load small sample for plots if needed
                df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=10000)
            sdf = df.sample(n=min(len(df), 5000), random_state=42) if len(df) > 5000 else df
            # Use storage-backed plotting helper
            try:
                from ..platform.storage import get_storage
                from ..eda.eda import generate_basic_plots_storage
                storage = get_storage()
                plots_meta = generate_basic_plots_storage(storage, job_id, sdf)
                eda["plots"] = plots_meta
            except Exception:
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
        # Write preliminary result so /result is available early when modeling proceeds automatically
        try:
            (job_dir / "result.json").write_text(json.dumps({"phase": "eda-only", "eda": eda}, indent=2))
        except Exception:
            pass
    _update({"progress": 48})

    # Clarify if needed (target/metric suggestions)
    framing = manifest.get("framing", {})
    target = framing.get("target")

    # Try to infer target automatically before gating
    if not target:
        q_raw = (manifest.get("question") or "")
        q = q_raw.lower()
        cols = [str(c) for c in (eda.get("columns") or [])]
        # patterns like "on y" or "predict y" or "target=y"
        m = re.search(r"\b(on|predict)\s+([A-Za-z0-9_]+)\b", q_raw)
        cand = None
        if m and m.group(2) in cols:
            cand = m.group(2)
        if not cand and "target=" in q:
            m2 = re.search(r"target\s*=\s*([A-Za-z0-9_]+)", q_raw, re.I)
            if m2 and m2.group(1) in cols:
                cand = m2.group(1)
        if not cand and "y" in cols:
            cand = "y"
        if not cand:
            for k in ("target","label","response"):
                if k in cols:
                    cand = k; break
        if cand:
            framing["target"] = cand
            manifest["framing"] = framing
            # persist manifest update
            try:
                (JOBS_DIR / job_id / "manifest.json").write_text(json.dumps(manifest, indent=2))
            except Exception:
                pass
            target = cand

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
        # Suggest likely targets (binary-like or known names) with class balances
        cols = list(eda.get("columns") or [])
        likely_names = [c for c in cols if str(c).lower() in {"target", "label", "survived", "churn", "defaulted", "clicked", "y"}]
        top3 = []
        if df is None:
            try:
                df = load_dataframe(dpath, file_format, sheet_name, delimiter, sample_rows=20000)
            except Exception:
                df = None
        if df is not None:
            for c in likely_names:
                if c in df.columns:
                    vc = df[c].value_counts(dropna=False)
                    total = int(vc.sum()) or 1
                    bal = ", ".join(f"{str(k)}:{int(v)}" for k, v in vc.head(3).items())
                    top3.append(f"{c} [{bal}]")
        msg = "I need a target column to model. Candidates: " + (", ".join(top3) if top3 else ", ".join(likely_names) or "please specify") + ". Reply 'target=<column>'"
        job.setdefault("messages", []).append({"role": "assistant", "content": msg})
        try:
            if job_store is not None:
                from ..platform.statemachine import transition_stage
                transition_stage(job_store, job_id, "clarify")
        except Exception:
            _update({"stage": "clarify"})
        return  # Gate progression until /clarify

    # Modeling delegated to modular pipeline (lazy import to avoid import-time failures)
    # Record modeling duration
    try:
        mstart = start
        dur_ms = int((time.time() - mstart) * 1000)
        job.setdefault("durations_ms", {})["modeling"] = dur_ms
        job_store.update(job_id, {"durations_ms": job.get("durations_ms")})
    except Exception:
        pass

    _update({"progress": 60})
    try:
        if job_store is not None:
            from ..platform.statemachine import transition_stage
            transition_stage(job_store, job_id, "modeling")
    except Exception:
        _update({"stage": "modeling"})
    eda_decision(job_id, "Starting modeling via modular pipeline")

    # Cancellation check before heavy work
    if job_store is not None:
        cur = job_store.get(job_id) or {}
        if cur.get("cancel"):
            try:
                from ..platform.statemachine import transition_stage
                transition_stage(job_store, job_id, "cancelled", {"status": "CANCELLED", "progress": 100})
            except Exception:
                _update({"stage": "cancelled", "status": "CANCELLED", "progress": 100})
            return
    # Modeling resumability: if modeling.done and minimal modeling.json exists, we could skip heavy retrain
    modeling_done = (job_dir / "modeling.done")
    modeling_json = (job_dir / "modeling.json")
    if modeling_done.exists() and modeling_json.exists():
        try:
            m = json.loads(modeling_json.read_text())
            if isinstance(m, dict) and m.get("metrics"):
                modeling = m
                explain = m.get("explain", {})
        except Exception:
            pass


    # Mark modeling done
    try:
        (job_dir / "modeling.done").write_text("ok")
    except Exception:
        pass

    try:
        from ..modeling.pipeline import run_modeling  # lazy import
        from ..core.config import MODEL_TIMEOUT_S
        start = time.time()
        deadline = start + (MODEL_TIMEOUT_S or 1e12)

        # Pick a target heuristically if not provided
        framing = manifest.get("framing", {}) or {}
        # Apply safe auto-actions (feature-flagged)
        try:
            from ..modeling.auto_actions import apply_safe_actions
            framing = apply_safe_actions(job_id, eda or {}, modeling or {}, framing or {})
        except Exception:
            pass

        if not framing.get("target"):
            candidates = [c for c in (eda.get("columns") or []) if re.search(r"^(y|target|label)$", str(c), re.I)]
            target = candidates[0] if candidates else (eda.get("columns") or [None])[-1]
            if target:
                framing["target"] = target
        # Run modeling
        modeling_result = run_modeling(job_id, df, eda, framing)
        if time.time() > deadline:
            raise TimeoutError(f"Modeling exceeded timeout {MODEL_TIMEOUT_S}s")
        modeling = modeling_result
        explain = modeling_result.get("explain", {})
    except Exception as e:
        modeling = {"error": f"Modeling failed: {e}"}
        job.setdefault("messages", []).append({"role": "system", "content": f"Modeling error: {e}"})
        explain = {}

    _update({"stage": "report", "progress": 85})
    time.sleep(0.1)
    _update({"stage": "qa", "progress": 95})
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

    # Pydantic validation (best-effort) for modeling/result contracts
    try:
        from ..core.models import ModelingOutput, ResultPayload
        _ = ModelingOutput(**(modeling or {}))
    except Exception as e:
        model_decision(job_id, f"Modeling schema validation warning: {e}")

    result = {
        "eda": eda,
        "modeling": modeling if isinstance(locals().get("modeling"), dict) else {"note": "skipped"},
        "explain": explain if isinstance(locals().get("explain"), dict) else {},
        "qa": {"issues": []},
    }
    # Record report duration start
    rmark = time.time()
    if validation_error:
        result.setdefault("error", {})["eda"] = f"schema validation failed: {validation_error}"

    # Post-model critique (optional)
    try:
        from ..critique.moe import CRITIQUE_POST_MODEL, critique_post_model
        if CRITIQUE_POST_MODEL:
            crit = critique_post_model(job_id, eda, modeling, explain, manifest)
            (job_dir / "critique.json").write_text(json.dumps(crit, indent=2))
            result["critique"] = crit
            job.setdefault("messages", []).append({"role": "assistant", "content": f"Critique: {crit.get('summary','')}"})
    except Exception as e:
        job.setdefault("messages", []).append({"role": "system", "content": f"Critique failed: {e}"})

    # Validate final result payload (non-fatal)
    try:
        from ..core.models import ResultPayload
        _ = ResultPayload(**result)
    except Exception as e:
        model_decision(job_id, f"Result schema validation warning: {e}")

    # Cancellation check before reporting
    if job_store is not None:
        cur = job_store.get(job_id) or {}
        if cur.get("cancel"):
            try:
                from ..platform.statemachine import transition_stage
                transition_stage(job_store, job_id, "cancelled", {"status": "CANCELLED", "progress": 100})
            except Exception:
                _update({"stage": "cancelled", "status": "CANCELLED", "progress": 100})
            return

    # Validate existing report artifact before generating, if present
    try:
        html_ok = isinstance(result.get("report_html"), str) and len(result.get("report_html") or "") > 50
        if not html_ok and (JOBS_DIR / job_id / "report.html").exists():
            html_ok = (JOBS_DIR / job_id / "report.html").stat().st_size > 50
        if not html_ok:
            raise ValueError("report_html missing or too short")
    except Exception as e:
        job.setdefault("messages", []).append({"role": "system", "content": f"Report validation: {e}"})

    # Enrich telemetry with selected tools and warnings before report
    try:
        from ..core.telemetry import append_run_telemetry
        sel = (modeling or {}).get("selected_tools")
        feats = (modeling or {}).get("features")
        warns = []
        if isinstance(eda, dict) and eda.get("warnings"):
            warns.extend(eda.get("warnings"))
        append_run_telemetry(job_id, {
            "selected_tools": sel,
            "features": feats,
            "warnings": warns[:10],
        })
    except Exception:
        pass

    # Generate report HTML (OpenAI or fallback)
    try:
        from ..core.config import REPORT_TIMEOUT_S
        rstart = time.time()
        report_html = reporting_expert(job_id, eda, modeling, explain)
        if REPORT_TIMEOUT_S and (time.time() - rstart) > REPORT_TIMEOUT_S:
            raise TimeoutError(f"Report exceeded timeout {REPORT_TIMEOUT_S}s")
        result["report_html"] = report_html
        # Record report duration and mark done
        try:
            dur_ms = int((time.time() - rmark) * 1000)
            job.setdefault("durations_ms", {})["report"] = dur_ms
            job_store.update(job_id, {"durations_ms": job.get("durations_ms")})
            (job_dir / "report.done").write_text("ok")
        except Exception:
            pass
    except Exception as e:
        model_decision(job_id, f"Reporting generation failed: {e}")

    # Append telemetry
    try:
        from ..core.telemetry import append_run_telemetry
        append_run_telemetry(job_id, {
            "durations_ms": job.get("durations_ms"),
            "status": "FAILED" if validation_error else "COMPLETED",
            "timings": result.get("timings"),
            "eda_error": result.get("error", {}).get("eda"),
        })
    except Exception:
        pass

    # Include simple stage timings if available
    try:
        timeline = None
        if job_store is not None:
            timeline = (job_store.get(job_id) or {}).get("timeline")
        if not timeline:
            timeline = job.get("timeline")
        if isinstance(timeline, dict) and timeline:
            result.setdefault("timings", {})["stage_starts"] = timeline
    except Exception:
        pass

    result["phase"] = "full"
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))
    # Final status: mark as FAILED if validation_error, else COMPLETED
    if validation_error:
        _update({"status": "FAILED", "progress": 100, "stage": "error"})
    else:
        _update({"status": "COMPLETED", "progress": 100, "stage": "done"})


def _is_large_file(path: Path) -> bool:
    try:
        return path.stat().st_size > LARGE_FILE_MB * 1024 * 1024
    except Exception:
        return False

