from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

import pandas as pd
import numpy as np
import pandas.api.types as ptypes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Defaults used by plotting helpers
PLOTS_PER_NUMERIC = 6
PLOTS_PER_CATEG = 6


def infer_format(path: Path) -> str:
    """Infer file format by extension and simple content sniffing.

    Args:
        path: File path to inspect.

    Returns:
        One of "csv", "tsv", or "excel".
    """
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".tsv":
        return "tsv"
    if ext in {".xlsx", ".xls"}:
        return "excel"
    try:
        with path.open("rb") as f:
            head = f.read(4096)
            if b"\t" in head and head.count(b"\t") > head.count(b",") * 1.5:
                return "tsv"
            return "csv"
    except Exception:
        return "csv"


def detect_delimiter(sample: str) -> str:
    """Heuristically choose a delimiter among [tab, comma, semicolon, pipe].
    Uses per-line field counts and chooses the delimiter with the highest median fields (>=2).
    Falls back to comma.
    """
    candidates = ["\t", ",", ";", "|"]
    lines = [ln for ln in sample.splitlines() if ln.strip()][:20] or [sample]
    best = (",", 1.0)
    for delim in candidates:
        try:
            fields_per_line = [len(ln.split(delim)) for ln in lines]
            # Ignore delimiters that almost never split
            if max(fields_per_line or [1]) <= 1:
                continue
            sorted_counts = sorted(fields_per_line)
            mid = len(sorted_counts) // 2
            median = (sorted_counts[mid] if len(sorted_counts) % 2 == 1 else (sorted_counts[mid - 1] + sorted_counts[mid]) / 2.0)
            if median > best[1]:
                best = (delim, float(median))
        except Exception:
            continue
    return best[0]


def is_large_file(path: Path, large_file_mb: int) -> bool:
    """Return True if file size > large_file_mb.

    Robust to stat() errors.
    """
    try:
        return path.stat().st_size > large_file_mb * 1024 * 1024
    except Exception:
        return False


def load_sampled_chunked_csv(path: Path, delimiter: str, sample_target: int) -> Dict[str, Any]:
    """Chunked reader: returns a sample dataframe, accurate missingness, and total rows."""
    chunks: List[pd.DataFrame] = []
    missing_sums: Dict[str, int] = {}
    total_rows = 0
    sample_remaining = sample_target
    for chunk in pd.read_csv(path, sep=delimiter, chunksize=20_000):
        total_rows += len(chunk)
        # missing
        miss = chunk.isna().sum()
        for c, v in miss.to_dict().items():
            missing_sums[c] = missing_sums.get(c, 0) + int(v)
        # sample
        need = min(sample_remaining, len(chunk))
        if need > 0:
            chunks.append(chunk.sample(n=need, random_state=42) if need < len(chunk) else chunk)
            sample_remaining -= need
        if sample_remaining <= 0:
            break
    df_sample = pd.concat(chunks, axis=0) if chunks else pd.DataFrame()
    # Build missing over union of columns seen during chunking (not only sampled)
    missing: Dict[str, Any] = {}
    for c in sorted(missing_sums.keys(), key=str):
        count = int(missing_sums.get(c, 0))
        pct = float((count / total_rows * 100) if total_rows else 0.0)
        missing[str(c)] = {"count": count, "pct": pct}
    return {"df": df_sample, "missing": missing, "total_rows": int(total_rows)}


def load_dataframe(path: Path, file_format: Optional[str], sheet_name: Optional[str], delimiter: Optional[str], sample_rows: Optional[int] = None) -> pd.DataFrame:
    """Load a dataframe from CSV/TSV/Excel; auto-detects delimiter when needed.

    Args:
        path: File path.
        file_format: Optional override (csv/tsv/excel).
        sheet_name: Excel sheet when reading Excel.
        delimiter: CSV delimiter override; if None, a small sample is sniffed.
        sample_rows: If set, reads only the first N rows.

    Returns:
        pandas.DataFrame of the loaded data.
    """
    fmt = (file_format or infer_format(path)).lower()
    if fmt == "excel":
        df = pd.read_excel(path, sheet_name=sheet_name, nrows=sample_rows)
    else:
        if fmt == "tsv":
            sep = "\t"
        elif fmt == "csv":
            if delimiter:
                sep = delimiter
            else:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(4096)
                sep = detect_delimiter(sample)
        else:
            sep = ","
        df = pd.read_csv(path, sep=sep, nrows=sample_rows)
    return df


def _to_py(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


def compute_eda(df: pd.DataFrame, max_topk: int = 5) -> Dict[str, Any]:
    eda: Dict[str, Any] = {"schema_version": "1.0"}
    rows, cols = df.shape
    eda["shape"] = {"rows": int(rows), "cols": int(cols)}
    eda["columns"] = [str(c) for c in df.columns]
    eda["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
    # Missingness & nunique
    miss = df.isna().sum()
    eda["missing"] = {str(c): {"count": int(miss[c]), "pct": float((miss[c] / rows * 100) if rows else 0.0)} for c in df.columns}
    nunq = df.nunique(dropna=True)
    eda["nunique"] = {str(c): int(nunq[c]) for c in df.columns}
    # ID-like/constant columns
    id_candidates = [str(c) for c in df.columns if nunq[c] >= rows * 0.9]
    const_cols = [str(c) for c in df.columns if nunq[c] <= 1]
    eda["id_candidates"] = id_candidates
    eda["constant_columns"] = const_cols
    # Time-like detection on object columns (sample-based)
    time_like = []
    try:
        for c in df.columns:
            if ptypes.is_object_dtype(df[c]) and nunq[c] > 5:
                sample = df[c].dropna().astype(str).head(100)
                if not sample.empty:
                    # Pandas >=2 default infers format; avoid deprecated infer_datetime_format
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        time_like.append(str(c))
    except Exception:
        pass
    # Numeric stats
    num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]
    if num_cols:
        desc = df[num_cols].describe().to_dict()
        eda["numeric_stats"] = {c: {k: _to_py(v.get(c)) for k, v in desc.items()} for c in num_cols}
        try:
            eda["skew"] = {str(c): _to_py(df[c].skew()) for c in num_cols}
            eda["kurtosis"] = {str(c): _to_py(df[c].kurt()) for c in num_cols}
        except Exception:
            pass
    # Top correlations
    try:
        if len(num_cols) >= 2 and rows <= 20000:
            corr = df[num_cols].corr().abs()
            tri = corr.where(~np.tril(np.ones(corr.shape), -1).astype(bool))
            pairs = []
            for i, a in enumerate(num_cols):
                for b in num_cols[i+1:]:
                    val = tri.loc[a, b]
                    if pd.notna(val):
                        pairs.append(((str(a), str(b)), float(val)))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            eda["top_correlations"] = [(a, b, v) for (a, b), v in pairs[:10]]
    except Exception as e:
        import logging; logging.getLogger(__name__).warning("EDA: top_correlations failed: %s", e)
    # Categorical top-k summary (for tests)
    try:
        from pandas.api.types import CategoricalDtype
        cat_cols = [c for c in df.columns if (ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))]
        topk: Dict[str, List[str]] = {}
        for c in cat_cols:
            vc = df[c].astype(str).value_counts().head(5)
            topk[str(c)] = [str(idx) for idx in vc.index]
        if topk:
            eda["categorical_topk"] = topk
    except Exception as e:
        import logging; logging.getLogger(__name__).warning("EDA: categorical_topk failed: %s", e)
    # Rare levels
    rare_levels: Dict[str, List[str]] = {}
    try:
        from pandas.api.types import CategoricalDtype
        cat_cols = [c for c in df.columns if (ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))]
        for c in cat_cols:
            vc = df[c].astype(str).value_counts()
            threshold = max(2, int(len(df) * 0.01))
            rare_levels[str(c)] = [str(k) for k, v in vc.items() if v <= threshold][:max_topk]
        eda["rare_levels"] = rare_levels
    except Exception:
        pass
    # Time columns candidates
    eda["time_like_candidates"] = time_like
    # Recommendations
    recs = []
    if any(m["pct"] > 30 for m in eda["missing"].values()):
        recs.append("High missingness detected; consider imputation or dropping columns >30% missing.")
    if any(len(v) > 0 for v in rare_levels.values()):
        recs.append("High-cardinality categories/rare levels present; consider top-k + 'Other' / hashing.")
    if eda.get("top_correlations"):
        recs.append("Strongly correlated numeric pairs; consider collinearity handling.")
    if id_candidates:
        recs.append("ID-like columns found; exclude from modeling to avoid leakage.")
    if time_like or eda.get("time_columns"):
        recs.append("Time data detected; prefer time-based split and check trend/seasonality.")
    eda["recommendations"] = recs
    return eda


def compute_target_relations(df: pd.DataFrame, target: str, max_features: int = 10) -> Dict[str, Any]:
    """Compute simple relationships between features and a target column.

    - Determines task type (classification if non-numeric or <=10 unique values)
    - For numeric features: ranks by absolute Pearson correlation with target (for classification uses numeric-cast of target)
    - For categorical features: lists top-5 frequent levels (as a heuristic indicator)

    Args:
        df: Input DataFrame containing the target.
        target: Target column name.
        max_features: Max numeric features to include in ranked output.

    Returns:
        Dict with keys: task (classification/regression), numeric[], categorical[]
    """
    rel: Dict[str, Any] = {"numeric": [], "categorical": []}
    if target not in df.columns:
        rel["task"] = "unknown"
        return rel
    y = df[target]
    y_is_numeric = ptypes.is_numeric_dtype(y)
    nunq = y.nunique(dropna=True)
    is_classification = (not y_is_numeric) or (nunq <= 10)
    # numeric features
    y_num = pd.to_numeric(y, errors="coerce") if is_classification else y
    num_cols = [c for c in df.columns if c != target and ptypes.is_numeric_dtype(df[c])]
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        s = pd.concat([x, y_num], axis=1).dropna()
        if len(s) < 5:
            continue
        try:
            corr = float(s.corr().iloc[0,1])
            rel["numeric"].append({"col": str(c), "score": abs(corr), "type": "corr"})
        except Exception:
            pass
    rel["numeric"].sort(key=lambda d: d["score"], reverse=True)
    rel["numeric"] = rel["numeric"][:max_features]
    # categorical
    from pandas.api.types import CategoricalDtype
    cat_cols = [c for c in df.columns if c != target and (ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))]
    for c in cat_cols:
        try:
            vc = df[c].value_counts(dropna=True)
            if vc.empty: continue
            top = vc.head(5).index.astype(str).tolist()
            rel["categorical"].append({"col": str(c), "top": top})
        except Exception:
            pass
    rel["task"] = "classification" if is_classification else "regression"
    return rel


def compute_timeseries_hints(df: pd.DataFrame, time_col: str, metric_col: Optional[str]) -> Dict[str, Any]:
    """Light TS hints: points count and coarse autocorr at lags 7/30.

    Args:
        df: Input DataFrame.
        time_col: Column with date/time values.
        metric_col: Optional numeric metric to aggregate by day (else uses daily counts).

    Returns:
        Dict with time_col, points, and optional acf7/acf30 values.
    """
    out: Dict[str, Any] = {"time_col": time_col}
    try:
        sdf = df[[time_col]].copy()
        sdf[time_col] = pd.to_datetime(sdf[time_col], errors="coerce")
        sdf = sdf.dropna(subset=[time_col]).sort_values(time_col)
        if metric_col and metric_col in df.columns and ptypes.is_numeric_dtype(df[metric_col]):
            sdf[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
            agg = sdf.set_index(time_col)[metric_col].resample('D').mean()
            series = agg.dropna()
        else:
            # fallback to count per day
            agg = sdf.set_index(time_col).resample('D').size()
            series = agg.astype(float)
        out["points"] = int(series.shape[0])
        if series.shape[0] >= 14:
            out["acf7"] = float(series.autocorr(lag=7))
        if series.shape[0] >= 30:
            out["acf30"] = float(series.autocorr(lag=30))
    except Exception:
        pass
    return out


def generate_basic_plots(job_id: str, df: pd.DataFrame, plots_dir: Path) -> Dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"/static/jobs/{job_id}/plots"
    def _slug(s: str) -> str: return re.sub(r"[^A-Za-z0-9_-]+","_", str(s))
    # Numeric histograms
    num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])][:PLOTS_PER_NUMERIC]
    hist_urls: List[str] = []
    for c in num_cols:
        plt.figure(figsize=(4,3)); sns.histplot(df[c].dropna(), kde=False, bins=30)
        plt.title(f"Hist: {c}"); plt.tight_layout()
        p = plots_dir / f"hist_{_slug(c)}.png"; plt.savefig(p); plt.close(); hist_urls.append(f"{base_url}/hist_{_slug(c)}.png")
    # Categorical bars
    from pandas.api.types import CategoricalDtype
    cat_cols = [c for c in df.columns if (ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))][:PLOTS_PER_CATEG]
    bar_urls: List[str] = []
    for c in cat_cols:
        plt.figure(figsize=(4,3)); df[c].value_counts().head(10).plot(kind='bar'); plt.title(f"Top: {c}"); plt.tight_layout()
        p = plots_dir / f"bar_{_slug(c)}.png"; plt.savefig(p); plt.close(); bar_urls.append(f"{base_url}/bar_{_slug(c)}.png")
    # Missingness bar
    plt.figure(figsize=(5,3));
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    miss_pct.plot(kind='bar'); plt.title('Missingness % (top 20)'); plt.tight_layout()
    miss_path = plots_dir / "missingness.png"; plt.savefig(miss_path); plt.close()
    return {"histograms": hist_urls, "categoricals": bar_urls, "missingness": f"{base_url}/missingness.png"}


def generate_basic_plots_storage(storage, job_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Generate plots and store via storage adapter. Returns URLs via storage.url_for."""
    from io import BytesIO

    def _slug(s: str) -> str: return re.sub(r"[^A-Za-z0-9_-]+","_", str(s))

    # Numeric histograms
    num_cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])][:PLOTS_PER_NUMERIC]
    hist_urls: List[str] = []
    for c in num_cols:
        buf = BytesIO()
        plt.figure(figsize=(4,3)); sns.histplot(df[c].dropna(), kde=False, bins=30)
        plt.title(f"Hist: {c}"); plt.tight_layout(); plt.savefig(buf, format='png'); plt.close()
        storage.put_file(job_id, f"plots/hist_{_slug(c)}.png", buf.getvalue(), content_type="image/png")
        hist_urls.append(storage.url_for(job_id, f"plots/hist_{_slug(c)}.png"))

    # Categorical bars
    from pandas.api.types import CategoricalDtype
    cat_cols = [c for c in df.columns if (ptypes.is_object_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype))][:PLOTS_PER_CATEG]
    bar_urls: List[str] = []
    for c in cat_cols:
        buf = BytesIO()
        plt.figure(figsize=(4,3)); df[c].value_counts().head(10).plot(kind='bar'); plt.title(f"Top: {c}"); plt.tight_layout(); plt.savefig(buf, format='png'); plt.close()
        storage.put_file(job_id, f"plots/bar_{_slug(c)}.png", buf.getvalue(), content_type="image/png")
        bar_urls.append(storage.url_for(job_id, f"plots/bar_{_slug(c)}.png"))

    # Missingness bar
    buf = BytesIO()
    plt.figure(figsize=(5,3));
    miss_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    miss_pct.plot(kind='bar'); plt.title('Missingness % (top 20)'); plt.tight_layout(); plt.savefig(buf, format='png'); plt.close()
    storage.put_file(job_id, "plots/missingness.png", buf.getvalue(), content_type="image/png")

    return {"histograms": hist_urls, "categoricals": bar_urls, "missingness": storage.url_for(job_id, "plots/missingness.png")}

