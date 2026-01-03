"""
Deterministic EDA tools for data analysis.

Each tool returns structured artifacts (no prose), with deterministic ordering
and stable rounding. Column validation is strict - never guess silently.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


class EDAToolError(Exception):
    """Raised when an EDA tool encounters an error."""
    pass


def _load_dataset(dataset_id: str, datasets_dir: Path) -> pd.DataFrame:
    """Load a dataset by ID from the datasets directory."""
    dataset_path = datasets_dir / dataset_id / "data.csv"
    if not dataset_path.exists():
        raise EDAToolError(f"Dataset not found: {dataset_id}")
    return pd.read_csv(dataset_path)


def _load_metadata(dataset_id: str, datasets_dir: Path) -> dict:
    """Load dataset metadata."""
    metadata_path = datasets_dir / dataset_id / "metadata.json"
    if not metadata_path.exists():
        raise EDAToolError(f"Metadata not found for dataset: {dataset_id}")
    with open(metadata_path) as f:
        return json.load(f)


def _load_profile(dataset_id: str, datasets_dir: Path) -> dict:
    """Load dataset profile."""
    profile_path = datasets_dir / dataset_id / "profile.json"
    if not profile_path.exists():
        raise EDAToolError(f"Profile not found for dataset: {dataset_id}")
    with open(profile_path) as f:
        return json.load(f)


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that all columns exist in the dataframe."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise EDAToolError(f"Columns not found in dataset: {missing}")


def dataset_overview(dataset_id: str, datasets_dir: Path) -> dict[str, Any]:
    """
    Get dataset overview.

    Returns:
        TextArtifact with summary + TableArtifact with columns/types/missingness
    """
    metadata = _load_metadata(dataset_id, datasets_dir)
    profile = _load_profile(dataset_id, datasets_dir)

    # Text summary
    text_content = (
        f"Dataset: {dataset_id}\n"
        f"Rows: {metadata['n_rows']}\n"
        f"Columns: {metadata['n_cols']}\n"
        f"Created: {metadata.get('created_at', 'unknown')}"
    )

    # Table with column info (deterministically sorted)
    headers = ["Column", "Type", "Missing %"]
    rows = []
    for col in sorted(metadata["column_names"]):
        col_type = metadata["inferred_types"].get(col, "unknown")
        col_profile = profile.get("columns", {}).get(col, {})
        missing_pct = col_profile.get("missing_pct", 0.0)
        rows.append([col, col_type, f"{missing_pct:.2f}%"])

    return {
        "text_artifact": {"type": "text", "content": text_content},
        "table_artifact": {"type": "table", "headers": headers, "rows": rows},
    }


def univariate_summary(dataset_id: str, column: str, datasets_dir: Path) -> dict[str, Any]:
    """
    Get univariate summary for a single column.

    Returns:
        TableArtifact with statistics or value counts
    """
    df = _load_dataset(dataset_id, datasets_dir)
    _validate_columns(df, [column])

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        # Numeric: return describe stats
        desc = series.describe()
        headers = ["Statistic", "Value"]
        rows = [
            ["count", str(int(desc.get("count", 0)))],
            ["mean", f"{desc.get('mean', 0):.4f}"],
            ["std", f"{desc.get('std', 0):.4f}"],
            ["min", f"{desc.get('min', 0):.4f}"],
            ["25%", f"{desc.get('25%', 0):.4f}"],
            ["50%", f"{desc.get('50%', 0):.4f}"],
            ["75%", f"{desc.get('75%', 0):.4f}"],
            ["max", f"{desc.get('max', 0):.4f}"],
        ]
    else:
        # Categorical: return top 10 value counts
        value_counts = series.value_counts().head(10)
        headers = ["Value", "Count", "Percentage"]
        total = len(series)
        rows = [
            [str(val), str(cnt), f"{100*cnt/total:.2f}%"]
            for val, cnt in value_counts.items()
        ]

    return {"table_artifact": {"type": "table", "headers": headers, "rows": rows}}


def groupby_aggregate(
    dataset_id: str,
    group_col: str,
    target_col: str,
    agg: str,
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Group by a column and aggregate another.

    Args:
        agg: One of 'sum', 'mean', 'count', 'min', 'max', 'median'

    Returns:
        TableArtifact with grouped results
    """
    valid_aggs = ["sum", "mean", "count", "min", "max", "median"]
    if agg not in valid_aggs:
        raise EDAToolError(f"Invalid aggregation: {agg}. Must be one of {valid_aggs}")

    df = _load_dataset(dataset_id, datasets_dir)
    _validate_columns(df, [group_col, target_col])

    # Perform groupby
    grouped = df.groupby(group_col, dropna=False)[target_col].agg(agg)
    grouped = grouped.sort_index()

    headers = [group_col, f"{agg}({target_col})"]
    rows = [[str(idx), f"{val:.4f}" if isinstance(val, float) else str(val)]
            for idx, val in grouped.items()]

    return {"table_artifact": {"type": "table", "headers": headers, "rows": rows}}


def time_trend(
    dataset_id: str,
    date_col: str,
    target_col: str,
    agg: str,
    freq: str,
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Compute time trend aggregation.

    Args:
        freq: One of 'D' (daily), 'W' (weekly), 'M' (monthly)
        agg: One of 'sum', 'mean', 'count', 'min', 'max', 'median'

    Returns:
        TableArtifact with time-aggregated results
    """
    valid_freqs = ["D", "W", "M"]
    valid_aggs = ["sum", "mean", "count", "min", "max", "median"]

    if freq not in valid_freqs:
        raise EDAToolError(f"Invalid frequency: {freq}. Must be one of {valid_freqs}")
    if agg not in valid_aggs:
        raise EDAToolError(f"Invalid aggregation: {agg}. Must be one of {valid_aggs}")

    df = _load_dataset(dataset_id, datasets_dir)
    _validate_columns(df, [date_col, target_col])

    # Parse dates
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        raise EDAToolError(f"Could not parse column {date_col} as datetime: {e}")

    # Set date as index and resample
    df_indexed = df.set_index(date_col)
    resampled = df_indexed[target_col].resample(freq).agg(agg)
    resampled = resampled.dropna().sort_index()

    freq_names = {"D": "Date", "W": "Week", "M": "Month"}
    headers = [freq_names[freq], f"{agg}({target_col})"]
    rows = [[idx.strftime("%Y-%m-%d"), f"{val:.4f}" if isinstance(val, float) else str(val)]
            for idx, val in resampled.items()]

    return {"table_artifact": {"type": "table", "headers": headers, "rows": rows}}


def correlation(
    dataset_id: str,
    columns: list[str],
    datasets_dir: Path,
) -> dict[str, Any]:
    """
    Compute correlation matrix for numeric columns.

    Args:
        columns: List of numeric columns to correlate

    Returns:
        TableArtifact with correlation matrix
    """
    df = _load_dataset(dataset_id, datasets_dir)
    _validate_columns(df, columns)

    # Validate all columns are numeric
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise EDAToolError(f"Column {col} is not numeric, cannot compute correlation")

    # Compute correlation (deterministic with sorted columns)
    sorted_cols = sorted(columns)
    corr_matrix = df[sorted_cols].corr()

    # Build table
    headers = [""] + sorted_cols
    rows = []
    for row_col in sorted_cols:
        row = [row_col]
        for col_col in sorted_cols:
            row.append(f"{corr_matrix.loc[row_col, col_col]:.4f}")
        rows.append(row)

    return {"table_artifact": {"type": "table", "headers": headers, "rows": rows}}

