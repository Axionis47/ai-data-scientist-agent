"""EDA Service Facade

Stable entry points for EDA-related operations. These delegate to app.eda.eda
so the pipeline can depend on this boundary instead of concrete modules.
"""

from pathlib import Path

from ..eda import eda as _eda

# Re-export with service-stable names
infer_format = _eda.infer_format
load_dataframe = _eda.load_dataframe
load_sampled_chunked_csv = _eda.load_sampled_chunked_csv
compute_eda = _eda.compute_eda
compute_target_relations = _eda.compute_target_relations
compute_timeseries_hints = _eda.compute_timeseries_hints


def is_large_file(path: Path, large_file_mb: int) -> bool:
    """Service-level helper for file size checks (simple passthrough).
    Exists to keep pipeline orchestration imports within the service layer.
    """
    try:
        return path.stat().st_size > large_file_mb * 1024 * 1024
    except Exception:
        return False
