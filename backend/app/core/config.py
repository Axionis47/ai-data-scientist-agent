import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# File and upload constraints
MAX_UPLOAD_MB = 512
ALLOWED_EXTS = {".csv", ".tsv", ".xlsx", ".xls"}
LARGE_FILE_MB = 50

# Stage timeouts and concurrency (0 disables)
EDA_TIMEOUT_S = int(os.getenv("EDA_TIMEOUT_S", "0"))
MODEL_TIMEOUT_S = int(os.getenv("MODEL_TIMEOUT_S", "0"))
REPORT_TIMEOUT_S = int(os.getenv("REPORT_TIMEOUT_S", "0"))
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))

# Env-tunable modeling flags
BLEND_DELTA = float(os.getenv("BLEND_DELTA", "0.02"))
EARLY_STOP_SAMPLE = int(os.getenv("EARLY_STOP_SAMPLE", "10000"))
HGB_MIN_ROWS = int(os.getenv("HGB_MIN_ROWS", "1000"))
CALIBRATE_ENABLED = os.getenv("CALIBRATE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))
PDP_TOP_NUM = int(os.getenv("PDP_TOP_NUM", "2"))
SEARCH_TIME_BUDGET = int(
    os.getenv("SEARCH_TIME_BUDGET", "0")
)  # seconds; 0 disables quick search

# SHAP toggles
SHAP_ENABLED = os.getenv("SHAP_ENABLED", "false").lower() in ("1", "true", "yes")
SHAP_MAX_ROWS = int(os.getenv("SHAP_MAX_ROWS", "20000"))

# Static exposure safety
STATIC_EXPOSE_ORIGINAL = os.getenv("STATIC_EXPOSE_ORIGINAL", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Brand/report tokens (optional)
REPORT_PRIMARY = os.getenv("REPORT_PRIMARY")
REPORT_ACCENT = os.getenv("REPORT_ACCENT")
REPORT_BG = os.getenv("REPORT_BG")
REPORT_SURFACE = os.getenv("REPORT_SURFACE")
REPORT_TEXT = os.getenv("REPORT_TEXT")
REPORT_MUTED = os.getenv("REPORT_MUTED")
REPORT_OK = os.getenv("REPORT_OK")
REPORT_WARN = os.getenv("REPORT_WARN")
REPORT_ERROR = os.getenv("REPORT_ERROR")
REPORT_FONT_FAMILY = os.getenv("REPORT_FONT_FAMILY")
REPORT_LOGO_URL = os.getenv("REPORT_LOGO_URL")

# Feature flags
REPORT_JSON_FIRST = os.getenv("REPORT_JSON_FIRST", "false").lower() in (
    "1",
    "true",
    "yes",
)
SAFE_AUTO_ACTIONS = os.getenv("SAFE_AUTO_ACTIONS", "false").lower() in (
    "1",
    "true",
    "yes",
)

REPORT_INLINE_ASSETS = os.getenv("REPORT_INLINE_ASSETS", "false").lower() in (
    "1",
    "true",
    "yes",
)
