import json
import logging
from typing import Any, Dict
from .config import JOBS_DIR, LOG_LEVEL


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        try:
            payload: Dict[str, Any] = {
                "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            job_id = getattr(record, "job_id", None)
            if isinstance(job_id, str) and job_id:
                payload["job_id"] = job_id
            stage = getattr(record, "stage", None)
            if isinstance(stage, str) and stage:
                payload["stage"] = stage
            dur = getattr(record, "duration_ms", None)
            if isinstance(dur, int):
                payload["duration_ms"] = dur
            if record.exc_info:
                try:
                    payload["exc"] = self.formatException(record.exc_info)
                except Exception:
                    pass
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return super().format(record)


class _JobIdFilter(logging.Filter):
    def filter(
        self, record: logging.LogRecord
    ) -> bool:  # pragma: no cover - formatting
        try:
            msg = record.getMessage()
            if isinstance(msg, str) and msg.startswith("[") and "]" in msg[:48]:
                # Extract leading [jobid]
                j = msg[1 : msg.index("]", 1)]
                if len(j) >= 8:  # heuristic; typical uuid hex length 32
                    setattr(record, "job_id", j)
            # Default stage from logger if not provided
            if not hasattr(record, "stage"):
                if record.name == "ai-ds-eda":
                    setattr(record, "stage", "eda")
                elif record.name == "ai-ds-model":
                    setattr(record, "stage", "modeling")
        except Exception:
            pass
        return True


def setup_json_logging() -> None:
    # Idempotent setup: avoid duplicate handlers
    root = logging.getLogger()
    want = _JsonFormatter
    has_json = any(isinstance(h.formatter, want) for h in root.handlers if h.formatter)
    if has_json:
        return
    level = getattr(logging, str(LOG_LEVEL or "INFO").upper(), logging.INFO)
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_JobIdFilter())
    root.handlers = [handler]


# Initialize module-level loggers
log = logging.getLogger("ai-ds-backend")
eda_logger = logging.getLogger("ai-ds-eda")
model_logger = logging.getLogger("ai-ds-model")
for _lg in (eda_logger, model_logger, log):
    _lg.setLevel(getattr(logging, str(LOG_LEVEL or "INFO").upper(), logging.INFO))


def eda_decision(
    job_id: str,
    message: str,
    *,
    stage: str | None = None,
    duration_ms: int | None = None,
):
    try:
        extra: Dict[str, Any] = {}
        if stage:
            extra["stage"] = stage
        if duration_ms is not None:
            extra["duration_ms"] = int(duration_ms)
        eda_logger.info(f"[{job_id}] {message}", extra=extra)
        job_dir = JOBS_DIR / job_id
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "eda_decisions.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        log.debug(f"eda_decision logging failed: {e}")


def model_decision(
    job_id: str,
    message: str,
    *,
    stage: str | None = None,
    duration_ms: int | None = None,
):
    try:
        extra: Dict[str, Any] = {}
        if stage:
            extra["stage"] = stage
        if duration_ms is not None:
            extra["duration_ms"] = int(duration_ms)
        model_logger.info(f"[{job_id}] {message}", extra=extra)
        job_dir = JOBS_DIR / job_id
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "model_decisions.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        log.debug(f"model_decision logging failed: {e}")
