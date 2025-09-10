import logging
from pathlib import Path
from .config import JOBS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("ai-ds-backend")
eda_logger = logging.getLogger("ai-ds-eda")
eda_logger.setLevel(logging.INFO)
model_logger = logging.getLogger("ai-ds-model")
model_logger.setLevel(logging.INFO)


def eda_decision(job_id: str, message: str):
    try:
        eda_logger.info(f"[{job_id}] {message}")
        job_dir = JOBS_DIR / job_id
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "eda_decisions.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        log.debug(f"eda_decision logging failed: {e}")


def model_decision(job_id: str, message: str):
    try:
        model_logger.info(f"[{job_id}] {message}")
        job_dir = JOBS_DIR / job_id
        logs_dir = job_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        with (logs_dir / "model_decisions.log").open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        log.debug(f"model_decision logging failed: {e}")

