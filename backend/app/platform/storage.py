from __future__ import annotations
from typing import Optional
from pathlib import Path
import os

try:
    from google.cloud import storage as gcs_storage  # type: ignore
except Exception:  # pragma: no cover
    gcs_storage = None  # type: ignore

from ..core.config import JOBS_DIR


class Storage:
    """Abstract storage interface."""

    def put_file(
        self,
        job_id: str,
        rel_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def url_for(self, job_id: str, rel_path: str, expires_seconds: int = 3600) -> str:
        """Return a URL for accessing the object. For local, returns /static path."""
        raise NotImplementedError


class LocalStorage(Storage):
    base: Path

    def __init__(self, base: Path | None = None):
        self.base = base or JOBS_DIR

    def put_file(
        self,
        job_id: str,
        rel_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        p = self.base / job_id / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return self.url_for(job_id, rel_path)

    def url_for(self, job_id: str, rel_path: str, expires_seconds: int = 3600) -> str:
        rel = str(Path(rel_path).as_posix())
        return f"/static/jobs/{job_id}/{rel}"


class GCSStorage(Storage):  # pragma: no cover - requires GCP env
    def __init__(self, bucket_name: str):
        if gcs_storage is None:
            raise RuntimeError("google-cloud-storage is not installed")
        self.client = gcs_storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def _blob(self, job_id: str, rel_path: str):
        key = f"jobs/{job_id}/{Path(rel_path).as_posix()}"
        return self.bucket.blob(key)

    def put_file(
        self,
        job_id: str,
        rel_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        blob = self._blob(job_id, rel_path)
        if content_type:
            blob.content_type = content_type
        blob.upload_from_string(data)
        return self.url_for(job_id, rel_path)

    def url_for(self, job_id: str, rel_path: str, expires_seconds: int = 3600) -> str:
        blob = self._blob(job_id, rel_path)
        try:
            return blob.generate_signed_url(
                version="v4", expiration=expires_seconds, method="GET"
            )
        except Exception:
            # Public bucket case (not recommended by default)
            return f"https://storage.googleapis.com/{self.bucket.name}/{blob.name}"


def get_storage() -> Storage:
    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "gcs":
        bucket = os.getenv("GCS_BUCKET")
        if not bucket:
            raise RuntimeError("GCS_BUCKET env var required for GCS storage")
        return GCSStorage(bucket)
    return LocalStorage()
