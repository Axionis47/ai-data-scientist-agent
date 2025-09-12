from __future__ import annotations
from typing import Dict, Any, Optional
import threading
import os

try:
    from google.cloud import firestore  # type: ignore
except Exception:  # pragma: no cover
    firestore = None  # type: ignore


class JobStore:
    def create(self, job_id: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def create(self, job_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self._data[job_id] = dict(data)

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(job_id, {})
            self._data[job_id].update(patch)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._data.get(job_id) or {})


class FirestoreJobStore(JobStore):  # pragma: no cover - requires GCP env
    def __init__(self, collection: str = "jobs"):
        if firestore is None:
            raise RuntimeError("google-cloud-firestore is not installed")
        self.client = firestore.Client()
        self.collection = self.client.collection(collection)

    def create(self, job_id: str, data: Dict[str, Any]) -> None:
        self.collection.document(job_id).set(data)

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        self.collection.document(job_id).set(patch, merge=True)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        doc = self.collection.document(job_id).get()
        return doc.to_dict() if doc.exists else None


def get_job_store() -> JobStore:
    backend = (os.getenv("JOBSTORE_BACKEND") or "memory").lower()
    if backend == "firestore":
        return FirestoreJobStore()
    return InMemoryJobStore()
