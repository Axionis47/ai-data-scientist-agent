from __future__ import annotations
from typing import Dict, Any, Callable
import threading
import queue
import os
import json

try:
    from google.cloud import pubsub_v1  # type: ignore
except Exception:  # pragma: no cover
    pubsub_v1 = None  # type: ignore


class JobQueue:
    def enqueue(self, payload: Dict[str, Any]) -> None:
        raise NotImplementedError


class LocalThreadQueue(JobQueue):
    """Simple local queue that executes a handler function on a background thread.
    For development only.
    """

    def __init__(self, handler: Callable[[Dict[str, Any]], None]):
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._handler = handler
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            item = self._q.get()
            try:
                self._handler(item)
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception("Job handler error: %s", e)
            finally:
                self._q.task_done()

    def enqueue(self, payload: Dict[str, Any]) -> None:
        self._q.put(payload)


class PubSubQueue(JobQueue):  # pragma: no cover - requires GCP env
    def __init__(self, topic: str):
        if pubsub_v1 is None:
            raise RuntimeError("google-cloud-pubsub is not installed")
        self._publisher = pubsub_v1.PublisherClient()
        self._topic_path = topic  # assume full topic path provided

    def enqueue(self, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._publisher.publish(self._topic_path, data=data)


def get_job_queue(local_handler: Callable[[Dict[str, Any]], None]) -> JobQueue:
    backend = (os.getenv("JOBQUEUE_BACKEND") or "local").lower()
    if backend == "pubsub":
        topic = os.getenv("PUBSUB_TOPIC_PATH")
        if not topic:
            raise RuntimeError("PUBSUB_TOPIC_PATH env var required for pubsub queue")
        return PubSubQueue(topic)
    return LocalThreadQueue(local_handler)

