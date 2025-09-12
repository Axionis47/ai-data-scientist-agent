from __future__ import annotations
import threading
import time
from typing import Dict, Any, Optional, Callable

from .jobqueue import get_job_queue
from .jobstore import get_job_store

# Simple concurrency limiter wrapper around LocalThreadQueue handler
class QueueRunner:
    def __init__(self, handler: Callable[[Dict[str, Any]], None], max_concurrent: int = 1):
        self._inflight = 0
        self._lock = threading.Lock()
        self._max = max(1, max_concurrent)
        def gated_handler(payload: Dict[str, Any]) -> None:
            with self._lock:
                if self._inflight >= self._max:
                    # Busy-wait backoff; LocalThreadQueue has only one worker anyway, but keep counter correct
                    # In a real queue, we'd push back into queue; here we sleep a bit
                    while self._inflight >= self._max:
                        time.sleep(0.01)
                self._inflight += 1
            try:
                handler(payload)
            finally:
                with self._lock:
                    self._inflight -= 1
        self._queue = get_job_queue(gated_handler)

    def enqueue(self, payload: Dict[str, Any]) -> None:
        self._queue.enqueue(payload)


def get_default_queue_runner(handler: Callable[[Dict[str, Any]], None]) -> QueueRunner:
    from ..core.config import MAX_CONCURRENT_JOBS
    return QueueRunner(handler, max_concurrent=MAX_CONCURRENT_JOBS)

