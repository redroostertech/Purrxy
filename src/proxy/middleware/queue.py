"""
Request queue for limiting concurrent backend requests.

Uses threading.Semaphore to enforce a maximum number of concurrent
requests to the llama.cpp backend, which typically has limited
concurrency (default: 1 slot).
"""

import logging
import threading
from typing import Optional

from ..monitoring.stats import Stats

logger = logging.getLogger("proxy.queue")


class RequestQueue:
    """Semaphore-based request queue for backend concurrency control.

    Usage:
        queue = RequestQueue(max_concurrent=1, timeout=120, stats=stats)

        if queue.acquire():
            try:
                # ... forward request to backend ...
            finally:
                queue.release()
        else:
            # Queue timeout — return 529 overloaded
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        timeout: float = 120.0,
        stats: Optional[Stats] = None,
    ):
        """Initialize the request queue.

        Args:
            max_concurrent: Maximum number of concurrent backend requests.
            timeout: Seconds to wait for a slot before timing out.
            stats: Optional Stats object for queue depth reporting.
        """
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be >= 1, got {max_concurrent}"
            )
        self._semaphore = threading.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._timeout = timeout
        self._stats = stats
        self._active = 0
        self._lock = threading.Lock()

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent requests allowed."""
        return self._max_concurrent

    @property
    def timeout(self) -> float:
        """Queue wait timeout in seconds."""
        return self._timeout

    def acquire(self) -> bool:
        """Acquire a slot in the queue.

        Blocks up to ``timeout`` seconds. Returns True if a slot was
        acquired, False if the wait timed out.
        """
        acquired = self._semaphore.acquire(timeout=self._timeout)
        if acquired:
            with self._lock:
                self._active += 1
                self._update_stats()
            logger.debug("Queue slot acquired (active=%d/%d)",
                         self._active, self._max_concurrent)
        else:
            logger.warning("Queue timeout after %.1fs (max_concurrent=%d)",
                           self._timeout, self._max_concurrent)
        return acquired

    def release(self) -> None:
        """Release a slot back to the queue."""
        with self._lock:
            self._active = max(0, self._active - 1)
            self._update_stats()
        self._semaphore.release()
        logger.debug("Queue slot released (active=%d/%d)",
                     self._active, self._max_concurrent)

    def _update_stats(self) -> None:
        """Push current queue depth to the stats object."""
        if self._stats is not None:
            self._stats.set_queue_depth(self._active, self._max_concurrent)
