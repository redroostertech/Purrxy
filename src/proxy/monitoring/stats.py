"""
Thread-safe statistics tracking for purrxy.

Collects request counts, latencies, token throughput, active connections,
and recent request summaries for the dashboard and admin endpoints.
"""

import threading
import time
from typing import Any, Dict, List, Optional


class RequestSummary:
    """Summary of a single completed request."""

    __slots__ = ("model", "input_tokens", "output_tokens", "latency_ms",
                 "stream", "timestamp", "error")

    def __init__(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        stream: bool,
        timestamp: float,
        error: bool = False,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.stream = stream
        self.timestamp = timestamp
        self.error = error

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "stream": self.stream,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class Stats:
    """Thread-safe statistics collector.

    All public methods acquire the internal lock, making them safe to call
    from any request-handler thread or the dashboard thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        # Counters
        self._total_requests = 0
        self._stream_requests = 0
        self._non_stream_requests = 0
        self._error_count = 0

        # Active connections
        self._active_connections = 0

        # Latency tracking (in milliseconds)
        self._latencies: List[float] = []

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Recent requests (ring buffer, last N)
        self._recent: List[RequestSummary] = []
        self._recent_max = 20

        # Queue depth (updated externally by the queue module)
        self._queue_depth = 0
        self._queue_max = 0

        # Discovered models from backends
        self._discovered_models: List[Dict[str, Any]] = []

    # ── Connection tracking ──────────────────────────────────────────

    def connection_opened(self) -> None:
        """Record a new active connection."""
        with self._lock:
            self._active_connections += 1

    def connection_closed(self) -> None:
        """Record a closed connection."""
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    # ── Request recording ────────────────────────────────────────────

    def record_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        stream: bool,
        error: bool = False,
    ) -> None:
        """Record a completed request."""
        summary = RequestSummary(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            stream=stream,
            timestamp=time.time(),
            error=error,
        )
        with self._lock:
            self._total_requests += 1
            if error:
                self._error_count += 1
            if stream:
                self._stream_requests += 1
            else:
                self._non_stream_requests += 1

            self._latencies.append(latency_ms)
            # Keep latency buffer bounded
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]

            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            self._recent.append(summary)
            if len(self._recent) > self._recent_max:
                self._recent = self._recent[-self._recent_max:]

    # ── Queue tracking ───────────────────────────────────────────────

    def set_queue_depth(self, depth: int, max_concurrent: int) -> None:
        """Update current queue depth."""
        with self._lock:
            self._queue_depth = depth
            self._queue_max = max_concurrent

    # ── Model discovery ──────────────────────────────────────────────

    def set_discovered_models(self, models: List[Dict[str, Any]]) -> None:
        """Store discovered backend models."""
        with self._lock:
            self._discovered_models = list(models)

    # ── Snapshot for dashboard / admin ───────────────────────────────

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of all stats.

        Returns:
            Dict with all current statistics.
        """
        with self._lock:
            uptime = time.monotonic() - self._start_time
            latencies = list(self._latencies)
            recent = [r.to_dict() for r in self._recent[-5:]]
            discovered = list(self._discovered_models)

            # Compute latency percentiles
            avg_latency = 0.0
            p50_latency = 0.0
            p95_latency = 0.0
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                sorted_lat = sorted(latencies)
                p50_latency = sorted_lat[len(sorted_lat) // 2]
                p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
                p95_latency = sorted_lat[p95_idx]

            # Token throughput (per minute)
            uptime_minutes = max(uptime / 60.0, 1.0 / 60.0)  # avoid div by zero
            input_tpm = self._total_input_tokens / uptime_minutes
            output_tpm = self._total_output_tokens / uptime_minutes

            return {
                "uptime_seconds": round(uptime, 1),
                "active_connections": self._active_connections,
                "total_requests": self._total_requests,
                "stream_requests": self._stream_requests,
                "non_stream_requests": self._non_stream_requests,
                "error_count": self._error_count,
                "avg_latency_ms": round(avg_latency, 1),
                "p50_latency_ms": round(p50_latency, 1),
                "p95_latency_ms": round(p95_latency, 1),
                "input_tokens_per_min": round(input_tpm, 1),
                "output_tokens_per_min": round(output_tpm, 1),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "queue_depth": self._queue_depth,
                "queue_max": self._queue_max,
                "recent_requests": recent,
                "discovered_models": discovered,
            }
