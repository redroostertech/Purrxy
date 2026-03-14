"""
Conversation analytics for lana-proxy.

Logs structured summaries after each request/response cycle to a JSONL file.
Provides session-level aggregation accessible via the /admin/analytics endpoint.

Thread-safe with threading.Lock.
"""

import json
import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger("proxy.analytics")


class Analytics:
    """Tracks and logs per-request analytics in JSONL format.

    Each entry contains:
        timestamp, model, input_tokens, output_tokens, latency_ms,
        stream, tool_calls_count, cache_hit, stop_reason
    """

    MAX_IN_MEMORY = 10000  # Cap in-memory entries to prevent unbounded growth

    def __init__(self, path: str):
        """Initialize analytics logging.

        Args:
            path: File path for the JSONL analytics log.
        """
        self.path = path
        self._entries: List[dict] = []
        self._lock = threading.Lock()
        logger.info("Analytics logging to: %s", path)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        stream: bool,
        tool_calls_count: int = 0,
        cache_hit: bool = False,
        stop_reason: str = "end_turn",
    ):
        """Record a completed request's analytics.

        Args:
            model: The model name from the request.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
            latency_ms: Request latency in milliseconds.
            stream: Whether this was a streaming request.
            tool_calls_count: Number of tool calls in the response.
            cache_hit: Whether the response came from cache.
            stop_reason: The stop reason from the response.
        """
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "stream": stream,
            "tool_calls_count": tool_calls_count,
            "cache_hit": cache_hit,
            "stop_reason": stop_reason,
        }

        line = json.dumps(entry, separators=(",", ":"))

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.MAX_IN_MEMORY:
                self._entries = self._entries[-self.MAX_IN_MEMORY:]
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.error("Failed to write analytics: %s", e)

        logger.debug("Analytics recorded: model=%s latency=%dms", model, latency_ms)

    def get_summary(self, cache_stats: Optional[dict] = None) -> dict:
        """Generate a session summary of all recorded analytics.

        Args:
            cache_stats: Optional cache statistics dict to include.

        Returns:
            Dict with aggregated session statistics.
        """
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return {
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "average_latency_ms": 0,
                "by_model": {},
                "tool_usage": {},
                "cache_hit_rate_percent": 0.0,
            }

        total_requests = len(entries)
        total_input = sum(e["input_tokens"] for e in entries)
        total_output = sum(e["output_tokens"] for e in entries)
        total_latency = sum(e["latency_ms"] for e in entries)
        avg_latency = total_latency / total_requests if total_requests > 0 else 0

        # Per-model breakdown
        by_model: Dict[str, dict] = {}
        for e in entries:
            model = e["model"]
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_latency_ms": 0,
                }
            by_model[model]["requests"] += 1
            by_model[model]["input_tokens"] += e["input_tokens"]
            by_model[model]["output_tokens"] += e["output_tokens"]
            by_model[model]["total_latency_ms"] += e["latency_ms"]

        # Add average latency per model
        for model, stats in by_model.items():
            if stats["requests"] > 0:
                stats["average_latency_ms"] = round(
                    stats["total_latency_ms"] / stats["requests"], 1
                )

        # Tool usage frequency (count of requests that used tools)
        tool_usage: Dict[str, int] = {}
        tool_requests = [e for e in entries if e["tool_calls_count"] > 0]
        tool_usage["requests_with_tools"] = len(tool_requests)
        tool_usage["total_tool_calls"] = sum(
            e["tool_calls_count"] for e in entries
        )

        # Cache hit rate
        cache_hits = sum(1 for e in entries if e.get("cache_hit", False))
        cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0.0

        summary = {
            "total_requests": total_requests,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "average_latency_ms": round(avg_latency, 1),
            "by_model": by_model,
            "tool_usage": tool_usage,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
        }

        if cache_stats:
            summary["cache_stats"] = cache_stats

        return summary
