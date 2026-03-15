"""
Prompt caching layer for purrxy.

Provides an in-memory LRU cache for identical non-streaming requests.
Cache key is a SHA-256 hash of (model + system + messages + tools).

Features:
- Configurable TTL (time-to-live) and max size
- Thread-safe with threading.Lock
- Optional disk persistence (JSON file saved on shutdown, loaded on startup)
- Tracks cache hits/misses for analytics
"""

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger("proxy.cache")


def _make_cache_key(body: dict) -> str:
    """Compute a SHA-256 cache key from deterministic request parts.

    Uses model, system, messages, and tools fields.

    Args:
        body: The Anthropic request body.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    key_parts = {
        "model": body.get("model", ""),
        "system": body.get("system"),
        "messages": body.get("messages", []),
        "tools": body.get("tools"),
    }
    serialized = json.dumps(key_parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PromptCache:
    """In-memory LRU cache for non-streaming Anthropic responses.

    Attributes:
        hits: Number of cache hits since startup.
        misses: Number of cache misses since startup.
    """

    def __init__(self, ttl: int = 300, max_size: int = 100, cache_dir: Optional[str] = None):
        """Initialize the prompt cache.

        Args:
            ttl: Time-to-live in seconds for cache entries (default 300).
            max_size: Maximum number of entries in the cache (default 100).
            cache_dir: Optional directory for disk persistence.
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache_dir = cache_dir
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

        if cache_dir:
            self._load_from_disk()

        logger.info("Cache initialized: ttl=%ds, max_size=%d, disk=%s",
                     ttl, max_size, cache_dir or "disabled")

    def get(self, body: dict) -> Optional[dict]:
        """Look up a cached response for the given request.

        Args:
            body: The Anthropic request body.

        Returns:
            The cached response dict, or None if not found or expired.
        """
        key = _make_cache_key(body)

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                logger.debug("Cache miss: %s", key[:16])
                return None

            entry = self._cache[key]
            # Check TTL (monotonic clock immune to NTP adjustments)
            if time.monotonic() - entry["timestamp"] > self.ttl:
                del self._cache[key]
                self.misses += 1
                logger.debug("Cache expired: %s", key[:16])
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            logger.debug("Cache hit: %s", key[:16])
            return entry["response"]

    def put(self, body: dict, response: dict):
        """Store a response in the cache.

        Args:
            body: The Anthropic request body (used to compute key).
            response: The Anthropic response body to cache.
        """
        key = _make_cache_key(body)

        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                removed_key, _ = self._cache.popitem(last=False)
                logger.debug("Cache evicted: %s", removed_key[:16])

            self._cache[key] = {
                "response": response,
                "timestamp": time.monotonic(),
            }
            # Move to end
            self._cache.move_to_end(key)

        logger.debug("Cache stored: %s (size=%d)", key[:16], len(self._cache))

    def save_to_disk(self):
        """Persist the cache to disk (if cache_dir is configured).

        Called on server shutdown to preserve cache across restarts.
        """
        if not self.cache_dir:
            return

        path = os.path.join(self.cache_dir, "prompt_cache.json")
        with self._lock:
            # Convert monotonic timestamps to wall-clock for disk persistence
            mono_now = time.monotonic()
            wall_now = time.time()
            entries = {}
            for key, entry in self._cache.items():
                age = mono_now - entry["timestamp"]
                entries[key] = {
                    "response": entry["response"],
                    "timestamp": wall_now - age,
                }

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, separators=(",", ":"))
            logger.info("Cache saved to disk: %d entries", len(entries))
        except Exception as e:
            logger.error("Failed to save cache to disk: %s", e)

    def _load_from_disk(self):
        """Load cache entries from disk if available."""
        if not self.cache_dir:
            return

        path = os.path.join(self.cache_dir, "prompt_cache.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                entries = json.load(f)

            wall_now = time.time()
            mono_now = time.monotonic()
            loaded = 0
            for key, entry in entries.items():
                # Skip expired entries (disk timestamps are wall-clock)
                age = wall_now - entry.get("timestamp", 0)
                if age > self.ttl:
                    continue
                if loaded >= self.max_size:
                    break
                # Convert wall-clock age to monotonic for runtime use
                entry["timestamp"] = mono_now - age
                self._cache[key] = entry
                loaded += 1

            logger.info("Cache loaded from disk: %d entries", loaded)
        except Exception as e:
            logger.warning("Failed to load cache from disk: %s", e)

    def get_stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with hits, misses, size, hit_rate.
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hit_rate_percent": round(hit_rate, 2),
        }
