"""
Request/response recording and replay for lana-proxy.

Provides two modes:
- Record: saves every request/response pair to a JSONL file for later replay.
- Replay: serves recorded responses from a JSONL file without contacting the
  backend. Matches requests by SHA-256 hash of the messages array.

Thread-safe: file writes are protected by a threading.Lock.
"""

import hashlib
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("proxy.recorder")


def _hash_messages(messages: list) -> str:
    """Compute a SHA-256 hash of the JSON-serialized messages array.

    Args:
        messages: The messages array from an Anthropic request.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class Recorder:
    """Records request/response pairs to a JSONL file.

    Each line contains a JSON object with:
        timestamp, request, response, latency_ms, stream, model
    """

    def __init__(self, path: str):
        """Initialize the recorder.

        Args:
            path: File path for the JSONL output.
        """
        self.path = path
        self._lock = threading.Lock()
        logger.info("Recording to: %s", path)

    def record(
        self,
        request: dict,
        response: dict,
        latency_ms: int,
        stream: bool,
        model: str,
        sse_events: Optional[List[str]] = None,
    ):
        """Record a request/response pair.

        Args:
            request: The original Anthropic request body.
            response: The Anthropic response body (for non-streaming).
            latency_ms: Request latency in milliseconds.
            stream: Whether this was a streaming request.
            model: The model name from the request.
            sse_events: For streaming requests, the list of SSE event strings.
        """
        entry = {
            "timestamp": time.time(),
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "stream": stream,
            "model": model,
        }
        if sse_events is not None:
            entry["sse_events"] = sse_events

        # Compute hash for replay matching
        messages = request.get("messages", [])
        entry["messages_hash"] = _hash_messages(messages)

        line = json.dumps(entry, separators=(",", ":"))
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        logger.debug("Recorded request: hash=%s stream=%s", entry["messages_hash"], stream)


class Replayer:
    """Serves recorded responses from a JSONL file.

    Loads all entries into memory on init and matches by messages hash.
    """

    def __init__(self, path: str):
        """Initialize the replayer by loading all recorded entries.

        Args:
            path: Path to the JSONL recording file.

        Raises:
            FileNotFoundError: If the recording file does not exist.
        """
        self.path = path
        self._entries: Dict[str, dict] = {}
        self._load(path)
        logger.info("Replay loaded %d entries from: %s", len(self._entries), path)

    def _load(self, path: str):
        """Load entries from the JSONL file, keyed by messages hash."""
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line %d in %s", line_num, path)
                    continue
                msg_hash = entry.get("messages_hash")
                if msg_hash:
                    self._entries[msg_hash] = entry

    def lookup(self, request: dict) -> Optional[dict]:
        """Look up a recorded entry matching the given request.

        Matches by SHA-256 hash of the messages array.

        Args:
            request: The Anthropic request body.

        Returns:
            The recorded entry dict, or None if no match found.
        """
        messages = request.get("messages", [])
        msg_hash = _hash_messages(messages)
        entry = self._entries.get(msg_hash)
        if entry:
            logger.debug("Replay hit: hash=%s", msg_hash)
        else:
            logger.debug("Replay miss: hash=%s", msg_hash)
        return entry
