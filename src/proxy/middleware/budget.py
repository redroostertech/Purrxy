"""
Token counting and budget tracking for purrxy.

Tracks cumulative token usage per session (input + output) and enforces
an optional maximum token budget. When the budget is exceeded, new
requests are rejected with an Anthropic 429 error.

Thread-safe with threading.Lock.
"""

import logging
import threading
from typing import Dict

logger = logging.getLogger("proxy.budget")


class BudgetTracker:
    """Tracks token usage and enforces budget limits.

    Attributes:
        max_tokens: Maximum allowed tokens (0 = unlimited).
        total_input: Cumulative input tokens used.
        total_output: Cumulative output tokens used.
    """

    def __init__(self, max_tokens: int = 0):
        """Initialize the budget tracker.

        Args:
            max_tokens: Maximum token budget (0 for unlimited).
        """
        self.max_tokens = max_tokens
        self.total_input = 0
        self.total_output = 0
        self._per_model: Dict[str, Dict[str, int]] = {}
        self._lock = threading.Lock()

        if max_tokens > 0:
            logger.info("Token budget: %d tokens", max_tokens)
        else:
            logger.info("Token budget: unlimited")

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input + self.total_output

    @property
    def remaining(self) -> int:
        """Remaining tokens in the budget (0 if unlimited)."""
        if self.max_tokens <= 0:
            return 0
        return max(0, self.max_tokens - self.total_tokens)

    def check_budget(self) -> bool:
        """Check whether the budget allows a new request.

        Returns:
            True if the request is allowed, False if budget is exceeded.
        """
        if self.max_tokens <= 0:
            return True
        with self._lock:
            return self.total_tokens < self.max_tokens

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage from a completed request.

        Args:
            model: The model name.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
        """
        with self._lock:
            self.total_input += input_tokens
            self.total_output += output_tokens

            if model not in self._per_model:
                self._per_model[model] = {"input_tokens": 0, "output_tokens": 0}
            self._per_model[model]["input_tokens"] += input_tokens
            self._per_model[model]["output_tokens"] += output_tokens

        logger.debug("Token usage: +%d/%d (total: %d/%d)",
                      input_tokens, output_tokens,
                      self.total_input, self.total_output)

    def reset(self):
        """Reset all token counters to zero."""
        with self._lock:
            self.total_input = 0
            self.total_output = 0
            self._per_model.clear()
        logger.info("Budget counters reset")

    def get_status(self) -> dict:
        """Return current budget status as a JSON-serializable dict.

        Returns:
            Dict with total usage, remaining budget, and per-model breakdown.
        """
        with self._lock:
            per_model = {
                model: dict(stats)
                for model, stats in self._per_model.items()
            }
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "remaining": self.remaining,
            "unlimited": self.max_tokens <= 0,
            "per_model": per_model,
        }
