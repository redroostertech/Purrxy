"""
Optional proxy features.

Each module is independent and can be enabled/disabled via CLI flags.
All classes are thread-safe.

Modules:
    cache       LRU prompt cache with optional disk persistence.
    queue       Semaphore-based request queuing for limited-concurrency backends.
    recorder    JSONL request/response recording and replay.
    budget      Token budget tracking and enforcement.
    analytics   Per-request JSONL analytics logging.
"""

from .cache import PromptCache
from .queue import RequestQueue
from .recorder import Recorder, Replayer
from .budget import BudgetTracker
from .analytics import Analytics

__all__ = [
    "PromptCache",
    "RequestQueue",
    "Recorder",
    "Replayer",
    "BudgetTracker",
    "Analytics",
]
