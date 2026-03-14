"""
Observability and monitoring.

Modules:
    stats       Thread-safe request/latency/token tracking.
    dashboard   Live ANSI terminal dashboard.
    discovery   Auto model discovery from backend /v1/models.
"""

from .stats import Stats
from .dashboard import Dashboard
from .discovery import discover_models

__all__ = [
    "Stats",
    "Dashboard",
    "discover_models",
]
