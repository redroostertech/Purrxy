"""
Generic process supervisor.

Reusable in any Python 3.9+ project. Zero external dependencies.

Classes:
    ProcessConfig   Declarative process definition.
    Supervisor      Multi-process orchestrator with health checks,
                    auto-restart, and dependency ordering.

Functions:
    load_dotenv              Parse a .env file into a dict.
    load_dotenv_into_environ Load .env values into os.environ.
"""

from .supervisor import ProcessConfig, Supervisor
from .env import load_dotenv, load_dotenv_into_environ

__all__ = [
    "ProcessConfig",
    "Supervisor",
    "load_dotenv",
    "load_dotenv_into_environ",
]
