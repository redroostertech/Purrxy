"""
Configuration management for purrxy.

Loads settings from CLI arguments, environment variables, and defaults.
Precedence: CLI args > environment variables > defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Config:
    """Runtime configuration for the proxy server."""

    host: str = "127.0.0.1"
    port: int = 5304
    backend_url: str = "http://127.0.0.1:8080"
    default_model: str = "local-model"
    model_map: Dict[str, str] = field(default_factory=dict)
    routes: Dict[str, str] = field(default_factory=dict)
    request_timeout: int = 300
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_concurrent: int = 1
    queue_timeout: float = 120.0
    dashboard: bool = False
    log_level: str = "INFO"
    record_path: Optional[str] = None
    replay_path: Optional[str] = None
    cache_enabled: bool = False
    cache_ttl: int = 300
    cache_max_size: int = 100
    cache_dir: Optional[str] = None
    token_budget: int = 0  # 0 = unlimited
    analytics_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Create config populated from environment variables."""
        return cls(
            host=os.environ.get("PURRXY_HOST", cls.host),
            port=int(os.environ.get("PURRXY_PORT", cls.port)),
            backend_url=os.environ.get("PURRXY_BACKEND", cls.backend_url),
            log_level=os.environ.get("PURRXY_LOG_LEVEL", cls.log_level),
            request_timeout=int(
                os.environ.get("PURRXY_TIMEOUT", cls.request_timeout)
            ),
        )

    def validate(self):
        """Validate configuration values.

        Raises:
            ValueError: If any config value is out of range.
        """
        if not (1 <= self.port <= 65535):
            raise ValueError(
                f"Port must be between 1 and 65535, got {self.port}"
            )
        if self.request_timeout <= 0:
            raise ValueError(
                f"Timeout must be greater than 0, got {self.request_timeout}"
            )

    def resolve_model(self, anthropic_model: str) -> str:
        """Map an Anthropic model name to a backend model name."""
        return self.model_map.get(anthropic_model, self.default_model)

    def resolve_backend_url(self, anthropic_model: str) -> str:
        """Resolve the backend URL for a given model name.

        Checks routes first (model_name -> backend_url), then falls back
        to the default backend_url.

        Args:
            anthropic_model: The model name from the Anthropic request.

        Returns:
            Backend base URL string.
        """
        return self.routes.get(anthropic_model, self.backend_url)

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (for admin endpoints)."""
        return {
            "host": self.host,
            "port": self.port,
            "backend_url": self.backend_url,
            "default_model": self.default_model,
            "model_map": dict(self.model_map),
            "routes": dict(self.routes),
            "request_timeout": self.request_timeout,
            "max_request_size": self.max_request_size,
            "max_concurrent": self.max_concurrent,
            "queue_timeout": self.queue_timeout,
            "dashboard": self.dashboard,
            "log_level": self.log_level,
            "record_path": self.record_path,
            "replay_path": self.replay_path,
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "cache_dir": self.cache_dir,
            "token_budget": self.token_budget,
            "analytics_path": self.analytics_path,
        }
