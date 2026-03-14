"""Tests for proxy configuration management."""

import os
import unittest
from unittest.mock import patch

from src.proxy.config import Config


class TestDefaultConfig(unittest.TestCase):
    """Test Config default values."""

    def test_default_config(self):
        """Config() has expected defaults."""
        cfg = Config()
        self.assertEqual(cfg.port, 5304)
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.backend_url, "http://127.0.0.1:8080")
        self.assertEqual(cfg.default_model, "local-model")
        self.assertEqual(cfg.model_map, {})
        self.assertEqual(cfg.request_timeout, 300)
        self.assertEqual(cfg.max_request_size, 10 * 1024 * 1024)
        self.assertEqual(cfg.log_level, "INFO")


class TestFromEnv(unittest.TestCase):
    """Test Config.from_env() reads environment variables."""

    def test_from_env(self):
        """Config.from_env() reads LANA_PROXY_PORT, LANA_PROXY_HOST, etc."""
        env = {
            "LANA_PROXY_PORT": "9999",
            "LANA_PROXY_HOST": "0.0.0.0",
            "LANA_PROXY_BACKEND": "http://localhost:1234",
            "LANA_PROXY_LOG_LEVEL": "DEBUG",
            "LANA_PROXY_TIMEOUT": "60",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.from_env()
        self.assertEqual(cfg.port, 9999)
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertEqual(cfg.backend_url, "http://localhost:1234")
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertEqual(cfg.request_timeout, 60)

    def test_from_env_defaults(self):
        """Config.from_env() uses defaults when env vars are absent."""
        # Clear any LANA_PROXY_* vars that might be set
        env_keys = [k for k in os.environ if k.startswith("LANA_PROXY_")]
        removals = {k: "" for k in env_keys}
        with patch.dict(os.environ, {}, clear=False):
            # Remove any LANA_PROXY_ keys
            for k in env_keys:
                os.environ.pop(k, None)
            cfg = Config.from_env()
        self.assertEqual(cfg.port, 5304)
        self.assertEqual(cfg.host, "127.0.0.1")


class TestResolveModel(unittest.TestCase):
    """Test model resolution via model_map."""

    def test_resolve_model_with_mapping(self):
        """model_map lookup works when a mapping exists."""
        cfg = Config(
            model_map={"claude-3-opus": "qwen2.5-coder", "claude-3-sonnet": "llama3"},
            default_model="fallback-model",
        )
        self.assertEqual(cfg.resolve_model("claude-3-opus"), "qwen2.5-coder")
        self.assertEqual(cfg.resolve_model("claude-3-sonnet"), "llama3")

    def test_resolve_model_without_mapping(self):
        """Falls back to default_model when no mapping exists."""
        cfg = Config(
            model_map={"claude-3-opus": "qwen2.5-coder"},
            default_model="fallback-model",
        )
        self.assertEqual(cfg.resolve_model("claude-3-haiku"), "fallback-model")
        self.assertEqual(cfg.resolve_model("unknown-model"), "fallback-model")


class TestValidate(unittest.TestCase):
    """Test Config.validate() method."""

    def test_validate_valid(self):
        """Valid config passes validation without raising."""
        cfg = Config(port=5304, request_timeout=300)
        # Should not raise
        cfg.validate()

    def test_validate_invalid_port_zero(self):
        """Port 0 raises ValueError."""
        cfg = Config(port=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_port_too_high(self):
        """Port 70000 raises ValueError."""
        cfg = Config(port=70000)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_timeout_zero(self):
        """Timeout 0 raises ValueError."""
        cfg = Config(request_timeout=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validate_invalid_timeout_negative(self):
        """Timeout -1 raises ValueError."""
        cfg = Config(request_timeout=-1)
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
