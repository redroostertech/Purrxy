"""
Tests for the process supervisor, env parser, and backend modules.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from src.process.env import load_dotenv, load_dotenv_into_environ
from src.backend.config import BackendConfig, find_llama_server, build_backend_command
from src.process.supervisor import (
    ProcessConfig,
    ProcessHandle,
    Supervisor,
    _is_pid_alive,
    _stop_pid,
)


# ── env.py tests ──────────────────────────────────────────────────────────────


class TestLoadDotenv(unittest.TestCase):
    """Tests for the .env file parser."""

    def _write_env(self, content: str) -> str:
        """Write content to a temp .env file, return path."""
        fd, path = tempfile.mkstemp(suffix=".env")
        os.write(fd, content.encode())
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    def test_basic_key_value(self):
        path = self._write_env("FOO=bar\nBAZ=qux\n")
        result = load_dotenv(path)
        self.assertEqual(result, {"FOO": "bar", "BAZ": "qux"})

    def test_comments_and_blanks(self):
        path = self._write_env("# comment\nFOO=bar\n\n# another\nBAZ=qux\n")
        result = load_dotenv(path)
        self.assertEqual(result, {"FOO": "bar", "BAZ": "qux"})

    def test_whitespace_stripping(self):
        path = self._write_env("  FOO  =  bar  \n")
        result = load_dotenv(path)
        self.assertEqual(result, {"FOO": "bar"})

    def test_quoted_values(self):
        path = self._write_env('FOO="hello world"\nBAR=\'single\'\n')
        result = load_dotenv(path)
        self.assertEqual(result["FOO"], "hello world")
        self.assertEqual(result["BAR"], "single")

    def test_tilde_expansion(self):
        path = self._write_env("DIR=~/models\n")
        result = load_dotenv(path)
        home = os.path.expanduser("~")
        self.assertEqual(result["DIR"], f"{home}/models")

    def test_value_with_equals(self):
        path = self._write_env("URL=http://host:8080/path?a=1&b=2\n")
        result = load_dotenv(path)
        self.assertEqual(result["URL"], "http://host:8080/path?a=1&b=2")

    def test_empty_value(self):
        path = self._write_env("EMPTY=\n")
        result = load_dotenv(path)
        self.assertEqual(result["EMPTY"], "")

    def test_missing_file(self):
        result = load_dotenv("/nonexistent/path/.env")
        self.assertEqual(result, {})

    def test_lines_without_equals(self):
        path = self._write_env("no_equals_here\nFOO=bar\n")
        result = load_dotenv(path)
        self.assertEqual(result, {"FOO": "bar"})


class TestLoadDotenvIntoEnviron(unittest.TestCase):
    """Tests for loading .env into os.environ."""

    def _write_env(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".env")
        os.write(fd, content.encode())
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    def test_sets_env_var(self):
        path = self._write_env("TEST_SUPERVISOR_VAR=hello\n")
        self.addCleanup(lambda: os.environ.pop("TEST_SUPERVISOR_VAR", None))
        load_dotenv_into_environ(path)
        self.assertEqual(os.environ["TEST_SUPERVISOR_VAR"], "hello")

    def test_no_overwrite_by_default(self):
        path = self._write_env("TEST_SUPERVISOR_VAR=new\n")
        os.environ["TEST_SUPERVISOR_VAR"] = "existing"
        self.addCleanup(lambda: os.environ.pop("TEST_SUPERVISOR_VAR", None))
        load_dotenv_into_environ(path)
        self.assertEqual(os.environ["TEST_SUPERVISOR_VAR"], "existing")

    def test_overwrite_when_requested(self):
        path = self._write_env("TEST_SUPERVISOR_VAR=new\n")
        os.environ["TEST_SUPERVISOR_VAR"] = "existing"
        self.addCleanup(lambda: os.environ.pop("TEST_SUPERVISOR_VAR", None))
        load_dotenv_into_environ(path, overwrite=True)
        self.assertEqual(os.environ["TEST_SUPERVISOR_VAR"], "new")


# ── backend.py tests ─────────────────────────────────────────────────────────


class TestFindLlamaServer(unittest.TestCase):
    """Tests for llama-server binary discovery."""

    def test_finds_standard_location(self):
        """Finds binary at build/bin/llama-server."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = os.path.join(tmpdir, "build", "bin")
            os.makedirs(bin_dir)
            server_path = os.path.join(bin_dir, "llama-server")
            with open(server_path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(server_path, 0o755)

            result = find_llama_server(tmpdir)
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("llama-server"))

    def test_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_llama_server(tmpdir)
            self.assertIsNone(result)

    def test_skips_non_executable(self):
        """Non-executable files are not returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = os.path.join(tmpdir, "build", "bin")
            os.makedirs(bin_dir)
            server_path = os.path.join(bin_dir, "llama-server")
            with open(server_path, "w") as f:
                f.write("not executable")
            # Don't chmod +x

            result = find_llama_server(tmpdir)
            self.assertIsNone(result)


class TestBuildBackendCommand(unittest.TestCase):
    """Tests for backend command construction."""

    def test_basic_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake binary
            bin_dir = os.path.join(tmpdir, "build", "bin")
            os.makedirs(bin_dir)
            server_path = os.path.join(bin_dir, "llama-server")
            with open(server_path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(server_path, 0o755)

            config = BackendConfig(
                llama_cpp_dir=tmpdir,
                model_dir="/models",
                model_name="test.gguf",
                context_size=4096,
                gpu_layers=50,
                parallel_slots=2,
                flash_attention=True,
                host="0.0.0.0",
                port=9090,
            )

            cmd = build_backend_command(config)

            self.assertIn("--model", cmd)
            self.assertIn("/models/test.gguf", cmd)
            self.assertIn("--ctx-size", cmd)
            self.assertIn("4096", cmd)
            self.assertIn("--n-gpu-layers", cmd)
            self.assertIn("50", cmd)
            self.assertIn("--parallel", cmd)
            self.assertIn("2", cmd)
            self.assertIn("--host", cmd)
            self.assertIn("0.0.0.0", cmd)
            self.assertIn("--port", cmd)
            self.assertIn("9090", cmd)
            self.assertIn("--flash-attn", cmd)

    def test_no_flash_attention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bin_dir = os.path.join(tmpdir, "build", "bin")
            os.makedirs(bin_dir)
            server_path = os.path.join(bin_dir, "llama-server")
            with open(server_path, "w") as f:
                f.write("#!/bin/sh\n")
            os.chmod(server_path, 0o755)

            config = BackendConfig(
                llama_cpp_dir=tmpdir,
                model_dir="/models",
                model_name="test.gguf",
                flash_attention=False,
            )

            cmd = build_backend_command(config)
            self.assertNotIn("--flash-attn", cmd)


class TestBackendConfig(unittest.TestCase):
    """Tests for BackendConfig."""

    def test_model_path(self):
        config = BackendConfig(
            llama_cpp_dir="/llama",
            model_dir="/models",
            model_name="test.gguf",
        )
        self.assertEqual(config.model_path, "/models/test.gguf")

    def test_health_url(self):
        config = BackendConfig(
            llama_cpp_dir="/llama",
            model_dir="/models",
            model_name="test.gguf",
            host="0.0.0.0",
            port=9090,
        )
        self.assertEqual(config.health_url, "http://0.0.0.0:9090/health")

    def test_validate_missing_binary(self):
        config = BackendConfig(
            llama_cpp_dir="/nonexistent",
            model_dir="/models",
            model_name="test.gguf",
        )
        with self.assertRaises(FileNotFoundError):
            config.validate()


# ── supervisor.py tests ───────────────────────────────────────────────────────


class TestProcessConfig(unittest.TestCase):
    """Tests for ProcessConfig defaults."""

    def test_defaults(self):
        config = ProcessConfig(name="test", cmd=["echo", "hi"])
        self.assertEqual(config.max_restarts, 5)
        self.assertEqual(config.backoff_base, 2.0)
        self.assertEqual(config.backoff_max, 60.0)
        self.assertTrue(config.auto_restart)
        self.assertIsNone(config.depends_on)
        self.assertIsNone(config.health_url)


class TestProcessHandle(unittest.TestCase):
    """Tests for ProcessHandle lifecycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {self.tmpdir}"))

    def test_start_and_stop(self):
        """Start a simple process and stop it."""
        config = ProcessConfig(
            name="sleeper",
            cmd=[sys.executable, "-c", "import time; time.sleep(60)"],
        )
        handle = ProcessHandle(config, self.tmpdir)
        handle.start()

        self.assertTrue(handle.is_alive())
        self.assertIsNotNone(handle.pid)

        # PID file should exist
        pid_path = os.path.join(self.tmpdir, "sleeper.pid")
        self.assertTrue(os.path.exists(pid_path))

        handle.stop(timeout=2)
        self.assertFalse(handle.is_alive())

        # PID file should be cleaned up
        self.assertFalse(os.path.exists(pid_path))

    def test_backoff_delay_increases(self):
        config = ProcessConfig(
            name="test",
            cmd=["echo"],
            backoff_base=2.0,
            backoff_max=60.0,
        )
        handle = ProcessHandle(config, self.tmpdir)

        # Restart count 0: base delay ~2s
        delay0 = handle.backoff_delay()
        self.assertGreaterEqual(delay0, 1.0)  # 2^0 = 1, with jitter
        self.assertLessEqual(delay0, 1.5)

        handle.increment_restart()
        # Restart count 1: ~2s
        delay1 = handle.backoff_delay()
        self.assertGreaterEqual(delay1, 2.0)
        self.assertLessEqual(delay1, 3.0)

        handle.increment_restart()
        # Restart count 2: ~4s
        delay2 = handle.backoff_delay()
        self.assertGreaterEqual(delay2, 4.0)
        self.assertLessEqual(delay2, 6.0)

    def test_should_restart_false_after_intentional_stop(self):
        config = ProcessConfig(name="test", cmd=["echo"])
        handle = ProcessHandle(config, self.tmpdir)
        handle._intentional_stop = True
        self.assertFalse(handle.should_restart())

    def test_should_restart_false_after_max_restarts(self):
        config = ProcessConfig(name="test", cmd=["echo"], max_restarts=3)
        handle = ProcessHandle(config, self.tmpdir)
        handle._restart_count = 3
        self.assertFalse(handle.should_restart())

    def test_should_restart_false_when_disabled(self):
        config = ProcessConfig(name="test", cmd=["echo"], auto_restart=False)
        handle = ProcessHandle(config, self.tmpdir)
        self.assertFalse(handle.should_restart())


class TestSupervisor(unittest.TestCase):
    """Tests for the Supervisor orchestrator."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.system(f"rm -rf {self.tmpdir}"))

    def test_add_process(self):
        sv = Supervisor(config_dir=self.tmpdir)
        sv.add_process(ProcessConfig(name="a", cmd=["echo"]))
        sv.add_process(ProcessConfig(name="b", cmd=["echo"]))
        self.assertEqual(len(sv._processes), 2)
        self.assertEqual(sv._start_order, ["a", "b"])

    def test_add_duplicate_raises(self):
        sv = Supervisor(config_dir=self.tmpdir)
        sv.add_process(ProcessConfig(name="a", cmd=["echo"]))
        with self.assertRaises(ValueError):
            sv.add_process(ProcessConfig(name="a", cmd=["echo"]))

    def test_resolve_start_order_all(self):
        sv = Supervisor(config_dir=self.tmpdir)
        sv.add_process(ProcessConfig(name="a", cmd=["echo"]))
        sv.add_process(ProcessConfig(name="b", cmd=["echo"]))
        order = sv._resolve_start_order(None)
        self.assertEqual(order, ["a", "b"])

    def test_resolve_start_order_with_dependency(self):
        sv = Supervisor(config_dir=self.tmpdir)
        sv.add_process(ProcessConfig(name="db", cmd=["echo"]))
        sv.add_process(ProcessConfig(name="api", cmd=["echo"], depends_on="db"))
        # Requesting only "api" should include "db" as dependency
        order = sv._resolve_start_order(["api"])
        self.assertEqual(order, ["db", "api"])

    def test_resolve_start_order_specific(self):
        sv = Supervisor(config_dir=self.tmpdir)
        sv.add_process(ProcessConfig(name="a", cmd=["echo"]))
        sv.add_process(ProcessConfig(name="b", cmd=["echo"]))
        sv.add_process(ProcessConfig(name="c", cmd=["echo"]))
        order = sv._resolve_start_order(["b"])
        self.assertEqual(order, ["b"])

    def test_status_no_processes(self):
        sv = Supervisor(config_dir=self.tmpdir)
        status = sv.status()
        # Should at least have supervisor entry
        self.assertIn("supervisor", status)

    def test_pid_file_operations(self):
        sv = Supervisor(config_dir=self.tmpdir)
        os.makedirs(self.tmpdir, exist_ok=True)

        # Write and read
        sv._write_supervisor_pid()
        pid = sv._read_pid_file("supervisor.pid")
        self.assertEqual(pid, os.getpid())

        # Remove
        sv._remove_pid_file("supervisor.pid")
        pid = sv._read_pid_file("supervisor.pid")
        self.assertIsNone(pid)

    def test_read_nonexistent_pid_file(self):
        sv = Supervisor(config_dir=self.tmpdir)
        pid = sv._read_pid_file("nonexistent.pid")
        self.assertIsNone(pid)


class TestCrossPlatformUtils(unittest.TestCase):
    """Tests for cross-platform process utilities."""

    def test_is_pid_alive_self(self):
        """Current process should be alive."""
        self.assertTrue(_is_pid_alive(os.getpid()))

    def test_is_pid_alive_nonexistent(self):
        """A very high PID should not be alive."""
        self.assertFalse(_is_pid_alive(99999999))


if __name__ == "__main__":
    unittest.main()
