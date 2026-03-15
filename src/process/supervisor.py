"""
Generic process supervisor.

Manages multiple child processes with health checks, auto-restart,
exponential backoff, log streaming, and clean shutdown. Pure Python
stdlib — works on macOS, Linux, and Windows.

Reusable in any Python project. Knows nothing about Purrxy,
llama.cpp, or any specific application.

Architecture
============

The supervisor has three layers:

    ProcessConfig   — Declarative process definition (what to run)
    ProcessHandle   — Single-process lifecycle (spawn, stop, restart logic)
    Supervisor      — Multi-process orchestrator (ordering, dependencies, monitor loop)

Lifecycle
---------

    1. Register processes via add_process() in dependency order.
    2. Call start() — this is blocking.
    3. Supervisor starts each process in order, waiting for dependencies
       to pass health checks before starting dependents.
    4. Monitor loop runs every 1s: detects crashes, auto-restarts with
       exponential backoff, resets backoff after sustained uptime.
    5. On SIGINT/SIGTERM, stops all processes in reverse order.

Crash Recovery
--------------

    - Each process gets max_restarts attempts (default 5).
    - Backoff: 2^N seconds with 25% jitter (capped at backoff_max).
    - If a process stays alive for healthy_reset_time (default 30s),
      the restart counter resets to 0.
    - If max_restarts is exceeded, the entire supervisor shuts down.

Cross-Platform
--------------

    - macOS/Linux: SIGTERM for graceful stop, SIGKILL as fallback.
    - Windows: subprocess.terminate() (TerminateProcess), no graceful
      shutdown for child processes (Windows limitation).
    - PID liveness: os.kill(pid, 0) on Unix, OpenProcess on Windows.

Log Streaming
-------------

    Each child process gets a daemon thread that reads its stdout pipe
    line by line. Output goes to both Python logging (visible in the
    supervisor's terminal) and a persistent log file in config_dir.

PID Files
---------

    Written to config_dir/{name}.pid for each process plus supervisor.pid.
    The --stop command reads these files and sends signals directly,
    even when the supervisor process itself is gone (orphan recovery).

Usage
=====

    from src.process import Supervisor, ProcessConfig

    sv = Supervisor(config_dir="~/.config/my-app")

    sv.add_process(ProcessConfig(
        name="database",
        cmd=["postgres", "-D", "/data"],
        health_url="http://localhost:5432/health",
    ))

    sv.add_process(ProcessConfig(
        name="api",
        cmd=["python3", "server.py"],
        depends_on="database",
        health_url="http://localhost:8000/health",
    ))

    sv.start()  # Blocks until signal or all processes crash out
"""

import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("supervisor")

# Cross-platform constants
IS_WINDOWS = sys.platform == "win32"


@dataclass
class ProcessConfig:
    """Configuration for a managed process.

    Attributes:
        name: Human-readable process name (used in logs, PID files).
        cmd: Command and arguments as a list (subprocess argv).
        env: Additional environment variables (merged with os.environ).
        cwd: Working directory for the process.
        health_url: URL to poll for readiness (GET, expects 2xx).
        health_timeout: Max seconds to wait for health check on startup.
        health_interval: Seconds between health check polls.
        depends_on: Name of another process that must be healthy first.
        max_restarts: Max consecutive restarts before giving up.
        backoff_base: Base seconds for exponential backoff.
        backoff_max: Maximum backoff delay in seconds.
        auto_restart: Whether to restart on crash.
        healthy_reset_time: Seconds of uptime before restart count resets.
    """
    name: str
    cmd: List[str]
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None
    health_url: Optional[str] = None
    health_timeout: int = 120
    health_interval: float = 2.0
    depends_on: Optional[str] = None
    max_restarts: int = 5
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    auto_restart: bool = True
    healthy_reset_time: float = 30.0


class ProcessHandle:
    """Manages a single child process lifecycle."""

    def __init__(self, config: ProcessConfig, config_dir: str):
        self.config = config
        self.name = config.name
        self._config_dir = config_dir
        self._process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._last_start: float = 0.0
        self._intentional_stop = False

    @property
    def pid(self) -> Optional[int]:
        """Current process PID, or None if not running."""
        return self._process.pid if self._process else None

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def start(self) -> None:
        """Spawn the process."""
        if self.is_alive():
            logger.warning("%s: already running (PID %d)", self.name, self.pid)
            return

        self._intentional_stop = False

        # Build environment
        env = os.environ.copy()
        if self.config.env:
            env.update(self.config.env)

        # Spawn
        try:
            self._process = subprocess.Popen(
                self.config.cmd,
                env=env,
                cwd=self.config.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            logger.error("%s: command not found: %s", self.name, self.config.cmd[0])
            raise
        except PermissionError:
            logger.error("%s: permission denied: %s", self.name, self.config.cmd[0])
            raise

        self._last_start = time.time()
        self._write_pid_file()

        # Start log streaming thread
        self._log_thread = threading.Thread(
            target=self._stream_output,
            name=f"log-{self.name}",
            daemon=True,
        )
        self._log_thread.start()

        logger.info("%s: started (PID %d)", self.name, self.pid)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the process gracefully, then force kill if needed."""
        self._intentional_stop = True

        if not self.is_alive():
            logger.info("%s: not running", self.name)
            self._remove_pid_file()
            return

        pid = self.pid
        logger.info("%s: stopping (PID %d)...", self.name, pid)

        # Graceful shutdown
        if IS_WINDOWS:
            self._process.terminate()
        else:
            self._process.send_signal(signal.SIGTERM)

        try:
            self._process.wait(timeout=timeout)
            logger.info("%s: stopped", self.name)
        except subprocess.TimeoutExpired:
            logger.warning("%s: force killing (PID %d)...", self.name, pid)
            self._process.kill()
            try:
                self._process.wait(timeout=3)
                logger.info("%s: killed", self.name)
            except subprocess.TimeoutExpired:
                logger.error("%s: could not be killed (PID %d)", self.name, pid)

        self._remove_pid_file()

    def is_alive(self) -> bool:
        """Check if the process is currently running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def should_restart(self) -> bool:
        """Whether this process should be auto-restarted."""
        if self._intentional_stop:
            return False
        if not self.config.auto_restart:
            return False
        if self._restart_count >= self.config.max_restarts:
            return False
        return True

    def backoff_delay(self) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(
            self.config.backoff_base ** self._restart_count,
            self.config.backoff_max,
        )
        # Add 0–25% jitter
        jitter = delay * 0.25 * random.random()
        return delay + jitter

    def increment_restart(self) -> None:
        """Increment the restart counter."""
        self._restart_count += 1

    def maybe_reset_backoff(self) -> None:
        """Reset restart count if process has been healthy long enough."""
        if self.is_alive() and self._last_start > 0:
            uptime = time.time() - self._last_start
            if uptime >= self.config.healthy_reset_time:
                if self._restart_count > 0:
                    logger.info(
                        "%s: stable for %.0fs, resetting restart counter",
                        self.name, uptime,
                    )
                    self._restart_count = 0

    def wait_healthy(self) -> bool:
        """Poll the health URL until ready or timeout.

        Returns:
            True if healthy, False if timed out.
        """
        if not self.config.health_url:
            return True

        logger.info("%s: waiting for health check (%s)...",
                     self.name, self.config.health_url)

        deadline = time.time() + self.config.health_timeout
        waited = 0

        while time.time() < deadline:
            if not self.is_alive():
                logger.error("%s: process died during health check", self.name)
                return False

            try:
                req = urllib.request.Request(self.config.health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        logger.info("%s: healthy (%.0fs)", self.name, waited)
                        return True
            except (urllib.error.URLError, OSError, ValueError):
                pass

            time.sleep(self.config.health_interval)
            waited += self.config.health_interval

            if int(waited) % 10 == 0 and waited > 0:
                logger.info("%s: still waiting... (%.0fs / %ds)",
                             self.name, waited, self.config.health_timeout)

        logger.error("%s: health check timed out after %ds",
                      self.name, self.config.health_timeout)
        return False

    def _stream_output(self) -> None:
        """Read process stdout line by line, forward to logging and log file."""
        proc_logger = logging.getLogger(f"supervisor.{self.name}")
        log_path = os.path.join(self._config_dir, f"{self.name}.log")

        try:
            log_file = open(log_path, "a")
        except OSError:
            log_file = None

        try:
            for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    proc_logger.info("%s", line)
                    if log_file:
                        log_file.write(line + "\n")
                        log_file.flush()
        except (ValueError, OSError):
            pass  # Pipe closed
        finally:
            if log_file:
                log_file.close()

    def _write_pid_file(self) -> None:
        """Write the process PID to a file."""
        pid_path = os.path.join(self._config_dir, f"{self.name}.pid")
        try:
            with open(pid_path, "w") as f:
                f.write(str(self.pid))
        except OSError as e:
            logger.warning("Failed to write PID file %s: %s", pid_path, e)

    def _remove_pid_file(self) -> None:
        """Remove the PID file."""
        pid_path = os.path.join(self._config_dir, f"{self.name}.pid")
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass


class Supervisor:
    """Orchestrates multiple processes with dependency ordering and auto-restart.

    Generic — not tied to any specific application. Configure processes via
    ProcessConfig objects and call start().
    """

    def __init__(self, config_dir: str):
        """
        Args:
            config_dir: Directory for PID files and logs.
                Will be created if it doesn't exist.
        """
        self._config_dir = os.path.expanduser(config_dir)
        self._processes: Dict[str, ProcessHandle] = {}
        self._start_order: List[str] = []
        self._running = False
        self._lock = threading.Lock()

    def add_process(self, config: ProcessConfig) -> None:
        """Register a process to be managed.

        Call this before start(). Processes are started in the order added,
        respecting depends_on relationships.

        Args:
            config: Process configuration.

        Raises:
            ValueError: If a process with the same name already exists.
        """
        if config.name in self._processes:
            raise ValueError(f"Process '{config.name}' already registered")

        self._processes[config.name] = ProcessHandle(config, self._config_dir)
        self._start_order.append(config.name)

    def start(self, names: Optional[List[str]] = None) -> None:
        """Start managed processes and monitor them. Blocks until shutdown.

        Args:
            names: Optional list of process names to start. If None, starts all.
                Dependencies are always started regardless.
        """
        os.makedirs(self._config_dir, exist_ok=True)
        self._setup_signal_handlers()

        # Resolve which processes to start
        to_start = self._resolve_start_order(names)

        if not to_start:
            logger.warning("No processes to start")
            return

        logger.info("Starting %d process(es): %s",
                      len(to_start), ", ".join(to_start))

        # Start processes in order
        for name in to_start:
            handle = self._processes[name]

            # Wait for dependency to be healthy first
            if handle.config.depends_on:
                dep_name = handle.config.depends_on
                dep = self._processes.get(dep_name)
                if dep and not dep.wait_healthy():
                    logger.error(
                        "Dependency '%s' not healthy, cannot start '%s'",
                        dep_name, name,
                    )
                    self.stop()
                    return

            try:
                handle.start()
            except (FileNotFoundError, PermissionError):
                logger.error("Failed to start '%s', shutting down", name)
                self.stop()
                return

        # Write supervisor PID
        self._write_supervisor_pid()

        logger.info("All processes started. Monitoring...")

        # Enter monitor loop
        self._running = True
        self._monitor_loop()

    def stop(self) -> None:
        """Stop all managed processes in reverse order."""
        self._running = False

        # Stop in reverse order (proxy before backend, etc.)
        for name in reversed(self._start_order):
            handle = self._processes.get(name)
            if handle and handle.is_alive():
                handle.stop()

        self._remove_supervisor_pid()
        logger.info("All processes stopped")

    def stop_from_pid_files(self) -> None:
        """Stop processes using PID files (for external --stop command).

        This works even when the supervisor itself isn't running — it reads
        PID files and sends signals directly.
        """
        # Stop in reverse order
        stopped = []
        for name in reversed(self._start_order):
            pid = self._read_pid_file(f"{name}.pid")
            if pid and _is_pid_alive(pid):
                logger.info("Stopping %s (PID %d)...", name, pid)
                _stop_pid(pid)
                stopped.append(name)
                self._remove_pid_file(f"{name}.pid")
            else:
                if pid:
                    logger.info("%s: not running (stale PID %d)", name, pid)
                    self._remove_pid_file(f"{name}.pid")
                else:
                    logger.info("%s: no PID file", name)

        # Also stop supervisor PID
        sv_pid = self._read_pid_file("supervisor.pid")
        if sv_pid and sv_pid != os.getpid() and _is_pid_alive(sv_pid):
            logger.info("Stopping supervisor (PID %d)...", sv_pid)
            _stop_pid(sv_pid)
        self._remove_pid_file("supervisor.pid")

        if stopped:
            logger.info("Stopped: %s", ", ".join(stopped))
        else:
            logger.info("No running processes found")

    def status(self) -> Dict[str, dict]:
        """Get status of all managed processes.

        Returns:
            Dict mapping process name to status info. Example:
            {
                "backend": {"running": True, "pid": 1234},
                "proxy": {"running": False, "pid": None},
            }
        """
        result = {}

        for name in self._start_order:
            pid = self._read_pid_file(f"{name}.pid")
            running = pid is not None and _is_pid_alive(pid)
            result[name] = {
                "running": running,
                "pid": pid if running else None,
            }

        # Supervisor itself
        sv_pid = self._read_pid_file("supervisor.pid")
        result["supervisor"] = {
            "running": sv_pid is not None and _is_pid_alive(sv_pid),
            "pid": sv_pid if sv_pid and _is_pid_alive(sv_pid) else None,
        }

        return result

    def _monitor_loop(self) -> None:
        """Main loop: check process health, restart crashed processes."""
        while self._running:
            with self._lock:
                for name in self._start_order:
                    handle = self._processes[name]

                    if not handle.is_alive() and not handle._intentional_stop:
                        # Process crashed
                        if handle.should_restart():
                            delay = handle.backoff_delay()
                            handle.increment_restart()
                            logger.warning(
                                "%s: crashed, restarting in %.1fs "
                                "(attempt %d/%d)",
                                name, delay,
                                handle.restart_count,
                                handle.config.max_restarts,
                            )
                            time.sleep(delay)
                            try:
                                handle.start()
                            except (FileNotFoundError, PermissionError):
                                logger.error(
                                    "%s: restart failed, giving up", name
                                )
                                self.stop()
                                return
                        else:
                            logger.error(
                                "%s: exceeded max restarts (%d), "
                                "shutting down",
                                name, handle.config.max_restarts,
                            )
                            self.stop()
                            return
                    else:
                        # Process running — maybe reset backoff
                        handle.maybe_reset_backoff()

            time.sleep(1)

    def _resolve_start_order(
        self, names: Optional[List[str]] = None,
    ) -> List[str]:
        """Determine process start order, respecting dependencies.

        Args:
            names: Specific process names to start. If None, start all.

        Returns:
            Ordered list of process names to start.
        """
        if names is None:
            return list(self._start_order)

        # Include dependencies
        needed = set()
        for name in names:
            if name not in self._processes:
                logger.warning("Unknown process: %s", name)
                continue
            needed.add(name)
            dep = self._processes[name].config.depends_on
            if dep and dep in self._processes:
                needed.add(dep)

        # Maintain original registration order
        return [n for n in self._start_order if n in needed]

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for clean shutdown."""
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.info("Received %s, shutting down...", sig_name)
        self.stop()

    def _write_supervisor_pid(self) -> None:
        pid_path = os.path.join(self._config_dir, "supervisor.pid")
        try:
            with open(pid_path, "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            pass

    def _remove_supervisor_pid(self) -> None:
        self._remove_pid_file("supervisor.pid")

    def _read_pid_file(self, filename: str) -> Optional[int]:
        """Read a PID from a file. Returns None if missing or invalid."""
        pid_path = os.path.join(self._config_dir, filename)
        try:
            with open(pid_path, "r") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _remove_pid_file(self, filename: str) -> None:
        pid_path = os.path.join(self._config_dir, filename)
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass


# ── Cross-platform process utilities ──────────────────────────────────────────


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Works on macOS, Linux, and Windows.
    """
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except (AttributeError, OSError):
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # Process exists but we can't signal it


def _stop_pid(pid: int, timeout: float = 5.0) -> bool:
    """Send SIGTERM to a PID, wait, then SIGKILL if needed.

    Returns True if the process was stopped.
    """
    try:
        if IS_WINDOWS:
            os.kill(pid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return True

    # Wait for termination
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.5)

    # Force kill
    try:
        os.kill(pid, signal.SIGKILL if not IS_WINDOWS else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    time.sleep(1)
    return not _is_pid_alive(pid)
