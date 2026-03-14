#!/usr/bin/env python3
"""
lana-proxy: Anthropic Messages API -> OpenAI Chat Completions translator.

Part of the LANA suite (Locally Autonomous Neural Agent).
Translates Anthropic Messages API requests into OpenAI-compatible format
and forwards them to a local llama.cpp server.

Usage:
    python3 lana-proxy.py [OPTIONS]
    python3 lana-proxy.py --start [--backend-only | --proxy-only]
    python3 lana-proxy.py --stop
    python3 lana-proxy.py --status

Examples:
    # Run proxy directly (foreground)
    python3 lana-proxy.py --port 5304 --backend http://127.0.0.1:8080

    # Manage full stack via supervisor
    python3 lana-proxy.py --start                    # Start backend + proxy
    python3 lana-proxy.py --start --backend-only     # Start only llama.cpp
    python3 lana-proxy.py --start --proxy-only       # Start only the proxy
    python3 lana-proxy.py --stop                     # Stop all
    python3 lana-proxy.py --status                   # Show process status
"""

import argparse
import logging
import os
import signal
import sys

__version__ = open(os.path.join(os.path.dirname(__file__), "VERSION")).read().strip()

from src.proxy.config import Config
from src.proxy.monitoring.dashboard import Dashboard
from src.proxy.monitoring.discovery import discover_models
from src.proxy.middleware.queue import RequestQueue
from src.proxy.server import ThreadedHTTPServer
from src.proxy.monitoring.stats import Stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="lana-proxy",
        description="Anthropic Messages API to OpenAI Chat Completions proxy",
    )

    # ── Supervisor actions ────────────────────────────────────────────────
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--start",
        action="store_true",
        help="Start backend + proxy under the process supervisor",
    )
    action.add_argument(
        "--stop",
        action="store_true",
        help="Stop all supervised processes",
    )
    action.add_argument(
        "--status",
        action="store_true",
        help="Show status of supervised processes",
    )
    action.add_argument(
        "--logs",
        action="store_true",
        help="Tail supervisor log output (like pm2 logs)",
    )
    action.add_argument(
        "--restart",
        action="store_true",
        help="Stop then start all supervised processes",
    )

    # Supervisor modifiers
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="With --start: only start the llama.cpp backend",
    )
    parser.add_argument(
        "--proxy-only",
        action="store_true",
        help="With --start: only start the proxy (backend assumed external)",
    )
    # Hidden flags: used internally when re-execing as a daemon
    parser.add_argument("--_daemon", action="store_true", help=argparse.SUPPRESS)

    # ── Proxy configuration ───────────────────────────────────────────────
    parser.add_argument(
        "--host",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to listen on (default: 5304)",
    )
    parser.add_argument(
        "--backend",
        help="llama.cpp backend URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Backend request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--model-map",
        nargs="*",
        metavar="ANTHROPIC=BACKEND",
        default=[],
        help="Model name mappings (e.g., claude-3-opus=qwen2.5-coder)",
    )
    parser.add_argument(
        "--default-model",
        help="Default backend model name when no mapping exists (default: local-model)",
    )
    parser.add_argument(
        "--route",
        nargs="*",
        metavar="MODEL=URL",
        default=[],
        help="Route model names to specific backend URLs (e.g., claude-3-opus=http://localhost:8080)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Max concurrent backend requests (default: 1)",
    )
    parser.add_argument(
        "--queue-timeout",
        type=float,
        help="Seconds to wait for a queue slot (default: 120)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable live terminal dashboard",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="Record request/response pairs to JSONL file",
    )
    parser.add_argument(
        "--replay",
        metavar="PATH",
        help="Replay responses from JSONL file (no backend needed)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable prompt cache",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        metavar="SECONDS",
        help="Cache TTL in seconds (default: 300)",
    )
    parser.add_argument(
        "--cache-max-size",
        type=int,
        metavar="N",
        help="Max cache entries (default: 100)",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Disk cache directory for persistence",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        metavar="N",
        help="Session token budget, 0=unlimited (default: 0)",
    )
    parser.add_argument(
        "--analytics",
        metavar="PATH",
        help="Analytics log file path (JSONL)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build Config from environment variables, then override with CLI args."""
    config = Config.from_env()

    # CLI overrides
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.backend:
        config.backend_url = args.backend
    if args.log_level:
        config.log_level = args.log_level
    if args.timeout:
        config.request_timeout = args.timeout
    if args.default_model:
        config.default_model = args.default_model
    if args.max_concurrent is not None:
        config.max_concurrent = args.max_concurrent
    if args.queue_timeout is not None:
        config.queue_timeout = args.queue_timeout
    if args.dashboard:
        config.dashboard = True
    if args.record:
        config.record_path = args.record
    if args.replay:
        config.replay_path = args.replay
    if args.cache:
        config.cache_enabled = True
    if args.cache_ttl is not None:
        config.cache_ttl = args.cache_ttl
    if args.cache_max_size is not None:
        config.cache_max_size = args.cache_max_size
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.token_budget is not None:
        config.token_budget = args.token_budget
    if args.analytics:
        config.analytics_path = args.analytics

    # Parse model mappings: "claude-3-opus=qwen2.5" -> {"claude-3-opus": "qwen2.5"}
    for mapping in args.model_map:
        if "=" in mapping:
            key, value = mapping.split("=", 1)
            config.model_map[key.strip()] = value.strip()

    # Parse routes: "claude-3-opus=http://localhost:8080"
    for route in args.route:
        if "=" in route:
            key, value = route.split("=", 1)
            config.routes[key.strip()] = value.strip()

    config.validate()

    return config


def setup_logging(level: str):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def _build_proxy_forward_args(args: argparse.Namespace) -> list:
    """Build CLI args to forward to the child proxy process.

    Extracts proxy-specific flags from the parsed args and returns
    them as a list suitable for subprocess argv.
    """
    forward = []
    if args.host:
        forward.extend(["--host", args.host])
    if args.port:
        forward.extend(["--port", str(args.port)])
    if args.backend:
        forward.extend(["--backend", args.backend])
    if args.log_level:
        forward.extend(["--log-level", args.log_level])
    if args.timeout:
        forward.extend(["--timeout", str(args.timeout)])
    if args.default_model:
        forward.extend(["--default-model", args.default_model])
    for mapping in args.model_map:
        forward.extend(["--model-map", mapping])
    for route in args.route:
        forward.extend(["--route", route])
    if args.max_concurrent is not None:
        forward.extend(["--max-concurrent", str(args.max_concurrent)])
    if args.queue_timeout is not None:
        forward.extend(["--queue-timeout", str(args.queue_timeout)])
    if args.dashboard:
        forward.append("--dashboard")
    if args.record:
        forward.extend(["--record", args.record])
    if args.replay:
        forward.extend(["--replay", args.replay])
    if args.cache:
        forward.append("--cache")
    if args.cache_ttl is not None:
        forward.extend(["--cache-ttl", str(args.cache_ttl)])
    if args.cache_max_size is not None:
        forward.extend(["--cache-max-size", str(args.cache_max_size)])
    if args.cache_dir:
        forward.extend(["--cache-dir", args.cache_dir])
    if args.token_budget is not None:
        forward.extend(["--token-budget", str(args.token_budget)])
    if args.analytics:
        forward.extend(["--analytics", args.analytics])
    return forward


def _get_supervisor_context(args: argparse.Namespace):
    """Set up common supervisor context: load .env, build Supervisor with processes."""
    from src.backend import BackendConfig, build_backend_command
    from src.process import ProcessConfig, Supervisor, load_dotenv_into_environ

    project_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "lana-proxy")

    env_path = os.path.join(project_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv_into_environ(env_path)

    start_backend = not getattr(args, "proxy_only", False)
    start_proxy = not getattr(args, "backend_only", False)

    sv = Supervisor(config_dir=config_dir)

    if start_backend:
        backend_cfg = BackendConfig.from_env()
        backend_cmd = build_backend_command(backend_cfg)
        sv.add_process(ProcessConfig(
            name="backend",
            cmd=backend_cmd,
            health_url=backend_cfg.health_url,
            health_timeout=120,
        ))

    if start_proxy:
        proxy_cmd = [
            sys.executable,
            os.path.join(project_dir, "lana-proxy.py"),
        ] + _build_proxy_forward_args(args)

        if start_backend:
            backend_cfg = BackendConfig.from_env()
            if "--backend" not in proxy_cmd:
                proxy_cmd.extend([
                    "--backend",
                    f"http://{backend_cfg.host}:{backend_cfg.port}",
                ])

        sv.add_process(ProcessConfig(
            name="proxy",
            cmd=proxy_cmd,
            depends_on="backend" if start_backend else None,
            health_url=f"http://127.0.0.1:{args.port or 5304}/health",
            health_timeout=30,
        ))

    return sv, config_dir, project_dir, start_backend, start_proxy


def _print_status(sv) -> None:
    """Print process status table."""
    status = sv.status()
    for name, info in status.items():
        state = "running" if info["running"] else "stopped"
        pid_str = f" (PID {info['pid']})" if info["pid"] else ""
        print(f"  {name:15s} {state}{pid_str}")


def _kill_stale_port(port: int) -> None:
    """Kill any process listening on the given port (macOS/Linux).

    Sends SIGTERM, waits up to 3 seconds, then SIGKILL any survivors.
    Blocks until the port is fully released.
    """
    import subprocess as _sp
    import signal
    import time as _time

    try:
        result = _sp.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, _sp.TimeoutExpired):
        return

    pids = set()
    for pid_str in result.stdout.strip().split("\n"):
        pid_str = pid_str.strip()
        if not pid_str:
            continue
        try:
            pids.add(int(pid_str))
        except ValueError:
            pass

    if not pids:
        return

    logger = logging.getLogger("lana-proxy")

    # SIGTERM first
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to PID %d on port %d", pid, port)
        except OSError:
            pass

    # Wait up to 3 seconds for processes to die
    deadline = _time.monotonic() + 3
    while _time.monotonic() < deadline:
        alive = set()
        for pid in pids:
            try:
                os.kill(pid, 0)  # Check if alive
                alive.add(pid)
            except OSError:
                pass
        if not alive:
            return
        _time.sleep(0.2)

    # SIGKILL survivors
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info("Sent SIGKILL to PID %d on port %d", pid, port)
        except OSError:
            pass

    # Brief wait for SIGKILL to take effect
    _time.sleep(0.5)


def _daemonize_start(args: argparse.Namespace) -> None:
    """Spawn the supervisor as a background daemon process.

    Re-execs lana-proxy.py with --_daemon flag in a fully detached subprocess.
    The parent process prints status and returns to the shell immediately.
    """
    import subprocess as _sp

    project_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "lana-proxy")
    os.makedirs(config_dir, exist_ok=True)

    # Kill stale processes on configured ports before starting
    from src.process.env import load_dotenv
    env = load_dotenv(os.path.join(project_dir, ".env"))
    proxy_port = int(env.get("PROXY_PORT", "5304"))
    backend_port = int(env.get("BACKEND_PORT", "8080"))

    if not args.backend_only:
        _kill_stale_port(proxy_port)
    if not args.proxy_only:
        _kill_stale_port(backend_port)

    # Build the daemon command: same args but with --_daemon instead of --start
    daemon_cmd = [sys.executable, os.path.join(project_dir, "lana-proxy.py"), "--_daemon"]
    if args.backend_only:
        daemon_cmd.append("--backend-only")
    if args.proxy_only:
        daemon_cmd.append("--proxy-only")
    # Forward proxy config flags
    daemon_cmd.extend(_build_proxy_forward_args(args))

    # Redirect daemon output to supervisor.log
    log_path = os.path.join(config_dir, "supervisor.log")
    log_file = open(log_path, "a")

    # Start fully detached (no stdin, stdout/stderr to log file)
    _sp.Popen(
        daemon_cmd,
        stdout=log_file,
        stderr=log_file,
        stdin=_sp.DEVNULL,
        start_new_session=True,
    )
    log_file.close()

    # Poll until all expected processes are running (up to 30s for model loading)
    import time as _time

    sv, *_ = _get_supervisor_context(args)
    expect_count = (1 if args.backend_only or args.proxy_only else 2) + 1  # +1 for supervisor

    print("lana-proxy starting...", end="", flush=True)
    deadline = _time.monotonic() + 30
    while _time.monotonic() < deadline:
        _time.sleep(1)
        status = sv.status()
        running = sum(1 for info in status.values() if info["running"])
        if running >= expect_count:
            break
        print(".", end="", flush=True)

    print()
    print()
    _print_status(sv)
    print()
    print(f"  Logs: lana-proxy --logs")
    print(f"  Stop: lana-proxy --stop")


def _tail_logs(config_dir: str) -> None:
    """Tail all log files in config_dir. Ctrl+C to stop."""
    import subprocess as _sp

    log_files = []
    for name in ["supervisor.log", "backend.log", "proxy.log"]:
        path = os.path.join(config_dir, name)
        if os.path.isfile(path):
            log_files.append(path)

    if not log_files:
        print("No log files found in", config_dir)
        return

    print(f"Tailing {len(log_files)} log file(s)... (Ctrl+C to stop)")
    print()

    try:
        # Use tail -f to follow all log files
        _sp.run(["tail", "-f"] + log_files)
    except KeyboardInterrupt:
        print()


def _run_supervisor(args: argparse.Namespace) -> None:
    """Handle --start, --stop, --status, --logs, and --restart commands."""
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "lana-proxy")

    # --logs: tail log files
    if args.logs:
        _tail_logs(config_dir)
        return

    # --status: print process table
    if args.status:
        sv, *_ = _get_supervisor_context(args)
        _print_status(sv)
        return

    # --stop: stop via PID files, then clean up any orphaned port listeners
    if args.stop:
        sv, *_ = _get_supervisor_context(args)
        sv.stop_from_pid_files()
        # Kill anything still holding the ports (orphaned processes)
        project_dir = os.path.dirname(os.path.realpath(__file__))
        from src.process.env import load_dotenv
        env = load_dotenv(os.path.join(project_dir, ".env"))
        _kill_stale_port(int(env.get("PROXY_PORT", "5304")))
        _kill_stale_port(int(env.get("BACKEND_PORT", "8080")))
        return

    # --restart: stop then start
    if args.restart:
        sv, *_ = _get_supervisor_context(args)
        sv.stop_from_pid_files()
        import time as _time
        _time.sleep(1)
        # Fall through to start logic below
        args.start = True

    # --start: launch processes
    if args.start:
        sv, config_dir, project_dir, start_backend, start_proxy = _get_supervisor_context(args)

        # Validate backend config before starting
        if start_backend:
            from src.backend import BackendConfig
            backend_cfg = BackendConfig.from_env()
            backend_cfg.validate()

        if args._daemon:
            # Daemon child: run supervisor directly (blocking)
            names = []
            if start_backend:
                names.append("backend")
            if start_proxy:
                names.append("proxy")
            sv.start(names=names)
        else:
            # Default: daemonize (spawn background process, return to shell)
            _daemonize_start(args)
        return

    # --_daemon: internal flag, run supervisor in foreground (called by daemonize)
    if args._daemon:
        sv, config_dir, project_dir, start_backend, start_proxy = _get_supervisor_context(args)
        names = []
        if start_backend:
            names.append("backend")
        if start_proxy:
            names.append("proxy")
        sv.start(names=names)
        return


def main():
    args = parse_args()

    # Supervisor mode
    if any([args.start, args.stop, args.status, args.logs, args.restart,
            getattr(args, "_daemon", False)]):
        log_level = args.log_level or "INFO"
        setup_logging(log_level)
        _run_supervisor(args)
        return

    # Direct proxy mode (original behavior)
    config = build_config(args)
    setup_logging(config.log_level)

    logger = logging.getLogger("proxy")

    # Create shared stats and queue
    stats = Stats()
    queue = RequestQueue(
        max_concurrent=config.max_concurrent,
        timeout=config.queue_timeout,
        stats=stats,
    )

    server = ThreadedHTTPServer(config, stats=stats, queue=queue)

    # Auto-discover models from all backend URLs
    backend_urls = [config.backend_url]
    for url in config.routes.values():
        if url not in backend_urls:
            backend_urls.append(url)
    discover_models(backend_urls, stats)

    # Start dashboard if enabled
    dashboard = None
    if config.dashboard:
        dashboard = Dashboard(stats)
        dashboard.start()

    # Graceful shutdown on SIGINT/SIGTERM
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        if dashboard is not None:
            dashboard.stop()
        server.shutdown_server()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("lana-proxy starting on %s:%d", config.host, config.port)
    logger.info("Backend: %s", config.backend_url)
    logger.info("Default model: %s", config.default_model)
    if config.model_map:
        logger.info("Model mappings: %s", config.model_map)
    if config.routes:
        logger.info("Routes: %s", config.routes)
    logger.info("Max concurrent: %d, queue timeout: %.0fs",
                config.max_concurrent, config.queue_timeout)
    if config.record_path:
        logger.info("Recording to: %s", config.record_path)
    if config.replay_path:
        logger.info("Replaying from: %s", config.replay_path)
    if config.cache_enabled:
        logger.info("Cache enabled: ttl=%ds, max=%d", config.cache_ttl, config.cache_max_size)
    if config.token_budget > 0:
        logger.info("Token budget: %d", config.token_budget)
    if config.analytics_path:
        logger.info("Analytics: %s", config.analytics_path)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        if dashboard is not None:
            dashboard.stop()
        server.shutdown_server()
        server.server_close()


if __name__ == "__main__":
    main()
