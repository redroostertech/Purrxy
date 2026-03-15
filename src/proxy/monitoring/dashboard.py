"""
Live terminal dashboard for purrxy.

Displays real-time statistics using ANSI escape codes (no curses, no
external dependencies). Runs in a background daemon thread, refreshing
every 1 second. Degrades gracefully when the terminal does not support
ANSI codes (detected via os.isatty).
"""

import os
import sys
import threading
import time
from typing import Optional

from .stats import Stats


def _format_uptime(seconds: float) -> str:
    """Format seconds into a human-readable uptime string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _bar(value: int, maximum: int, width: int = 20) -> str:
    """Render a simple ASCII progress bar."""
    if maximum <= 0:
        return "[" + " " * width + "]"
    filled = min(int((value / maximum) * width), width)
    return "[" + "#" * filled + " " * (width - filled) + "]"


class Dashboard:
    """Background dashboard that renders stats to the terminal.

    Usage:
        dashboard = Dashboard(stats)
        dashboard.start()
        # ... server runs ...
        dashboard.stop()
    """

    def __init__(self, stats: Stats, refresh_interval: float = 1.0):
        self._stats = stats
        self._refresh = refresh_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_tty = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, 'fileno') else False

    def start(self) -> None:
        """Start the dashboard background thread."""
        if not self._is_tty:
            return  # Silently skip if not a terminal
        self._thread = threading.Thread(
            target=self._run, name="dashboard", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the dashboard to stop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        """Main loop: clear screen and render stats every interval."""
        # Hide cursor
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        try:
            while not self._stop_event.is_set():
                self._render()
                self._stop_event.wait(self._refresh)
        finally:
            # Show cursor on exit
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()

    def _render(self) -> None:
        """Render one frame of the dashboard."""
        snap = self._stats.snapshot()

        lines = []
        lines.append("\033[2J\033[H")  # Clear screen + move to top-left

        # Header
        lines.append("\033[1;36m" + "=" * 60 + "\033[0m")
        lines.append("\033[1;36m  Purrxy — Live Dashboard\033[0m")
        lines.append("\033[1;36m" + "=" * 60 + "\033[0m")
        lines.append("")

        # Uptime and connections
        uptime_str = _format_uptime(snap["uptime_seconds"])
        lines.append(f"  \033[1mUptime:\033[0m {uptime_str}")
        lines.append(f"  \033[1mActive connections:\033[0m {snap['active_connections']}")
        lines.append("")

        # Request counts
        lines.append("\033[1;33m  Requests\033[0m")
        lines.append(f"    Total:         {snap['total_requests']}")
        lines.append(f"    Streaming:     {snap['stream_requests']}")
        lines.append(f"    Non-streaming: {snap['non_stream_requests']}")
        lines.append(f"    Errors:        {snap['error_count']}")
        lines.append("")

        # Latency
        lines.append("\033[1;33m  Latency (ms)\033[0m")
        lines.append(f"    Average: {snap['avg_latency_ms']:>8.1f}")
        lines.append(f"    p50:     {snap['p50_latency_ms']:>8.1f}")
        lines.append(f"    p95:     {snap['p95_latency_ms']:>8.1f}")
        lines.append("")

        # Token throughput
        lines.append("\033[1;33m  Token Throughput (per minute)\033[0m")
        lines.append(f"    Input:  {snap['input_tokens_per_min']:>8.1f} tok/min")
        lines.append(f"    Output: {snap['output_tokens_per_min']:>8.1f} tok/min")
        lines.append(f"    Total:  in={snap['total_input_tokens']}  out={snap['total_output_tokens']}")
        lines.append("")

        # Queue
        if snap["queue_max"] > 0:
            lines.append("\033[1;33m  Queue\033[0m")
            bar = _bar(snap["queue_depth"], snap["queue_max"])
            lines.append(f"    Depth: {snap['queue_depth']}/{snap['queue_max']}  {bar}")
            lines.append("")

        # Backend models
        if snap["discovered_models"]:
            lines.append("\033[1;33m  Backend Models\033[0m")
            for m in snap["discovered_models"]:
                name = m.get("id", "unknown")
                lines.append(f"    - {name}")
            lines.append("")

        # Recent requests
        lines.append("\033[1;33m  Recent Requests\033[0m")
        if snap["recent_requests"]:
            for req in reversed(snap["recent_requests"]):
                stream_flag = "S" if req["stream"] else "N"
                err_flag = " ERR" if req["error"] else ""
                lines.append(
                    f"    [{stream_flag}] {req['model']:<24} "
                    f"{req['latency_ms']:>7.0f}ms  "
                    f"in={req['input_tokens']:<5} out={req['output_tokens']:<5}"
                    f"{err_flag}"
                )
        else:
            lines.append("    (none yet)")
        lines.append("")

        lines.append("\033[2m  Press Ctrl+C to stop\033[0m")

        output = "\n".join(lines)
        try:
            sys.stdout.write(output)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            self._stop_event.set()
