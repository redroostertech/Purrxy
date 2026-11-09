"""
A normal stop signal (SIGTERM from a service manager, SIGINT from Ctrl+C)
must stop the running proxy cleanly and quickly.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port: int, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError("proxy exited before it started listening")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("proxy never started listening")


class TestStopSignals(unittest.TestCase):

    def _run_and_signal(self, signum: int) -> None:
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "lana-proxy.py"),
             "--port", str(port), "--backend", f"http://127.0.0.1:{_free_port()}"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            _wait_listening(port, proc)
            proc.send_signal(signum)
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.fail(f"proxy ignored signal {signum} for 5 seconds")
            self.assertEqual(code, 0)
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_sigterm_stops_the_proxy(self):
        self._run_and_signal(signal.SIGTERM)

    def test_sigint_stops_the_proxy(self):
        self._run_and_signal(signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
