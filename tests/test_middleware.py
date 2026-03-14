"""
Tests for middleware components: cache, queue, budget, analytics, recorder.

Covers thread safety, edge cases, and core behavior for each middleware.
"""

import json
import os
import tempfile
import threading
import time
import unittest

from src.proxy.middleware.cache import PromptCache, _make_cache_key
from src.proxy.middleware.budget import BudgetTracker
from src.proxy.middleware.analytics import Analytics
from src.proxy.middleware.recorder import Recorder, Replayer, _hash_messages
from src.proxy.middleware.queue import RequestQueue


# ─── Cache Tests ──────────────────────────────────────────────────────────────


class TestMakeCacheKey(unittest.TestCase):
    """Tests for the cache key generation function."""

    def test_same_body_same_key(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(_make_cache_key(body), _make_cache_key(body))

    def test_different_bodies_different_keys(self):
        body1 = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        body2 = {"model": "m", "messages": [{"role": "user", "content": "bye"}]}
        self.assertNotEqual(_make_cache_key(body1), _make_cache_key(body2))

    def test_key_is_hex_sha256(self):
        body = {"model": "m", "messages": []}
        key = _make_cache_key(body)
        self.assertEqual(len(key), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in key))

    def test_order_independent(self):
        """Key should be the same regardless of dict key order in body."""
        body1 = {"model": "m", "system": "s", "messages": [], "tools": None}
        body2 = {"tools": None, "messages": [], "model": "m", "system": "s"}
        self.assertEqual(_make_cache_key(body1), _make_cache_key(body2))

    def test_missing_fields_use_defaults(self):
        """Body with missing fields should still produce a valid key."""
        key = _make_cache_key({})
        self.assertEqual(len(key), 64)


class TestPromptCache(unittest.TestCase):
    """Tests for the PromptCache class."""

    def setUp(self):
        self.cache = PromptCache(ttl=10, max_size=3)

    def test_miss_on_empty_cache(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.assertIsNone(self.cache.get(body))
        self.assertEqual(self.cache.misses, 1)

    def test_hit_after_put(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        response = {"type": "message", "content": [{"type": "text", "text": "hello"}]}
        self.cache.put(body, response)
        result = self.cache.get(body)
        self.assertEqual(result, response)
        self.assertEqual(self.cache.hits, 1)

    def test_eviction_at_max_size(self):
        """Oldest entry should be evicted when cache is full."""
        bodies = [
            {"model": "m", "messages": [{"role": "user", "content": f"msg{i}"}]}
            for i in range(4)
        ]
        for i, body in enumerate(bodies):
            self.cache.put(body, {"id": i})

        # First entry should be evicted
        self.assertIsNone(self.cache.get(bodies[0]))
        # Last three should still be there
        for i in range(1, 4):
            self.assertIsNotNone(self.cache.get(bodies[i]))

    def test_ttl_expiration(self):
        """Expired entries should return None."""
        cache = PromptCache(ttl=0, max_size=10)  # 0 second TTL
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        cache.put(body, {"response": True})
        time.sleep(0.01)
        self.assertIsNone(cache.get(body))
        self.assertEqual(cache.misses, 1)

    def test_get_stats(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.cache.put(body, {"r": 1})
        self.cache.get(body)  # hit
        self.cache.get({"model": "x", "messages": []})  # miss

        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["hit_rate_percent"], 50.0)

    def test_stats_zero_requests(self):
        stats = self.cache.get_stats()
        self.assertEqual(stats["hit_rate_percent"], 0.0)

    def test_disk_persistence(self):
        """Cache should save to and load from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache1 = PromptCache(ttl=300, max_size=10, cache_dir=tmpdir)
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            response = {"type": "message", "content": [{"type": "text", "text": "hello"}]}
            cache1.put(body, response)
            cache1.save_to_disk()

            # Verify file was created
            cache_file = os.path.join(tmpdir, "prompt_cache.json")
            self.assertTrue(os.path.exists(cache_file))

            # Load into a new cache instance
            cache2 = PromptCache(ttl=300, max_size=10, cache_dir=tmpdir)
            result = cache2.get(body)
            self.assertEqual(result, response)

    def test_thread_safety(self):
        """Concurrent puts and gets should not raise exceptions."""
        cache = PromptCache(ttl=60, max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    body = {"model": "m", "messages": [{"role": "user", "content": f"t{thread_id}-{i}"}]}
                    cache.put(body, {"id": f"{thread_id}-{i}"})
                    cache.get(body)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


# ─── Queue Tests ──────────────────────────────────────────────────────────────


class TestRequestQueue(unittest.TestCase):
    """Tests for the RequestQueue class."""

    def test_invalid_max_concurrent(self):
        with self.assertRaises(ValueError):
            RequestQueue(max_concurrent=0)

    def test_acquire_and_release(self):
        queue = RequestQueue(max_concurrent=1, timeout=5)
        self.assertTrue(queue.acquire())
        queue.release()

    def test_timeout_when_full(self):
        """Should return False when queue is full and timeout expires."""
        queue = RequestQueue(max_concurrent=1, timeout=0.1)
        self.assertTrue(queue.acquire())
        # Second acquire should timeout
        self.assertFalse(queue.acquire())
        queue.release()

    def test_release_allows_next(self):
        """After release, another acquire should succeed."""
        queue = RequestQueue(max_concurrent=1, timeout=1)
        self.assertTrue(queue.acquire())
        queue.release()
        self.assertTrue(queue.acquire())
        queue.release()

    def test_concurrent_slots(self):
        """Multiple slots should allow multiple concurrent acquires."""
        queue = RequestQueue(max_concurrent=3, timeout=1)
        self.assertTrue(queue.acquire())
        self.assertTrue(queue.acquire())
        self.assertTrue(queue.acquire())
        # 4th should timeout
        self.assertFalse(queue.acquire())
        queue.release()
        queue.release()
        queue.release()

    def test_properties(self):
        queue = RequestQueue(max_concurrent=5, timeout=30)
        self.assertEqual(queue.max_concurrent, 5)
        self.assertEqual(queue.timeout, 30)

    def test_thread_safety(self):
        """Concurrent acquire/release should not corrupt state."""
        queue = RequestQueue(max_concurrent=2, timeout=5)
        results = {"acquired": 0, "errors": []}
        lock = threading.Lock()

        def worker():
            try:
                if queue.acquire():
                    with lock:
                        results["acquired"] += 1
                    time.sleep(0.01)
                    queue.release()
            except Exception as e:
                with lock:
                    results["errors"].append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(results["errors"], [])
        self.assertEqual(results["acquired"], 10)


# ─── Budget Tests ─────────────────────────────────────────────────────────────


class TestBudgetTracker(unittest.TestCase):
    """Tests for the BudgetTracker class."""

    def test_unlimited_budget(self):
        tracker = BudgetTracker(max_tokens=0)
        self.assertTrue(tracker.check_budget())
        tracker.record_usage("model", 1000, 500)
        self.assertTrue(tracker.check_budget())
        self.assertEqual(tracker.remaining, 0)

    def test_budget_enforcement(self):
        tracker = BudgetTracker(max_tokens=100)
        self.assertTrue(tracker.check_budget())
        tracker.record_usage("model", 60, 50)
        self.assertFalse(tracker.check_budget())

    def test_remaining_tokens(self):
        tracker = BudgetTracker(max_tokens=1000)
        tracker.record_usage("model", 300, 200)
        self.assertEqual(tracker.remaining, 500)
        self.assertEqual(tracker.total_tokens, 500)

    def test_remaining_never_negative(self):
        tracker = BudgetTracker(max_tokens=100)
        tracker.record_usage("model", 200, 200)
        self.assertEqual(tracker.remaining, 0)

    def test_per_model_tracking(self):
        tracker = BudgetTracker(max_tokens=0)
        tracker.record_usage("model-a", 100, 50)
        tracker.record_usage("model-b", 200, 100)
        tracker.record_usage("model-a", 50, 25)

        status = tracker.get_status()
        self.assertEqual(status["per_model"]["model-a"]["input_tokens"], 150)
        self.assertEqual(status["per_model"]["model-a"]["output_tokens"], 75)
        self.assertEqual(status["per_model"]["model-b"]["input_tokens"], 200)

    def test_reset(self):
        tracker = BudgetTracker(max_tokens=1000)
        tracker.record_usage("model", 500, 300)
        tracker.reset()
        self.assertEqual(tracker.total_input, 0)
        self.assertEqual(tracker.total_output, 0)
        self.assertEqual(tracker.total_tokens, 0)
        self.assertTrue(tracker.check_budget())

        status = tracker.get_status()
        self.assertEqual(status["per_model"], {})

    def test_get_status(self):
        tracker = BudgetTracker(max_tokens=1000)
        tracker.record_usage("model", 100, 50)

        status = tracker.get_status()
        self.assertEqual(status["total_input_tokens"], 100)
        self.assertEqual(status["total_output_tokens"], 50)
        self.assertEqual(status["total_tokens"], 150)
        self.assertEqual(status["max_tokens"], 1000)
        self.assertEqual(status["remaining"], 850)
        self.assertFalse(status["unlimited"])

    def test_unlimited_status(self):
        tracker = BudgetTracker(max_tokens=0)
        status = tracker.get_status()
        self.assertTrue(status["unlimited"])

    def test_thread_safety(self):
        tracker = BudgetTracker(max_tokens=0)
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    tracker.record_usage(f"model-{thread_id}", 10, 5)
                    tracker.check_budget()
                    tracker.get_status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(tracker.total_input, 4000)
        self.assertEqual(tracker.total_output, 2000)


# ─── Analytics Tests ──────────────────────────────────────────────────────────


class TestAnalytics(unittest.TestCase):
    """Tests for the Analytics class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "analytics.jsonl")
        self.analytics = Analytics(self.log_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_writes_jsonl(self):
        self.analytics.record(
            model="model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            stream=False,
        )

        with open(self.log_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["model"], "model")
        self.assertEqual(entry["input_tokens"], 100)
        self.assertEqual(entry["output_tokens"], 50)
        self.assertEqual(entry["latency_ms"], 200)
        self.assertFalse(entry["stream"])

    def test_record_with_tool_calls_and_cache(self):
        self.analytics.record(
            model="model",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            stream=True,
            tool_calls_count=3,
            cache_hit=True,
            stop_reason="tool_use",
        )

        with open(self.log_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["tool_calls_count"], 3)
        self.assertTrue(entry["cache_hit"])
        self.assertEqual(entry["stop_reason"], "tool_use")
        self.assertTrue(entry["stream"])

    def test_empty_summary(self):
        summary = self.analytics.get_summary()
        self.assertEqual(summary["total_requests"], 0)
        self.assertEqual(summary["total_input_tokens"], 0)
        self.assertEqual(summary["average_latency_ms"], 0)

    def test_summary_aggregation(self):
        self.analytics.record("model-a", 100, 50, 200, False)
        self.analytics.record("model-a", 200, 100, 400, True)
        self.analytics.record("model-b", 50, 25, 100, False, tool_calls_count=2)

        summary = self.analytics.get_summary()
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["total_input_tokens"], 350)
        self.assertEqual(summary["total_output_tokens"], 175)
        self.assertAlmostEqual(summary["average_latency_ms"], 233.3, places=1)

        # Per-model breakdown
        self.assertEqual(summary["by_model"]["model-a"]["requests"], 2)
        self.assertEqual(summary["by_model"]["model-b"]["requests"], 1)

        # Tool usage
        self.assertEqual(summary["tool_usage"]["requests_with_tools"], 1)
        self.assertEqual(summary["tool_usage"]["total_tool_calls"], 2)

    def test_summary_with_cache_stats(self):
        self.analytics.record("m", 10, 5, 50, False, cache_hit=True)
        cache_stats = {"hits": 5, "misses": 3}
        summary = self.analytics.get_summary(cache_stats=cache_stats)
        self.assertEqual(summary["cache_stats"], cache_stats)

    def test_in_memory_cap(self):
        """Should cap in-memory entries at MAX_IN_MEMORY."""
        analytics = Analytics(self.log_path)
        analytics.MAX_IN_MEMORY = 10
        for i in range(20):
            analytics.record("m", 10, 5, 50, False)
        summary = analytics.get_summary()
        self.assertEqual(summary["total_requests"], 10)

    def test_thread_safety(self):
        errors = []

        def worker(thread_id):
            try:
                for i in range(25):
                    self.analytics.record(f"model-{thread_id}", 10, 5, 50, False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        summary = self.analytics.get_summary()
        self.assertEqual(summary["total_requests"], 100)


# ─── Recorder/Replayer Tests ─────────────────────────────────────────────────


class TestHashMessages(unittest.TestCase):
    """Tests for the message hashing function."""

    def test_same_messages_same_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(_hash_messages(msgs), _hash_messages(msgs))

    def test_different_messages_different_hash(self):
        msgs1 = [{"role": "user", "content": "hello"}]
        msgs2 = [{"role": "user", "content": "goodbye"}]
        self.assertNotEqual(_hash_messages(msgs1), _hash_messages(msgs2))

    def test_empty_messages(self):
        h = _hash_messages([])
        self.assertEqual(len(h), 64)


class TestRecorderReplayer(unittest.TestCase):
    """Tests for the Recorder and Replayer classes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.record_path = os.path.join(self.tmpdir, "recording.jsonl")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_and_replay(self):
        """Record a request/response and replay it."""
        recorder = Recorder(self.record_path)
        request = {"messages": [{"role": "user", "content": "hello"}], "model": "m"}
        response = {"type": "message", "content": [{"type": "text", "text": "hi"}]}

        recorder.record(request, response, latency_ms=100, stream=False, model="m")

        replayer = Replayer(self.record_path)
        result = replayer.lookup(request)
        self.assertIsNotNone(result)
        self.assertEqual(result["response"], response)
        self.assertEqual(result["model"], "m")

    def test_replay_miss(self):
        """Lookup should return None for unrecorded requests."""
        recorder = Recorder(self.record_path)
        request1 = {"messages": [{"role": "user", "content": "hello"}]}
        recorder.record(request1, {"r": 1}, latency_ms=50, stream=False, model="m")

        replayer = Replayer(self.record_path)
        request2 = {"messages": [{"role": "user", "content": "different"}]}
        self.assertIsNone(replayer.lookup(request2))

    def test_record_with_sse_events(self):
        recorder = Recorder(self.record_path)
        request = {"messages": [{"role": "user", "content": "hi"}]}
        sse = ["event: message_start\ndata: {}\n\n", "event: message_stop\ndata: {}\n\n"]
        recorder.record(request, {}, latency_ms=50, stream=True, model="m", sse_events=sse)

        with open(self.record_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["sse_events"], sse)
        self.assertTrue(entry["stream"])

    def test_multiple_recordings_last_wins(self):
        """If same messages are recorded twice, replayer uses the last one."""
        recorder = Recorder(self.record_path)
        request = {"messages": [{"role": "user", "content": "hello"}]}
        recorder.record(request, {"version": 1}, latency_ms=50, stream=False, model="m")
        recorder.record(request, {"version": 2}, latency_ms=50, stream=False, model="m")

        replayer = Replayer(self.record_path)
        result = replayer.lookup(request)
        self.assertEqual(result["response"], {"version": 2})

    def test_replayer_skips_malformed_lines(self):
        """Malformed JSONL lines should be skipped without error."""
        with open(self.record_path, "w") as f:
            f.write("not valid json\n")
            entry = {
                "messages_hash": _hash_messages([{"role": "user", "content": "hi"}]),
                "response": {"ok": True},
            }
            f.write(json.dumps(entry) + "\n")

        replayer = Replayer(self.record_path)
        request = {"messages": [{"role": "user", "content": "hi"}]}
        result = replayer.lookup(request)
        self.assertIsNotNone(result)

    def test_replayer_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Replayer("/nonexistent/path.jsonl")

    def test_thread_safety(self):
        recorder = Recorder(self.record_path)
        errors = []

        def worker(thread_id):
            try:
                for i in range(25):
                    request = {"messages": [{"role": "user", "content": f"t{thread_id}-{i}"}]}
                    recorder.record(request, {"id": i}, latency_ms=10, stream=False, model="m")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        with open(self.record_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 100)


if __name__ == "__main__":
    unittest.main()
