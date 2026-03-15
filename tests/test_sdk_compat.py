"""Anthropic SDK compatibility tests for purrxy.

These tests verify that purrxy correctly handles the exact HTTP requests
sent by the Anthropic Python SDK and returns responses in the exact format
the SDK expects to parse.

Tests that require a running proxy + backend server are decorated with
@unittest.skipUnless and will skip gracefully when the server isn't running.

To run with a live server:
    1. Start everything:   purrxy --start
    2. Run tests:          python3 -m unittest tests.test_sdk_compat -v
"""

import json
import unittest
import urllib.error
import urllib.request

from src.proxy.config import Config
from src.proxy.errors import make_error_body
from src.proxy.translation.request import translate_request
from src.proxy.translation.response import translate_response
from src.proxy.translation.stream import translate_stream

# ── Configuration ────────────────────────────────────────────────────────────

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 5304
PROXY_BASE = "http://{}:{}".format(PROXY_HOST, PROXY_PORT)


def _proxy_is_running():
    """Check if purrxy is running and reachable."""
    try:
        req = urllib.request.Request(
            "{}/health".format(PROXY_BASE),
            method="GET",
        )
        resp = urllib.request.urlopen(req, timeout=2)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception:
        return False


PROXY_AVAILABLE = _proxy_is_running()


# ── Offline Tests (no server needed) ────────────────────────────────────────

class TestSDKRequestTranslation(unittest.TestCase):
    """Verify that requests matching the Anthropic SDK format translate correctly.

    These tests exercise the translation layer directly, no server needed.
    """

    def setUp(self):
        self.config = Config(default_model="local-model")

    def test_sdk_style_simple_request(self):
        """Anthropic SDK sends this shape for a simple messages.create() call."""
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, Claude"}
            ],
        }
        result = translate_request(body, self.config)

        self.assertEqual(result["model"], "local-model")
        self.assertEqual(result["max_tokens"], 1024)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["content"], "Hello, Claude")

    def test_sdk_style_with_system_and_temperature(self):
        """SDK sends system as a string and temperature as a float."""
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 4096,
            "system": "You are a helpful coding assistant.",
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": "Write a hello world in Python"}
            ],
        }
        result = translate_request(body, self.config)

        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"],
                         "You are a helpful coding assistant.")
        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(result["temperature"], 0.0)

    def test_sdk_style_streaming_request(self):
        """SDK sets stream=True for streaming responses."""
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "stream": True,
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
        }
        result = translate_request(body, self.config)

        self.assertTrue(result["stream"])
        self.assertEqual(result["stream_options"], {"include_usage": True})

    def test_sdk_style_multi_turn_with_tools(self):
        """SDK sends multi-turn conversation with tool use/result blocks."""
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "tools": [{
                "name": "read_file",
                "description": "Read a file from disk",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                    },
                    "required": ["path"],
                },
            }],
            "messages": [
                {"role": "user", "content": "Read my config file"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll read that file for you."},
                        {
                            "type": "tool_use",
                            "id": "toolu_01A",
                            "name": "read_file",
                            "input": {"path": "/etc/hosts"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_01A",
                        "content": "127.0.0.1 localhost",
                    }],
                },
            ],
        }
        result = translate_request(body, self.config)

        # Should have tool definitions
        self.assertEqual(len(result["tools"]), 1)
        self.assertEqual(result["tools"][0]["type"], "function")
        self.assertEqual(result["tools"][0]["function"]["name"], "read_file")

        # Messages: user, assistant (with tool_calls), tool
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][1]["role"], "assistant")
        self.assertIn("tool_calls", result["messages"][1])
        self.assertEqual(result["messages"][2]["role"], "tool")


class TestSDKResponseFormat(unittest.TestCase):
    """Verify response format matches what the Anthropic SDK expects to parse.

    The SDK expects these exact fields in non-streaming responses:
    - id: string (starts with "msg_")
    - type: "message"
    - role: "assistant"
    - model: string
    - content: list of content blocks
    - stop_reason: string or null
    - stop_sequence: string or null
    - usage: {input_tokens: int, output_tokens: int}
    """

    def test_response_has_all_required_fields(self):
        """SDK parses these fields — all must be present."""
        openai_body = {
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = translate_response(openai_body, "claude-3-5-sonnet-20241022")

        # All required fields must be present
        self.assertIn("id", result)
        self.assertTrue(result["id"].startswith("msg_"),
                        "id must start with 'msg_'")
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["model"], "claude-3-5-sonnet-20241022")
        self.assertIn("content", result)
        self.assertIsInstance(result["content"], list)
        self.assertIn("stop_reason", result)
        self.assertIn("stop_sequence", result)
        self.assertIsNone(result["stop_sequence"])
        self.assertIn("usage", result)
        self.assertIn("input_tokens", result["usage"])
        self.assertIn("output_tokens", result["usage"])
        self.assertIsInstance(result["usage"]["input_tokens"], int)
        self.assertIsInstance(result["usage"]["output_tokens"], int)

    def test_text_content_block_format(self):
        """SDK expects content blocks with {type: "text", text: string}."""
        openai_body = {
            "choices": [{
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        result = translate_response(openai_body, "test-model")

        block = result["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertIn("text", block)
        self.assertIsInstance(block["text"], str)

    def test_tool_use_content_block_format(self):
        """SDK expects tool_use blocks with {type, id, name, input}."""
        openai_body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
        result = translate_response(openai_body, "test-model")

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        block = tool_blocks[0]
        self.assertEqual(block["type"], "tool_use")
        self.assertIn("id", block)
        self.assertIn("name", block)
        self.assertIn("input", block)
        self.assertIsInstance(block["input"], dict)


class TestSDKStreamEventSequence(unittest.TestCase):
    """Verify the SSE event sequence matches what the Anthropic SDK expects.

    The SDK expects this exact event sequence:
    1. event: message_start (with message object)
    2. event: content_block_start (for each block)
    3. event: content_block_delta (text_delta or input_json_delta)
    4. event: content_block_stop
    5. event: message_delta (with stop_reason and usage)
    6. event: message_stop
    """

    def _make_lines(self, chunks):
        """Convert chunks to SSE line generator."""
        lines = []
        for chunk in chunks:
            if chunk == "[DONE]":
                lines.append("data: [DONE]")
            else:
                lines.append("data: {}".format(json.dumps(chunk)))
            lines.append("")
        return iter(lines)

    def _parse_events(self, events):
        """Parse SSE strings into (event_type, data) tuples."""
        parsed = []
        for event_str in events:
            lines = event_str.strip().split("\n")
            event_type = None
            data = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
            parsed.append((event_type, data))
        return parsed

    def test_stream_event_sequence(self):
        """Verify the exact event sequence the SDK expects."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            },
            "[DONE]",
        ]

        events = list(translate_stream(self._make_lines(chunks), "claude-3-5-sonnet-20241022", "msg_test"))
        parsed = self._parse_events(events)
        event_types = [e[0] for e in parsed]

        # Must start with message_start
        self.assertEqual(event_types[0], "message_start")

        # Must end with message_delta then message_stop
        self.assertEqual(event_types[-2], "message_delta")
        self.assertEqual(event_types[-1], "message_stop")

        # message_start must contain full message object
        msg_start = parsed[0][1]
        self.assertEqual(msg_start["type"], "message_start")
        msg = msg_start["message"]
        self.assertIn("id", msg)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertIn("model", msg)
        self.assertEqual(msg["content"], [])
        self.assertIsNone(msg["stop_reason"])
        self.assertIsNone(msg["stop_sequence"])
        self.assertIn("usage", msg)

        # message_delta must have stop_reason and usage
        msg_delta = [e for e in parsed if e[0] == "message_delta"][0][1]
        self.assertEqual(msg_delta["type"], "message_delta")
        self.assertIn("delta", msg_delta)
        self.assertIn("stop_reason", msg_delta["delta"])
        self.assertIn("stop_sequence", msg_delta["delta"])
        self.assertIn("usage", msg_delta)
        self.assertIn("output_tokens", msg_delta["usage"])

        # message_stop
        msg_stop = [e for e in parsed if e[0] == "message_stop"][0][1]
        self.assertEqual(msg_stop["type"], "message_stop")

    def test_stream_content_block_lifecycle(self):
        """Each content block must have start, delta(s), and stop events."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
            "[DONE]",
        ]

        events = list(translate_stream(self._make_lines(chunks), "model", "msg_id"))
        parsed = self._parse_events(events)

        # Find block lifecycle events
        block_starts = [e for e in parsed if e[0] == "content_block_start"]
        block_deltas = [e for e in parsed if e[0] == "content_block_delta"]
        block_stops = [e for e in parsed if e[0] == "content_block_stop"]

        # Must have at least one complete block lifecycle
        self.assertTrue(len(block_starts) >= 1)
        self.assertTrue(len(block_deltas) >= 1)
        self.assertTrue(len(block_stops) >= 1)

        # content_block_start must have index and content_block
        start = block_starts[0][1]
        self.assertIn("index", start)
        self.assertIn("content_block", start)
        self.assertEqual(start["content_block"]["type"], "text")

        # content_block_delta must have index and delta
        delta = block_deltas[0][1]
        self.assertIn("index", delta)
        self.assertIn("delta", delta)
        self.assertEqual(delta["delta"]["type"], "text_delta")
        self.assertIn("text", delta["delta"])


class TestSDKErrorFormat(unittest.TestCase):
    """Verify error responses match the Anthropic error schema.

    The SDK expects errors in this format:
    {
        "type": "error",
        "error": {
            "type": "<error_type>",
            "message": "<human-readable message>"
        }
    }
    """

    def test_error_response_schema(self):
        """Error body must have type='error' and error={type, message}."""
        for status in (400, 401, 403, 404, 429, 500, 529):
            body = make_error_body(status, "Test error message")

            self.assertEqual(body["type"], "error",
                             "Error body type must be 'error' for status {}".format(status))
            self.assertIn("error", body)
            self.assertIn("type", body["error"])
            self.assertIn("message", body["error"])
            self.assertIsInstance(body["error"]["type"], str)
            self.assertIsInstance(body["error"]["message"], str)

    def test_400_error_type(self):
        """400 maps to invalid_request_error."""
        body = make_error_body(400, "bad request")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_401_error_type(self):
        """401 maps to authentication_error."""
        body = make_error_body(401, "unauthorized")
        self.assertEqual(body["error"]["type"], "authentication_error")

    def test_429_error_type(self):
        """429 maps to rate_limit_error."""
        body = make_error_body(429, "too many requests")
        self.assertEqual(body["error"]["type"], "rate_limit_error")

    def test_500_error_type(self):
        """500 maps to api_error."""
        body = make_error_body(500, "internal error")
        self.assertEqual(body["error"]["type"], "api_error")

    def test_529_error_type(self):
        """529 maps to overloaded_error."""
        body = make_error_body(529, "overloaded")
        self.assertEqual(body["error"]["type"], "overloaded_error")


class TestSDKHeaderAcceptance(unittest.TestCase):
    """Verify the proxy accepts all headers the Anthropic SDK sends.

    The SDK sends these headers:
    - x-api-key: <api_key>
    - anthropic-version: 2023-06-01
    - anthropic-beta: <beta_features>
    - content-type: application/json
    - accept: application/json
    - user-agent: Anthropic/Python ...

    The proxy must accept all of these without error (especially x-api-key,
    which should be ignored rather than validated).
    """

    def test_request_translation_ignores_headers(self):
        """Translation layer works regardless of what headers were sent.

        Headers are handled at the HTTP layer, not the translation layer.
        This test verifies translation is header-agnostic.
        """
        config = Config(default_model="local-model")
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Test"}],
        }
        # This should succeed — translation doesn't look at headers
        result = translate_request(body, config)
        self.assertEqual(result["model"], "local-model")


# ── Live Server Tests (skip when server not running) ─────────────────────────

@unittest.skipUnless(PROXY_AVAILABLE, "purrxy not running on port 5304")
class TestLiveSDKHeaders(unittest.TestCase):
    """Test that a running proxy accepts Anthropic SDK headers."""

    def test_health_endpoint(self):
        """GET /health returns status ok."""
        req = urllib.request.Request(
            "{}/health".format(PROXY_BASE),
            method="GET",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        self.assertEqual(data["status"], "ok")

    def test_accepts_anthropic_headers(self):
        """Proxy accepts all headers the SDK sends without rejecting them."""
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "{}/v1/messages".format(PROXY_BASE),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Api-Key": "sk-ant-test-key-not-validated",
                "Anthropic-Version": "2023-06-01",
                "Anthropic-Beta": "messages-2023-12-15",
                "User-Agent": "Anthropic/Python 0.40.0",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            # Should get a valid Anthropic response (or backend error, not header error)
            self.assertIn("type", data)
        except urllib.error.HTTPError as e:
            # Even if the backend is down, we should NOT get a 401 (auth error)
            # from the proxy. 529 (backend unavailable) is acceptable.
            self.assertNotEqual(e.code, 401,
                                "Proxy should not validate x-api-key")
            self.assertNotEqual(e.code, 403,
                                "Proxy should not reject SDK headers")

    def test_response_format_matches_sdk_expectations(self):
        """Live response has all fields the SDK needs."""
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "{}/v1/messages".format(PROXY_BASE),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": "sk-ant-test",
                "Anthropic-Version": "2023-06-01",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())

            # Verify all SDK-required fields
            self.assertEqual(data["type"], "message")
            self.assertEqual(data["role"], "assistant")
            self.assertTrue(data["id"].startswith("msg_"))
            self.assertIsInstance(data["content"], list)
            self.assertIn("stop_reason", data)
            self.assertIn("stop_sequence", data)
            self.assertIn("usage", data)
            self.assertIn("input_tokens", data["usage"])
            self.assertIn("output_tokens", data["usage"])
        except urllib.error.HTTPError as e:
            # Backend may not be running — that's OK for this test
            if e.code == 529:
                self.skipTest("Backend not available")
            else:
                raise

    def test_streaming_response_format(self):
        """Live streaming response has correct SSE event sequence."""
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hi"}],
        }).encode("utf-8")

        req = urllib.request.Request(
            "{}/v1/messages".format(PROXY_BASE),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": "sk-ant-test",
                "Anthropic-Version": "2023-06-01",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=60)

            # Read SSE events until message_stop (connection stays open)
            event_types = []
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("event: "):
                    event_types.append(line[7:])
                    if line[7:] == "message_stop":
                        break

            # Verify sequence
            self.assertTrue(len(event_types) >= 4,
                            "Expected at least 4 events, got {}".format(len(event_types)))
            self.assertEqual(event_types[0], "message_start")
            self.assertEqual(event_types[-2], "message_delta")
            self.assertEqual(event_types[-1], "message_stop")

        except urllib.error.HTTPError as e:
            if e.code == 529:
                self.skipTest("Backend not available")
            else:
                raise

    def test_error_response_format(self):
        """Error response matches SDK expected error schema."""
        # Send an invalid request (missing required fields)
        body = json.dumps({"invalid": "request"}).encode("utf-8")

        req = urllib.request.Request(
            "{}/v1/messages".format(PROXY_BASE),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": "sk-ant-test",
                "Anthropic-Version": "2023-06-01",
            },
            method="POST",
        )

        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("Expected HTTPError for invalid request")
        except urllib.error.HTTPError as e:
            data = json.loads(e.read())
            self.assertEqual(data["type"], "error")
            self.assertIn("error", data)
            self.assertIn("type", data["error"])
            self.assertIn("message", data["error"])

    def test_xapi_key_not_validated(self):
        """Proxy accepts any x-api-key value without validation."""
        body = json.dumps({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "test"}],
        }).encode("utf-8")

        # Try with a clearly fake key
        req = urllib.request.Request(
            "{}/v1/messages".format(PROXY_BASE),
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": "completely-fake-key-12345",
                "Anthropic-Version": "2023-06-01",
            },
            method="POST",
        )

        try:
            resp = urllib.request.urlopen(req, timeout=30)
            # If we get here, the key was not validated (good)
            self.assertTrue(True)
        except urllib.error.HTTPError as e:
            # 529 (backend down) is fine — 401/403 means key was validated (bad)
            self.assertNotEqual(e.code, 401,
                                "Proxy validated x-api-key — it should not")
            self.assertNotEqual(e.code, 403,
                                "Proxy rejected fake x-api-key — it should not")


if __name__ == "__main__":
    unittest.main()
