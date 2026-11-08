"""End-to-end provider gateway tests.

Runs the real proxy server and a fake upstream HTTP server, both on
127.0.0.1 with ephemeral ports. No external network access.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.proxy.config import Config
from src.proxy.server import ThreadedHTTPServer

KEY = "sk-e2e-SECRET-4f9c2b7a1d"


# -- Fake upstream ---------------------------------------------------------


class FakeUpstream:
    """Scriptable upstream. routes maps (method, path) -> (status, headers, body).

    body is bytes, a dict (JSON), or a list of str (SSE lines written in order).
    Every received request is appended to self.requests.
    """

    def __init__(self):
        self.routes = {}
        self.requests = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                fake.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(raw) if raw else None,
                    }
                )
                route = fake.routes.get((self.command, self.path.split("?")[0]))
                if route is None:
                    status, headers, body = 404, {}, {"error": {"message": "no route"}}
                else:
                    status, headers, body = route
                if isinstance(body, list):
                    self.send_response(status)
                    self.send_header("Content-Type", "text/event-stream")
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.end_headers()
                    for line in body:
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    return
                data = json.dumps(body).encode() if isinstance(body, dict) else body
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            do_GET = _serve
            do_POST = _serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()

    def url(self, path=""):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# -- Fixtures ----------------------------------------------------------------


def openai_completion(**extra):
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello from upstream"},
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }
    body.update(extra)
    return body


def openai_sse(chunks, done=True):
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return lines


def openai_chunk(delta, finish=None, **extra):
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-x",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    chunk.update(extra)
    return chunk


ANTHROPIC_MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 9},
}


def anthropic_sse(*events):
    lines = []
    for etype, data in events:
        lines.append(f"event: {etype}\ndata: {json.dumps(dict(data, type=etype))}\n\n")
    return lines


ANTHROPIC_STREAM = anthropic_sse(
    (
        "message_start",
        {
            "message": {
                "id": "msg_s",
                "type": "message",
                "role": "assistant",
                "model": "claude-x",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 8, "output_tokens": 1},
            }
        },
    ),
    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
    ("ping", {}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "there"}}),
    ("content_block_stop", {"index": 0}),
    (
        "content_block_start",
        {
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_9", "name": "lookup", "input": {}},
        },
    ),
    (
        "content_block_delta",
        {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"q": "x"}'}},
    ),
    ("content_block_stop", {"index": 1}),
    ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}}),
    ("message_stop", {}),
)

CHAT_BODY = {
    "model": "some-model",
    "messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Weather in Paris?"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        }
    ],
}


def parse_sse(raw):
    chunks, done = [], False
    for line in raw.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
            else:
                chunks.append(json.loads(data))
    return chunks, done


# -- Base --------------------------------------------------------------------


class GatewayTestBase(unittest.TestCase):
    def setUp(self):
        self.upstream = FakeUpstream()
        self.addCleanup(self.upstream.close)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.record_path = os.path.join(self.tmp, "rec.jsonl")
        self.analytics_path = os.path.join(self.tmp, "analytics.jsonl")
        self.config = Config(
            host="127.0.0.1",
            port=0,
            backend_url=self.upstream.url(),
            request_timeout=10,
            record_path=self.record_path,
            analytics_path=self.analytics_path,
            cache_enabled=True,
        )
        self.proxy = ThreadedHTTPServer(self.config)
        self.port = self.proxy.server_address[1]
        threading.Thread(target=self.proxy.serve_forever, args=(0.05,), daemon=True).start()
        self.addCleanup(self.proxy.server_close)
        self.addCleanup(self.proxy.shutdown)

    def gw_headers(self, provider, key=KEY, url=None, path="/v1"):
        headers = {"X-Lana-Upstream-Provider": provider}
        if key is not None:
            headers["X-Lana-Upstream-Key"] = key
        headers["X-Lana-Upstream-Url"] = url if url is not None else self.upstream.url(path)
        return headers

    def call(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            method=method,
            headers=dict(headers or {}),
        )
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read().decode()

    def last_upstream(self):
        self.assertTrue(self.upstream.requests, "upstream received no request")
        return self.upstream.requests[-1]

    def assert_no_lana_headers_forwarded(self):
        for r in self.upstream.requests:
            leaked = [h for h in r["headers"] if h.startswith("x-lana")]
            self.assertEqual(leaked, [], f"X-Lana headers forwarded: {leaked}")


# -- OpenAI dialect upstreams -----------------------------------------------


class TestOpenAIDialect(GatewayTestBase):
    def test_non_streaming_passthrough_with_bearer(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, {}, openai_completion())
        status, _, raw = self.call(
            "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai")
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), openai_completion())
        req = self.last_upstream()
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {KEY}")
        self.assertEqual(req["body"], CHAT_BODY)  # forwarded unchanged
        self.assert_no_lana_headers_forwarded()

    def test_streaming_passthrough_adds_missing_done(self):
        chunks = [
            openai_chunk({"role": "assistant", "content": ""}),
            openai_chunk({"content": "Hi"}),
            openai_chunk({}, finish="stop"),
        ]
        for upstream_done in (True, False):
            with self.subTest(upstream_sends_done=upstream_done):
                self.upstream.routes[("POST", "/v1/chat/completions")] = (
                    200,
                    {},
                    [": keepalive\n\n"] + openai_sse(chunks, done=upstream_done),
                )
                body = dict(CHAT_BODY, stream=True)
                status, ctype, raw = self.call(
                    "POST", "/v1/chat/completions", body, self.gw_headers("groq")
                )
                self.assertEqual(status, 200)
                self.assertIn("text/event-stream", ctype)
                got, done = parse_sse(raw)
                self.assertEqual(got, chunks)
                self.assertTrue(done)
                self.assertEqual(raw.count("[DONE]"), 1)

    def test_perplexity_extra_fields_pass_through(self):
        citations = ["https://a.example/1", "https://b.example/2"]
        search_results = [{"title": "A", "url": "https://a.example/1", "date": "2025-01-01"}]
        self.upstream.routes[("POST", "/chat/completions")] = (
            200,
            {},
            openai_completion(citations=citations, search_results=search_results),
        )
        status, _, raw = self.call(
            "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("perplexity", path="")
        )
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["citations"], citations)
        self.assertEqual(body["search_results"], search_results)

        final = openai_chunk({}, finish="stop", citations=citations, search_results=search_results)
        self.upstream.routes[("POST", "/chat/completions")] = (
            200,
            {},
            openai_sse([openai_chunk({"content": "x"}), final]),
        )
        status, _, raw = self.call(
            "POST",
            "/v1/chat/completions",
            dict(CHAT_BODY, stream=True),
            self.gw_headers("perplexity", path=""),
        )
        chunks, _ = parse_sse(raw)
        self.assertEqual(chunks[-1], final)

    def test_azure_api_key_header_and_query_preserved(self):
        self.upstream.routes[("POST", "/openai/deployments/gpt4o/chat/completions")] = (
            200,
            {},
            openai_completion(),
        )
        url = self.upstream.url("/openai/deployments/gpt4o?api-version=2024-10-21")
        status, _, _ = self.call(
            "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("azure_openai", url=url)
        )
        self.assertEqual(status, 200)
        req = self.last_upstream()
        self.assertEqual(
            req["path"], "/openai/deployments/gpt4o/chat/completions?api-version=2024-10-21"
        )
        self.assertEqual(req["headers"]["api-key"], KEY)
        self.assertNotIn("authorization", req["headers"])

    def test_models_passthrough(self):
        listing = {"object": "list", "data": [{"id": "gpt-x", "object": "model"}]}
        self.upstream.routes[("GET", "/v1/models")] = (200, {}, listing)
        status, _, raw = self.call("GET", "/v1/models", headers=self.gw_headers("openai"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), listing)

    def test_models_unsupported_provider(self):
        status, _, raw = self.call(
            "GET", "/v1/models", headers=self.gw_headers("perplexity", path="")
        )
        self.assertEqual(status, 404)
        self.assertIn("model listing", json.loads(raw)["error"]["message"])
        self.assertEqual(self.upstream.requests, [])


# -- Anthropic dialect upstream ---------------------------------------------


class TestAnthropicDialect(GatewayTestBase):
    def test_non_streaming_translation(self):
        self.upstream.routes[("POST", "/v1/messages")] = (200, {}, ANTHROPIC_MESSAGE)
        status, _, raw = self.call(
            "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("anthropic")
        )
        self.assertEqual(status, 200)
        req = self.last_upstream()
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["headers"]["x-api-key"], KEY)
        self.assertEqual(req["headers"]["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", req["headers"])
        self.assertEqual(req["body"]["system"], "Be terse.")
        self.assertEqual(req["body"]["max_tokens"], 4096)
        self.assertEqual(req["body"]["model"], "some-model")
        self.assertEqual(req["body"]["tools"][0]["name"], "get_weather")
        self.assertNotIn("stream", req["body"])
        self.assert_no_lana_headers_forwarded()

        body = json.loads(raw)
        self.assertEqual(body["object"], "chat.completion")
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "Checking.")
        self.assertEqual(
            json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]),
            {"city": "Paris"},
        )
        self.assertEqual(
            body["usage"], {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20}
        )

    def test_streaming_translation_with_usage(self):
        self.upstream.routes[("POST", "/v1/messages")] = (200, {}, ANTHROPIC_STREAM)
        body = dict(CHAT_BODY, stream=True, stream_options={"include_usage": True})
        status, ctype, raw = self.call(
            "POST", "/v1/chat/completions", body, self.gw_headers("anthropic")
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", ctype)
        self.assertTrue(self.last_upstream()["body"]["stream"])
        self.assertEqual(self.last_upstream()["headers"]["accept"], "text/event-stream")
        chunks, done = parse_sse(raw)
        self.assertTrue(done)
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "Hi there")
        tool = [
            tc
            for c in chunks
            if c["choices"]
            for tc in c["choices"][0]["delta"].get("tool_calls", [])
        ]
        self.assertEqual(tool[0]["id"], "toolu_9")
        self.assertEqual(tool[0]["function"]["name"], "lookup")
        self.assertEqual("".join(t["function"]["arguments"] for t in tool), '{"q": "x"}')
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            chunks[-1]["usage"], {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14}
        )

    def test_stream_error_event_becomes_openai_error_chunk(self):
        self.upstream.routes[("POST", "/v1/messages")] = (
            200,
            {},
            anthropic_sse(
                ("message_start", {"message": {"id": "m", "model": "claude-x", "usage": {}}}),
                ("error", {"error": {"type": "overloaded_error", "message": "Overloaded"}}),
            ),
        )
        status, _, raw = self.call(
            "POST",
            "/v1/chat/completions",
            dict(CHAT_BODY, stream=True),
            self.gw_headers("anthropic"),
        )
        self.assertEqual(status, 200)
        chunks, done = parse_sse(raw)
        self.assertTrue(done)
        self.assertIn("Overloaded", chunks[-1]["error"]["message"])

    def test_models_mapped(self):
        self.upstream.routes[("GET", "/v1/models")] = (
            200,
            {},
            {"data": [{"type": "model", "id": "claude-x", "created_at": "2025-02-19T00:00:00Z"}]},
        )
        status, _, raw = self.call("GET", "/v1/models", headers=self.gw_headers("anthropic"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["data"][0]["id"], "claude-x")
        self.assertEqual(self.last_upstream()["headers"]["x-api-key"], KEY)

    def test_anthropic_inbound_to_anthropic_upstream(self):
        """/v1/messages with gateway headers reaches the selected provider."""
        self.upstream.routes[("POST", "/v1/messages")] = (200, {}, ANTHROPIC_MESSAGE)
        body = {
            "model": "claude-x",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        for _ in range(2):  # second call must not be served from the prompt cache
            status, _, raw = self.call("POST", "/v1/messages", body, self.gw_headers("anthropic"))
            self.assertEqual(status, 200)
            out = json.loads(raw)
            self.assertEqual(out["type"], "message")
            self.assertEqual(out["stop_reason"], "tool_use")
            self.assertEqual(out["content"][1]["input"], {"city": "Paris"})
        self.assertEqual(len(self.upstream.requests), 2)
        self.assertEqual(self.last_upstream()["body"]["model"], "claude-x")

    def test_anthropic_inbound_streaming_via_openai_upstream(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (
            200,
            {},
            openai_sse(
                [
                    openai_chunk({"role": "assistant", "content": "Yo"}),
                    openai_chunk({}, finish="stop"),
                ]
            ),
        )
        body = {
            "model": "gpt-x",
            "max_tokens": 50,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status, _, raw = self.call("POST", "/v1/messages", body, self.gw_headers("openai"))
        self.assertEqual(status, 200)
        self.assertIn("event: message_start", raw)
        self.assertIn('"text": "Yo"', raw)
        self.assertIn("event: message_stop", raw)
        self.assertEqual(self.last_upstream()["body"]["model"], "gpt-x")


# -- Errors and security -----------------------------------------------------


class TestErrorsAndSecrecy(GatewayTestBase):
    def test_upstream_status_mapping_and_key_redaction(self):
        cases = [
            (401, 401),
            (403, 403),
            (404, 404),
            (429, 429),
            (400, 400),
            (422, 400),
            (500, 502),
            (502, 502),
            (503, 503),
            (529, 503),
            (504, 504),
        ]
        for upstream_status, expected in cases:
            with self.subTest(upstream_status=upstream_status):
                # Upstream echoes the key back in its error body.
                self.upstream.routes[("POST", "/v1/chat/completions")] = (
                    upstream_status,
                    {},
                    {"error": {"message": f"Incorrect API key provided: {KEY}"}},
                )
                status, ctype, raw = self.call(
                    "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai")
                )
                self.assertEqual(status, expected)
                self.assertIn("application/json", ctype)
                err = json.loads(raw)["error"]
                self.assertIn(f"HTTP {upstream_status}", err["message"])
                self.assertIn("Incorrect API key provided", err["message"])
                self.assertNotIn(KEY, raw)
                self.assertEqual(set(err), {"message", "type", "param", "code"})

    def test_error_types(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (
            401,
            {},
            {"error": {"message": "nope"}},
        )
        _, _, raw = self.call("POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai"))
        self.assertEqual(json.loads(raw)["error"]["type"], "authentication_error")

        # /v1/messages gets the Anthropic error envelope instead
        self.upstream.routes[("POST", "/v1/messages")] = (429, {}, {"error": {"message": "slow"}})
        status, _, raw = self.call(
            "POST",
            "/v1/messages",
            {"model": "c", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
            self.gw_headers("anthropic"),
        )
        self.assertEqual(status, 429)
        self.assertEqual(json.loads(raw)["type"], "error")
        self.assertEqual(json.loads(raw)["error"]["type"], "rate_limit_error")

    def test_header_validation_errors(self):
        cases = [
            ("unknown provider", {"X-Lana-Upstream-Provider": "nope", "X-Lana-Upstream-Key": KEY}),
            ("key without provider", {"X-Lana-Upstream-Key": KEY}),
            (
                "custom without url",
                {"X-Lana-Upstream-Provider": "custom", "X-Lana-Upstream-Key": KEY},
            ),
            (
                "http to public host",
                {
                    "X-Lana-Upstream-Provider": "openai",
                    "X-Lana-Upstream-Key": KEY,
                    "X-Lana-Upstream-Url": "http://api.openai.com/v1",
                },
            ),
            (
                "private address for preset",
                {
                    "X-Lana-Upstream-Provider": "openai",
                    "X-Lana-Upstream-Key": KEY,
                    "X-Lana-Upstream-Url": "https://10.0.0.8/v1",
                },
            ),
            ("missing key", {"X-Lana-Upstream-Provider": "anthropic"}),
        ]
        for name, headers in cases:
            with self.subTest(case=name):
                status, _, raw = self.call("POST", "/v1/chat/completions", CHAT_BODY, headers)
                self.assertEqual(status, 400)
                self.assertIn("message", json.loads(raw)["error"])
                self.assertNotIn(KEY, raw)
        self.assertEqual(self.upstream.requests, [])

    def test_missing_model_with_upstream(self):
        body = {"messages": [{"role": "user", "content": "x"}]}
        status, _, raw = self.call("POST", "/v1/chat/completions", body, self.gw_headers("openai"))
        self.assertEqual(status, 400)
        self.assertIn("model", json.loads(raw)["error"]["message"])

    def test_redirect_is_refused(self):
        other = FakeUpstream()
        self.addCleanup(other.close)
        self.upstream.routes[("POST", "/v1/chat/completions")] = (
            307,
            {"Location": other.url("/steal")},
            b"",
        )
        status, _, raw = self.call(
            "POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai")
        )
        self.assertEqual(status, 502)
        self.assertIn("redirect", json.loads(raw)["error"]["message"])
        self.assertEqual(other.requests, [])

    def test_unreachable_upstream(self):
        status, _, raw = self.call(
            "POST",
            "/v1/chat/completions",
            CHAT_BODY,
            self.gw_headers("openai", url="http://127.0.0.1:1/v1"),
        )
        self.assertEqual(status, 502)
        self.assertIn("unreachable", json.loads(raw)["error"]["message"])

    def test_key_never_in_recorder_analytics_or_logs(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, {}, openai_completion())
        self.upstream.routes[("POST", "/v1/messages")] = (200, {}, ANTHROPIC_STREAM)
        with self.assertLogs("proxy", level=logging.DEBUG) as captured:
            self.call("POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai"))
            self.call(
                "POST",
                "/v1/chat/completions",
                dict(CHAT_BODY, stream=True),
                self.gw_headers("anthropic"),
            )
            self.call(
                "POST",
                "/v1/messages",
                {"model": "c", "max_tokens": 5, "messages": [{"role": "user", "content": "x"}]},
                self.gw_headers("anthropic"),
            )
            self.upstream.routes[("POST", "/v1/chat/completions")] = (
                401,
                {},
                {"error": {"message": f"bad key {KEY}"}},
            )
            self.call("POST", "/v1/chat/completions", CHAT_BODY, self.gw_headers("openai"))
        log_text = "\n".join(captured.output)
        self.assertIn("upstream=openai", log_text)  # logs were captured
        self.assertNotIn(KEY, log_text)
        with open(self.record_path, encoding="utf-8") as f:
            recorded = f.read()
        with open(self.analytics_path, encoding="utf-8") as f:
            analytics = f.read()
        self.assertTrue(recorded and analytics)
        self.assertNotIn(KEY, recorded)
        self.assertNotIn(KEY, analytics)
        self.assertNotIn("X-Lana", recorded)
        # the key reached the upstream, and only via its auth header
        for r in self.upstream.requests:
            self.assertNotIn(KEY, json.dumps(r["body"]))
        self.assert_no_lana_headers_forwarded()


# -- Behavior without gateway headers ----------------------------------------


class TestConfigRoutingUnchanged(GatewayTestBase):
    def test_messages_without_headers_uses_config_backend(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, {}, openai_completion())
        body = {
            "model": "claude-3-opus",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        status, _, raw = self.call("POST", "/v1/messages", body)
        self.assertEqual(status, 200)
        out = json.loads(raw)
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["content"][0]["text"], "Hello from upstream")
        req = self.last_upstream()
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["body"]["model"], "local-model")  # config model mapping applies
        self.assertNotIn("authorization", req["headers"])
        # prompt cache still applies to config-routed requests
        self.call("POST", "/v1/messages", body)
        self.assertEqual(len(self.upstream.requests), 1)

    def test_chat_completions_without_headers_uses_config_backend(self):
        self.upstream.routes[("POST", "/v1/chat/completions")] = (200, {}, openai_completion())
        status, _, raw = self.call("POST", "/v1/chat/completions", CHAT_BODY)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), openai_completion())
        req = self.last_upstream()
        self.assertEqual(req["body"], CHAT_BODY)
        self.assertNotIn("authorization", req["headers"])

    def test_models_without_headers_uses_config_backend(self):
        listing = {"object": "list", "data": [{"id": "local", "object": "model"}]}
        self.upstream.routes[("GET", "/v1/models")] = (200, {}, listing)
        status, _, raw = self.call("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), listing)

    def test_chat_completions_validation_uses_openai_error_shape(self):
        for body in ({"model": "m"}, {"model": "m", "messages": []}):
            with self.subTest(body=body):
                status, _, raw = self.call("POST", "/v1/chat/completions", body)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(raw))
                self.assertNotIn("type", json.loads(raw))


if __name__ == "__main__":
    unittest.main()
