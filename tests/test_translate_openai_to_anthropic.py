"""Tests for OpenAI (hub) -> Anthropic upstream translation: request, response, stream."""

import json
import unittest

from src.proxy.translation.openai_to_anthropic import (
    DEFAULT_MAX_TOKENS,
    AnthropicStreamError,
    anthropic_models_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai,
    openai_request_to_anthropic,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class TestRequestTranslation(unittest.TestCase):
    def test_simple_request_defaults_max_tokens(self):
        out = openai_request_to_anthropic(
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        self.assertEqual(
            out,
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": DEFAULT_MAX_TOKENS,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            },
        )

    def test_max_tokens_sources(self):
        cases = [
            ({"max_tokens": 100}, 100),
            ({"max_completion_tokens": 200}, 200),
            ({"max_completion_tokens": 300, "max_tokens": 50}, 300),
            ({}, DEFAULT_MAX_TOKENS),
        ]
        for extra, expected in cases:
            with self.subTest(extra=extra):
                body = dict({"model": "m", "messages": [{"role": "user", "content": "x"}]}, **extra)
                self.assertEqual(openai_request_to_anthropic(body)["max_tokens"], expected)

    def test_system_and_developer_hoisted(self):
        out = openai_request_to_anthropic(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "developer", "content": [{"type": "text", "text": "Use metric."}]},
                    {"role": "user", "content": "Hi"},
                    {"role": "system", "content": "Late rule."},
                ],
            }
        )
        self.assertEqual(out["system"], "Be brief.\n\nUse metric.\n\nLate rule.")
        self.assertEqual([m["role"] for m in out["messages"]], ["user"])

    def test_sampling_stop_user_stream(self):
        out = openai_request_to_anthropic(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "temperature": 1.7,
                "top_p": 0.9,
                "stop": "END",
                "user": "u-1",
                "stream": True,
                "stream_options": {"include_usage": True},
                "response_format": {"type": "json_object"},
            }
        )
        self.assertEqual(out["temperature"], 1.0)  # clamped to Anthropic range
        self.assertEqual(out["top_p"], 0.9)
        self.assertEqual(out["stop_sequences"], ["END"])
        self.assertEqual(out["metadata"], {"user_id": "u-1"})
        self.assertTrue(out["stream"])
        for key in ("stream_options", "response_format", "user", "stop"):
            self.assertNotIn(key, out)

    def test_images(self):
        out = openai_request_to_anthropic(
            {
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64,QUJD"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            },
                            {"type": "image_url", "image_url": {"url": "ftp://nope"}},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(
            out["messages"][0]["content"],
            [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
                },
                {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
            ],
        )

    def test_tools_and_tool_choice(self):
        cases = [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "any"}),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "tool", "name": "get_weather"},
            ),
            (None, None),
        ]
        for choice, expected in cases:
            with self.subTest(choice=choice):
                body = {
                    "model": "m",
                    "messages": [{"role": "user", "content": "x"}],
                    "tools": [WEATHER_TOOL],
                }
                if choice is not None:
                    body["tool_choice"] = choice
                out = openai_request_to_anthropic(body)
                self.assertEqual(
                    out["tools"],
                    [
                        {
                            "name": "get_weather",
                            "input_schema": WEATHER_TOOL["function"]["parameters"],
                            "description": "Get weather",
                        }
                    ],
                )
                self.assertEqual(out.get("tool_choice"), expected)

    def test_parallel_tool_calls_false(self):
        out = openai_request_to_anthropic(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [WEATHER_TOOL],
                "parallel_tool_calls": False,
            }
        )
        self.assertEqual(out["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})

    def test_multi_turn_tool_results(self):
        out = openai_request_to_anthropic(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "Weather in Paris and Rome?"},
                    {
                        "role": "assistant",
                        "content": "Checking.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "not json"},
                            },
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
                    {
                        "role": "tool",
                        "tool_call_id": "call_2",
                        "content": [{"type": "text", "text": "21C"}],
                    },
                    {"role": "user", "content": "Thanks, which is warmer?"},
                    {"role": "assistant", "content": None},
                    {"role": "assistant", "content": "Rome."},
                ],
            }
        )
        msgs = out["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual(
            msgs[1]["content"],
            [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
                {"type": "tool_use", "id": "call_2", "name": "get_weather", "input": {}},
            ],
        )
        # Tool results merge into one user turn, before the follow-up text.
        self.assertEqual(
            msgs[2]["content"],
            [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "18C"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "21C"},
                {"type": "text", "text": "Thanks, which is warmer?"},
            ],
        )
        self.assertEqual(msgs[3]["content"], [{"type": "text", "text": "Rome."}])


class TestResponseTranslation(unittest.TestCase):
    def _msg(self, content, stop_reason="end_turn", usage=None):
        return {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        }

    def test_text_response(self):
        out = anthropic_response_to_openai(
            self._msg([{"type": "text", "text": "Hello"}]), created=1
        )
        self.assertEqual(
            out,
            {
                "id": "msg_01",
                "object": "chat.completion",
                "created": 1,
                "model": "claude-x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def test_tool_use_response(self):
        out = anthropic_response_to_openai(
            self._msg(
                [
                    {"type": "thinking", "thinking": "hmm", "signature": "s"},
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
                stop_reason="tool_use",
            )
        )
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "Let me check.")
        self.assertEqual(
            choice["message"]["tool_calls"],
            [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        )

    def test_tool_only_response_has_null_content(self):
        out = anthropic_response_to_openai(
            self._msg(
                [{"type": "tool_use", "id": "t", "name": "f", "input": {}}], stop_reason="tool_use"
            )
        )
        self.assertIsNone(out["choices"][0]["message"]["content"])

    def test_stop_reasons(self):
        cases = [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("refusal", "content_filter"),
            ("pause_turn", "stop"),
            ("something_new", "stop"),
            (None, "stop"),
        ]
        for stop_reason, finish in cases:
            with self.subTest(stop_reason=stop_reason):
                out = anthropic_response_to_openai(self._msg([], stop_reason=stop_reason))
                self.assertEqual(out["choices"][0]["finish_reason"], finish)

    def test_usage_with_cache(self):
        out = anthropic_response_to_openai(
            self._msg(
                [],
                usage={
                    "input_tokens": 10,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 20,
                },
            )
        )
        self.assertEqual(
            out["usage"],
            {
                "prompt_tokens": 33,
                "completion_tokens": 7,
                "total_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        )

    def test_models_listing(self):
        out = anthropic_models_to_openai(
            {
                "data": [
                    {
                        "type": "model",
                        "id": "claude-a",
                        "display_name": "A",
                        "created_at": "2025-02-19T00:00:00Z",
                    },
                    {"type": "model", "id": "claude-b", "created_at": "garbage"},
                ],
                "has_more": False,
            }
        )
        self.assertEqual(out["object"], "list")
        self.assertEqual(
            out["data"][0],
            {"id": "claude-a", "object": "model", "created": 1739923200, "owned_by": "anthropic"},
        )
        self.assertEqual(out["data"][1]["created"], 0)


def _sse(event_type, data):
    return [f"event: {event_type}\n", f"data: {json.dumps(dict(data, type=event_type))}\n", "\n"]


def _anthropic_stream(*events):
    lines = []
    for event_type, data in events:
        lines.extend(_sse(event_type, data))
    return lines


def _chunks(output):
    """Parse translator output into (list_of_chunk_dicts, saw_done)."""
    chunks, done = [], False
    for item in output:
        for line in item.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
            else:
                chunks.append(json.loads(data))
    return chunks, done


MESSAGE_START = (
    "message_start",
    {
        "message": {
            "id": "msg_9",
            "type": "message",
            "role": "assistant",
            "model": "claude-x",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 1},
        }
    },
)


class TestStreamTranslation(unittest.TestCase):
    def test_text_stream_with_usage(self):
        lines = _anthropic_stream(
            MESSAGE_START,
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("ping", {}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 12},
                },
            ),
            ("message_stop", {}),
        )
        chunks, done = _chunks(anthropic_stream_to_openai(lines, include_usage=True, created=5))
        self.assertTrue(done)
        self.assertTrue(all(c["id"] == "msg_9" and c["model"] == "claude-x" for c in chunks))
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
        self.assertEqual(deltas[0], {"role": "assistant", "content": ""})
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Hello")
        finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(
            chunks[-1]["usage"], {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}
        )

    def test_usage_chunk_only_when_requested(self):
        lines = _anthropic_stream(MESSAGE_START, ("message_stop", {}))
        chunks, done = _chunks(anthropic_stream_to_openai(lines))
        self.assertTrue(done)
        self.assertFalse(any("usage" in c for c in chunks))

    def test_tool_call_stream(self):
        lines = _anthropic_stream(
            MESSAGE_START,
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Ok."}}),
            ("content_block_stop", {"index": 0}),
            (
                "content_block_start",
                {
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_A",
                        "name": "get_weather",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
            ),
            (
                "content_block_delta",
                {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'}},
            ),
            ("content_block_stop", {"index": 1}),
            (
                "content_block_start",
                {
                    "index": 2,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_B",
                        "name": "get_time",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {"index": 2, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
            ),
            ("content_block_stop", {"index": 2}),
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}},
            ),
            ("message_stop", {}),
        )
        chunks, done = _chunks(anthropic_stream_to_openai(lines))
        self.assertTrue(done)
        tool_deltas = [
            tc
            for c in chunks
            if c["choices"]
            for tc in c["choices"][0]["delta"].get("tool_calls", [])
        ]
        self.assertEqual(
            tool_deltas[0],
            {
                "index": 0,
                "id": "toolu_A",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            },
        )
        args = {}
        for tc in tool_deltas:
            args[tc["index"]] = args.get(tc["index"], "") + tc["function"].get("arguments", "")
        self.assertEqual(json.loads(args[0]), {"city": "Paris"})
        self.assertEqual(json.loads(args[1]), {})
        self.assertEqual(tool_deltas[-2]["id"], "toolu_B")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_stop_reason_mapping_in_stream(self):
        for stop_reason, finish in [
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("refusal", "content_filter"),
        ]:
            with self.subTest(stop_reason=stop_reason):
                lines = _anthropic_stream(
                    MESSAGE_START,
                    ("message_delta", {"delta": {"stop_reason": stop_reason}}),
                    ("message_stop", {}),
                )
                chunks, _ = _chunks(anthropic_stream_to_openai(lines))
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], finish)

    def test_error_event_raises(self):
        lines = _anthropic_stream(
            MESSAGE_START,
            ("error", {"error": {"type": "overloaded_error", "message": "Overloaded"}}),
        )
        gen = anthropic_stream_to_openai(lines)
        with self.assertRaises(AnthropicStreamError) as ctx:
            list(gen)
        self.assertEqual(ctx.exception.error_type, "overloaded_error")
        self.assertEqual(ctx.exception.message, "Overloaded")

    def test_truncated_stream_still_terminates(self):
        lines = _anthropic_stream(
            MESSAGE_START,
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        )
        chunks, done = _chunks(anthropic_stream_to_openai(lines))
        self.assertTrue(done)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_empty_stream(self):
        chunks, done = _chunks(anthropic_stream_to_openai([]))
        self.assertTrue(done)
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
