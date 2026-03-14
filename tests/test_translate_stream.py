"""Tests for OpenAI → Anthropic streaming SSE translation."""

import json
import unittest

from src.proxy.translation.stream import translate_stream


def _make_lines(chunks):
    """Convert a list of SSE data payloads into line strings."""
    lines = []
    for chunk in chunks:
        if chunk == "[DONE]":
            lines.append("data: [DONE]")
        else:
            lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # blank line between events
    return iter(lines)


def _parse_events(events):
    """Parse SSE event strings into (event_type, data) tuples."""
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


class TestTextStreaming(unittest.TestCase):
    """Test text-only streaming translation."""

    def test_simple_text_stream(self):
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
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            "[DONE]",
        ]

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_test123"))
        parsed = _parse_events(events)

        # Verify event sequence
        event_types = [e[0] for e in parsed]
        self.assertEqual(event_types[0], "message_start")
        self.assertIn("content_block_start", event_types)
        self.assertIn("content_block_delta", event_types)
        self.assertIn("content_block_stop", event_types)
        self.assertEqual(event_types[-2], "message_delta")
        self.assertEqual(event_types[-1], "message_stop")

        # Verify message_start
        msg_start = parsed[0][1]
        self.assertEqual(msg_start["message"]["role"], "assistant")
        self.assertEqual(msg_start["message"]["model"], "claude-3-opus")

        # Verify text deltas contain our content
        text_deltas = [
            e[1]["delta"]["text"]
            for e in parsed
            if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "text_delta"
        ]
        self.assertEqual("".join(text_deltas), "Hello world")

        # Verify message_delta has stop_reason
        msg_delta = [e for e in parsed if e[0] == "message_delta"][0][1]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")


class TestToolCallStreaming(unittest.TestCase):
    """Test tool call streaming translation."""

    def test_tool_call_stream(self):
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }],
                }, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"city"'},
                    }],
                }, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": ': "London"}'},
                    }],
                }, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
            "[DONE]",
        ]

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_test456"))
        parsed = _parse_events(events)

        event_types = [e[0] for e in parsed]

        # Should have tool_use content_block_start
        tool_starts = [
            e for e in parsed
            if e[0] == "content_block_start"
            and e[1]["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(tool_starts[0][1]["content_block"]["name"], "get_weather")
        self.assertEqual(tool_starts[0][1]["content_block"]["id"], "call_abc")

        # Should have input_json_delta events
        json_deltas = [
            e[1]["delta"]["partial_json"]
            for e in parsed
            if e[0] == "content_block_delta"
            and e[1]["delta"]["type"] == "input_json_delta"
        ]
        self.assertTrue(len(json_deltas) >= 1)

        # Verify stop_reason is tool_use
        msg_delta = [e for e in parsed if e[0] == "message_delta"][0][1]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")


class TestMixedStreaming(unittest.TestCase):
    """Test mixed text + tool call streaming."""

    def test_text_then_tool(self):
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": "Let me check."}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_mix",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "test"}'},
                    }],
                }, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10},
            },
            "[DONE]",
        ]

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_mix"))
        parsed = _parse_events(events)

        event_types = [e[0] for e in parsed]

        # Should see: message_start, text block start, text delta, text block stop,
        # tool block start, tool delta, tool block stop, message_delta, message_stop
        self.assertEqual(event_types[0], "message_start")

        # Text block should come before tool block
        text_start_idx = next(
            i for i, e in enumerate(parsed)
            if e[0] == "content_block_start" and e[1]["content_block"]["type"] == "text"
        )
        tool_start_idx = next(
            i for i, e in enumerate(parsed)
            if e[0] == "content_block_start" and e[1]["content_block"]["type"] == "tool_use"
        )
        self.assertLess(text_start_idx, tool_start_idx)


class TestMultipleToolCallsStream(unittest.TestCase):
    """Test streaming with 2 concurrent tool calls at different indices."""

    def test_multiple_tool_calls_stream(self):
        """Two concurrent tool calls (different indices) should produce
        2 content_block_start events for tool_use, each with different
        names/ids, and separate input_json_delta events for each."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
            },
            # First tool call start
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_first",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }],
                }, "finish_reason": None}],
            },
            # Second tool call start
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_second",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": ""},
                    }],
                }, "finish_reason": None}],
            },
            # Arguments for first tool call
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"city": "London"}'},
                    }],
                }, "finish_reason": None}],
            },
            # Arguments for second tool call
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": 1,
                        "function": {"arguments": '{"zone": "UTC"}'},
                    }],
                }, "finish_reason": None}],
            },
            # Finish
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 20},
            },
            "[DONE]",
        ]

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_multi"))
        parsed = _parse_events(events)

        # Should have 2 tool_use content_block_start events
        tool_starts = [
            e for e in parsed
            if e[0] == "content_block_start"
            and e[1]["content_block"]["type"] == "tool_use"
        ]
        self.assertEqual(len(tool_starts), 2)
        self.assertEqual(tool_starts[0][1]["content_block"]["name"], "get_weather")
        self.assertEqual(tool_starts[0][1]["content_block"]["id"], "call_first")
        self.assertEqual(tool_starts[1][1]["content_block"]["name"], "get_time")
        self.assertEqual(tool_starts[1][1]["content_block"]["id"], "call_second")

        # Should have input_json_delta events for each tool
        json_deltas = [
            e for e in parsed
            if e[0] == "content_block_delta"
            and e[1]["delta"]["type"] == "input_json_delta"
        ]
        self.assertTrue(len(json_deltas) >= 2)

        # Verify arguments end up in the right places
        delta_texts = [e[1]["delta"]["partial_json"] for e in json_deltas]
        combined = "".join(delta_texts)
        self.assertIn("London", combined)
        self.assertIn("UTC", combined)

        # Verify stop reason
        msg_delta = [e for e in parsed if e[0] == "message_delta"][0][1]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")


class TestEmptyContentDelta(unittest.TestCase):
    """Test handling of empty content delta."""

    def test_empty_content_delta(self):
        """Chunk with delta.content='' (empty string) should still emit events."""
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

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_empty_delta"))
        parsed = _parse_events(events)

        event_types = [e[0] for e in parsed]

        # Should have content_block_start (triggered by the empty string content)
        self.assertIn("content_block_start", event_types)

        # Should have content_block_delta events
        text_deltas = [
            e for e in parsed
            if e[0] == "content_block_delta" and e[1]["delta"]["type"] == "text_delta"
        ]
        self.assertTrue(len(text_deltas) >= 1)


class TestUsageInFinalChunk(unittest.TestCase):
    """Test that usage tokens from the final chunk appear in message_delta."""

    def test_usage_in_final_chunk(self):
        """Usage tokens from the final chunk should appear in message_delta event."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 17},
            },
            "[DONE]",
        ]

        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_usage"))
        parsed = _parse_events(events)

        # Find message_delta event
        msg_delta_events = [e for e in parsed if e[0] == "message_delta"]
        self.assertEqual(len(msg_delta_events), 1)
        msg_delta = msg_delta_events[0][1]
        self.assertEqual(msg_delta["usage"]["output_tokens"], 17)


class TestEmptyStream(unittest.TestCase):
    """Test edge case of empty stream."""

    def test_done_only(self):
        chunks = ["[DONE]"]
        events = list(translate_stream(_make_lines(chunks), "claude-3-opus", "msg_empty"))
        parsed = _parse_events(events)

        event_types = [e[0] for e in parsed]
        self.assertIn("message_start", event_types)
        self.assertIn("message_delta", event_types)
        self.assertIn("message_stop", event_types)


if __name__ == "__main__":
    unittest.main()
