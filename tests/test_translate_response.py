"""Tests for OpenAI → Anthropic response translation."""

import unittest

from src.proxy.translation.response import translate_response


class TestTextResponse(unittest.TestCase):
    """Test text-only response translation."""

    def test_simple_text(self):
        openai_body = {
            "id": "chatcmpl-abc123",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
            },
        }
        result = translate_response(openai_body, "claude-3-opus")

        self.assertEqual(result["type"], "message")
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["model"], "claude-3-opus")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "Hello! How can I help?")
        self.assertEqual(result["usage"]["input_tokens"], 10)
        self.assertEqual(result["usage"]["output_tokens"], 7)

    def test_empty_response(self):
        openai_body = {
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        result = translate_response(openai_body, "claude-3-opus")
        # Should have at least one content block
        self.assertTrue(len(result["content"]) >= 1)


class TestToolCallResponse(unittest.TestCase):
    """Test tool call response translation."""

    def test_single_tool_call(self):
        openai_body = {
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        result = translate_response(openai_body, "claude-3-opus")

        self.assertEqual(result["stop_reason"], "tool_use")
        # Should have one tool_use block
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "get_weather")
        self.assertEqual(tool_blocks[0]["input"], {"city": "London"})
        self.assertEqual(tool_blocks[0]["id"], "call_abc123")

    def test_mixed_text_and_tool(self):
        openai_body = {
            "id": "chatcmpl-abc",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me look that up.",
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "weather"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = translate_response(openai_body, "claude-3-opus")

        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "Let me look that up.")
        self.assertEqual(result["content"][1]["type"], "tool_use")


class TestStopReasonMapping(unittest.TestCase):
    """Test finish_reason → stop_reason mapping."""

    def _make_response(self, finish_reason):
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "x"},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def test_stop(self):
        result = translate_response(self._make_response("stop"), "m")
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_length(self):
        result = translate_response(self._make_response("length"), "m")
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_tool_calls(self):
        result = translate_response(self._make_response("tool_calls"), "m")
        self.assertEqual(result["stop_reason"], "tool_use")

    def test_unknown(self):
        result = translate_response(self._make_response("something_new"), "m")
        self.assertEqual(result["stop_reason"], "end_turn")


class TestMultipleToolCalls(unittest.TestCase):
    """Test response with multiple tool calls."""

    def test_multiple_tool_calls(self):
        """OpenAI response with 2 tool_calls should produce 2 tool_use content blocks."""
        openai_body = {
            "id": "chatcmpl-multi",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }
        result = translate_response(openai_body, "claude-3-opus")

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(tool_blocks[0]["id"], "call_1")
        self.assertEqual(tool_blocks[0]["name"], "get_weather")
        self.assertEqual(tool_blocks[0]["input"], {"city": "London"})
        self.assertEqual(tool_blocks[1]["id"], "call_2")
        self.assertEqual(tool_blocks[1]["name"], "get_weather")
        self.assertEqual(tool_blocks[1]["input"], {"city": "Paris"})
        self.assertEqual(result["stop_reason"], "tool_use")


class TestMalformedToolArguments(unittest.TestCase):
    """Test handling of malformed tool call arguments."""

    def test_malformed_tool_arguments(self):
        """tool_call with invalid JSON arguments should fallback to empty dict {}."""
        openai_body = {
            "id": "chatcmpl-bad",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "do_something",
                            "arguments": "{this is not valid json!!",
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        result = translate_response(openai_body, "claude-3-opus")

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["input"], {})


class TestMissingUsage(unittest.TestCase):
    """Test handling of missing usage field."""

    def test_missing_usage(self):
        """Response with no usage field should default to 0/0 tokens."""
        openai_body = {
            "id": "chatcmpl-nousage",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
        }
        result = translate_response(openai_body, "claude-3-opus")
        self.assertEqual(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["usage"]["output_tokens"], 0)


class TestMissingToolCallId(unittest.TestCase):
    """Test handling of tool_call without id."""

    def test_missing_tool_call_id(self):
        """tool_call without id should generate a fallback id starting with 'toolu_'."""
        openai_body = {
            "id": "chatcmpl-noid",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"key": "value"}',
                        },
                        # Note: no "id" field
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        }
        result = translate_response(openai_body, "claude-3-opus")

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertTrue(tool_blocks[0]["id"].startswith("toolu_"),
                        f"Expected id starting with 'toolu_', got: {tool_blocks[0]['id']}")


class TestContentFilterStopReason(unittest.TestCase):
    """Test content_filter finish_reason mapping."""

    def test_content_filter_stop_reason(self):
        """finish_reason 'content_filter' maps to 'end_turn'."""
        openai_body = {
            "choices": [{
                "message": {"role": "assistant", "content": "Filtered."},
                "finish_reason": "content_filter",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        result = translate_response(openai_body, "claude-3-opus")
        self.assertEqual(result["stop_reason"], "end_turn")


class TestNoChoices(unittest.TestCase):
    """Test handling of malformed responses."""

    def test_empty_choices(self):
        openai_body = {"choices": [], "usage": {}}
        result = translate_response(openai_body, "claude-3-opus")
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["stop_reason"], "end_turn")


if __name__ == "__main__":
    unittest.main()
