"""Tests for Anthropic → OpenAI request translation."""

import json
import unittest

from src.proxy.config import Config
from src.proxy.translation.request import translate_request


class TestSimpleMessages(unittest.TestCase):
    """Test basic message translation."""

    def setUp(self):
        self.config = Config(default_model="test-model")

    def test_simple_text_message(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = translate_request(body, self.config)

        self.assertEqual(result["model"], "test-model")
        self.assertEqual(result["max_tokens"], 1024)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["content"], "Hello")

    def test_model_mapping(self):
        config = Config(model_map={"claude-3-opus": "qwen2.5-coder"})
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, config)
        self.assertEqual(result["model"], "qwen2.5-coder")

    def test_multi_turn(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        result = translate_request(body, self.config)
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["messages"][1]["role"], "assistant")


class TestSystemPrompt(unittest.TestCase):
    """Test system prompt extraction."""

    def setUp(self):
        self.config = Config()

    def test_string_system(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(result["messages"][0]["content"], "You are helpful.")
        self.assertEqual(result["messages"][1]["role"], "user")

    def test_block_array_system(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertIn("Line one.", result["messages"][0]["content"])
        self.assertIn("Line two.", result["messages"][0]["content"])

    def test_no_system(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertEqual(result["messages"][0]["role"], "user")


class TestToolUse(unittest.TestCase):
    """Test tool_use and tool_result translation."""

    def setUp(self):
        self.config = Config()

    def test_tool_definitions(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }],
        }
        result = translate_request(body, self.config)
        tool = result["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "get_weather")
        self.assertIn("properties", tool["function"]["parameters"])

    def test_assistant_tool_use_blocks(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    },
                ],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Let me check.")
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(msg["tool_calls"][0]["id"], "toolu_123")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(msg["tool_calls"][0]["function"]["arguments"]),
            {"city": "London"},
        )

    def test_tool_result_blocks(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Sunny, 22C",
                    },
                ],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "toolu_123")
        self.assertEqual(msg["content"], "Sunny, 22C")

    def test_mixed_tool_result_and_text(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "Result data",
                    },
                    {"type": "text", "text": "Now do something else."},
                ],
            }],
        }
        result = translate_request(body, self.config)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "tool")
        self.assertEqual(result["messages"][1]["role"], "user")


class TestToolResultEdgeCases(unittest.TestCase):
    """Test tool_result edge cases: is_error, image blocks, multiple tool_use."""

    def setUp(self):
        self.config = Config()

    def test_tool_result_is_error(self):
        """tool_result with is_error=true should prepend '[ERROR] ' to content."""
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_err1",
                    "content": "File not found",
                    "is_error": True,
                }],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "toolu_err1")
        self.assertTrue(msg["content"].startswith("[ERROR] "),
                        f"Expected content to start with '[ERROR] ', got: {msg['content']}")
        self.assertIn("File not found", msg["content"])

    def test_tool_result_is_error_false(self):
        """tool_result with is_error=false should NOT prepend '[ERROR] '."""
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_ok1",
                    "content": "Success result",
                    "is_error": False,
                }],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertFalse(msg["content"].startswith("[ERROR] "),
                         f"Content should NOT start with '[ERROR] ', got: {msg['content']}")
        self.assertIn("Success result", msg["content"])

    def test_tool_result_content_blocks_with_image(self):
        """tool_result with content list containing text + image blocks.

        Image blocks should produce '[image: image/png]' placeholder text.
        """
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_img1",
                    "content": [
                        {"type": "text", "text": "Screenshot captured"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                    ],
                }],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertIn("Screenshot captured", msg["content"])
        self.assertIn("[image: image/png]", msg["content"])

    def test_multiple_tool_use_blocks(self):
        """Assistant message with 2+ tool_use blocks produces a single
        OpenAI message with multiple tool_calls."""
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll check both."},
                    {
                        "type": "tool_use",
                        "id": "toolu_a",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_b",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            }],
        }
        result = translate_request(body, self.config)
        msg = result["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "I'll check both.")
        self.assertEqual(len(msg["tool_calls"]), 2)
        self.assertEqual(msg["tool_calls"][0]["id"], "toolu_a")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(msg["tool_calls"][0]["function"]["arguments"]),
            {"city": "London"},
        )
        self.assertEqual(msg["tool_calls"][1]["id"], "toolu_b")
        self.assertEqual(msg["tool_calls"][1]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(msg["tool_calls"][1]["function"]["arguments"]),
            {"city": "Paris"},
        )


class TestStreamingFlag(unittest.TestCase):
    """Test streaming configuration."""

    def setUp(self):
        self.config = Config()

    def test_stream_true(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertTrue(result["stream"])
        self.assertEqual(result["stream_options"], {"include_usage": True})

    def test_stream_false(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertNotIn("stream", result)


class TestOptionalParams(unittest.TestCase):
    """Test optional parameter passthrough."""

    def setUp(self):
        self.config = Config()

    def test_temperature_and_top_p(self):
        body = {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = translate_request(body, self.config)
        self.assertEqual(result["temperature"], 0.7)
        self.assertEqual(result["top_p"], 0.9)


if __name__ == "__main__":
    unittest.main()
