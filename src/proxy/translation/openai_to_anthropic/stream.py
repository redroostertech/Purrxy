"""
Translate an Anthropic Messages SSE stream into OpenAI chat.completion.chunk
SSE lines (the hub streaming format).

This is the inverse of translation/stream.py.

Anthropic events consumed (dispatched on the JSON "type" field):
    message_start, content_block_start, content_block_delta (text_delta,
    input_json_delta), content_block_stop, message_delta, message_stop,
    ping, error

OpenAI output sequence:
    role chunk -> content / tool_calls delta chunks -> finish_reason chunk
    -> usage chunk (only when include_usage) -> data: [DONE]
"""

import json
import logging
import time
from typing import Any, Dict, Iterable, Iterator, Optional

from .response import map_finish_reason, map_usage

logger = logging.getLogger("proxy.translate_openai_to_anthropic.stream")

DONE = "data: [DONE]\n\n"


class AnthropicStreamError(Exception):
    """Raised when the Anthropic stream carries an error event."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def anthropic_stream_to_openai(
    lines: Iterable[str],
    model: str = "",
    include_usage: bool = False,
    created: Optional[int] = None,
) -> Iterator[str]:
    """Translate Anthropic SSE lines into OpenAI SSE chunk strings.

    Args:
        lines: Raw SSE lines from the Anthropic upstream.
        model: Fallback model name (message_start's model wins when present).
        include_usage: Emit a final usage chunk (stream_options.include_usage).
        created: Fixed creation timestamp (defaults to now).

    Yields:
        "data: {...}\\n\\n" strings, ending with "data: [DONE]\\n\\n".

    Raises:
        AnthropicStreamError: when the upstream sends an error event.
    """
    state = _State(model, created if created is not None else int(time.time()))
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse Anthropic SSE data line")
            continue
        if not isinstance(event, dict):
            continue
        yield from state.handle(event)
        if state.stopped:
            break
    yield from state.finish(include_usage)


class _State:
    def __init__(self, model: str, created: int):
        self.id = ""
        self.model = model
        self.created = created
        self.started = False
        self.stopped = False
        self.finish_reason: Optional[str] = None
        self.finish_emitted = False
        self.usage: Dict[str, Any] = {}
        self.blocks: Dict[int, Optional[int]] = {}  # anthropic index -> tool index (None for text)
        self.next_tool_index = 0

    def _chunk(self, delta: dict, finish_reason: Optional[str] = None) -> str:
        return _sse(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                        "logprobs": None,
                    }
                ],
            }
        )

    def _start(self) -> Iterator[str]:
        if not self.started:
            self.started = True
            yield self._chunk({"role": "assistant", "content": ""})

    def handle(self, event: dict) -> Iterator[str]:
        etype = event.get("type")
        if etype == "message_start":
            message = event.get("message") or {}
            self.id = message.get("id", self.id)
            self.model = message.get("model") or self.model
            self.usage.update(message.get("usage") or {})
            yield from self._start()
        elif etype == "content_block_start":
            yield from self._start()
            yield from self._block_start(event)
        elif etype == "content_block_delta":
            yield from self._start()
            yield from self._block_delta(event)
        elif etype == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                self.finish_reason = map_finish_reason(delta["stop_reason"])
            # message_delta usage is cumulative; later values win.
            self.usage.update(
                {k: v for k, v in (event.get("usage") or {}).items() if v is not None}
            )
        elif etype == "message_stop":
            self.stopped = True
        elif etype == "error":
            err = event.get("error") or {}
            raise AnthropicStreamError(
                err.get("type", "api_error"), err.get("message", "Upstream stream error")
            )
        # ping and content_block_stop need no output.

    def _block_start(self, event: dict) -> Iterator[str]:
        index = event.get("index", 0)
        block = event.get("content_block") or {}
        btype = block.get("type")
        if btype == "tool_use":
            tool_index = self.next_tool_index
            self.next_tool_index += 1
            self.blocks[index] = tool_index
            yield self._chunk(
                {
                    "tool_calls": [
                        {
                            "index": tool_index,
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {"name": block.get("name", ""), "arguments": ""},
                        }
                    ]
                }
            )
        elif btype == "text":
            self.blocks[index] = None
            if block.get("text"):
                yield self._chunk({"content": block["text"]})
        # thinking and other block types are not represented in the hub.

    def _block_delta(self, event: dict) -> Iterator[str]:
        index = event.get("index", 0)
        delta = event.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta" and delta.get("text"):
            yield self._chunk({"content": delta["text"]})
        elif dtype == "input_json_delta" and self.blocks.get(index) is not None:
            fragment = delta.get("partial_json", "")
            if fragment:
                yield self._chunk(
                    {
                        "tool_calls": [
                            {
                                "index": self.blocks[index],
                                "function": {"arguments": fragment},
                            }
                        ]
                    }
                )

    def finish(self, include_usage: bool) -> Iterator[str]:
        yield from self._start()
        if not self.finish_emitted:
            self.finish_emitted = True
            yield self._chunk({}, self.finish_reason or "stop")
        if include_usage:
            yield _sse(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": map_usage(self.usage),
                }
            )
        yield DONE


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
