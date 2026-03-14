"""
Translate OpenAI streaming SSE chunks to Anthropic streaming SSE events.

Implements a state machine that reads OpenAI chat.completion.chunk events
line-by-line and yields properly sequenced Anthropic SSE events:

    message_start → (content_block_start → content_block_delta* → content_block_stop)*
    → message_delta → message_stop

Handles:
- Text content streaming (text_delta events)
- Tool call streaming with incremental argument assembly (input_json_delta events)
- Mixed text + tool_use responses
- Usage token reporting
"""

import json
import logging
import uuid
from typing import Generator

logger = logging.getLogger("proxy.translate_stream")


def translate_stream(
    lines: Generator[str, None, None],
    original_model: str,
    request_id: str,
) -> Generator[str, None, None]:
    """Translate a stream of OpenAI SSE lines into Anthropic SSE events.

    Args:
        lines: Generator yielding raw SSE lines from the OpenAI backend.
        original_model: Model name from the original Anthropic request.
        request_id: Unique ID for this message (used in message_start).

    Yields:
        Formatted SSE event strings (each ending with double newline).
    """
    state = _StreamState(original_model, request_id)

    for line in lines:
        line = line.strip()

        # Skip empty lines and non-data lines
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()

        # End of stream
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Failed to parse SSE chunk: %s", data[:200])
            continue

        for event in state.process_chunk(chunk):
            yield event

    # Finalize: close any open blocks, emit message_delta and message_stop
    for event in state.finalize():
        yield event


class _StreamState:
    """State machine for OpenAI-to-Anthropic stream translation.

    Tracks which content blocks are open and manages the event sequence.
    """

    def __init__(self, model: str, request_id: str):
        self.model = model
        self.request_id = request_id
        self.message_started = False
        self.block_index = 0
        self.text_block_open = False
        self.tool_blocks_started: dict = {}  # openai_index → our_block_index
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason = "end_turn"

    def process_chunk(self, chunk: dict) -> Generator[str, None, None]:
        """Process a single OpenAI SSE chunk and yield Anthropic events."""
        # Emit message_start on first chunk
        if not self.message_started:
            yield from self._emit_message_start(chunk)
            self.message_started = True

        # Extract usage if present (usually in final chunk)
        usage = chunk.get("usage")
        if usage:
            self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
            self.output_tokens = usage.get("completion_tokens", self.output_tokens)

        choices = chunk.get("choices", [])
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")

        # Text content delta
        text = delta.get("content")
        if text is not None:
            if not self.text_block_open:
                yield _format_event("content_block_start", {
                    "type": "content_block_start",
                    "index": self.block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                self.text_block_open = True

            yield _format_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        # Tool call deltas
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            yield from self._process_tool_delta(tc_index, tc)

        # Track finish reason for message_delta
        if finish:
            self.finish_reason = _map_stop_reason(finish)

    def finalize(self) -> Generator[str, None, None]:
        """Close all open blocks and emit final events."""
        if not self.message_started:
            # Never got any chunks — emit minimal valid sequence
            yield from self._emit_message_start({})
            self.message_started = True

        # Close text block if open
        if self.text_block_open:
            yield _format_event("content_block_stop", {
                "type": "content_block_stop",
                "index": self.block_index,
            })
            self.block_index += 1
            self.text_block_open = False

        # Close any open tool blocks
        for tc_index in sorted(self.tool_blocks_started.keys()):
            block_idx = self.tool_blocks_started[tc_index]
            yield _format_event("content_block_stop", {
                "type": "content_block_stop",
                "index": block_idx,
            })

        # message_delta with stop reason and output usage
        yield _format_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self.finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        })

        # message_stop
        yield _format_event("message_stop", {"type": "message_stop"})

    def _emit_message_start(self, chunk: dict) -> Generator[str, None, None]:
        """Emit the message_start event."""
        yield _format_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.request_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            },
        })

    def _process_tool_delta(
        self, tc_index: int, tc: dict
    ) -> Generator[str, None, None]:
        """Process a single tool call delta from OpenAI format."""
        # Close text block before starting tool blocks
        if self.text_block_open:
            yield _format_event("content_block_stop", {
                "type": "content_block_stop",
                "index": self.block_index,
            })
            self.block_index += 1
            self.text_block_open = False

        # Start new tool block on first delta for this index
        if tc_index not in self.tool_blocks_started:
            func = tc.get("function", {})
            tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
            tool_name = func.get("name", "")

            self.tool_blocks_started[tc_index] = self.block_index

            yield _format_event("content_block_start", {
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {},
                },
            })
            self.block_index += 1

        # Emit argument fragments as input_json_delta
        func = tc.get("function", {})
        args_fragment = func.get("arguments", "")
        if args_fragment:
            yield _format_event("content_block_delta", {
                "type": "content_block_delta",
                "index": self.tool_blocks_started[tc_index],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": args_fragment,
                },
            })


def _format_event(event_type: str, data: dict) -> str:
    """Format an SSE event string.

    Returns:
        Formatted SSE event ending with double newline.
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _map_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    # Note: "content_filter" has no direct Anthropic equivalent; we map it to
    # "end_turn" because the generation was terminated by the backend, similar
    # to a natural stop.
    mapping = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


def read_sse_lines(response) -> Generator[str, None, None]:
    """Read SSE lines from an HTTP response object.

    Handles both http.client.HTTPResponse and similar file-like objects.
    Reads line-by-line for memory efficiency.

    Args:
        response: An object with a readline() method (e.g., HTTPResponse).

    Yields:
        Individual lines decoded as UTF-8.
    """
    while True:
        line = response.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        yield line
