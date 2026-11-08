"""
Translate Anthropic Messages API responses to OpenAI Chat Completions
responses (the hub format). Non-streaming only; see stream.py for SSE.

This is the inverse of translation/response.py.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Anthropic stop_reason -> OpenAI finish_reason
FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def map_finish_reason(stop_reason: Optional[str]) -> str:
    """Map an Anthropic stop_reason to an OpenAI finish_reason."""
    return FINISH_REASON_MAP.get(stop_reason or "", "stop")


def map_usage(usage: Optional[dict]) -> dict:
    """Map Anthropic usage to OpenAI usage.

    OpenAI's prompt_tokens includes cached tokens, so Anthropic's cache
    creation and cache read counts are added to input_tokens.
    """
    usage = usage or {}
    cache_read = usage.get("cache_read_input_tokens") or 0
    prompt = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + cache_read
    )
    completion = usage.get("output_tokens") or 0
    out: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cache_read:
        out["prompt_tokens_details"] = {"cached_tokens": cache_read}
    return out


def anthropic_response_to_openai(msg: dict, created: Optional[int] = None) -> dict:
    """Convert an Anthropic Messages response into a chat.completion object."""
    text_parts: List[str] = []
    tool_calls: List[dict] = []
    for block in msg.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        # thinking, redacted_thinking and server tool blocks have no
        # Chat Completions equivalent and are dropped.

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": msg.get("id", ""),
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": msg.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": map_finish_reason(msg.get("stop_reason")),
                "logprobs": None,
            }
        ],
        "usage": map_usage(msg.get("usage")),
    }


def _epoch(created_at: Any) -> int:
    if not isinstance(created_at, str):
        return 0
    try:
        return int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def anthropic_models_to_openai(body: dict) -> dict:
    """Convert an Anthropic GET /v1/models listing into the OpenAI list shape."""
    return {
        "object": "list",
        "data": [
            {
                "id": m.get("id", ""),
                "object": "model",
                "created": _epoch(m.get("created_at")),
                "owned_by": "anthropic",
            }
            for m in body.get("data") or []
        ],
    }
