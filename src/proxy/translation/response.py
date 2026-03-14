"""
Translate OpenAI Chat Completions responses to Anthropic Messages API format.

Handles non-streaming responses only. For streaming, see translate_stream.py.

Maps:
- OpenAI choices/message → Anthropic content blocks
- OpenAI tool_calls → Anthropic tool_use blocks
- OpenAI finish_reason → Anthropic stop_reason
- OpenAI usage → Anthropic usage
"""

import json
import logging
import uuid
from typing import Any, Dict, List

logger = logging.getLogger("proxy.translate_response")

# OpenAI finish_reason → Anthropic stop_reason
# Note: "content_filter" has no direct Anthropic equivalent; we map it to
# "end_turn" because the generation was terminated by the backend, similar
# to a natural stop.
STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def translate_response(openai_body: dict, original_model: str) -> dict:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format.

    Args:
        openai_body: Parsed OpenAI response body.
        original_model: The model name from the original Anthropic request,
                        echoed back in the response.

    Returns:
        Dict matching the Anthropic Messages API response schema.
    """
    choice = _get_first_choice(openai_body)
    message = choice.get("message", {})

    content = _build_content_blocks(message)
    stop_reason = _map_stop_reason(choice.get("finish_reason", "stop"))
    usage = _map_usage(openai_body.get("usage", {}))

    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }

    logger.debug("Translated response: blocks=%d, stop_reason=%s",
                 len(content), stop_reason)

    return response


def _get_first_choice(body: dict) -> dict:
    """Extract the first choice from an OpenAI response."""
    choices = body.get("choices", [])
    if not choices:
        logger.warning("OpenAI response has no choices, returning empty")
        return {"message": {}, "finish_reason": "stop"}
    choice = choices[0]
    if "message" not in choice:
        logger.warning("OpenAI response choice missing 'message' field, returning empty")
        choice["message"] = {}
    return choice


def _build_content_blocks(message: dict) -> List[dict]:
    """Build Anthropic content blocks from an OpenAI message.

    Text content becomes a text block. Tool calls become tool_use blocks.
    Text block is placed first if both are present.
    """
    blocks: List[dict] = []

    # Text content
    text = message.get("content")
    if text:
        blocks.append({
            "type": "text",
            "text": text,
        })

    # Tool calls
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}
            logger.warning("Failed to parse tool call arguments for %s",
                           func.get("name", "unknown"))

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": input_data,
        })

    # Ensure at least one content block (Anthropic requires non-empty content)
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    return blocks


def _map_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    return STOP_REASON_MAP.get(finish_reason, "end_turn")


def _map_usage(usage: dict) -> dict:
    """Map OpenAI usage fields to Anthropic usage fields."""
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
