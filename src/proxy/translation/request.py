"""
Translate Anthropic Messages API requests to OpenAI Chat Completions format.

Handles:
- System prompt extraction (top-level string or block array)
- Message role and content block translation
- Tool use / tool result mapping
- Tool definition schema conversion
- Streaming flag passthrough
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..config import Config

logger = logging.getLogger("proxy.translate_request")


def translate_request(body: dict, config: Config) -> dict:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Args:
        body: Parsed Anthropic request body.
        config: Proxy configuration (for model mapping).

    Returns:
        Dict ready to be JSON-serialized and sent to the OpenAI-compatible backend.
    """
    _warn_unsupported_features(body)

    openai_req: Dict[str, Any] = {}

    # Model
    openai_req["model"] = config.resolve_model(body.get("model", ""))

    # Max tokens
    if "max_tokens" in body:
        openai_req["max_tokens"] = body["max_tokens"]

    # Temperature, top_p
    for param in ("temperature", "top_p", "top_k"):
        if param in body:
            openai_req[param] = body[param]

    # Messages: start with system prompt, then translate message array
    messages = []
    system_msg = _extract_system(body.get("system"))
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    for msg in body.get("messages", []):
        messages.extend(_translate_message(msg))

    openai_req["messages"] = messages

    # Tools
    if "tools" in body:
        openai_req["tools"] = [_translate_tool(t) for t in body["tools"]]

    # Streaming
    if body.get("stream"):
        openai_req["stream"] = True
        openai_req["stream_options"] = {"include_usage": True}

    logger.debug("Translated request: model=%s, messages=%d, stream=%s",
                 openai_req["model"], len(messages), openai_req.get("stream", False))

    return openai_req


def _warn_unsupported_features(body: dict):
    """Log warnings for Anthropic features that are not supported through the proxy.

    These features are silently ignored during translation — this function
    ensures the user is at least notified via logs.
    """
    if "thinking" in body:
        logger.warning("Unsupported feature 'thinking' in request — will be ignored")
    if "budget_tokens" in body:
        logger.warning("Unsupported feature 'budget_tokens' in request — will be ignored")

    # Check content blocks in messages for unsupported block types
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "document":
                    logger.warning(
                        "Unsupported content block type 'document' — will be ignored"
                    )
                elif btype == "model_result":
                    logger.warning(
                        "Unsupported content block type 'model_result' — will be ignored"
                    )


def _extract_system(system: Any) -> Optional[str]:
    """Extract system prompt text from Anthropic's system field.

    The system field can be:
    - A plain string
    - A list of content blocks (each with type "text")
    - None / absent

    Returns:
        Combined system prompt string, or None if absent.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts) if parts else None
    return None


def _translate_message(msg: dict) -> List[dict]:
    """Translate a single Anthropic message to one or more OpenAI messages.

    A single Anthropic message may produce multiple OpenAI messages because:
    - tool_result blocks become separate role="tool" messages
    - An assistant message with text + tool_use maps to one message with both fields

    Args:
        msg: A single Anthropic message dict.

    Returns:
        List of OpenAI message dicts.
    """
    role = msg.get("role", "user")
    content = msg.get("content")

    # Simple string content
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    # Content is a list of blocks
    if not isinstance(content, list):
        return [{"role": role, "content": str(content) if content else ""}]

    if role == "assistant":
        return _translate_assistant_blocks(content)
    elif role == "user":
        return _translate_user_blocks(content)
    else:
        # Fallback: concatenate text blocks
        text = " ".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        )
        return [{"role": role, "content": text}]


def _translate_assistant_blocks(blocks: List[dict]) -> List[dict]:
    """Translate assistant content blocks to a single OpenAI assistant message.

    Combines text blocks into content and tool_use blocks into tool_calls.
    """
    text_parts = []
    tool_calls = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    message: Dict[str, Any] = {"role": "assistant"}

    if text_parts:
        message["content"] = "\n".join(text_parts)
    else:
        message["content"] = None

    if tool_calls:
        message["tool_calls"] = tool_calls

    return [message]


def _translate_user_blocks(blocks: List[dict]) -> List[dict]:
    """Translate user content blocks, splitting tool_result into separate messages.

    Each tool_result block becomes a role="tool" message.
    Text and image blocks are grouped into a single role="user" message.
    """
    messages = []
    user_parts: List[Any] = []

    for block in blocks:
        btype = block.get("type")

        if btype == "tool_result":
            # Flush any accumulated user content first
            if user_parts:
                messages.append(_build_user_message(user_parts))
                user_parts = []
            messages.append(_translate_tool_result(block))

        elif btype == "text":
            user_parts.append(block.get("text", ""))

        elif btype == "image":
            # Pass through image blocks in OpenAI vision format
            source = block.get("source", {})
            if source.get("type") == "base64":
                user_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}",
                    },
                })

    # Flush remaining user content
    if user_parts:
        messages.append(_build_user_message(user_parts))

    return messages


def _build_user_message(parts: List[Any]) -> dict:
    """Build a user message from accumulated content parts.

    If all parts are strings, join them. If mixed with image parts, use
    the OpenAI content array format.
    """
    if all(isinstance(p, str) for p in parts):
        return {"role": "user", "content": "\n".join(parts)}

    # Mixed content: use array format
    content = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            content.append(part)
    return {"role": "user", "content": content}


def _translate_tool_result(block: dict) -> dict:
    """Convert an Anthropic tool_result block to an OpenAI tool message.

    When is_error is true, prepends "[ERROR] " to the content string so the
    LLM knows the tool invocation failed (OpenAI has no direct equivalent of
    the is_error flag).
    """
    content = block.get("content", "")

    # Content can be a string or a list of content blocks
    if isinstance(content, list):
        parts = []
        for b in content:
            btype = b.get("type")
            if btype == "text":
                parts.append(b.get("text", ""))
            elif btype == "image":
                # OpenAI tool messages only accept string content, so describe
                # the image rather than passing binary data.
                media_type = b.get("source", {}).get("media_type", "image/png")
                parts.append(f"[image: {media_type}]")
        content = "\n".join(parts)

    content = str(content)

    # Prepend error marker when the tool reported failure
    if block.get("is_error"):
        content = f"[ERROR] {content}"

    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": content,
    }


def _translate_tool(tool: dict) -> dict:
    """Convert an Anthropic tool definition to OpenAI function tool format.

    Anthropic uses "input_schema" while OpenAI uses "parameters" nested
    inside a "function" wrapper.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }
