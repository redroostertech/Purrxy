"""
Translate OpenAI Chat Completions requests (the hub format) to Anthropic
Messages API requests.

This is the inverse of translation/request.py.

Handles:
- system / developer messages hoisted to the top-level "system" field
- user content parts (text, image_url data URIs and http(s) URLs)
- assistant tool_calls -> tool_use blocks
- role "tool" messages -> tool_result blocks inside a user turn
- merging consecutive same-role turns (tool results first, then text)
- tools, tool_choice, parallel_tool_calls
- max_tokens (required by Anthropic; defaulted when absent), stop, sampling
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("proxy.translate_openai_to_anthropic.request")

DEFAULT_MAX_TOKENS = 4096

_DATA_URI = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def openai_request_to_anthropic(body: dict, default_max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Convert an OpenAI Chat Completions request into an Anthropic Messages request.

    Args:
        body: Parsed OpenAI request body.
        default_max_tokens: Used when the request sets neither max_tokens nor
            max_completion_tokens (Anthropic requires max_tokens).

    Returns:
        Dict ready to POST to an Anthropic /v1/messages endpoint.
    """
    _warn_unsupported(body)
    req: Dict[str, Any] = {"model": body.get("model", "")}

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    req["max_tokens"] = int(max_tokens) if max_tokens else default_max_tokens

    system_parts: List[str] = []
    turns: List[dict] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role in ("system", "developer"):
            text = _text_of(msg.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            _append_turn(turns, "user", [_tool_result_block(msg)])
        elif role == "assistant":
            blocks = _assistant_blocks(msg)
            if blocks:
                _append_turn(turns, "assistant", blocks)
        elif role == "user":
            blocks = _user_blocks(msg.get("content"))
            if blocks:
                _append_turn(turns, "user", blocks)
        else:
            logger.warning("Unsupported message role '%s' ignored", role)

    if system_parts:
        req["system"] = "\n\n".join(system_parts)
    req["messages"] = turns

    if "temperature" in body and body["temperature"] is not None:
        # OpenAI accepts 0..2, Anthropic 0..1.
        req["temperature"] = max(0.0, min(1.0, float(body["temperature"])))
    for param in ("top_p", "top_k"):
        if body.get(param) is not None:
            req[param] = body[param]

    stop = body.get("stop")
    if isinstance(stop, str):
        req["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        req["stop_sequences"] = [s for s in stop if isinstance(s, str)]

    if body.get("user"):
        req["metadata"] = {"user_id": str(body["user"])}

    tools = [_translate_tool(t) for t in body.get("tools") or [] if _is_function_tool(t)]
    if tools:
        req["tools"] = tools
    tool_choice = _translate_tool_choice(body.get("tool_choice"))
    if body.get("parallel_tool_calls") is False and tools:
        tool_choice = tool_choice or {"type": "auto"}
        if tool_choice.get("type") != "none":
            tool_choice["disable_parallel_tool_use"] = True
    if tool_choice and tools:
        req["tool_choice"] = tool_choice

    if body.get("stream"):
        req["stream"] = True

    return req


def _warn_unsupported(body: dict):
    for key in ("response_format", "logprobs", "seed", "audio", "modalities"):
        if body.get(key) is not None:
            logger.warning("Unsupported field '%s' for Anthropic upstream; ignored", key)
    if (body.get("n") or 1) > 1:
        logger.warning("n > 1 is not supported by Anthropic upstream; returning one choice")


def _append_turn(turns: List[dict], role: str, blocks: List[dict]):
    """Append blocks, merging into the previous turn when the role repeats."""
    if turns and turns[-1]["role"] == role:
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": list(blocks)})


def _text_of(content: Any) -> str:
    """Flatten string or text-part-array content to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "input_text")
        )
    return str(content)


def _user_blocks(content: Any) -> List[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}] if content else []
    blocks: List[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif ptype == "image_url":
            block = _image_block(part.get("image_url"))
            if block:
                blocks.append(block)
        else:
            logger.warning("Unsupported user content part '%s' ignored", ptype)
    return blocks


def _image_block(image_url: Any) -> Optional[dict]:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        return None
    match = _DATA_URI.match(url)
    if match:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)},
        }
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    logger.warning("Unsupported image_url (not base64 data URI or http URL) ignored")
    return None


def _assistant_blocks(msg: dict) -> List[dict]:
    blocks: List[dict] = []
    text = _text_of(msg.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function") or {}
        args = func.get("arguments")
        if isinstance(args, dict):
            tool_input = args
        else:
            try:
                tool_input = json.loads(args) if args else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Tool call arguments for '%s' are not valid JSON", func.get("name", "unknown")
                )
                tool_input = {}
            if not isinstance(tool_input, dict):
                tool_input = {"value": tool_input}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": tool_input,
            }
        )
    return blocks


def _tool_result_block(msg: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": msg.get("tool_call_id", ""),
        "content": _text_of(msg.get("content")),
    }


def _is_function_tool(tool: Any) -> bool:
    return isinstance(tool, dict) and tool.get("type", "function") == "function"


def _translate_tool(tool: dict) -> dict:
    func = tool.get("function") or {}
    out = {
        "name": func.get("name", ""),
        "input_schema": func.get("parameters") or dict(_EMPTY_SCHEMA),
    }
    if func.get("description"):
        out["description"] = func["description"]
    return out


def _translate_tool_choice(choice: Any) -> Optional[dict]:
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    logger.warning("Unsupported tool_choice %r ignored", choice)
    return None
