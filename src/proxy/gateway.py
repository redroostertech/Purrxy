"""
Upstream client for the provider gateway.

Takes a hub (OpenAI Chat Completions) request plus a resolved Upstream and
returns a hub response, whatever dialect the upstream speaks:

    openai     request and response pass through unchanged (extra provider
               fields such as Perplexity's citations survive)
    anthropic  translated via translation/openai_to_anthropic

Security:
    - outgoing headers are built from scratch (providers.upstream_headers);
      nothing from the client request is copied
    - redirects are refused, so a key can never follow a 3xx to another host
    - every error message is passed through providers.redact() with the key
    - nothing here logs headers, URLs with credentials, or bodies
"""

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Iterator, Optional

from .providers import DIALECT_ANTHROPIC, Upstream, endpoint_url, redact, upstream_headers
from .translation.openai_to_anthropic import (
    AnthropicStreamError,
    anthropic_models_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai,
    openai_request_to_anthropic,
)
from .translation.stream import read_sse_lines

logger = logging.getLogger("proxy.gateway")

DONE = "data: [DONE]\n\n"


class UpstreamError(Exception):
    """An upstream call failed. status is the HTTP status to return to the client."""

    def __init__(self, status: int, message: str, upstream_status: Optional[int] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.upstream_status = upstream_status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def map_upstream_status(status: int) -> int:
    """Map an upstream HTTP error status to the status returned to the client."""
    if status in (400, 401, 403, 404, 409, 413, 429):
        return status
    if status == 422:
        return 400
    if status in (408, 504):
        return 504
    if 300 <= status < 400:
        return 502
    if status in (503, 529):
        return 503
    if 400 <= status < 500:
        return 400
    return 502


def _extract_message(raw: str) -> str:
    """Pull a human-readable message out of an upstream error body."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()[:500]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:500]
        if isinstance(err, str):
            return err[:500]
        if data.get("message"):
            return str(data["message"])[:500]
    return raw.strip()[:500]


def _open(
    upstream: Upstream,
    method: str,
    url: str,
    payload: Optional[dict],
    timeout: float,
    stream: bool = False,
):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers=upstream_headers(upstream, stream=stream),
        method=method,
    )
    try:
        return _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        if 300 <= e.code < 400:
            detail = "redirects are not followed"
        else:
            detail = _extract_message(raw) or e.reason or "no detail"
        message = redact(
            f"Upstream provider '{upstream.name}' returned HTTP {e.code}: {detail}",
            upstream.key,
        )
        logger.warning("Upstream %s (%s) HTTP %d", upstream.name, upstream.host, e.code)
        raise UpstreamError(map_upstream_status(e.code), message, e.code) from None
    except (socket.timeout, TimeoutError):
        logger.warning("Upstream %s (%s) timed out", upstream.name, upstream.host)
        raise UpstreamError(504, f"Upstream provider '{upstream.name}' timed out") from None
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise UpstreamError(504, f"Upstream provider '{upstream.name}' timed out") from None
        logger.warning("Upstream %s (%s) unreachable", upstream.name, upstream.host)
        raise UpstreamError(
            502,
            redact(f"Upstream provider '{upstream.name}' is unreachable: {reason}", upstream.key),
        ) from None


def _read_json(upstream: Upstream, resp) -> dict:
    with resp:
        raw = resp.read()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise UpstreamError(
            502, f"Upstream provider '{upstream.name}' returned a non-JSON response"
        ) from None
    if not isinstance(data, dict):
        raise UpstreamError(
            502, f"Upstream provider '{upstream.name}' returned a non-object response"
        )
    return data


def chat_completion(upstream: Upstream, body: dict, timeout: float) -> dict:
    """Send a non-streaming hub request and return a hub (chat.completion) response."""
    if upstream.provider.dialect == DIALECT_ANTHROPIC:
        payload = openai_request_to_anthropic(dict(body, stream=False))
        payload.pop("stream", None)
        resp = _open(
            upstream, "POST", endpoint_url(upstream.base_url, "/messages"), payload, timeout
        )
        data = _read_json(upstream, resp)
        if data.get("type") == "error":
            raise UpstreamError(
                502,
                redact(
                    f"Upstream provider '{upstream.name}' error: {_extract_message(json.dumps(data))}",
                    upstream.key,
                ),
            )
        return anthropic_response_to_openai(data)

    resp = _open(
        upstream, "POST", endpoint_url(upstream.base_url, "/chat/completions"), body, timeout
    )
    data = _read_json(upstream, resp)
    if "choices" not in data:
        if "error" in data:
            raise UpstreamError(
                502,
                redact(
                    f"Upstream provider '{upstream.name}' error: {_extract_message(json.dumps(data))}",
                    upstream.key,
                ),
            )
        raise UpstreamError(
            502, f"Upstream provider '{upstream.name}' response is missing 'choices'"
        )
    return data


def open_chat_stream(upstream: Upstream, body: dict, timeout: float) -> Iterator[str]:
    """Open a streaming hub request.

    The connection is opened eagerly, so HTTP and connection errors raise
    UpstreamError here, before the caller has sent any response headers.

    Returns:
        A generator of OpenAI SSE strings ("data: ...\\n\\n"), always ending
        with "data: [DONE]\\n\\n". Mid-stream upstream errors raise
        UpstreamError from the generator. Close the generator to release the
        connection early.
    """
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
    if upstream.provider.dialect == DIALECT_ANTHROPIC:
        payload = openai_request_to_anthropic(dict(body, stream=True))
        resp = _open(
            upstream,
            "POST",
            endpoint_url(upstream.base_url, "/messages"),
            payload,
            timeout,
            stream=True,
        )
        events = anthropic_stream_to_openai(
            read_sse_lines(resp),
            model=body.get("model", ""),
            include_usage=include_usage,
        )
    else:
        resp = _open(
            upstream,
            "POST",
            endpoint_url(upstream.base_url, "/chat/completions"),
            body,
            timeout,
            stream=True,
        )
        events = _passthrough_sse(read_sse_lines(resp))
    return _guarded(upstream, resp, events)


def _passthrough_sse(lines: Iterator[str]) -> Iterator[str]:
    """Relay OpenAI SSE data lines verbatim; guarantee a trailing [DONE]."""
    for line in lines:
        line = line.strip()
        if line.startswith(":"):
            yield line + "\n\n"  # keepalive comment (e.g. OpenRouter)
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        yield f"data: {data}\n\n"
    yield DONE


def _guarded(upstream: Upstream, resp, events: Iterator[str]) -> Iterator[str]:
    try:
        yield from events
    except AnthropicStreamError as e:
        raise UpstreamError(
            502,
            redact(f"Upstream provider '{upstream.name}' stream error: {e.message}", upstream.key),
        ) from None
    except (socket.timeout, TimeoutError):
        raise UpstreamError(
            504, f"Upstream provider '{upstream.name}' timed out mid-stream"
        ) from None
    finally:
        resp.close()


def list_models(upstream: Upstream, timeout: float) -> dict:
    """Return the upstream's model list in OpenAI shape."""
    if not upstream.provider.models:
        raise UpstreamError(404, f"Provider '{upstream.name}' does not expose a model listing")
    resp = _open(upstream, "GET", endpoint_url(upstream.base_url, "/models"), None, timeout)
    data = _read_json(upstream, resp)
    if upstream.provider.dialect == DIALECT_ANTHROPIC:
        return anthropic_models_to_openai(data)
    return data
