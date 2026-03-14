"""
API format translation between Anthropic Messages and OpenAI Chat Completions.

Modules:
    request     Anthropic request -> OpenAI request.
    response    OpenAI response -> Anthropic response (non-streaming).
    stream      OpenAI SSE -> Anthropic SSE (streaming state machine).

All translation functions are pure — no side effects, no shared state.
"""

from .request import translate_request
from .response import translate_response
from .stream import translate_stream, read_sse_lines

__all__ = [
    "translate_request",
    "translate_response",
    "translate_stream",
    "read_sse_lines",
]
