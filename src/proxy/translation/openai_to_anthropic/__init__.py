"""
Hub (OpenAI Chat Completions) <-> Anthropic Messages upstream translation.

The inverse of the flat modules in translation/: those adapt an Anthropic
client to an OpenAI backend; these adapt an OpenAI client (or the hub) to an
Anthropic provider.

Modules:
    request     OpenAI request -> Anthropic request.
    response    Anthropic response -> OpenAI response (non-streaming).
    stream      Anthropic SSE -> OpenAI SSE (streaming state machine).
"""

from .request import DEFAULT_MAX_TOKENS, openai_request_to_anthropic
from .response import (
    anthropic_models_to_openai,
    anthropic_response_to_openai,
    map_finish_reason,
    map_usage,
)
from .stream import AnthropicStreamError, anthropic_stream_to_openai

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "AnthropicStreamError",
    "anthropic_models_to_openai",
    "anthropic_response_to_openai",
    "anthropic_stream_to_openai",
    "map_finish_reason",
    "map_usage",
    "openai_request_to_anthropic",
]
