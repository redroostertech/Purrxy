"""
Error handling and Anthropic-format error response generation.

Errors on /v1/messages follow the Anthropic error envelope:
{
    "type": "error",
    "error": {
        "type": "<error_type>",
        "message": "<human-readable message>"
    }
}

Errors on /v1/chat/completions and /v1/models use the OpenAI error object
(see make_openai_error_body).
"""

import logging

logger = logging.getLogger("proxy.errors")

# Anthropic error types mapped to HTTP status codes
ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def make_error_body(status: int, message: str) -> dict:
    """Build an Anthropic-format error response body.

    Args:
        status: HTTP status code.
        message: Human-readable error description.

    Returns:
        Dict matching the Anthropic error envelope schema.
    """
    error_type = ERROR_TYPES.get(status, "api_error")
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


# OpenAI error types for the /v1/chat/completions inbound dialect
OPENAI_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "invalid_request_error",
    429: "rate_limit_error",
}


def make_openai_error_body(status: int, message: str, code=None) -> dict:
    """Build an OpenAI-format error response body.

    Args:
        status: HTTP status code.
        message: Human-readable error description.
        code: Optional machine-readable code.

    Returns:
        Dict matching the OpenAI error object schema.
    """
    if status in OPENAI_ERROR_TYPES:
        error_type = OPENAI_ERROR_TYPES[status]
    elif status >= 500:
        error_type = "upstream_error" if status in (502, 503, 504) else "server_error"
    else:
        error_type = "invalid_request_error"
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        },
    }
