"""
Error handling and Anthropic-format error response generation.

All errors returned to clients follow the Anthropic error envelope:
{
    "type": "error",
    "error": {
        "type": "<error_type>",
        "message": "<human-readable message>"
    }
}
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
