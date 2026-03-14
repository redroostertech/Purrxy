"""
lana-proxy: Anthropic Messages API translation proxy.

Subpackages:
    translation     Anthropic <-> OpenAI format conversion.
    middleware      Optional features (cache, queue, recording, budget, analytics).
    monitoring      Observability (stats, dashboard, model discovery).

Core modules:
    config          Proxy configuration management.
    server          Threaded HTTP server and request routing.
    errors          Anthropic error envelope formatting.
"""
