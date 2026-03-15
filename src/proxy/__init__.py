"""
purrxy: Anthropic Messages API translation proxy and provider gateway.

Subpackages:
    translation     Anthropic <-> OpenAI format conversion.
    middleware      Optional features (cache, queue, recording, budget, analytics).
    monitoring      Observability (stats, dashboard, model discovery).

Core modules:
    config          Proxy configuration management.
    server          Threaded HTTP server and request routing.
    errors          Anthropic and OpenAI error formatting.
    providers       Provider presets and per-request upstream selection.
    gateway         Upstream client for the provider gateway.
"""
