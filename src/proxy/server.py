"""
HTTP server and request handler for lana-proxy.

Routes incoming Anthropic Messages API requests to the appropriate translator,
forwards to the llama.cpp backend, and returns translated responses.

Provider gateway: when a request carries X-Lana-Upstream-* headers (see
providers.py) it is sent to the selected cloud provider instead of the
configured backend. Without those headers routing is unchanged.

Endpoints:
    POST /v1/messages          - Anthropic Messages API (streaming + non-streaming)
    POST /v1/chat/completions  - OpenAI Chat Completions API (streaming + non-streaming)
    GET  /v1/models            - Model listing (OpenAI shape)
    GET  /health         - Health check
    GET  /admin/config   - Return current config as JSON (localhost only)
    POST /admin/reload   - Hot-swap config fields (localhost only)
"""

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional, Tuple

from . import gateway
from .config import Config
from .errors import make_error_body, make_openai_error_body
from .gateway import UpstreamError
from .middleware.analytics import Analytics
from .middleware.budget import BudgetTracker
from .middleware.cache import PromptCache
from .middleware.queue import RequestQueue
from .middleware.recorder import Recorder, Replayer
from .monitoring.stats import Stats
from .providers import LOCAL_BACKEND, Upstream, UpstreamConfigError, resolve_upstream
from .translation.request import translate_request
from .translation.response import translate_response
from .translation.stream import read_sse_lines, translate_stream

logger = logging.getLogger("proxy.server")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: Config, stats: Optional[Stats] = None,
                 queue: Optional[RequestQueue] = None):
        self.config = config
        self.stats = stats or Stats()
        self.queue = queue

        # Recorder / Replayer
        self.recorder: Optional[Recorder] = None
        self.replayer: Optional[Replayer] = None
        if config.record_path:
            self.recorder = Recorder(config.record_path)
        if config.replay_path:
            self.replayer = Replayer(config.replay_path)

        # Prompt cache
        self.cache: Optional[PromptCache] = None
        if config.cache_enabled:
            self.cache = PromptCache(
                ttl=config.cache_ttl,
                max_size=config.cache_max_size,
                cache_dir=config.cache_dir,
            )

        # Budget tracker
        self.budget: Optional[BudgetTracker] = None
        if config.token_budget > 0:
            self.budget = BudgetTracker(max_tokens=config.token_budget)

        # Analytics
        self.analytics: Optional[Analytics] = None
        if config.analytics_path:
            self.analytics = Analytics(config.analytics_path)

        super().__init__((config.host, config.port), ProxyRequestHandler)

    def shutdown_server(self):
        """Clean shutdown: save cache to disk if configured."""
        if self.cache is not None:
            self.cache.save_to_disk()
        self.shutdown()


class ProxyRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests and routes them through the translation pipeline."""

    # Suppress default access log — we handle our own logging
    def log_message(self, format, *args):
        logger.debug("HTTP: %s", format % args)

    @property
    def config(self) -> Config:
        return self.server.config

    @property
    def stats(self) -> Stats:
        return self.server.stats

    @property
    def queue(self) -> Optional[RequestQueue]:
        return self.server.queue

    @property
    def recorder(self) -> Optional[Recorder]:
        return self.server.recorder

    @property
    def replayer(self) -> Optional[Replayer]:
        return self.server.replayer

    @property
    def cache(self) -> Optional[PromptCache]:
        return self.server.cache

    @property
    def budget(self) -> Optional[BudgetTracker]:
        return self.server.budget

    @property
    def analytics(self) -> Optional[Analytics]:
        return self.server.analytics

    # -- Routing -----------------------------------------------------------

    def _path_without_query(self) -> str:
        """Return the request path without query string."""
        path = self.path
        idx = path.find("?")
        return path[:idx] if idx != -1 else path

    def do_POST(self):
        path = self._path_without_query()
        if path == "/v1/messages":
            self._handle_messages()
        elif path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif path == "/admin/reload":
            self._handle_admin_reload()
        elif path == "/admin/budget/reset":
            self._handle_budget_reset()
        else:
            self._send_error(404, f"Unknown endpoint: {self.path}")

    def do_GET(self):
        path = self._path_without_query()
        if path == "/health":
            self._send_json(200, {"status": "ok"})
        elif path == "/v1/models":
            self._handle_models()
        elif path == "/admin/config":
            self._handle_admin_config()
        elif path == "/admin/budget":
            self._handle_budget_status()
        elif path == "/admin/analytics":
            self._handle_analytics_summary()
        else:
            self._send_error(404, f"Unknown endpoint: {self.path}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    # -- Admin Endpoints ---------------------------------------------------

    def _is_localhost(self) -> bool:
        """Check if the request originates from localhost."""
        client_ip = self.client_address[0]
        return client_ip in ("127.0.0.1", "::1")

    def _handle_admin_config(self):
        """GET /admin/config -- return current config as JSON."""
        if not self._is_localhost():
            self._send_error(403, "Admin endpoints are localhost-only")
            return
        self._send_json(200, self.config.to_dict())

    def _handle_admin_reload(self):
        """POST /admin/reload -- hot-swap config fields.

        Accepts JSON body with optional fields: backend_url, default_model,
        model_map, routes. Updates the running config without restart.
        """
        if not self._is_localhost():
            self._send_error(403, "Admin endpoints are localhost-only")
            return

        body = self._read_json_body()
        if body is None:
            return

        updated = []
        config = self.config

        if "backend_url" in body:
            config.backend_url = body["backend_url"]
            updated.append("backend_url")

        if "default_model" in body:
            config.default_model = body["default_model"]
            updated.append("default_model")

        if "model_map" in body and isinstance(body["model_map"], dict):
            config.model_map = body["model_map"]
            updated.append("model_map")

        if "routes" in body and isinstance(body["routes"], dict):
            config.routes = body["routes"]
            updated.append("routes")

        logger.info("Admin reload: updated fields %s", updated)
        self._send_json(200, {"status": "ok", "updated": updated})

    def _handle_budget_status(self):
        """GET /admin/budget -- return current budget status."""
        if not self._is_localhost():
            self._send_error(403, "Admin endpoints are localhost-only")
            return
        if self.budget is None:
            self._send_json(200, {"status": "disabled", "message": "No token budget configured"})
            return
        self._send_json(200, self.budget.get_status())

    def _handle_budget_reset(self):
        """POST /admin/budget/reset -- reset budget counters."""
        if not self._is_localhost():
            self._send_error(403, "Admin endpoints are localhost-only")
            return
        if self.budget is None:
            self._send_json(200, {"status": "disabled", "message": "No token budget configured"})
            return
        self.budget.reset()
        self._send_json(200, {"status": "ok", "message": "Budget counters reset"})

    def _handle_analytics_summary(self):
        """GET /admin/analytics -- return analytics summary."""
        if not self._is_localhost():
            self._send_error(403, "Admin endpoints are localhost-only")
            return
        if self.analytics is None:
            self._send_json(200, {"status": "disabled", "message": "Analytics not configured"})
            return
        cache_stats = self.cache.get_stats() if self.cache else None
        self._send_json(200, self.analytics.get_summary(cache_stats=cache_stats))

    # -- Messages Endpoint -------------------------------------------------

    def _handle_messages(self):
        """Process a POST /v1/messages request."""
        start = time.monotonic()
        self.stats.connection_opened()

        # Read and parse request body
        body = self._read_json_body()
        if body is None:
            self.stats.connection_closed()
            return  # Error already sent

        # Validate required fields
        error = self._validate_messages_request(body)
        if error:
            self._send_error(400, error)
            self.stats.connection_closed()
            return

        # Per-request upstream selection (provider gateway). None means the
        # config-based backend routing applies unchanged.
        try:
            upstream = resolve_upstream(self.headers)
        except UpstreamConfigError as e:
            self._send_error(e.status, str(e))
            self.stats.connection_closed()
            return

        # Default model if not provided (some SDK calls omit it)
        if "model" not in body:
            body["model"] = self.config.default_model
        original_model = body.get("model", "unknown")
        is_stream = body.get("stream", False)

        logger.info("Request: model=%s stream=%s", original_model, is_stream)

        # Budget check: reject if token budget exceeded
        if self.budget is not None and not self.budget.check_budget():
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self.stats.connection_closed()
            self._send_error(429, "Token budget exceeded")
            return

        # Replay: serve recorded response if available. Replay and cache are
        # skipped for gateway requests so responses never cross upstreams.
        if self.replayer is not None and upstream is None:
            entry = self.replayer.lookup(body)
            if entry is not None:
                elapsed = time.monotonic() - start
                response_body = entry.get("response", {})
                input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
                output_tokens = response_body.get("usage", {}).get("output_tokens", 0)
                self.stats.record_request(
                    model=original_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed * 1000,
                    stream=is_stream,
                )
                if self.analytics is not None:
                    tool_count = sum(
                        1 for b in response_body.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    )
                    self.analytics.record(
                        model=original_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=int(elapsed * 1000),
                        stream=is_stream,
                        stop_reason=response_body.get("stop_reason", "end_turn"),
                        tool_calls_count=tool_count,
                    )
                self._send_json(200, response_body)
                self.stats.connection_closed()
                return

        # Cache check: for non-streaming, try cache before backend
        if not is_stream and self.cache is not None and upstream is None:
            cached = self.cache.get(body)
            if cached is not None:
                elapsed = time.monotonic() - start
                input_tokens = cached.get("usage", {}).get("input_tokens", 0)
                output_tokens = cached.get("usage", {}).get("output_tokens", 0)
                self.stats.record_request(
                    model=original_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=elapsed * 1000,
                    stream=False,
                )
                if self.analytics is not None:
                    tool_count = sum(
                        1 for b in cached.get("content", [])
                        if isinstance(b, dict) and b.get("type") == "tool_use"
                    )
                    self.analytics.record(
                        model=original_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        latency_ms=int(elapsed * 1000),
                        stream=False,
                        cache_hit=True,
                        stop_reason=cached.get("stop_reason", "end_turn"),
                        tool_calls_count=tool_count,
                    )
                logger.info("Cache hit: model=%s %dms", original_model, int(elapsed * 1000))
                self._send_json(200, cached)
                self.stats.connection_closed()
                return

        # Translate to OpenAI format
        try:
            openai_payload = translate_request(body, self.config)
            if upstream is not None:
                # Cloud providers get the client's model name, not the local map.
                openai_payload["model"] = original_model
        except Exception as e:
            logger.error("Request translation failed: %s", e)
            self._send_error(400, f"Failed to translate request: {e}")
            self.stats.connection_closed()
            return

        # Acquire queue slot if queue is configured. The queue protects the
        # local backend; gateway requests to cloud providers bypass it.
        use_queue = self.queue is not None and upstream is None
        if use_queue:
            if not self.queue.acquire():
                elapsed = time.monotonic() - start
                self.stats.record_request(
                    model=original_model,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=elapsed * 1000,
                    stream=is_stream,
                    error=True,
                )
                self.stats.connection_closed()
                self._send_error(529, "Backend queue full -- request timed out waiting for a slot")
                return

        # Forward to backend
        try:
            if is_stream:
                self._handle_streaming(openai_payload, original_model, start, body, upstream)
            else:
                self._handle_non_streaming(openai_payload, original_model, start, body, upstream)
        except UpstreamError as e:
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self._send_error(e.status, e.message)
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = str(e)
            logger.error("Backend HTTP error %d: %s", e.code, error_body)
            status = _map_backend_http_status(e.code)
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self._send_error(status, f"Backend error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            logger.error("Backend connection failed: %s", e)
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self._send_error(529, f"Backend unavailable: {e.reason}")
        except TimeoutError:
            logger.error("Backend request timed out")
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self._send_error(529, "Backend request timed out")
        except Exception as e:
            logger.error("Unexpected error: %s", e, exc_info=True)
            elapsed = time.monotonic() - start
            self.stats.record_request(
                model=original_model, input_tokens=0, output_tokens=0,
                latency_ms=elapsed * 1000, stream=is_stream, error=True,
            )
            self._send_error(500, f"Internal proxy error: {e}")
        finally:
            if use_queue:
                self.queue.release()
            self.stats.connection_closed()

    def _handle_non_streaming(
        self, payload: dict, original_model: str, start: float,
        original_body: Optional[dict] = None, upstream: Optional[Upstream] = None,
    ):
        """Forward request to backend (or gateway upstream) and return translated response."""
        if upstream is not None:
            openai_body = gateway.chat_completion(upstream, payload, self.config.request_timeout)
        else:
            backend_response = self._forward_to_backend(payload, original_model)
            openai_body = json.loads(backend_response)

        # Validate backend response has expected structure
        if not isinstance(openai_body, dict):
            self._send_error(502, "Backend returned non-object response")
            return
        if "choices" not in openai_body:
            self._send_error(502, "Backend response missing 'choices' field")
            return

        anthropic_body = translate_response(openai_body, original_model)

        elapsed = time.monotonic() - start
        input_tokens = anthropic_body.get("usage", {}).get("input_tokens", 0)
        output_tokens = anthropic_body.get("usage", {}).get("output_tokens", 0)

        logger.info("Response: %dms, tokens=%d/%d",
                     int(elapsed * 1000), input_tokens, output_tokens)

        self.stats.record_request(
            model=original_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed * 1000,
            stream=False,
        )

        # Record request/response pair
        if self.recorder is not None and original_body is not None:
            self.recorder.record(
                request=original_body,
                response=anthropic_body,
                latency_ms=int(elapsed * 1000),
                stream=False,
                model=original_model,
            )

        # Store in cache
        if self.cache is not None and original_body is not None and upstream is None:
            self.cache.put(original_body, anthropic_body)

        # Update budget token counts
        if self.budget is not None:
            self.budget.record_usage(original_model, input_tokens, output_tokens)

        # Record analytics
        if self.analytics is not None:
            tool_count = sum(
                1 for b in anthropic_body.get("content", [])
                if isinstance(b, dict) and b.get("type") == "tool_use"
            )
            self.analytics.record(
                model=original_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=int(elapsed * 1000),
                stream=False,
                stop_reason=anthropic_body.get("stop_reason", "end_turn"),
                tool_calls_count=tool_count,
            )

        self._send_json(200, anthropic_body)

    def _handle_streaming(
        self, payload: dict, original_model: str, start: float,
        original_body: Optional[dict] = None, upstream: Optional[Upstream] = None,
    ):
        """Forward request to backend (or gateway upstream) and stream translated SSE events."""
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Open streaming connection to backend (may raise HTTPError/URLError
        # or UpstreamError, which is handled by the caller in _handle_messages)
        if upstream is not None:
            lines = gateway.open_chat_stream(upstream, payload, self.config.request_timeout)
            close_upstream = lines.close
        else:
            req = self._build_backend_request(payload, original_model)
            response = urllib.request.urlopen(req, timeout=self.config.request_timeout)
            lines = read_sse_lines(response)
            close_upstream = response.close

        # Send response headers. The body has no Content-Length or chunked
        # framing, so the connection must close to mark the end of the stream.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._set_cors_headers()
        self.end_headers()

        # Stream translated events
        try:
            for event in translate_stream(lines, original_model, request_id):
                self.wfile.write(event.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during streaming")
        except Exception as e:
            logger.error("Error during streaming: %s", e, exc_info=True)
            try:
                error_event = (
                    f"event: error\n"
                    f"data: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
                )
                self.wfile.write(error_event.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                logger.debug("Failed to write error SSE event to client")
        finally:
            close_upstream()

        elapsed = time.monotonic() - start
        logger.info("Stream complete: %dms", int(elapsed * 1000))

        self.stats.record_request(
            model=original_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=elapsed * 1000,
            stream=True,
        )

        # Record streaming request (response body not available for streams)
        if self.recorder is not None and original_body is not None:
            self.recorder.record(
                request=original_body,
                response={},
                latency_ms=int(elapsed * 1000),
                stream=True,
                model=original_model,
            )

        # Update budget (tokens unknown for streaming, record 0)
        if self.budget is not None:
            self.budget.record_usage(original_model, 0, 0)

        # Record analytics for streaming
        if self.analytics is not None:
            self.analytics.record(
                model=original_model,
                input_tokens=0,
                output_tokens=0,
                latency_ms=int(elapsed * 1000),
                stream=True,
            )

    # -- Chat Completions Endpoint (OpenAI inbound dialect) ----------------

    def _local_upstream(self, model: str) -> Upstream:
        """The config-routed backend as an Upstream (no key, trusted URL)."""
        base = self.config.resolve_backend_url(model).rstrip("/")
        return Upstream(provider=LOCAL_BACKEND, base_url=f"{base}/v1")

    def _resolve_upstream_openai(self) -> Tuple[bool, Optional[Upstream]]:
        """Resolve per-request upstream; returns (ok, upstream_or_None)."""
        try:
            return True, resolve_upstream(self.headers)
        except UpstreamConfigError as e:
            self._send_openai_error(e.status, str(e))
            return False, None

    def _handle_chat_completions(self):
        """Process a POST /v1/chat/completions request.

        With X-Lana-Upstream-* headers the request goes to the selected
        provider; otherwise it is forwarded unchanged to the configured backend
        (routes are applied by model name). Prompt cache and replay apply only
        to /v1/messages.
        """
        start = time.monotonic()
        self.stats.connection_opened()
        try:
            self._chat_completions(start)
        finally:
            self.stats.connection_closed()

    def _chat_completions(self, start: float):
        body = self._read_json_body(send_error=self._send_openai_error)
        if body is None:
            return
        if not isinstance(body, dict):
            self._send_openai_error(400, "Request body must be a JSON object")
            return
        ok, upstream = self._resolve_upstream_openai()
        if not ok:
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_openai_error(400, "Missing or empty field: messages")
            return
        if not body.get("model"):
            if upstream is not None:
                self._send_openai_error(400, "Missing required field: model")
                return
            body["model"] = self.config.default_model
        model = body["model"]
        is_stream = bool(body.get("stream"))
        target = upstream or self._local_upstream(model)

        logger.info("Chat request: upstream=%s model=%s stream=%s",
                    target.name, model, is_stream)

        if self.budget is not None and not self.budget.check_budget():
            self._record_chat(model, start, is_stream, error=True)
            self._send_openai_error(429, "Token budget exceeded")
            return

        use_queue = self.queue is not None and upstream is None
        if use_queue and not self.queue.acquire():
            self._record_chat(model, start, is_stream, error=True)
            self._send_openai_error(503, "Backend queue full -- request timed out waiting for a slot")
            return

        try:
            if is_stream:
                self._chat_stream(target, body, model, start)
            else:
                self._chat_once(target, body, model, start)
        except UpstreamError as e:
            self._record_chat(model, start, is_stream, error=True)
            self._send_openai_error(e.status, e.message)
        except Exception as e:
            logger.error("Unexpected gateway error: %s", type(e).__name__, exc_info=True)
            self._record_chat(model, start, is_stream, error=True)
            self._send_openai_error(500, "Internal proxy error")
        finally:
            if use_queue:
                self.queue.release()

    def _chat_once(self, target: Upstream, body: dict, model: str, start: float):
        hub = gateway.chat_completion(target, body, self.config.request_timeout)
        usage = hub.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        choice = (hub.get("choices") or [{}])[0]
        tool_count = len((choice.get("message") or {}).get("tool_calls") or [])
        self._record_chat(
            model, start, False, input_tokens=input_tokens, output_tokens=output_tokens,
            stop_reason=choice.get("finish_reason") or "stop", tool_calls_count=tool_count,
            request=body, response=hub,
        )
        self._send_json(200, hub)

    def _chat_stream(self, target: Upstream, body: dict, model: str, start: float):
        events = gateway.open_chat_stream(target, body, self.config.request_timeout)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length or chunked framing, so the stream ends at close.
        self.send_header("Connection", "close")
        self._set_cors_headers()
        self.end_headers()
        try:
            for event in events:
                self.wfile.write(event.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client disconnected during streaming")
        except UpstreamError as e:
            logger.warning("Upstream stream error %d", e.status)
            self._write_openai_stream_error(e.status, e.message)
        except Exception as e:
            logger.error("Error during streaming: %s", type(e).__name__, exc_info=True)
            self._write_openai_stream_error(500, "Internal proxy error during streaming")
        finally:
            events.close()
        self._record_chat(model, start, True, request=body, response={})

    def _write_openai_stream_error(self, status: int, message: str):
        try:
            body = make_openai_error_body(status, message)
            self.wfile.write(f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
        except Exception:
            logger.debug("Failed to write error SSE event to client")

    def _record_chat(self, model: str, start: float, stream: bool, error: bool = False,
                     input_tokens: int = 0, output_tokens: int = 0,
                     stop_reason: str = "stop", tool_calls_count: int = 0,
                     request: Optional[dict] = None, response: Optional[dict] = None):
        """Stats, recorder, budget and analytics for /v1/chat/completions.

        Only the request body is recorded; headers (and therefore the
        upstream key) never reach the recorder or analytics.
        """
        elapsed_ms = (time.monotonic() - start) * 1000
        self.stats.record_request(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=elapsed_ms, stream=stream, error=error,
        )
        if error:
            return
        if self.recorder is not None and request is not None:
            self.recorder.record(
                request=request, response=response or {}, latency_ms=int(elapsed_ms),
                stream=stream, model=model,
            )
        if self.budget is not None:
            self.budget.record_usage(model, input_tokens, output_tokens)
        if self.analytics is not None:
            self.analytics.record(
                model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=int(elapsed_ms), stream=stream, stop_reason=stop_reason,
                tool_calls_count=tool_calls_count,
            )

    def _handle_models(self):
        """GET /v1/models -- list models from the selected upstream or backend."""
        ok, upstream = self._resolve_upstream_openai()
        if not ok:
            return
        target = upstream or self._local_upstream("")
        try:
            self._send_json(200, gateway.list_models(target, self.config.request_timeout))
        except UpstreamError as e:
            self._send_openai_error(e.status, e.message)

    # -- Backend Communication ---------------------------------------------

    def _forward_to_backend(self, payload: dict,
                            original_model: str = "") -> bytes:
        """Send a non-streaming request to the backend and return the response body."""
        req = self._build_backend_request(payload, original_model)
        with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
            return resp.read()

    def _build_backend_request(self, payload: dict,
                               original_model: str = "") -> urllib.request.Request:
        """Build an HTTP request for the llama.cpp backend.

        Uses route-based URL resolution if the model has a configured route,
        otherwise falls back to the default backend_url.
        """
        backend_url = self.config.resolve_backend_url(original_model)
        url = f"{backend_url}/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    # -- Request Parsing ---------------------------------------------------

    def _read_json_body(self, send_error=None) -> Optional[dict]:
        """Read and parse the JSON request body.

        Args:
            send_error: Error responder (defaults to the Anthropic-format
                _send_error; /v1/chat/completions passes _send_openai_error).

        Returns:
            Parsed dict, or None if an error was sent to the client.
        """
        send_error = send_error or self._send_error
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            send_error(400, "Invalid Content-Length header")
            return None

        if content_length > self.config.max_request_size:
            send_error(400, f"Request body too large ({content_length} bytes)")
            return None

        if content_length == 0:
            send_error(400, "Empty request body")
            return None

        try:
            raw = self.rfile.read(content_length)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            send_error(400, f"Invalid JSON: {e}")
            return None

    def _validate_messages_request(self, body: dict) -> Optional[str]:
        """Validate required fields in an Anthropic Messages request.

        Returns:
            Error message string if validation fails, None if valid.
        """
        if "max_tokens" not in body:
            return "Missing required field: max_tokens"
        if "messages" not in body or not isinstance(body["messages"], list):
            return "Missing or invalid field: messages"
        if len(body["messages"]) == 0:
            return "messages must not be empty"
        return None

    # -- Response Helpers --------------------------------------------------

    def _send_json(self, status: int, body: dict):
        """Send a JSON response."""
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str):
        """Send an Anthropic-format error response."""
        body = make_error_body(status, message)
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(data)
        logger.warning("Error %d: %s", status, message)

    def _send_openai_error(self, status: int, message: str):
        """Send an OpenAI-format error response (for /v1/chat/completions)."""
        self._send_json(status, make_openai_error_body(status, message))
        logger.warning("Error %d: %s", status, message)

    def _set_cors_headers(self):
        """Set permissive CORS headers for local development."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")


def _map_backend_http_status(backend_status: int) -> int:
    """Map a backend HTTP error status to an appropriate Anthropic API status.

    Args:
        backend_status: HTTP status code from the backend.

    Returns:
        An HTTP status code suitable for the Anthropic error response.
    """
    if 400 <= backend_status < 500:
        return 400
    if backend_status == 503:
        return 529
    return 500
