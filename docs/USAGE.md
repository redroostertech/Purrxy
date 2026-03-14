# Usage Guide

## Basic Usage

```bash
# Start with defaults
lana-proxy

# Or with full stack
./start.sh
```

The proxy listens on `http://127.0.0.1:5304` and forwards to `http://127.0.0.1:8080`.

## Running Modes

### Standard Mode

```bash
lana-proxy
```

Runs in foreground, logs to stderr. Press Ctrl+C to stop.

### Dashboard Mode

```bash
lana-proxy --dashboard
```

Shows a live-updating terminal UI with:
- Active connections and request counts
- Latency percentiles (avg/p50/p95)
- Token throughput (input/output per minute)
- Backend status and discovered models
- Last 5 request summaries

### Debug Mode

```bash
lana-proxy --log-level DEBUG
```

Logs full request/response bodies, SSE chunks, and translation details.

### Supervised Mode

```bash
lana-proxy --start              # Start backend + proxy with auto-restart
lana-proxy --start --proxy-only # Proxy only (backend managed externally)
lana-proxy --status             # Check process status
lana-proxy --logs               # Tail log output (like pm2 logs)
lana-proxy --restart            # Stop then start all processes
lana-proxy --stop               # Stop all
```

## Connecting Clients

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:5304 claude
```

Or set it permanently:

```bash
echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:5304' >> ~/.zshrc
```

### LANA CODER

```bash
LANA_USE_PROXY=true lana-code
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="not-needed",       # Proxy ignores auth
    base_url="http://127.0.0.1:5304"
)

message = client.messages.create(
    model="model",        # Any name works — backend uses its loaded model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

### curl

```bash
# Non-streaming
curl -s http://127.0.0.1:5304/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | python3 -m json.tool

# Streaming
curl -N http://127.0.0.1:5304/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "max_tokens": 256,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Model Routing

### Single Backend (Default)

All requests go to one llama.cpp instance:

```bash
lana-proxy --backend http://127.0.0.1:8080
```

### Model Name Mapping

Map Anthropic model names to backend model names:

```bash
lana-proxy --model-map claude-3-opus=qwen2.5-coder claude-3-haiku=qwen3-coder
```

The proxy replaces the model name before forwarding. The original name is echoed back in the response.

### Multi-Backend Routing

Route different models to different llama.cpp instances:

```bash
lana-proxy \
  --route qwen2.5=http://localhost:8080 \
  --route qwen3=http://localhost:8081
```

Unmatched models fall back to `--backend`.

### Hot-Swap

Change routing without restarting:

```bash
# Update backend URL
curl -X POST http://127.0.0.1:5304/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"backend_url": "http://localhost:9090"}'

# Update model mapping
curl -X POST http://127.0.0.1:5304/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"model_map": {"claude-3-opus": "new-model"}}'

# View current config
curl -s http://127.0.0.1:5304/admin/config | python3 -m json.tool
```

## Request Queuing

llama.cpp typically handles one request at a time. The proxy queues excess requests:

```bash
lana-proxy --max-concurrent 1 --queue-timeout 120
```

- `--max-concurrent 1` — Only 1 request forwarded to backend at a time (default)
- `--queue-timeout 120` — Waiting requests timeout after 120 seconds with a 529 error

Increase `--max-concurrent` if your backend supports parallel slots.

## Caching

Cache identical non-streaming requests to avoid redundant inference:

```bash
# In-memory only
lana-proxy --cache --cache-ttl 300 --cache-max-size 100

# With disk persistence (survives restarts)
lana-proxy --cache --cache-dir ~/.config/lana-proxy/cache
```

Cache key is a hash of: model + system prompt + messages + tools. Only non-streaming responses are cached.

Useful during development when you're iterating on prompts and hitting the same requests repeatedly.

## Recording & Replay

### Recording

Save every request/response pair for debugging or offline testing:

```bash
lana-proxy --record ./sessions/debug.jsonl
```

Each line in the JSONL file contains:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "request": { ... },
  "response": { ... },
  "latency_ms": 1234,
  "stream": false,
  "model": "qwen2.5"
}
```

### Replay

Serve recorded responses without a running backend:

```bash
lana-proxy --replay ./sessions/debug.jsonl
```

Requests are matched by hashing the messages content. Streaming responses are replayed with realistic timing delays.

Use cases:
- Offline development and testing
- Demo mode
- Reproducing bugs from recorded sessions

## Token Budget

Limit total token consumption per session:

```bash
lana-proxy --token-budget 100000
```

When the budget is exhausted, new requests are rejected with HTTP 429:
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Token budget exceeded (used 100234 of 100000 tokens)"
  }
}
```

Monitor and reset via admin endpoints:

```bash
# Check usage
curl -s http://127.0.0.1:5304/admin/budget | python3 -m json.tool

# Reset counter
curl -X POST http://127.0.0.1:5304/admin/budget/reset
```

## Analytics

Log structured analytics for every request:

```bash
lana-proxy --analytics ~/.config/lana-proxy/analytics/session.jsonl
```

Each entry:
```json
{
  "timestamp": "...",
  "model": "model",
  "input_tokens": 150,
  "output_tokens": 89,
  "latency_ms": 2340,
  "stream": true,
  "tool_calls_count": 2,
  "cache_hit": false,
  "stop_reason": "end_turn"
}
```

View session summary:

```bash
curl -s http://127.0.0.1:5304/admin/analytics | python3 -m json.tool
```

Returns aggregated stats: total requests, tokens, average latency, per-model breakdown, tool usage frequency, and cache hit rate.

## Admin Endpoints

All admin endpoints are localhost-only.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/admin/config` | Current runtime configuration |
| `POST` | `/admin/reload` | Update config without restart |
| `GET` | `/admin/budget` | Token usage and remaining budget |
| `POST` | `/admin/budget/reset` | Reset token counter to zero |
| `GET` | `/admin/analytics` | Aggregated session analytics |

## Full Example

A typical production-like setup:

```bash
lana-proxy \
  --port 5304 \
  --backend http://127.0.0.1:8080 \
  --model-map claude-3-opus=qwen2.5 claude-3-haiku=qwen2.5 \
  --max-concurrent 1 \
  --cache \
  --cache-ttl 600 \
  --cache-dir ~/.config/lana-proxy/cache \
  --analytics ~/.config/lana-proxy/analytics/$(date +%Y%m%d).jsonl \
  --token-budget 500000 \
  --dashboard \
  --log-level INFO
```

This gives you:
- Model name translation for Claude Code compatibility
- Request queuing (1 at a time)
- 10-minute prompt cache with disk persistence
- Daily analytics logs
- 500K token session budget
- Live dashboard
- Standard logging

## Combining with start.sh

For a simpler workflow, configure everything in `.env` and use the scripts:

```bash
# Start everything
./start.sh

# Monitor
curl -s http://127.0.0.1:5304/admin/analytics | python3 -m json.tool

# Stop everything
./stop.sh
```
