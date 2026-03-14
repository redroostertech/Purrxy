# lana-proxy

Lightweight API translation proxy that enables Anthropic Messages API clients to communicate with llama.cpp and other OpenAI-compatible backends. Part of the **LANA** suite (Locally Autonomous Neural Agent).

**Zero dependencies** — pure Python 3.9+ stdlib only.

## Quick Start

```bash
# Install everything (llama.cpp, model, symlinks)
bash install.sh

# Start the full stack (backend + proxy)
./start.sh

# Or start components individually
./start.sh --backend
./start.sh --proxy
```

Then point any Anthropic client at the proxy:

```bash
# Claude Code
ANTHROPIC_BASE_URL=http://127.0.0.1:5304 claude

# LANA CODER
LANA_USE_PROXY=true lana-code
```

## How It Works

```
┌──────────────┐     Anthropic API     ┌─────────────┐     OpenAI API      ┌──────────────┐
│              │ ──────────────────────>│             │ ──────────────────> │              │
│  API Client  │   POST /v1/messages   │ lana-proxy  │ /v1/chat/completions│  llama.cpp   │
│              │ <──────────────────── │  :5304      │ <────────────────── │  :8080       │
└──────────────┘   Anthropic format    └─────────────┘    OpenAI format    └──────────────┘
```

**Requests:** Anthropic Messages format -> OpenAI Chat Completions format
**Responses:** OpenAI format -> Anthropic format (streaming and non-streaming)

## Installation

```bash
bash install.sh
```

The install script is idempotent and handles:

1. **Prerequisites check** — git, cmake, curl, python3 3.9+
2. **llama.cpp** — clones and builds with Metal GPU support (Apple Silicon)
3. **Model download** — fetches the model specified in `.env` (default: Qwen2.5-Coder-14B-Instruct Q5_K_M)
4. **Symlinks** — creates `~/bin/lana-proxy` for PATH access
5. **Config directories** — sets up `~/.config/lana-proxy/` for analytics, cache, recordings

## Configuration

### Environment File (.env)

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf` | GGUF model filename |
| `MODEL_URL` | *(Hugging Face URL)* | Download URL for the model |
| `MODEL_DIR` | `~/llama.cpp/models` | Where models are stored |
| `CONTEXT_SIZE` | `8192` | Context window (tokens) |
| `PARALLEL_SLOTS` | `1` | Concurrent inference slots |
| `GPU_LAYERS` | `99` | Layers offloaded to Metal GPU |
| `BACKEND_PORT` | `8080` | llama.cpp server port |
| `PROXY_PORT` | `5304` | lana-proxy listen port |

### CLI Flags

Settings resolve in order: **CLI args > environment variables > defaults**.

#### Core

| Flag | Env Var | Default | Description |
|---|---|---|---|
| `--host` | `LANA_PROXY_HOST` | `127.0.0.1` | Bind address |
| `--port` | `LANA_PROXY_PORT` | `5304` | Listen port |
| `--backend` | `LANA_PROXY_BACKEND` | `http://127.0.0.1:8080` | Backend URL |
| `--log-level` | `LANA_PROXY_LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `--timeout` | `LANA_PROXY_TIMEOUT` | `300` | Backend timeout (seconds) |
| `--default-model` | — | `local-model` | Fallback model name |
| `--model-map` | — | — | Name mappings (e.g., `claude-3-opus=qwen2.5-coder`) |

#### Multi-Backend Routing

| Flag | Default | Description |
|---|---|---|
| `--route` | — | Route models to backends (e.g., `claude-3-opus=http://localhost:8080`) |
| `--max-concurrent` | `1` | Max concurrent backend requests |
| `--queue-timeout` | `120` | Seconds to wait for a queue slot |

#### Caching & Recording

| Flag | Default | Description |
|---|---|---|
| `--cache` | off | Enable in-memory prompt cache |
| `--cache-ttl` | `300` | Cache entry TTL (seconds) |
| `--cache-max-size` | `100` | Max cached entries |
| `--cache-dir` | — | Disk persistence directory |
| `--record` | — | Record requests/responses to JSONL file |
| `--replay` | — | Replay from JSONL (no backend needed) |

#### Monitoring

| Flag | Default | Description |
|---|---|---|
| `--dashboard` | off | Live terminal dashboard |
| `--analytics` | — | Analytics log file (JSONL) |
| `--token-budget` | `0` | Session token limit (0 = unlimited) |

## Features

### API Translation

Full Anthropic Messages API translation:

| Feature | Status |
|---|---|
| Text messages (single and multi-turn) | Supported |
| System prompts (string and block array) | Supported |
| Streaming (`stream: true`) | Supported |
| Non-streaming | Supported |
| Tool definitions | Supported |
| Tool use (assistant -> tool_use blocks) | Supported |
| Tool results (user -> tool_result blocks) | Supported |
| Tool result errors (`is_error: true`) | Supported |
| Image content blocks (base64) | Supported |
| Temperature, top_p, top_k, max_tokens | Supported |

### Multi-Backend Routing

Route different models to different llama.cpp instances:

```bash
python3 lana-proxy.py \
  --route claude-3-opus=http://localhost:8080 \
  --route claude-3-haiku=http://localhost:8081
```

### Request Queuing

llama.cpp has limited concurrency. The proxy queues excess requests:

```bash
python3 lana-proxy.py --max-concurrent 2 --queue-timeout 120
```

### Prompt Caching

Cache identical non-streaming requests to save GPU cycles during development:

```bash
python3 lana-proxy.py --cache --cache-ttl 600 --cache-max-size 200
```

### Request/Response Recording & Replay

Record full conversations for debugging or offline testing:

```bash
# Record
python3 lana-proxy.py --record ./sessions/debug.jsonl

# Replay without a backend
python3 lana-proxy.py --replay ./sessions/debug.jsonl
```

### Live Dashboard

Real-time terminal UI showing requests, latency, throughput, and backend status:

```bash
python3 lana-proxy.py --dashboard
```

### Token Budget

Limit total token consumption per session:

```bash
python3 lana-proxy.py --token-budget 100000
```

### Conversation Analytics

Structured JSONL logs for every request — tokens, latency, tool usage, cache hits:

```bash
python3 lana-proxy.py --analytics ./logs/analytics.jsonl
```

### Auto Model Discovery

On startup, queries the backend's `/v1/models` endpoint and logs available models.

### Hot-Swap Configuration

Change backend, model mappings, or routes without restarting:

```bash
# View current config
curl http://127.0.0.1:5304/admin/config

# Update backend URL
curl -X POST http://127.0.0.1:5304/admin/reload \
  -H "Content-Type: application/json" \
  -d '{"backend_url": "http://localhost:9090"}'
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages API (main endpoint) |
| `GET` | `/health` | Health check |
| `GET` | `/admin/config` | Current configuration |
| `POST` | `/admin/reload` | Hot-swap configuration |
| `GET` | `/admin/budget` | Token usage stats |
| `POST` | `/admin/budget/reset` | Reset token counter |
| `GET` | `/admin/analytics` | Session analytics summary |

## Translation Reference

### Request Mapping

| Anthropic | OpenAI | Notes |
|---|---|---|
| `system` (top-level) | `messages[0].role: "system"` | String or block array |
| `content[].type: "text"` | `content` (string) | Text blocks joined |
| `content[].type: "tool_use"` | `tool_calls[].function` | Arguments JSON-serialized |
| `content[].type: "tool_result"` | `role: "tool"` message | Split per result |
| `tools[].input_schema` | `tools[].function.parameters` | Renamed + wrapped |

### Response Mapping

| OpenAI | Anthropic |
|---|---|
| `choices[0].message.content` | `content[].type: "text"` |
| `choices[0].message.tool_calls` | `content[].type: "tool_use"` |
| `finish_reason: "stop"` | `stop_reason: "end_turn"` |
| `finish_reason: "tool_calls"` | `stop_reason: "tool_use"` |
| `finish_reason: "length"` | `stop_reason: "max_tokens"` |
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |

### Streaming Event Sequence

```
message_start
  -> content_block_start (text)
    -> content_block_delta (text_delta) xN
  -> content_block_stop
  -> content_block_start (tool_use)
    -> content_block_delta (input_json_delta) xN
  -> content_block_stop
-> message_delta (stop_reason + usage)
-> message_stop
```

## Architecture

```
lana-proxy.py                          Entry point

src/
  process/                             Generic process management (reusable)
    supervisor.py                      Multi-process orchestrator
    env.py                             .env file parser

  backend/                             llama.cpp-specific glue
    config.py                          Binary discovery + command building

  proxy/                               Proxy core
    config.py                          Runtime configuration
    server.py                          Threaded HTTP server + routing
    errors.py                          Anthropic error envelope formatting

    translation/                       API format conversion
      request.py                       Anthropic -> OpenAI
      response.py                      OpenAI -> Anthropic (non-streaming)
      stream.py                        OpenAI SSE -> Anthropic SSE

    middleware/                         Optional features
      cache.py                         LRU prompt cache + disk persistence
      queue.py                         Semaphore-based request queuing
      recorder.py                      JSONL recording + replay
      budget.py                        Token budget enforcement
      analytics.py                     Per-request JSONL analytics

    monitoring/                        Observability
      stats.py                         Thread-safe metrics
      dashboard.py                     Live ANSI terminal UI
      discovery.py                     Backend model auto-discovery

tests/                                 106 tests
```

`src/process/` is fully generic — reusable in any Python 3.9+ project for process management with zero dependencies. Translation modules are pure functions. All subsystems are thread-safe and independently testable.

## Process Supervisor

Built-in cross-platform process management (macOS, Linux, Windows):

```bash
# Start full stack (backend + proxy) with auto-restart
lana-proxy --start

# Start components individually
lana-proxy --start --backend-only
lana-proxy --start --proxy-only

# Check status
lana-proxy --status

# Tail logs (like pm2 logs)
lana-proxy --logs

# Restart all
lana-proxy --restart

# Stop all
lana-proxy --stop
```

Features:
- Auto-restart on crash with exponential backoff
- Health check polling before starting dependents
- Stale port cleanup on start and stop (kills orphaned processes)
- Unified log streaming to terminal and log files
- PID file management with orphan recovery
- Clean shutdown on SIGINT/SIGTERM

## Scripts

| Script | Description |
|---|---|
| `install.sh` | Full setup: llama.cpp, model, symlinks, config |
| `start.sh` | Wrapper for `lana-proxy --start`. Use `--backend` or `--proxy` for individual |
| `stop.sh` | Wrapper for `lana-proxy --stop` |

## Testing

```bash
# Run all tests (106 total)
python3 -m unittest discover tests/ -v

# Individual test modules
python3 -m unittest tests.test_translate_request -v
python3 -m unittest tests.test_translate_response -v
python3 -m unittest tests.test_translate_stream -v
python3 -m unittest tests.test_config -v
python3 -m unittest tests.test_sdk_compat -v
python3 -m unittest tests.test_supervisor -v
```

### Manual Testing

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

# Health check
curl http://127.0.0.1:5304/health
```

## Requirements

- Python 3.9.6+ (stdlib only)
- llama.cpp server with OpenAI-compatible API
- macOS with Apple Silicon (for Metal GPU acceleration) or any platform with a compatible backend

## License

Part of the LANA suite. [MIT License](LICENSE). Copyright (c) 2025 RedRooster Technologies Inc.
