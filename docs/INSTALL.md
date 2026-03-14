# Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | macOS 13+ / Linux | macOS 14+ (Apple Silicon) |
| Python | 3.9.6+ | 3.11+ |
| RAM | 16 GB | 24 GB+ |
| Disk | 15 GB free | 30 GB+ free |

No Homebrew, Node.js, or pip packages required.

### Required Tools

| Tool | How to get it |
|---|---|
| `python3` (3.9+) | Pre-installed on macOS |
| `git` | `xcode-select --install` |
| `curl` | Pre-installed on macOS |
| `cmake` | Downloaded by installer if missing |
| `clang++` | `xcode-select --install` |

## Quick Install

```bash
git clone <repo-url> ~/Desktop/lana-proxy
cd ~/Desktop/lana-proxy
bash install.sh
source ~/.zshrc
```

## What the Installer Does

The install script is **idempotent** — safe to run multiple times.

### 1. Prerequisites Check

Verifies git, cmake, curl, and Python 3.9+ are available.

### 2. llama.cpp Build

- Clones `https://github.com/ggerganov/llama.cpp` to `~/llama.cpp/`
- Builds with Metal GPU support (`-DGGML_METAL=ON`)
- Compiles using all available CPU cores
- Skips if already built

### 3. Model Download

- Reads model URL from `.env` (or `.env.example` defaults)
- Downloads to the configured `MODEL_DIR` (default: `~/llama.cpp/models/`)
- Supports resume on interrupted downloads
- Skips if model file already exists

**Default model:** Qwen2.5-Coder-14B-Instruct Q5_K_M (10.5 GB)

This was chosen for the best balance of quality and speed on 24GB Apple Silicon machines with 8K context.

### 4. lana-proxy Setup

- Creates `.env` from `.env.example` if missing
- Symlinks `lana-proxy.py` to `~/bin/lana-proxy`
- Creates `~/.config/lana-proxy/` with subdirectories for:
  - `analytics/` — Analytics logs
  - `cache/` — Disk cache
  - `recordings/` — Request/response recordings

### 5. Scripts

Makes `start.sh` and `stop.sh` executable. These are thin wrappers around the built-in Python process supervisor which reads settings from `.env`:
- Model path
- Context size (default 8192)
- Parallel slots (default 1)
- GPU layers (default 99 — full Metal offload)
- Flash attention
- Host/port binding

## Configuration (.env)

After install, customize `~/Desktop/lana-proxy/.env`:

```bash
# Model
MODEL_NAME=Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf
MODEL_URL=https://huggingface.co/...
MODEL_DIR=~/llama.cpp/models

# Backend
CONTEXT_SIZE=8192        # Token context window
PARALLEL_SLOTS=1         # Concurrent inference slots
GPU_LAYERS=99            # 99 = all layers on GPU
FLASH_ATTENTION=true     # Flash attention optimization
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8080

# Proxy
PROXY_PORT=5304
PROXY_LOG_LEVEL=INFO
DEFAULT_MODEL=local-model
```

### Context Size Guidelines

| Model | RAM | Recommended Context |
|---|---|---|
| 14B Q5_K_M (10.5 GB) | 24 GB | 8192 — 16384 |
| 14B Q5_K_M (10.5 GB) | 32 GB | 16384 — 32768 |
| 30B Q4_K_M (18.6 GB) | 24 GB | 4096 — 8192 |
| 30B Q4_K_M (18.6 GB) | 32 GB+ | 8192 — 16384 |

The default (8192) is conservative and works reliably on 24 GB machines.

## Directory Layout After Install

```
~/Desktop/lana-proxy/               Project root
  lana-proxy.py                     Entry point
  .env                              Your configuration
  install.sh                        Installer
  start.sh                          Start wrapper (delegates to supervisor)
  stop.sh                           Stop wrapper (delegates to supervisor)
  src/
    process/                        Generic process management (reusable)
      supervisor.py                 Multi-process orchestrator
      env.py                        .env file parser
    backend/                        llama.cpp-specific glue
      config.py                     Binary discovery + command building
    proxy/                          Proxy core
      config.py                     Runtime configuration
      server.py                     Threaded HTTP server
      errors.py                     Error formatting
      translation/                  API format conversion
      middleware/                   Cache, queue, recording, budget, analytics
      monitoring/                   Stats, dashboard, discovery
  tests/                            Test suite (106 tests)

~/llama.cpp/                        llama.cpp source + build
  build/bin/llama-server            Inference server
  models/
    Qwen2.5-Coder-14B-*.gguf       Default model

~/.config/lana-proxy/               Runtime data
  analytics/
  cache/
  recordings/
  backend.pid
  proxy.pid
  supervisor.pid
  backend.log
  proxy.log

~/bin/
  lana-proxy -> ~/Desktop/lana-proxy/lana-proxy.py
```

## Starting the Stack

### Option A: Supervisor (Recommended)

```bash
lana-proxy --start
```

This:
1. Kills any stale processes on the configured ports
2. Starts the llama.cpp backend
3. Waits for the backend to pass health checks
4. Starts lana-proxy
5. Monitors both processes with auto-restart on crash

### Option B: Components Separately

```bash
lana-proxy --start --backend-only
lana-proxy --start --proxy-only
```

### Option C: Shell Script Wrappers

```bash
./start.sh               # Same as lana-proxy --start
./start.sh --backend     # Same as lana-proxy --start --backend-only
./start.sh --proxy       # Same as lana-proxy --start --proxy-only
```

### Option D: Foreground with Dashboard

```bash
# Start backend via supervisor
lana-proxy --start --backend-only

# Run proxy in foreground with live dashboard
lana-proxy --dashboard --log-level DEBUG
```

## Process Status

```bash
lana-proxy --status
```

Output:
```
  backend         running (PID 10902)
  proxy           running (PID 10910)
  supervisor      running (PID 10900)
```

## Logs

```bash
lana-proxy --logs
```

Tails all log files (supervisor, backend, proxy). Press Ctrl+C to stop.

## Stopping

```bash
# Stop everything
lana-proxy --stop

# Or via script wrapper
./stop.sh
```

Sends SIGTERM first, falls back to SIGKILL after 5 seconds. Auto-restarts on crash (up to 5 attempts with exponential backoff).

## Post-Install Verification

```bash
# Check proxy is accessible
curl http://127.0.0.1:5304/health

# Test a request (requires backend running)
curl -s http://127.0.0.1:5304/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }' | python3 -m json.tool

# Run tests
python3 -m unittest discover tests/ -v
```

## Connecting Clients

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:5304 claude
```

### LANA CODER

```bash
LANA_USE_PROXY=true lana-code
```

### Any Anthropic SDK Client

```python
import anthropic
client = anthropic.Anthropic(
    api_key="not-needed",
    base_url="http://127.0.0.1:5304"
)
```

## Updating

```bash
cd ~/Desktop/lana-proxy
git pull
# No build step — pure Python
```

To update llama.cpp:

```bash
cd ~/llama.cpp
git pull
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

## Uninstall

```bash
# Stop services
lana-proxy --stop

# Remove symlink
rm ~/bin/lana-proxy

# Remove source
rm -rf ~/Desktop/lana-proxy

# Remove runtime data
rm -rf ~/.config/lana-proxy

# Remove llama.cpp and models (optional — shared with lana-code)
rm -rf ~/llama.cpp ~/models
```

## Troubleshooting

### "Backend unavailable" errors

The llama.cpp server isn't running or hasn't finished loading:

```bash
# Check if it's running
curl http://127.0.0.1:8080/health

# Check logs
cat ~/.config/lana-proxy/backend.log

# Restart
lana-proxy --restart
```

### Model takes too long to load

Large models (18GB+) take 30-60 seconds. The supervisor polls for up to 120 seconds.

### Port already in use

```bash
# Find what's using port 5304
lsof -i :5304

# Or port 8080
lsof -i :8080

# Kill it
kill $(lsof -ti :5304)
```

### Python version too old

```bash
python3 --version
# Needs 3.9.6+
```

macOS ships with Python 3.9.6. If yours is older, you're on a very old macOS version.
