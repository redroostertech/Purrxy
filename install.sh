#!/bin/bash
# =============================================================================
# purrxy Install Script
# =============================================================================
# Installs llama.cpp, downloads a model, and sets up purrxy.
# Safe to run multiple times (idempotent).
#
# Requirements: git, cmake, curl, python3 (3.9+)
# Compatible with bash 3.2 (macOS default)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.config/purrxy"
BIN_DIR="$HOME/bin"

# ── Helpers ──────────────────────────────────────────────────────────────────

log_ok()   { printf "[OK]   %s\n" "$1"; }
log_skip() { printf "[SKIP] %s\n" "$1"; }
log_fail() { printf "[FAIL] %s\n" "$1"; }
log_info() { printf "[....] %s\n" "$1"; }

die() {
    log_fail "$1"
    exit 1
}

# Load .env file (source it with variable expansion)
load_env() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        # Read line by line, skip comments and empty lines, export valid vars
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            case "$line" in
                \#*|"") continue ;;
            esac
            # Only process lines that look like VAR=VALUE
            case "$line" in
                *=*)
                    # Strip leading/trailing whitespace from key
                    key="$(echo "$line" | cut -d= -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                    val="$(echo "$line" | cut -d= -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                    # Expand ~ to HOME
                    val="$(echo "$val" | sed "s|^~|$HOME|")"
                    eval "export $key=\"$val\""
                    ;;
            esac
        done < "$env_file"
    fi
}

# Expand ~ in a path
expand_path() {
    echo "$1" | sed "s|^~|$HOME|"
}

# ── Check Prerequisites ─────────────────────────────────────────────────────

log_info "Checking prerequisites..."

# git
if command -v git >/dev/null 2>&1; then
    log_ok "git found: $(git --version | head -1)"
else
    die "git is required but not found. Install Xcode Command Line Tools: xcode-select --install"
fi

# cmake (auto-install if missing)
if command -v cmake >/dev/null 2>&1; then
    log_ok "cmake found: $(cmake --version | head -1)"
else
    log_info "cmake not found — attempting to install..."
    if command -v brew >/dev/null 2>&1; then
        brew install cmake || die "Failed to install cmake via Homebrew"
        log_ok "cmake installed via Homebrew: $(cmake --version | head -1)"
    else
        # No Homebrew — download cmake directly
        log_info "No Homebrew found. Downloading cmake binary..."
        CMAKE_VERSION="3.31.6"
        CMAKE_TMP="/tmp/lana-cmake-$$.tar.gz"
        mkdir -p "$HOME/local-tools/bin"
        curl -sL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-macos-universal.tar.gz" \
            -o "$CMAKE_TMP" || die "Failed to download cmake"
        tar -xzf "$CMAKE_TMP" -C /tmp/
        cp "/tmp/cmake-${CMAKE_VERSION}-macos-universal/CMake.app/Contents/bin/cmake" "$HOME/local-tools/bin/"
        cp -r "/tmp/cmake-${CMAKE_VERSION}-macos-universal/CMake.app/Contents/share" "$HOME/local-tools/"
        rm -rf "$CMAKE_TMP" "/tmp/cmake-${CMAKE_VERSION}-macos-universal"
        export PATH="$HOME/local-tools/bin:$PATH"
        if command -v cmake >/dev/null 2>&1; then
            log_ok "cmake ${CMAKE_VERSION} installed to ~/local-tools/bin/"
        else
            die "cmake installation failed"
        fi
    fi
fi

# curl
if command -v curl >/dev/null 2>&1; then
    log_ok "curl found"
else
    die "curl is required but not found"
fi

# python3
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PYTHON_MAJOR="$(echo "$PYTHON_VERSION" | cut -d. -f1)"
    PYTHON_MINOR="$(echo "$PYTHON_VERSION" | cut -d. -f2)"
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        log_ok "python3 found: $PYTHON_VERSION"
    else
        die "Python 3.9+ required, found $PYTHON_VERSION"
    fi
else
    die "python3 is required but not found"
fi

echo ""

# ── Load Configuration ───────────────────────────────────────────────────────

# Create .env from .env.example if it doesn't exist
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
        log_ok "Created .env from .env.example"
    else
        log_fail "No .env or .env.example found"
    fi
else
    log_skip ".env already exists"
fi

# Load configuration
load_env "$SCRIPT_DIR/.env"

# Set defaults for anything not in .env
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/llama.cpp/models}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q5_k_m.gguf}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"
PARALLEL_SLOTS="${PARALLEL_SLOTS:-1}"
GPU_LAYERS="${GPU_LAYERS:-99}"
FLASH_ATTENTION="${FLASH_ATTENTION:-true}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-5304}"

LLAMA_CPP_DIR="$(expand_path "$LLAMA_CPP_DIR")"
MODEL_DIR="$(expand_path "$MODEL_DIR")"

echo ""

# ── Install llama.cpp ────────────────────────────────────────────────────────

log_info "Checking llama.cpp installation..."

# Look for llama-server binary in common locations
LLAMA_SERVER=""
for candidate in \
    "$LLAMA_CPP_DIR/build/bin/llama-server" \
    "$LLAMA_CPP_DIR/build/bin/server" \
    "$LLAMA_CPP_DIR/build/llama-server" \
    "$LLAMA_CPP_DIR/llama-server"; do
    if [ -x "$candidate" ]; then
        LLAMA_SERVER="$candidate"
        break
    fi
done

if [ -n "$LLAMA_SERVER" ]; then
    log_skip "llama.cpp already installed: $LLAMA_SERVER"
else
    # Clone if needed
    if [ -d "$LLAMA_CPP_DIR/.git" ]; then
        log_skip "llama.cpp repo already cloned at $LLAMA_CPP_DIR"
    else
        log_info "Cloning llama.cpp to $LLAMA_CPP_DIR..."
        git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR" || die "Failed to clone llama.cpp"
        log_ok "Cloned llama.cpp"
    fi

    # Build with Metal GPU support
    log_info "Building llama.cpp with Metal GPU support (this may take a few minutes)..."
    NUM_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

    cd "$LLAMA_CPP_DIR"
    cmake -B build -DGGML_METAL=ON || die "cmake configuration failed"
    cmake --build build --config Release -j"$NUM_CORES" || die "cmake build failed"
    cd "$SCRIPT_DIR"

    # Verify build
    LLAMA_SERVER=""
    for candidate in \
        "$LLAMA_CPP_DIR/build/bin/llama-server" \
        "$LLAMA_CPP_DIR/build/bin/server" \
        "$LLAMA_CPP_DIR/build/llama-server"; do
        if [ -x "$candidate" ]; then
            LLAMA_SERVER="$candidate"
            break
        fi
    done

    if [ -n "$LLAMA_SERVER" ]; then
        log_ok "llama.cpp built successfully: $LLAMA_SERVER"
    else
        die "llama.cpp build completed but llama-server binary not found"
    fi
fi

echo ""

# ── Download Model ───────────────────────────────────────────────────────────

log_info "Checking model installation..."

MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
    MODEL_SIZE="$(du -h "$MODEL_PATH" | cut -f1)"
    log_skip "Model already downloaded: $MODEL_PATH ($MODEL_SIZE)"
else
    # Create model directory
    mkdir -p "$MODEL_DIR" || die "Failed to create model directory: $MODEL_DIR"

    log_info "Downloading model: $MODEL_NAME"
    log_info "URL: $MODEL_URL"
    log_info "Destination: $MODEL_PATH"
    log_info "This is a large download (~10GB) — please be patient..."
    echo ""

    # Download with progress bar and resume support
    curl -L -C - --progress-bar -o "$MODEL_PATH" "$MODEL_URL" || {
        # Clean up partial download on failure
        rm -f "$MODEL_PATH"
        die "Model download failed"
    }

    # Verify download
    if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
        MODEL_SIZE="$(du -h "$MODEL_PATH" | cut -f1)"
        log_ok "Model downloaded: $MODEL_PATH ($MODEL_SIZE)"
    else
        rm -f "$MODEL_PATH"
        die "Model download produced empty file"
    fi
fi

echo ""

# ── Setup purrxy ─────────────────────────────────────────────────────────

log_info "Setting up purrxy..."

# Create config directory
if [ ! -d "$CONFIG_DIR" ]; then
    mkdir -p "$CONFIG_DIR" || die "Failed to create config directory: $CONFIG_DIR"
    log_ok "Created config directory: $CONFIG_DIR"
else
    log_skip "Config directory exists: $CONFIG_DIR"
fi

# Create subdirectories for analytics, cache, recordings
for subdir in analytics cache recordings; do
    if [ ! -d "$CONFIG_DIR/$subdir" ]; then
        mkdir -p "$CONFIG_DIR/$subdir"
        log_ok "Created $CONFIG_DIR/$subdir"
    else
        log_skip "$CONFIG_DIR/$subdir exists"
    fi
done

# Make main script executable
chmod +x "$SCRIPT_DIR/purrxy.py"

# Create bin directory if needed
if [ ! -d "$BIN_DIR" ]; then
    mkdir -p "$BIN_DIR"
    log_ok "Created $BIN_DIR"
fi

# Symlink to ~/bin/purrxy
if [ -L "$BIN_DIR/purrxy" ]; then
    EXISTING_TARGET="$(readlink "$BIN_DIR/purrxy")"
    if [ "$EXISTING_TARGET" = "$SCRIPT_DIR/purrxy.py" ]; then
        log_skip "Symlink already exists: $BIN_DIR/purrxy -> $SCRIPT_DIR/purrxy.py"
    else
        ln -sf "$SCRIPT_DIR/purrxy.py" "$BIN_DIR/purrxy"
        log_ok "Updated symlink: $BIN_DIR/purrxy -> $SCRIPT_DIR/purrxy.py"
    fi
elif [ -e "$BIN_DIR/purrxy" ]; then
    log_skip "$BIN_DIR/purrxy exists but is not a symlink — not overwriting"
else
    ln -s "$SCRIPT_DIR/purrxy.py" "$BIN_DIR/purrxy"
    log_ok "Created symlink: $BIN_DIR/purrxy -> $SCRIPT_DIR/purrxy.py"
fi

echo ""

# ── Make Scripts Executable ──────────────────────────────────────────────────

for script in start.sh stop.sh; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        chmod +x "$SCRIPT_DIR/$script"
        log_ok "Made $script executable"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo "============================================================================="
echo " Purrxy installation complete!"
echo "============================================================================="
echo ""
echo " Quick start:"
echo "   ./start.sh                Start backend + proxy"
echo "   ./start.sh --backend      Start only llama.cpp"
echo "   ./start.sh --proxy        Start only the proxy"
echo ""
echo " Configuration:  $SCRIPT_DIR/.env"
echo " Config dir:     $CONFIG_DIR"
echo " Model:          $MODEL_PATH"
echo " Proxy command:  purrxy (in ~/bin/)"
echo ""
echo " Test the proxy:"
echo "   curl http://127.0.0.1:$PROXY_PORT/health"
echo ""
echo " Stop:"
echo "   ./stop.sh"
echo "============================================================================="
