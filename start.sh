#!/bin/bash
# =============================================================================
# Start lana-proxy services
#
# Thin wrapper around the Python process supervisor.
#
# Usage:
#   ./start.sh              Start backend + proxy (default)
#   ./start.sh --backend    Start only the llama.cpp backend
#   ./start.sh --proxy      Start only the lana-proxy server
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Map bash-style flags to supervisor flags
ARGS="--start"
for arg in "$@"; do
    case "$arg" in
        --backend) ARGS="$ARGS --backend-only" ;;
        --proxy)   ARGS="$ARGS --proxy-only" ;;
        --help|-h)
            echo "Usage: ./start.sh [--backend] [--proxy]"
            echo ""
            echo "  No flags     Start backend + proxy (default)"
            echo "  --backend    Start only the llama.cpp backend"
            echo "  --proxy      Start only the lana-proxy server"
            echo ""
            echo "Delegates to: python3 lana-proxy.py --start"
            exit 0
            ;;
        *) ARGS="$ARGS $arg" ;;
    esac
done

exec python3 "$SCRIPT_DIR/lana-proxy.py" $ARGS
