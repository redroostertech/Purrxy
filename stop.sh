#!/bin/bash
# =============================================================================
# Stop purrxy services
#
# Thin wrapper around the Python process supervisor.
#
# Usage:
#   ./stop.sh               Stop all services
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

exec python3 "$SCRIPT_DIR/purrxy.py" --stop
