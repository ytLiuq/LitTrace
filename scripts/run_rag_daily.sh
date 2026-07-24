#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${LITTRACE_VENV_BIN:-$ROOT_DIR/.venv/bin}"
LOG_DIR="${LITTRACE_LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

exec "$VENV_BIN/littrace" rag daily
