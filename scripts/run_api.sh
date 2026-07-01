#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/uvicorn src.api.main:app --reload --port 8000
