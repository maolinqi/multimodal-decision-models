#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python --torch-backend cu128 -r requirements.lock.txt
mkdir -p run logs outputs
