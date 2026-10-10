#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null; then
 [[ -x .venv/bin/python ]] || uv venv --python "${PYTHON_VERSION:-3.12}" .venv
 uv pip install --python .venv/bin/python --torch-backend cu128 -e '.[dev]'
else
 "${PYTHON:-python3}" -m venv .venv
 .venv/bin/python -m pip install -e '.[dev]'
fi
printf '%s\n' 'Environment ready. Download models, then run ./start_all.sh.'
