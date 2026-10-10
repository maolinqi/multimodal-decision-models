#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv and Python 3.12 before setup'; exit 1; }
[[ -x .venv-gemma4/bin/python ]] || uv venv --python 3.12 .venv-gemma4
uv pip install --python .venv-gemma4/bin/python --torch-backend cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv-gemma4/bin/python transformers==5.19.0 pillow==12.3.0 accelerate==1.15.0 fastapi==0.118.0 uvicorn==0.37.0 httpx==0.28.1 python-multipart==0.0.20 sentencepiece==0.2.1
# Use PYTHONPATH at startup: do not install the legacy pinned environment into this runtime.
printf '%s\n' 'Gemma 4 runtime ready; existing .venv was preserved. Run ./start_all.sh.'
