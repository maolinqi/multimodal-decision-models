#!/usr/bin/env bash
# Bootstrap only project-local tools and the three compatible runtimes.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
 "${PYTHON:-python3}" -m venv .tools
 .tools/bin/python -m pip install 'uv==0.12.17'
 export PATH="$PWD/.tools/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
# uv reuses an existing Python 3.12, downloading it only when unavailable.
./setup.sh
./scripts/setup_qwen35.sh
./scripts/setup_gemma4.sh
