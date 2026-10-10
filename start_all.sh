#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -x .venv/bin/python ]] || { echo 'Run ./setup.sh first'; exit 1; }
mkdir -p run logs
export MODEL_ROOT="${MODEL_ROOT:-$PWD/models}"
export DECISION_URL="${DECISION_URL:-http://127.0.0.1:8457}"
start() {
 local name="$1" module="$2" port="$3" runtime="${4:-.venv/bin/python}"
 if [[ -f run/$name.pid ]] && kill -0 "$(cat "run/$name.pid")" 2>/dev/null; then
  echo "$name already running"; return
 fi
 nohup "$runtime" -m uvicorn "$module:app" --host 127.0.0.1 --port "$port" >"logs/$name.log" 2>&1 &
 echo "$!" >"run/$name.pid"
}
start backend multimodal_decision.service 8457
if [[ -x .venv-qwen35/bin/python ]]; then
 export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
 start qwen35 multimodal_decision.qwen35_service 8460 .venv-qwen35/bin/python
fi
if [[ -x .venv-gemma4/bin/python ]]; then
 export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
 start gemma4 multimodal_decision.gemma4_service 8461 .venv-gemma4/bin/python
fi
start gateway multimodal_decision.gateway 8456
# Startup succeeds only after each started service answers its health endpoint.
.venv/bin/python scripts/check_services.py
printf '%s\n' 'UI: http://127.0.0.1:8456; GPU model loads on first request.'
