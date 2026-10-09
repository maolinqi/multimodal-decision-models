#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for name in gateway backend; do
 file="run/$name.pid"
 [[ -f "$file" ]] || continue
 pid="$(cat "$file")"
 if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
  if [[ -d /proc/$pid ]] && [[ "$(readlink "/proc/$pid/cwd")" != "$PWD" ]]; then
   echo "Refusing to stop process outside this project: $pid"; continue
  fi
  args="$(ps -p "$pid" -o args=)"
  if [[ "$args" == *"multimodal_decision."* ]]; then kill "$pid"; else
   echo "Refusing to stop unrelated process $pid"; continue
  fi
 fi
 rm -f "$file"
done
