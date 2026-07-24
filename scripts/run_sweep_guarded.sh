#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${LOG:-/tmp/sft_sweep.log}"
STALL_S="${STALL_S:-240}"
JSON="$ROOT/outputs/regret_sweep.json"
LOCK="/tmp/online_sft_sweep.lock"
PY="$ROOT/.venv/bin/python"
JOB="/tmp/regret_job_main.py"

echo "==== guarded sweep $(date) stall=${STALL_S}s ====" | tee -a "$LOG"

while true; do
  rm -f "$LOCK"
  "$PY" -u "$JOB" >>"$LOG" 2>&1 &
  SPID=$!
  echo "$SPID" > /tmp/sft_sweep.pid
  echo "$(date) started job pid=$SPID ($JOB)" | tee -a "$LOG"
  last_mt=0
  [[ -f "$JSON" ]] && last_mt=$(stat -f '%m' "$JSON")
  stall_since=$(date +%s)
  while kill -0 "$SPID" 2>/dev/null; do
    sleep 10
    if [[ -f "$JSON" ]]; then
      mt=$(stat -f '%m' "$JSON")
      if [[ "$mt" -gt "$last_mt" ]]; then
        last_mt=$mt
        stall_since=$(date +%s)
      fi
    fi
    now=$(date +%s)
    age=$((now - stall_since))
    if [[ "$age" -ge "$STALL_S" ]]; then
      echo "$(date) STALL ${age}s — kill -9 $SPID and resume" | tee -a "$LOG"
      kill -9 "$SPID" 2>/dev/null || true
      pkill -9 -P "$SPID" 2>/dev/null || true
      sleep 2
      break
    fi
  done
  wait "$SPID" 2>/dev/null || true
  if tail -n 40 "$LOG" | grep -q '^DONE$'; then
    echo "$(date) sweep DONE — guard exiting" | tee -a "$LOG"
    exit 0
  fi
  echo "$(date) job pid=$SPID exited — restarting in 5s" | tee -a "$LOG"
  sleep 5
done
