#!/bin/bash
# B10 (2026-11-15): baselines with the energy-basis fix (see scripts/turbo_arm_v4.py, scripts/adaptive_local_v3.py headers).
#   lanes A,B: TuRBO v4 seeds 0-9 -> turbo_fix ; lane C: adaptive local 10% v3 seeds 0-19 -> adapt_fix. Resumable.
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"; S3="$V3/scripts"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2; mkdir -p "$V3/turbo_fix" "$V3/adapt_fix"; cd "$S3"
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
tl() { for s in "$@"; do run "$V3/turbo_fix/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$V3/turbo_fix" $s $((s+1)); done; }
( tl 0 1 2 3 4 ) > "$V3/logs/B10_laneA.log" 2>&1 &
( tl 5 6 7 8 9 ) > "$V3/logs/B10_laneB.log" 2>&1 &
( for s in $(seq 0 19); do run "$V3/adapt_fix/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local_v3.py "$V3/adapt_fix" 10 $s $((s+1)); done ) > "$V3/logs/B10_laneC.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B10DONE
