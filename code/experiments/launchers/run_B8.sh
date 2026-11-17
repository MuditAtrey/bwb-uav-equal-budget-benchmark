#!/bin/bash
# B8: AL (sigmoid weight, as published) with the washout repair that every other arm already has. Seeds 0-9 -> results_v3/al_repair. Resumable.
# usage: bash run_B8.sh      (3 lanes x NPROC=3)
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=3; mkdir -p "$V3/al_repair"; cd "$V3/scripts"
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
lane() { for s in "$@"; do run "$V3/al_repair/al_seed$s.csv" nice -n 10 "$PY" -I al_arm_v3.py "$V3/al_repair" $s; done; }
lane 0 1 2 3 > "$V3/logs/B8_laneA.log" 2>&1 &
lane 4 5 6   > "$V3/logs/B8_laneB.log" 2>&1 &
lane 7 8 9   > "$V3/logs/B8_laneC.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B8DONE
