#!/bin/bash
# B9 (2026-11-15): doubles the seed count of the three leading arms and closes the 2x2 design {classifier weight} x {washout repair}.
#   lanes A-C: AL + washout repair (sigmoid weight), seeds 10-19        -> results_v3/al_repair_extra
#   lanes D-F: AL + washout repair, FEAS_MODE=none (no classifier weight), seeds 0-9 -> results_v3/al_repair_nofeas
#   lane  G  : plain TuRBO seeds 10-19 -> turbo_extra2 ; adaptive local 10% seeds 10-19 -> controls_extra2
# Resumable (every arm resumes from its CSV / state file). usage: bash run_B9.sh
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"
S3="$V3/scripts"; SR="$REPO/revision_runs/results_revision/scripts"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2
mkdir -p "$V3/al_repair_extra" "$V3/al_repair_nofeas" "$V3/turbo_extra2" "$V3/controls_extra2"
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
alane() { d="$1"; shift; cd "$S3"; for s in "$@"; do run "$V3/$d/al_seed$s.csv" nice -n 10 "$PY" -I al_arm_v3.py "$V3/$d" $s; done; }
( alane al_repair_extra 10 11 12 13 ) > "$V3/logs/B9_laneA.log" 2>&1 &
( alane al_repair_extra 14 15 16 )    > "$V3/logs/B9_laneB.log" 2>&1 &
( alane al_repair_extra 17 18 19 )    > "$V3/logs/B9_laneC.log" 2>&1 &
( export FEAS_MODE=none; alane al_repair_nofeas 0 1 2 3 ) > "$V3/logs/B9_laneD.log" 2>&1 &
( export FEAS_MODE=none; alane al_repair_nofeas 4 5 6 )   > "$V3/logs/B9_laneE.log" 2>&1 &
( export FEAS_MODE=none; alane al_repair_nofeas 7 8 9 )   > "$V3/logs/B9_laneF.log" 2>&1 &
( cd "$S3"; for s in 10 11 12 13 14 15 16 17 18 19; do run "$V3/turbo_extra2/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v3.py "$V3/turbo_extra2" $s $((s+1)); done
  cd "$SR"; for s in 10 11 12 13 14 15 16 17 18 19; do run "$V3/controls_extra2/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local.py "$V3/controls_extra2" 10 $s $((s+1)); done ) > "$V3/logs/B9_laneG.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B9DONE
