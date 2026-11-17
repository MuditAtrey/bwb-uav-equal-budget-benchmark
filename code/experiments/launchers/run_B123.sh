#!/bin/bash
# REVISION_PLAN_v3 B1, B2, B3. Resumable: safe to Ctrl-C and rerun (every arm resumes from its CSV / state file).
#   lane A: B1 AL with FEAS_MODE=none, seeds 0-9   -> results_v3/al_nofeas
#   lane B: B1 AL with FEAS_MODE=hard, seeds 0-9   -> results_v3/al_hard
#   lane C: B2 plain TuRBO seeds 7-9 -> turbo_extra; TuRBO+feasibility seeds 0-9 -> turbo_feas; B3 controls seeds 3-9 -> controls_extra
# usage: bash run_B123.sh      (uses 8 of 10 cores; stop with: pkill -f results_v3/scripts; pkill -f results_revision/scripts)
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"
PY="$REPO/.venv/bin/python3"; S3="$V3/scripts"; SR="$REPO/revision_runs/results_revision/scripts"
export DRONE_ROOT="$REPO" TBUDGET=100000
N="nice -n 10"
run() { f="$1"; shift; prev=-1   # repeat until 288 lines (header + 287) or no progress
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
( cd "$SR"; export NPROC=3
  for s in 0 1 2 3 4 5 6 7 8 9; do FEAS_MODE=none run "$V3/al_nofeas/al_seed$s.csv" $N "$PY" -I al_arm.py "$V3/al_nofeas" $s; done ) > "$V3/logs/laneA.log" 2>&1 &
( cd "$SR"; export NPROC=3
  for s in 0 1 2 3 4 5 6 7 8 9; do FEAS_MODE=hard run "$V3/al_hard/al_seed$s.csv" $N "$PY" -I al_arm.py "$V3/al_hard" $s; done ) > "$V3/logs/laneB.log" 2>&1 &
( export NPROC=2
  cd "$S3"
  for s in 7 8 9; do run "$V3/turbo_extra/turbo_seed$s.csv" $N "$PY" -I turbo_arm_v3.py "$V3/turbo_extra" $s $((s+1)); done
  for s in 0 1 2 3 4 5 6 7 8 9; do TURBO_FEAS=1 run "$V3/turbo_feas/turbo_seed$s.csv" $N "$PY" -I turbo_arm_v3.py "$V3/turbo_feas" $s $((s+1)); done
  cd "$SR"
  for s in 3 4 5 6 7 8 9; do for a in local5 lhs uniform; do run "$V3/controls_extra/${a}_seed$s.csv" $N "$PY" -I controls_runner.py "$V3/controls_extra" $s $((s+1)) $a; done; done ) > "$V3/logs/laneC.log" 2>&1 &
echo "started $(date). progress: tail -f $V3/logs/lane*.log"
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait
echo ALLDONE
