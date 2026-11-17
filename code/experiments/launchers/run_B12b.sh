#!/bin/bash
# B12b (2026-11-16): corrected pool-size runs (see pool_size_v2/README.txt). Pool sizes 288, 1000, 3000; subsample seed 7000+seed.
# Fix relative to run_B12.sh: run() no longer clobbers the pool-size variable, and each run prints the POOL_N it used.
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"; W="$V3/pool_size_v2"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2
run() { local f="$1" cnt prev=-1; shift
  while true; do cnt=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$cnt" -ge 288 ] && break
    [ "$cnt" -eq "$prev" ] && { echo "stalled $f at $cnt"; break; }; prev=$cnt; "$@"; done; echo "$(date +%H:%M) done $f"; }
tl() { local ps="$1" s; shift; cd "$V3/scripts"; for s in "$@"; do echo "POOL_N=$ps turbo seed $s"; POOL_N=$ps run "$W/turbo_$ps/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$W/turbo_$ps" $s $((s+1)); done; }
ad() { local ps="$1" s; shift; cd "$V3/scripts"; for s in "$@"; do echo "POOL_N=$ps adapt seed $s"; POOL_N=$ps run "$W/adapt_$ps/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local_v3.py "$W/adapt_$ps" 10 $s $((s+1)); done; }
al() { local ps="$1" s; shift; cd "$V3/scripts"; for s in "$@"; do echo "POOL_N=$ps alr seed $s"; POOL_N=$ps run "$W/alr_$ps/al_seed$s.csv" nice -n 10 "$PY" -I al_arm_v3.py "$W/alr_$ps" $s; done; }
( tl 288 0 5; tl 1000 1 2 3 4 ) > "$V3/logs/B12b_laneA.log" 2>&1 &
( tl 1000 6 7 8 9; tl 3000 1 2 ) > "$V3/logs/B12b_laneB.log" 2>&1 &
( tl 3000 3 4 6 7 8 9 ) > "$V3/logs/B12b_laneC.log" 2>&1 &
( ad 288 0; ad 1000 1 2 3 4 5 6 7 8 9 ) > "$V3/logs/B12b_laneD.log" 2>&1 &
( ad 3000 1 2 3 4 5 6 7 8 9 ) > "$V3/logs/B12b_laneE.log" 2>&1 &
( al 288 0; al 1000 1 2 3 4 ) > "$V3/logs/B12b_laneF.log" 2>&1 &
( al 3000 1 2 3 4 ) > "$V3/logs/B12b_laneG.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B12bDONE
