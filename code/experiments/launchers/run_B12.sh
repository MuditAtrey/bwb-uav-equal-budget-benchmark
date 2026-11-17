#!/bin/bash
# B12 (2026-11-15): pool-size sensitivity. Each strategy starts from a random subsample of POOL_N designs of the Phase 1 pool
# (subsample seed 7000+seed, identical across strategies for a given seed) and then spends the same 287 evaluations.
#   TuRBO (v4) and adaptive local 10% (v3): POOL_N = 300, 1000, 3000, seeds 0-9.  AL-R: POOL_N = 1000, 3000, seeds 0-4.
# Outputs: results_v3/pool_size/<arm>_<N>/. Resumable. usage: bash run_B12.sh
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
tl() { n="$1"; shift; d="$V3/pool_size/turbo_$n"; mkdir -p "$d"; cd "$V3/scripts"; for s in "$@"; do POOL_N=$n run "$d/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$d" $s $((s+1)); done; }
ad() { n="$1"; shift; d="$V3/pool_size/adapt_$n"; mkdir -p "$d"; cd "$V3/scripts"; for s in "$@"; do POOL_N=$n run "$d/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local_v3.py "$d" 10 $s $((s+1)); done; }
al() { n="$1"; shift; d="$V3/pool_size/alr_$n"; mkdir -p "$d"; cd "$V3/scripts"; for s in "$@"; do POOL_N=$n run "$d/al_seed$s.csv" nice -n 10 "$PY" -I al_arm_v3.py "$d" $s; done; }
( for n in 300 1000 3000; do tl $n 0 1 2 3 4; done ) > "$V3/logs/B12_laneA.log" 2>&1 &
( for n in 300 1000 3000; do tl $n 5 6 7 8 9; done ) > "$V3/logs/B12_laneB.log" 2>&1 &
( for n in 300 1000 3000; do ad $n 0 1 2 3 4 5 6 7 8 9; done; al 1000 0 1 2 3 4; al 3000 0 1 2 3 4 ) > "$V3/logs/B12_laneC.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B12DONE
