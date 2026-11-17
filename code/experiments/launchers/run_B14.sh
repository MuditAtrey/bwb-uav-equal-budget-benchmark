#!/bin/bash
# B14 (2026-11-16): independent pools. Ten fresh 1,000-design Latin hypercube pools (LHS seed 9000+s), then TuRBO (v4) and adapt10 (v3)
# seeds 0-9, each from its own pool, 287 evaluations. Outputs: results_v3/indep_pool/{pools,turbo,adapt}. Resumable.
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"; W="$V3/indep_pool"
export DRONE_ROOT="$REPO" TBUDGET=100000; mkdir -p "$W/pools" "$W/turbo" "$W/adapt"; cd "$V3/scripts"
NPROC=9 nice -n 10 "$PY" -I indep_pool.py "$W/pools" 1000 0 10 > "$V3/logs/B14_pools.log" 2>&1
export NPROC=2 POOL_FILE="$W/pools/pool_seed{seed}.csv"
run() { local f="$1" cnt prev=-1; shift
  while true; do cnt=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$cnt" -ge 288 ] && break
    [ "$cnt" -eq "$prev" ] && { echo "stalled $f at $cnt"; break; }; prev=$cnt; "$@"; done; echo "$(date +%H:%M) done $f"; }
tl() { local s; for s in "$@"; do run "$W/turbo/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$W/turbo" $s $((s+1)); done; }
ad() { local s; for s in "$@"; do run "$W/adapt/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local_v3.py "$W/adapt" 10 $s $((s+1)); done; }
( tl 0 1 2 ) > "$V3/logs/B14_laneA.log" 2>&1 &
( tl 3 4 5 ) > "$V3/logs/B14_laneB.log" 2>&1 &
( tl 6 7 ) > "$V3/logs/B14_laneC.log" 2>&1 &
( tl 8 9 ) > "$V3/logs/B14_laneD.log" 2>&1 &
( ad 0 1 2 3 4 5 6 7 8 9 ) > "$V3/logs/B14_laneE.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B14DONE
