#!/bin/bash
# B15 (2026-11-17). (i) TuRBO (v4) seeds 10-19 on the full pool -> results_v3/turbo_fix_extra (brings TuRBO to n = 20, as AL-R and adapt10).
# (ii) Double budget: the first 287 evaluations of TuRBO seeds 0-9, adapt10 seeds 0-9 and AL-R seeds 0-4 are copied to results_v3/long_budget
#      and each run is continued to 574 evaluations with N_EXTRA=574. Resumable.
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"; W="$V3/long_budget"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2; cd "$V3/scripts"
run() { local tgt="$1" f="$2" cnt prev=-1; shift 2
  while true; do cnt=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$cnt" -ge "$tgt" ] && break
    [ "$cnt" -eq "$prev" ] && { echo "stalled $f at $cnt"; break; }; prev=$cnt; "$@"; done; echo "$(date +%H:%M) done $f"; }
tx() { local s; for s in "$@"; do run 288 "$V3/turbo_fix_extra/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$V3/turbo_fix_extra" $s $((s+1)); done; }
tl() { local s; for s in "$@"; do N_EXTRA=574 run 575 "$W/turbo/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$W/turbo" $s $((s+1)); done; }
ad() { local s; for s in "$@"; do N_EXTRA=574 run 575 "$W/adapt/adapt10_seed$s.csv" nice -n 10 "$PY" -I adaptive_local_v3.py "$W/adapt" 10 $s $((s+1)); done; }
al() { local s; for s in "$@"; do N_EXTRA=574 run 575 "$W/alr/al_seed$s.csv" nice -n 10 "$PY" -I al_arm_v3.py "$W/alr" $s; done; }
( tx 10 11 12 13 14 ) > "$V3/logs/B15_laneA.log" 2>&1 &
( tx 15 16 17 18 19 ) > "$V3/logs/B15_laneB.log" 2>&1 &
( tl 0 1 2 3 4 ) > "$V3/logs/B15_laneC.log" 2>&1 &
( tl 5 6 7 8 9 ) > "$V3/logs/B15_laneD.log" 2>&1 &
( ad 0 1 2 3 4 5 6 7 8 9 ) > "$V3/logs/B15_laneE.log" 2>&1 &
( al 0 1 2 ) > "$V3/logs/B15_laneF.log" 2>&1 &
( al 3 4 ) > "$V3/logs/B15_laneG.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B15DONE
