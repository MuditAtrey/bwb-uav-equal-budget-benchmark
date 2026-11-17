#!/bin/bash
# B11 (2026-11-15): repeats B5 (single-bound prices, seeds 0-4) and B2 (TuRBO + sigmoid weight, seeds 0-9) with the basis-corrected TuRBO (turbo_arm_v4.py).
# Outputs: results_v3/bound_prices_fix/<bound>/, results_v3/turbo_feas_fix/. Resumable. usage: bash run_B11.sh
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"; PY="$REPO/.venv/bin/python3"
export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC=2
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
bp() { name="$1"; ovr="$2"; mkdir -p "$V3/bound_prices_fix/$name"; cd "$V3/scripts"
  for s in 0 1 2 3 4; do BOUNDS_OVERRIDE="$ovr" ARM_TAG="turbo_$name" run "$V3/bound_prices_fix/$name/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$V3/bound_prices_fix/$name" $s $((s+1)); done; }
tf() { mkdir -p "$V3/turbo_feas_fix"; cd "$V3/scripts"; for s in "$@"; do TURBO_FEAS=1 run "$V3/turbo_feas_fix/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v4.py "$V3/turbo_feas_fix" $s $((s+1)); done; }
( bp taper_lo '{"taper_ratio":[0.525,0.8]}'; tf 0 1 2 ) > "$V3/logs/B11_laneA.log" 2>&1 &
( bp wtwist_hi '{"winglet_twist":[-10.0,1.0]}'; tf 3 4 5 ) > "$V3/logs/B11_laneB.log" 2>&1 &
( bp bmass_hi '{"battery_mass":[1.5,3.7]}'; tf 6 7 ) > "$V3/logs/B11_laneC.log" 2>&1 &
( bp speed_lo '{"cruise_speed_ms":[33.61,61.11]}'; tf 8 9 ) > "$V3/logs/B11_laneD.log" 2>&1 &
if command -v caffeinate >/dev/null; then caffeinate -i -w $$ & fi
wait; echo B11DONE
