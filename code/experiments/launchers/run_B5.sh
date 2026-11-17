#!/bin/bash
# B5: range cost of each bound ("shadow prices"). TuRBO (turbo_arm_v3.py) with ONE bound widened by 10% of its range, seeds 0-4, 287 evaluations each.
#   taper lower bound   0.55 -> 0.525     winglet twist upper bound  0 -> +1 deg
#   battery mass upper  3.5 -> 3.7 kg     cruise speed lower bound   36.11 -> 33.61 m/s (130 km/h is a mission requirement: report as a requirement trade)
# c_blend lower bound is NOT relaxed (geometry clip hard-codes c_blend >= 0.25).
# Resumable. usage: bash run_B5.sh [NPROC=2]
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"
PY="$REPO/.venv/bin/python3"; export DRONE_ROOT="$REPO" TBUDGET=100000 NPROC="${1:-2}"
cd "$V3/scripts"
run() { f="$1"; shift; prev=-1
  while true; do n=$(wc -l < "$f" 2>/dev/null || echo 0); [ "$n" -ge 288 ] && break
    [ "$n" -eq "$prev" ] && { echo "stalled $f at $n"; break; }; prev=$n; "$@"; done; echo "$(date +%H:%M) done $f"; }
declare -a NAMES=(taper_lo wtwist_hi bmass_hi speed_lo)
declare -a OVR=('{"taper_ratio":[0.525,0.8]}' '{"winglet_twist":[-10.0,1.0]}' '{"battery_mass":[1.5,3.7]}' '{"cruise_speed_ms":[33.61,61.11]}')
for i in 0 1 2 3; do
  for s in 0 1 2 3 4; do
    BOUNDS_OVERRIDE="${OVR[$i]}" ARM_TAG="turbo_${NAMES[$i]}" run "$V3/bound_prices/${NAMES[$i]}/turbo_seed$s.csv" nice -n 10 "$PY" -I turbo_arm_v3.py "$V3/bound_prices/${NAMES[$i]}" $s $((s+1))
  done
done
echo B5DONE
