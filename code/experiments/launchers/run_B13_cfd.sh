#!/bin/bash
# B13 (2026-11-15): out-of-sample test of the static-margin calibration. Euler CFD (h = 7 mm, same setup as B4) on 12 best-of-run designs
# from runs made AFTER the calibration was fitted (corrected TuRBO seeds 0-5, AL-R seeds 10-12, corrected adapt10 seeds 10-12).
# Kept in cfd_oos/ so that b4_calibrate.py (which globs cfd/) does not refit on them. Resumable. usage: bash run_B13_cfd.sh [NPAR]
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"
export DRONE_ROOT="$REPO" SU2_CFD="$REPO/scratch/su2/bin/SU2_CFD" DESIGNS_JSON="$V3/cfd_oos/designs.json" NITER=800
NP="${1:-2}"
tail -n +2 "$V3/cfd_oos/cfd_selection.csv" | cut -d, -f1 > "$V3/cfd_oos/queue.txt"
while read -r n; do
  while [ "$(jobs -r | wc -l)" -ge "$NP" ]; do sleep 5; done
  ( echo "$(date +%H:%M:%S) start $n"
    nice -n 5 "$REPO/.venv/bin/python3" -I "$V3/scripts/cfd_run_v3.py" "$n" "$V3/cfd_oos/${n}_h7" 7 > "$V3/cfd_oos/${n}_h7.log" 2>&1
    echo "$(date +%H:%M:%S) done $n rc=$?" ) &
done < "$V3/cfd_oos/queue.txt"
wait; echo B13DONE
