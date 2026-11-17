#!/bin/bash
# B4 step 3: Euler CFD (h = 7 mm) for the 15 selected designs, NPAR cases at a time (SU2 binary is x86 under Rosetta, serial).
# Resumable: cfd_run_v3.py skips cases whose result.csv exists. usage: bash run_B4_cfd.sh [NPAR]
V3="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$V3/../.." && pwd)"
export DRONE_ROOT="$REPO" SU2_CFD="$REPO/scratch/su2/bin/SU2_CFD" DESIGNS_JSON="$V3/cfd/designs.json" NITER=800
NP="${1:-2}"
"$REPO/.venv/bin/python3" -c "
import pandas as pd
s = pd.read_csv('$V3/cfd/cfd_selection.csv'); s = s[~s.name.str.contains(r'_b\d|topB')]
print('\n'.join(s.name))" > "$V3/cfd/queue.txt"
while read -r n; do
  while [ "$(jobs -r | wc -l)" -ge "$NP" ]; do sleep 5; done
  ( echo "$(date +%H:%M:%S) start $n"
    nice -n 5 "$REPO/.venv/bin/python3" -I "$V3/scripts/cfd_run_v3.py" "$n" "$V3/cfd/${n}_h7" 7 > "$V3/cfd/${n}_h7.log" 2>&1
    echo "$(date +%H:%M:%S) done $n rc=$?" ) &
done < "$V3/cfd/queue.txt"
wait; echo B4QUEUEDONE
