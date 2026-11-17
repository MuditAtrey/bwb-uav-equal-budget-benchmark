# Energy basis built bottom-up from the 5000 mAh 21700 cell (final, supersedes the 213.9 version)

Cell: Samsung INR21700-50E, 5000 mAh typical (4753 mAh min), 3.6 V nominal, 69 g, 9.8 A continuous (14.7 A pulse).
Cell energy = 5.0 Ah x 3.6 V = 18.0 Wh, 260.9 Wh/kg at cell level.

Pack used in the paper: 6S8P = 48 cells, 864 Wh nominal.
Mass: 48 x 69 g = 3.312 kg cells + 88 g overhead (assumed, not weighed: wiring, strip, insulation, connector) = 3.400 kg.
Usable fraction 0.80 (mission reserve assumption).
Usable specific energy e_u = 864 Wh x 0.80 / 3.400 kg = 203.3 Wh/kg.
Database basis was 160 Wh/kg, so every database range is multiplied by F = 203.3 / 160 = 1.2706. The argmax of the optimisation is unchanged (range is linear in e_u at fixed design).

Plausible range of e_u (all linear in the reported ranges):
- 195 Wh/kg: 3 g/cell wiring and 80 g fixed overhead
- 216 Wh/kg: 85 % usable instead of 80 %

Checks that need the real aircraft (listed as open items in the paper):
- Cruise current about 1.4 A per cell (0.28 C), far below 9.8 A. Takeoff peak current must be checked against the 14.7 A pulse rating per cell (with P = 8 and an 80 A ESC limit the per-cell current is 10 A).
- Pack mass 3.400 kg is a cap in this paper; designs with a larger battery mass need a different pack and are snapped back.
- Replace the 88 g overhead by a weighed pack and rerun make_numbers.py.
