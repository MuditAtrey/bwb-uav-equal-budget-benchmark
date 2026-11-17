"""Bottom-up 21700 pack model from the Samsung INR21700-50E cell (5000 mAh, 3.6 V nominal, 69 g, 9.8 A continuous).
Assumed (NOT datasheet, replace by a weighed pack): 1.5 g/cell for nickel strip, insulation and wrap; 50 g fixed for leads, connector, balance lead."""
import numpy as np
CELL_AH, CELL_V, CELL_G = 5.0, 3.6, 69.0
OVH_CELL_G, OVH_FIXED_G, S, DOD = 1.5, 50.0, 6, 0.85
WH_CELL = CELL_AH * CELL_V
def pack(P, ovh_cell=OVH_CELL_G, ovh_fixed=OVH_FIXED_G):
    n = S * P; m = (n * (CELL_G + ovh_cell) + ovh_fixed) / 1000.0; e = n * WH_CELL
    return dict(P=P, cells=n, V_nom=S * CELL_V, Ah=P * CELL_AH, Wh_nom=e, mass_kg=m, Wh_kg_nom=e / m, Wh_kg_usable=DOD * e / m,
                I_cell_at_80A=80.0 / P)
def usable_wh_per_kg(m_kg, ovh_cell=OVH_CELL_G, ovh_fixed=OVH_FIXED_G):
    """Continuous-mass version used when the optimiser treats pack mass as a real variable (cells = (m - fixed)/per-cell mass)."""
    n = (m_kg * 1000 - ovh_fixed) / (CELL_G + ovh_cell); return DOD * n * WH_CELL / m_kg
if __name__ == "__main__":
    print("cell: %.1f Wh, %.1f Wh/kg" % (WH_CELL, WH_CELL / CELL_G * 1000))
    for P in range(3, 10):
        r = pack(P); print({k: round(v, 3) for k, v in r.items()})
    print("sensitivity 6S8P usable Wh/kg vs overhead (g/cell, fixed g):")
    for oc, of in [(0, 0), (1.5, 50), (3, 80), (5, 120)]:
        print(oc, of, round(pack(8, oc, of)["mass_kg"], 3), round(pack(8, oc, of)["Wh_kg_usable"], 1))
    print("continuous e(m):", {m: round(usable_wh_per_kg(m), 1) for m in (1.5, 2.0, 2.5, 3.0, 3.43, 3.5)})
    e8 = pack(8)["Wh_kg_usable"]
    for name, r160 in [("paper champion", 298.7), ("AL replication mean (3 complete seeds)", 305.23), ("local search 10 pct", 288.6), ("local search 5 pct", 286.3)]:
        print(name, r160, "->", round(r160 * e8 / 160, 1), "km at %.1f Wh/kg;" % e8, round(r160 * 208.25 / 160, 1), "at 208.25")
