# 3D Euler CFD vs VLM (SU2 8.1.0, Gmsh 4.15.2, half model, Roe + MUSCL + Venkat)

Setup: inviscid, symmetry plane at y=0, tet mesh with surface size h = 7 mm (118k to 127k nodes),
cruise AoA and AoA+2 deg, Mach from the design cruise speed. Designs are the ones in the Phase 1 / AL database.
Viscous effects are NOT modelled, so drag is not validated here.

| Design | CL cfd / vlm | SM cfd / vlm | SM ratio | CL slope (1/deg) |
|---|---|---|---|---|
| AL champion | 0.2478 / 0.2561 | 0.221 / 0.183 | 1.21 | 0.0657 |
| P1 best | 0.2168 / 0.2203 | 0.228 / 0.195 | 1.17 | 0.0629 |
| P1 median | 0.1216 / 0.1142 | 0.132 / 0.104 | 1.28 | 0.0645 |
| P1 p10 | 0.0827 / 0.0693 | 0.175 / 0.146 | 1.20 | 0.0639 |

Spearman rank agreement over the 4 designs: CL 1.0, SM 1.0 (n=4, weak evidence on its own).

Mesh sensitivity (champion): h=10 mm gives CL 0.262, SM 0.230, CD +0.0051; h=7 mm gives CL 0.248, SM 0.221, CD -0.0016.
Euler drag is mesh dominated (numerical dissipation of the order of the induced drag) and goes negative, so it is not reported.
The h=5 mm champion run was not completed (batch hit its time limit).

Reading: the VLM lift is within about 3 to 6 percent (19 percent for the lowest-lift design, where CL is small), and the VLM
underestimates the static margin by 17 to 28 percent, consistently across designs. The champion margin in CFD is about 0.22, close to the 0.25 upper constraint.
