"""Figure 2: the planform exactly as the geometry builder lofts it (cubic leading edge and straight trailing edge over the center body,
linear outer taper, winglet junction), drawn from build_wing_geometry for the best design of the original campaign.
usage: python3 make_fig_planform.py"""
import os, sys, warnings; warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(f"{HERE}/../../../.."); sys.path.insert(0, REPO)
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from flying_wing_optimizer.geometry import build_wing_geometry
db = pd.read_csv(f"{REPO}/flying_wing_optimizer/results/database.csv", low_memory=False); v = db[db.is_valid.astype(bool)]; r = v.loc[v.range_calm_km.idxmax()]
K = ["sweep_inner", "sweep_outer", "y_blend", "c_root", "c_blend", "taper", "theta0", "theta1", "theta2", "tip_twist"]
ap, _ = build_wing_geometry(*[float(r[k]) for k in K], battery_mass=float(r.battery_mass), winglet_sweep_deg=float(r.winglet_sweep_deg), winglet_twist_deg=float(r.winglet_twist_deg))
xs = ap.wings[0].xsecs; y = np.array([float(s.xyz_le[1]) for s in xs]); le = np.array([float(s.xyz_le[0]) for s in xs]); te = le + np.array([float(s.chord) for s in xs])
yb, cr, cb = float(r.y_blend), float(r.c_root), float(r.c_blend); nm = y <= 0.70 + 1e-9
# dense center-body curves from the closed form (Eq. le_te) to show the cubic
t = np.tan(np.radians(float(r.sweep_inner))); xi = np.linspace(0, 1, 100); yl = xi * yb; xle = t * yb * (2 * xi**2 - xi**3); xte = cr + (t * yb + cb - cr) * xi
sys.path.insert(0, HERE); import figstyle as _fs
INK, BL, MUT, OR = _fs.K, _fs.NAVY, _fs.GREY, _fs.CRIMSON   # color scheme of the trim-sensitivity figure (2026-11-16)
_fs.apply(8); plt.rcParams.update({"axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "xtick.top": False, "ytick.right": False})
fig, ax = plt.subplots(figsize=(6.2, 3.6))
Y = np.r_[yl, y[(y > yb) & nm]]; LE = np.r_[xle, le[(y > yb) & nm]]; TE = np.r_[xte, te[(y > yb) & nm]]
ax.fill_between(Y, LE, TE, color=BL, alpha=0.10, lw=0); ax.plot(Y, LE, color=BL); ax.plot(Y, TE, color=BL)
ax.plot(y[~nm | (y >= 0.70 - 1e-9)], le[~nm | (y >= 0.70 - 1e-9)], color=BL, ls="--", lw=1); ax.plot(y[~nm | (y >= 0.70 - 1e-9)], te[~nm | (y >= 0.70 - 1e-9)], color=BL, ls="--", lw=1)
ax.plot([0, 0], [0, cr], color=OR, lw=2); ax.plot([yb, yb], [np.interp(yb, Y, LE), np.interp(yb, Y, TE)], color=MUT, ls=":", lw=1)
ct = te[nm][-1] - le[nm][-1]; ax.plot([0.70, 0.70], [le[nm][-1], te[nm][-1]], color=MUT, ls=":", lw=1)
ax.plot([0, 0], [cr, cr + 0.19], color=MUT, lw=3, alpha=0.5); ax.text(0.015, cr + 0.15, "pusher nacelle (not in the lattice)", fontsize=6.5, color=MUT, va="center")
ax.text(-0.012, cr / 2, r"$c_r$", ha="right", va="center", color=OR); ax.text(yb - 0.008, np.interp(yb, Y, LE) + cb / 2, r"$c_b$", va="center", ha="right", color=MUT)
ax.text(0.692, le[nm][-1] + ct / 2, r"$c_t=\lambda c_b$", va="center", ha="right", color=MUT)
ar = dict(arrowstyle="-", color=MUT, lw=0.5)
ax.annotate(r"cubic leading edge, zero slope at $y=0$," + "\n" + r"slope $\tan\Lambda_\mathrm{in}$ at $y_b$", xy=(yb * 0.45, np.interp(yb * 0.45, Y, LE)), xytext=(0.22, 0.0), fontsize=7, color=INK, arrowprops=ar, va="center")
ax.annotate("straight trailing edge", xy=(yb * 0.55, np.interp(yb * 0.55, Y, TE)), xytext=(0.06, 0.54), fontsize=7, color=INK, arrowprops=ar, va="center")
ax.annotate(r"straight leading edge, sweep $\Lambda_\mathrm{out}$;" + "\nlinear taper", xy=(0.52, np.interp(0.52, Y, LE)), xytext=(0.56, 0.22), fontsize=7, color=INK, arrowprops=ar, va="center")
ax.text(yb, np.interp(yb, Y, TE) + 0.035, r"$y_b$", ha="center", color=MUT); ax.text(0.02, 0.27, "MH91", fontsize=7, color=OR); ax.text(0.45, np.interp(0.45, Y, LE) + 0.09, "MH60", fontsize=7, color=MUT)
ax.text(0.80, le[-1] - 0.03, "winglet\n(projected)", fontsize=6.5, color=MUT, ha="center", va="bottom")
ax.set_xlabel("Spanwise coordinate $y$ (m)"); ax.set_ylabel("Chordwise coordinate $x$ (m)"); ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xlim(-0.06, 0.86)
fig.tight_layout(); _fs.save(fig, f"{HERE}/figures/figure2_geometric_manifold.pdf"); fig.savefig(f"{HERE}/figures/figure2_planform_preview.png", dpi=140)
