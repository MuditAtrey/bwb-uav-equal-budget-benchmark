"""Battery-balance solver (the 'balance loop' the preprint described but the code did not implement).
For a design, the pitching moment about the CG and the static margin both move linearly with the CG at fixed cruise CL:
   Cm(x_cg) = Cm_np - CL*SM,   SM = (x_np - x_cg)/mac,   x_cg shift = m_b * c_root * d(fb) / m_total.
The solver places the battery so that the aircraft trims with zero pitching moment (no elevon deflection, no trim drag),
SM* = SM0 + Cm0/CL, clipped to a margin window [SM_LO, SM_HI] and to the physical bay limits of the pack, then re-evaluates
the design with the full pipeline at that battery position (so every check is applied, not a linear estimate).
SM_HI = 0.195 keeps the Euler-corrected margin (VLM x 1.28, the largest ratio found by CFD) below the 0.25 limit.
Pack: 6S8P of 21700 cells, 0.13 m along the chord; payload bay depth 0.12 m from the nose; so
   fb in [(0.12 + 0.065)/c_root, (c_root - 0.065)/c_root]."""
import os, sys
ROOT = os.environ.get("DRONE_ROOT2", "/tmp/work2"); sys.path.insert(0, ROOT)
import numpy as np
import flying_wing_optimizer.config as C, flying_wing_optimizer.geometry as G
from flying_wing_optimizer.main import evaluate_design_vector
FB0, SM_LO, SM_HI, L_PACK, NOSE = 0.82, 0.06, 0.195, 0.13, 0.12

def setfb(fb):
    object.__setattr__(C.MASS_PLACEMENT, "battery_centroid_fraction", float(fb))
    object.__setattr__(G.MASS_PLACEMENT, "battery_centroid_fraction", float(fb))

def fb_limits(c_root):
    return (NOSE + L_PACK / 2) / c_root, (c_root - L_PACK / 2) / c_root

def target_fb(c_root, m_b, m_tot, SM0, Cm0, CL, x_np, x_cg, fb0=FB0):
    """Return (fb, SM_target) or None if the first evaluation did not reach the stability analysis."""
    if not (x_np > 0 and CL > 0 and abs(SM0) > 1e-9):
        return None
    mac = (x_np - x_cg) / SM0
    if not (mac > 0.05):
        return None
    sm_t = float(np.clip(SM0 + Cm0 / CL, SM_LO, SM_HI))
    fb = fb0 + (SM0 - sm_t) * m_tot * mac / (m_b * c_root)
    lo, hi = fb_limits(c_root)
    return float(np.clip(fb, lo, hi)), sm_t

def total_mass(vec):
    """Total mass from the same mass model as the pipeline (needed for designs the pipeline rejects, whose mass it does not report)."""
    import flying_wing_optimizer.geometry as GG
    kw = dict(zip(["sweep_inner","sweep_outer","y_blend","c_root","c_blend","taper","theta0","theta1","theta2","tip_twist","winglet_sweep_deg","winglet_twist_deg","battery_mass","cruise_speed_ms"], vec))
    try:
        ap = GG.build_wing_geometry(**{k: float(kw[k]) for k in kw if k != "cruise_speed_ms"})
        ap = ap[0] if isinstance(ap, tuple) else ap
        S = float(ap.S_ref)
    except Exception:
        return None
    M = C.MASS
    return M.payload_mass + (1.0 + 1.5 * S) + M.propulsion_mass + M.avionics_mass + float(vec[12])

def evaluate_at(vec, fb, tag="bal"):
    setfb(fb); r = evaluate_design_vector(list(vec), 0, tag); setfb(FB0); return r

def balanced(vec, tag="bal"):
    """Two full evaluations: at the nominal position, then at the trimmed position. Returns (r0, r1, fb)."""
    r0 = evaluate_at(vec, FB0, tag)
    mt = r0["m_total_kg"] if r0.get("m_total_kg", 0) > 0 else (total_mass(vec) or 0.0)
    t = target_fb(vec[3], vec[12], mt, r0["SM"], r0["Cm_trim"], r0["CL"], r0["x_np"], r0["x_cg"])
    if t is None:
        return r0, r0, FB0
    r1 = evaluate_at(vec, t[0], tag)
    return r0, r1, t[0]
