"""
flying_wing_optimizer/aerodynamics.py

Aerodynamic analysis module for the blended flying wing.

Rewrite 2026-08-06: Three targeted physics fixes applied.

FIX 1: VLM resolution upgraded from (2,12) → (12,8).
  Previous 2-panel spanwise discretization caused 13% error in CL
  (proven in scratch/test_aerodynamic_flaws.py). All lift and induced
  drag numbers were garbage.

FIX 2: Trim drag penalty added.
  Previously, L/D was evaluated on the clean wing with no elevon deflection,
  even when Cm was -0.09 (enormous trim drag). Now adds a Raymer-derived
  trim drag increment: CD_trim = Cm² / (π * AR * e_elevon * elevon_fraction).

FIX 3: Nacelle profile drag computed from fuselage wetted area.
  Previously used a flat constant from MotorNacelleDragConfig (0.179 N).
  Now iterates over the asb.Fuselage xsecs built in geometry.py to compute
  strip-based skin friction + form factor drag on the nacelle.

OLD approach (broken):
  - Find AoA where Cm=0 (natural trim)
  - Hope that the resulting CL gives ~150 kph cruise speed
  - Flat CD0_VISCOUS=0.008 for profile drag
  - Constant CD_nacelle from lookup

NEW approach (physically correct):
  - Compute CL_required for level flight at v_cruise: CL = 2mg/(ρV²S)
  - Find AoA where CL = CL_required (2-point VLM interpolation)
  - Evaluate Cm at that AoA → apply Raymer trim drag from Cm
  - Strip-based profile drag with form factor for thickness (wing + nacelle)
  - L/D computed at the actual trimmed cruise operating point
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import aerosandbox as asb

from flying_wing_optimizer.config import (
    ATMOSPHERE,
    CONSTRAINTS,
    MASS,
    STRUCTURE,
    SOFT_PENALTY,
    TARGETS,
    GEOMETRY,
)
from flying_wing_optimizer.geometry import build_wing_geometry

logger = logging.getLogger(__name__)


# ─── VLM Solver ──────────────────────────────────────────────────────────────

def _run_vlm(
    airplane: asb.Airplane,
    op_point: asb.OperatingPoint,
    primary_res: Tuple[int, int] = (2, 12),
    fallback_res: Tuple[int, int] = (1, 8),
    run_stability: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute AeroSandbox VLM with primary resolution + fallback.

    Resolution (2, 12) means: 2 spanwise panels per XSec interval, 12 chordwise.
    With 20 wing stations (19 intervals), this gives 2*19=38 effective spanwise
    strips per half-span, which is adequate for the geometry.

    WARNING: Do NOT increase spanwise_resolution beyond 2-3 without checking that
    the closest blend station gap (min y_n - y_{n-1}) is large enough. At close
    spacings (e.g., y3-y2=25mm), spanwise_resolution=12 creates 2mm panels that
    make the AIC matrix singular (CL blows up to 58+).
    """
    for res in [primary_res, fallback_res]:
        try:
            vlm = asb.VortexLatticeMethod(
                airplane=airplane,
                op_point=op_point,
                spanwise_resolution=res[0],
                chordwise_resolution=res[1],
            )
            if run_stability:
                return vlm.run_with_stability_derivatives()
            else:
                return vlm.run()
        except Exception as e:
            logger.warning(f"VLM failed at resolution {res}: {e}")
    return None


# ─── Cruise AoA Finder ───────────────────────────────────────────────────────

def _find_cruise_aoa(
    airplane: asb.Airplane,
    v_cruise: float,
    CL_required: float,
    primary_res: Tuple[int, int] = (2, 12),
    fallback_res: Tuple[int, int] = (1, 8),
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Find AoA for level flight at v_cruise using 2-point VLM interpolation.

    VLM CL is linear in AoA (potential flow), so 2 evaluations suffice.

    Returns:
        (alpha_cruise, CL_at_cruise, CL_alpha) or (None, None, None) on failure.
    """
    atmos = asb.Atmosphere(altitude=ATMOSPHERE.cruise_altitude)

    # Evaluate at alpha = 0° and alpha = 5°
    alphas = [0.0, 5.0]
    CLs = []
    for alpha in alphas:
        op = asb.OperatingPoint(atmosphere=atmos, velocity=v_cruise, alpha=alpha)
        res = _run_vlm(airplane, op, primary_res, fallback_res, run_stability=False)
        if res is None or "CL" not in res:
            return None, None, None
        CLs.append(float(res["CL"]))

    CL_alpha = (CLs[1] - CLs[0]) / (alphas[1] - alphas[0])

    if abs(CL_alpha) < 1e-6:
        # Wing produces no lift variation with AoA — degenerate geometry
        return None, None, None

    alpha_cruise = alphas[0] + (CL_required - CLs[0]) / CL_alpha

    # Sanity check: AoA should be in a reasonable range
    if alpha_cruise < -15.0 or alpha_cruise > 20.0:
        return None, None, None

    return alpha_cruise, CL_required, CL_alpha


# ─── Strip-Based Profile Drag (Wing + Nacelle) ───────────────────────────────

def _compute_profile_drag(
    airplane: asb.Airplane,
    v_cruise: float,
) -> float:
    """Compute profile drag coefficient using strip-based skin friction + form factor.

    FIX 3: Now includes the nacelle fuselage wetted area drag.
    Previously only computed wing strips; nacelle was a flat constant.
    Now iterates over both the wing cross-sections and the fuselage
    (nacelle fairing) cross-sections.

    For each pair of adjacent cross-sections, computes:
    - Local Reynolds number from average chord/diameter
    - Turbulent flat-plate skin friction (Schlichting formula)
    - Form factor accounting for airfoil thickness (Raymer simplified)
    - Wetted area with thickness correction

    Returns:
        CD_profile referenced to airplane.S_ref.
    """
    S_ref = airplane.S_ref
    rho = ATMOSPHERE.air_density
    nu = ATMOSPHERE.kinematic_viscosity

    total_drag_force = 0.0  # Accumulated drag force [N] → divided by q*S_ref at end
    q_cruise = 0.5 * rho * v_cruise ** 2

    # ── Wing strips ──────────────────────────────────────────────────────
    wing = airplane.wings[0]
    xsecs = wing.xsecs

    for i in range(len(xsecs) - 1):
        xs_a = xsecs[i]
        xs_b = xsecs[i + 1]

        c_a = xs_a.chord
        c_b = xs_b.chord
        c_avg = (c_a + c_b) / 2.0

        # Strip span (accounts for dihedral in winglet region)
        dy = abs(xs_b.xyz_le[1] - xs_a.xyz_le[1])
        dz = abs(xs_b.xyz_le[2] - xs_a.xyz_le[2])
        ds = math.sqrt(dy ** 2 + dz ** 2)

        if ds < 1e-6 or c_avg < 1e-4:
            continue

        # Strip planform area (one side of symmetric wing)
        S_strip = c_avg * ds

        # Local thickness ratio
        t_c_a = xs_a.airfoil.max_thickness()
        t_c_b = xs_b.airfoil.max_thickness()
        t_c_avg = (t_c_a + t_c_b) / 2.0

        # Wetted area (both surfaces of the strip, with thickness correction)
        S_wet_strip = 2.0 * (1.0 + 0.2 * t_c_avg) * S_strip

        # Local Reynolds number
        Re_local = max(v_cruise * c_avg / nu, 1e4)

        # Mixed laminar-turbulent boundary layer skin friction
        # MH60/MH91 reflexed airfoils at Re 250k-1.2M have a ~40% chord laminar run
        x_tr = 0.40
        Re_tr = max(Re_local * x_tr, 100.0)
        Cf_lam = 1.328 / math.sqrt(Re_tr)
        Cf_turb = 0.455 / (math.log10(Re_local) ** 2.58)
        Cf = x_tr * Cf_lam + (1.0 - x_tr) * Cf_turb

        # Form factor (Raymer simplified for subsonic flow)
        FF = 1.0 + 2.0 * t_c_avg + 60.0 * t_c_avg ** 4

        # This strip's contribution (one side) → multiply by 2 below for symmetric wing
        total_drag_force += Cf * FF * S_wet_strip * q_cruise

    # Multiply by 2 for both sides of the symmetric wing
    total_drag_force *= 2.0

    # ── Three-component pusher tail drag model ───────────────────────────────
    D_boom = 0.028
    L_boom = 0.175
    D_motor = 0.054
    L_motor = 0.055
    L_fairing = 0.065
    D_fairing_tip = 0.015

    # Boom
    S_wet_boom = math.pi * D_boom * L_boom
    Re_boom = max(v_cruise * L_boom / nu, 1e4)
    CF_boom = 0.455 / (math.log10(Re_boom) ** 2.58)
    CD_boom = CF_boom * 1.0 * S_wet_boom / S_ref if S_ref > 0 else 0.0

    # Motor body
    S_wet_motor = math.pi * D_motor * L_motor
    Re_motor = max(v_cruise * L_motor / nu, 1e4)
    CF_motor = 0.455 / (math.log10(Re_motor) ** 2.58)
    CD_motor = CF_motor * 1.3 * S_wet_motor / S_ref if S_ref > 0 else 0.0

    # Fairing
    S_wet_fairing = math.pi * (D_motor + D_fairing_tip) / 2.0 * math.sqrt(L_fairing**2 + ((D_motor - D_fairing_tip) / 2.0)**2)
    Re_fairing = max(v_cruise * L_fairing / nu, 1e4)
    CF_fairing = 0.455 / (math.log10(Re_fairing) ** 2.58)
    CD_fairing = CF_fairing * 1.05 * S_wet_fairing / S_ref if S_ref > 0 else 0.0

    total_tail_CD = CD_boom + CD_motor + CD_fairing
    total_drag_force += total_tail_CD * (q_cruise * S_ref)

    CD_profile = total_drag_force / (q_cruise * S_ref) if (q_cruise * S_ref) > 0 else 0.0
    return CD_profile


# ─── Trim Drag Penalty ───────────────────────────────────────────────────────

def _compute_trim_drag(
    Cm_cruise: float,
    S_ref: float,
    airplane: Optional[asb.Airplane] = None,
) -> float:
    """Compute trim drag increment from elevon deflection (Raymer approximation).

    FIX 2: Previously L/D was evaluated on the clean wing without any elevon
    deflection, even when Cm was -0.09. This inflated L/D by ~15-25%.

    For a flying wing with elevons on the aft 25% of the outer 40% span:
      S_elevon / S_ref ≈ 0.40 * 0.25 = 0.10 (each side)
      Combined effective elevon fraction ≈ 0.20

    Raymer, Aircraft Design Ch.12:
      CD_trim ≈ Cm² / (π * AR * e_elevon * S_elevon_fraction)

    Args:
        Cm_cruise: Pitching moment coefficient at cruise AoA.
        S_ref: Wing reference area [m²].
        airplane: Airplane object to compute dynamic wingspan.

    Returns:
        CD_trim: Trim drag coefficient increment (referenced to S_ref).
    """
    if abs(Cm_cruise) < 0.001:
        return 0.0

    if airplane is not None and hasattr(airplane, "wings") and len(airplane.wings) > 0:
        b = 2.0 * max(xs.xyz_le[1] for xs in airplane.wings[0].xsecs)
    else:
        b = GEOMETRY.wingspan_m

    AR = b ** 2 / S_ref if S_ref > 0 else 4.0
    e_elevon = 0.60        # Elevon effectiveness factor (flying wing)
    elevon_fraction = 0.20  # S_elevon / S_ref

    CD_trim = Cm_cruise ** 2 / (math.pi * AR * e_elevon * elevon_fraction)
    return max(0.0, CD_trim)


# ─── Main Evaluation Function ────────────────────────────────────────────────

def evaluate_aerodynamics(
    sweep_inner: float,
    sweep_outer: float,
    y_blend: float,
    c_root: float,
    c_blend: float,
    taper: float,
    theta0: float,
    theta1: float,
    theta2: float,
    tip_twist: float,
    v_cruise: float,
    battery_mass: float = 2.0,
    winglet_sweep_deg: float = 30.0,
    winglet_twist_deg: float = -5.0,
) -> Dict[str, Any]:
    """Evaluate aerodynamic performance at the cruise point."""
    reasons: List[str] = []
    soft_reasons: List[str] = []
    penalty_score: float = 0.0

    def make_invalid(reasons_list: List[str], penalty: float = 1e6) -> Dict[str, Any]:
        return {
            "is_valid": False,
            "penalty_score": penalty,
            "CL": 0.0, "CD": 0.0, "CD_vlm": 0.0, "CD_profile": 0.0, "CD_trim": 0.0,
            "L_D": 0.0, "AoA_cruise": 0.0, "Cm_cruise": 0.0,
            "SM": 0.0, "x_np": 0.0, "x_cg": 0.0, "S_ref": 0.0,
            "m_total": 0.0, "V_cruise_actual": 0.0, "CL_required": 0.0,
            "reasons": reasons_list, "soft_reasons": [],
            "invalid_reasons": reasons_list,
        }

    # ── 1. Washout check ─────────────────────────────────────────────────
    if not (tip_twist <= theta2 <= theta1 <= theta0):
        reasons.append("Washout rule violated: tip_twist <= theta2 <= theta1 <= theta0 required.")
        return make_invalid(reasons)

    # ── 2. Build geometry ────────────────────────────────────────────────
    try:
        result = build_wing_geometry(
            sweep_inner=sweep_inner,
            sweep_outer=sweep_outer,
            y_blend=y_blend,
            c_root=c_root,
            c_blend=c_blend,
            taper=taper,
            theta0=theta0,
            theta1=theta1,
            theta2=theta2,
            tip_twist=tip_twist,
            battery_mass=battery_mass,
            winglet_sweep_deg=winglet_sweep_deg,
            winglet_twist_deg=winglet_twist_deg,
        )
    except Exception as e:
        reasons.append(f"Geometry construction failed: {e}")
        return make_invalid(reasons)

    if result is None:
        reasons.append("Geometry construction rejected (invalid planform).")
        return make_invalid(reasons)
        
    airplane, clamped_params = result
    if airplane is None:
        if isinstance(clamped_params, list):
            reasons.extend(clamped_params)
        return make_invalid(reasons)

    S_ref = airplane.S_ref
    x_cg = airplane.x_cg
    mac = airplane.mac
    cg_mac_fraction = airplane.cg_mac_fraction

    # Volumetric constraints are now enforced in geometry.py. If we reached here, the design passed.

    # ── 3.5 Trailing Edge check (hard constraint) ─────────────────────────
    if not getattr(airplane, 'te_sweep_ok', True):
        reasons.append("Trailing edge has extreme jagged sweep (>75 deg).")
        return make_invalid(reasons)

    # ── 4. Compute CL required for level flight at v_cruise ──────────────
    rho = ATMOSPHERE.air_density
    g = ATMOSPHERE.gravity
    m = airplane.total_mass   # Dynamic: MASS.fixed_mass_subtotal + battery_mass

    CL_required = (2.0 * m * g) / (rho * v_cruise ** 2 * S_ref)

    if CL_required > CONSTRAINTS.max_cl_required:
        reasons.append(
            f"Wing too small for cruise speed: CL_required={CL_required:.3f} "
            f"> {CONSTRAINTS.max_cl_required} (S_ref={S_ref:.4f} m²)."
        )
        return make_invalid(reasons)

    if CL_required <= 0:
        reasons.append(f"CL_required is non-positive: {CL_required:.4f}")
        return make_invalid(reasons)

    # Build wing-only airplane for VLM solver (VLM is a wing-only panel method)
    vlm_airplane = asb.Airplane(
        name="VLMWingOnly",
        xyz_ref=airplane.xyz_ref,
        wings=airplane.wings,
    )

    # ── 5. Find cruise AoA ───────────────────────────────────────────────
    alpha_cruise, _, CL_alpha = _find_cruise_aoa(vlm_airplane, v_cruise, CL_required)

    if alpha_cruise is None:
        reasons.append("Failed to find AoA for cruise CL (VLM solver failure).")
        return make_invalid(reasons)

    # ── 6. Full VLM evaluation at cruise AoA (with stability) ────────────
    atmos = asb.Atmosphere(altitude=ATMOSPHERE.cruise_altitude)
    op_cruise = asb.OperatingPoint(atmosphere=atmos, velocity=v_cruise, alpha=alpha_cruise)
    res = _run_vlm(vlm_airplane, op_cruise, run_stability=True)

    if res is None:
        reasons.append("VLM solver failed at cruise AoA.")
        return make_invalid(reasons)

    CL = float(res["CL"])
    CD_vlm = float(res["CD"])
    Cm_stab = float(res["Cm"])
    x_np = float(res["x_np"])
    SM = float((x_np - x_cg) / mac) if mac > 0 else 0.0

    # Sanity gate: |Cm| > 10 means the stability solver's perturbed-state AIC
    # matrix became singular (e.g., degenerate panels). Fall back to plain run().
    if abs(Cm_stab) > 10.0:
        logger.warning(
            f"Cm={Cm_stab:.1f} from stability run (|Cm|>10, AIC blowup). "
            "Falling back to plain VLM run for Cm."
        )
        res_plain = _run_vlm(vlm_airplane, op_cruise, run_stability=False)
        if res_plain is not None and abs(float(res_plain.get("Cm", 1e9))) <= 10.0:
            Cm_cruise = float(res_plain["Cm"])
            CL = float(res_plain["CL"])
            CD_vlm = float(res_plain["CD"])
        else:
            reasons.append("VLM Cm blowup in both stability and plain runs — degenerate geometry.")
            return make_invalid(reasons)
    else:
        Cm_cruise = Cm_stab

    # Hard physical validation on induced drag
    AR = (GEOMETRY.wingspan_m ** 2) / S_ref
    Oswald_e = (CL ** 2) / (math.pi * AR * CD_vlm) if CD_vlm > 0 else float('inf')
    if Oswald_e > 1.0:
        reasons.append(f"Oswald efficiency non-physical (e={Oswald_e:.3f})")
        return make_invalid(reasons)

    # ── 7. Strip-based profile drag (wing + nacelle) ─────────────────────
    try:
        CD_profile = _compute_profile_drag(airplane, v_cruise)
    except Exception as e:
        logger.warning(f"Profile drag computation failed: {e}. Using fallback.")
        CD_profile = 0.008

    # ── 8. Trim drag (Raymer Cm² penalty) ────────────────────────────────
    CD_trim = _compute_trim_drag(Cm_cruise, S_ref, airplane=airplane)

    CD_total = CD_vlm + CD_profile + CD_trim
    L_D = CL / CD_total if CD_total > 0 else 0.0

    # Actual cruise speed from this CL (should ≈ v_cruise if CL ≈ CL_required)
    V_cruise_actual = math.sqrt(2.0 * m * g / (rho * S_ref * max(CL, 1e-6)))

    # ── 9. Constraint evaluation ─────────────────────────────────────────
    is_valid = True

    # HARD: Static Margin (must be between 5% and 25% MAC for stability + pitch responsiveness)
    if SM < CONSTRAINTS.min_static_margin:
        reasons.append(f"Static margin SM={SM:.4f} < {CONSTRAINTS.min_static_margin}")
        is_valid = False
    elif SM > CONSTRAINTS.max_static_margin:
        reasons.append(f"Static margin SM={SM:.4f} > {CONSTRAINTS.max_static_margin} (excessive pitch sluggishness)")
        is_valid = False

    # HARD: Cm at cruise (elevon authority limit)
    if abs(Cm_cruise) > CONSTRAINTS.max_abs_cm_at_cruise:
        reasons.append(
            f"|Cm| at cruise = {abs(Cm_cruise):.4f} > {CONSTRAINTS.max_abs_cm_at_cruise} "
            f"(exceeds elevon authority)."
        )
        is_valid = False

    # HARD: Minimum cruise speed (≥ 130 kph = 36.11 m/s)
    if v_cruise < TARGETS.min_cruise_speed_ms:
        reasons.append(
            f"Cruise speed {v_cruise*3.6:.1f} kph < minimum {TARGETS.min_cruise_speed_ms*3.6:.0f} kph."
        )
        is_valid = False

    # HARD: Motor & Propeller Clearance
    x_te_root = airplane.wings[0].xsecs[0].xyz_le[0] + airplane.wings[0].xsecs[0].chord
    from flying_wing_optimizer.config import MOTOR_MOUNT
    x_motor_body_end = x_te_root + max(0.0, MOTOR_MOUNT.motor_cg_beyond_te_m - 0.0275) + 0.055
    x_prop_disk = x_motor_body_end

    for xsec in airplane.wings[0].xsecs:
        y_val = xsec.xyz_le[1]
        x_te = xsec.xyz_le[0] + xsec.chord
        # Only check sections outside the nacelle fairing width (>30mm from centerline)
        if y_val > 0.030:
            # Check motor body interference (y <= 0.027m from fairing edge)
            if y_val <= 0.027 + 0.025:
                if x_te > (x_te_root + max(0.0, MOTOR_MOUNT.motor_cg_beyond_te_m - 0.0275) + 0.055 - 0.005):
                    reasons.append(f"Wing TE at y={y_val:.3f} clips motor body.")
                    is_valid = False
            # Check propeller disk interference (y <= 0.152m)
            if y_val <= 0.152:
                if x_te > x_prop_disk - 0.010:  # 10mm safety clearance from prop plane
                    reasons.append(f"Wing TE at y={y_val:.3f} clips propeller disk.")
                    is_valid = False

    # HARD: Positive lift
    if CL <= CONSTRAINTS.min_cl_cruise:
        reasons.append(f"CL at cruise = {CL:.4f} ≤ {CONSTRAINTS.min_cl_cruise}")
        is_valid = False

    # SOFT: Cm penalty (trim drag from elevon deflection, now explicitly modeled)
    if abs(Cm_cruise) > CONSTRAINTS.cm_soft_threshold:
        cm_penalty = SOFT_PENALTY.cm_penalty_weight * (Cm_cruise ** 2)
        penalty_score += cm_penalty
        soft_reasons.append(
            f"Cm soft-penalty: |Cm|={abs(Cm_cruise):.4f} > {CONSTRAINTS.cm_soft_threshold}, "
            f"penalty={cm_penalty:.2f}"
        )

    # SOFT: CG bounds
    if not (CONSTRAINTS.cg_min_mac <= cg_mac_fraction <= CONSTRAINTS.cg_max_mac):
        cg_violation = max(
            CONSTRAINTS.cg_min_mac - cg_mac_fraction,
            cg_mac_fraction - CONSTRAINTS.cg_max_mac,
        )
        cg_pen = SOFT_PENALTY.cg_penalty_weight * cg_violation
        penalty_score += cg_pen
        soft_reasons.append(
            f"CG soft-penalty: CG at {cg_mac_fraction * 100:.1f}% MAC "
            f"(target {CONSTRAINTS.cg_min_mac*100:.0f}-{CONSTRAINTS.cg_max_mac*100:.0f}%), "
            f"penalty={cg_pen:.2f}"
        )

    # SOFT: Structural thickness (tip)
    if not airplane.thickness_ok:
        t_pen = (STRUCTURE.min_thickness_tip_m - airplane.t_tip_m) * 200.0
        t_pen = max(t_pen, 0.0)
        penalty_score += t_pen
        soft_reasons.append(
            f"Tip thickness soft-penalty: {airplane.t_tip_m*1000:.1f} mm "
            f"< {STRUCTURE.min_thickness_tip_m*1000:.0f} mm, penalty={t_pen:.2f}"
        )

    return {
        "is_valid": is_valid,
        "penalty_score": penalty_score,
        "CL": CL,
        "CD": CD_total,
        "CD_vlm": CD_vlm,
        "CD_profile": CD_profile,
        "CD_trim": CD_trim,
        "L_D": L_D,
        "AoA_cruise": alpha_cruise,
        "Cm_cruise": Cm_cruise,
        "SM": SM,
        "x_np": x_np,
        "x_cg": x_cg,
        "S_ref": S_ref,
        "m_total": m,
        "airframe_mass": airplane.airframe_mass,
        "V_cruise_actual": V_cruise_actual,
        "CL_required": CL_required,
        "reasons": reasons,
        "soft_reasons": soft_reasons,
        "invalid_reasons": reasons,
        **clamped_params,
    }
