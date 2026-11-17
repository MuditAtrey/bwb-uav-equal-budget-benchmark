"""
flying_wing_optimizer/geometry.py

True Blended Wing Body (BWB) Parametric Geometry Builder.

Architecture:
  - Dual Leading-Edge Sweep: High inner sweep (45°-65°) for the wide, deep
    center body; lower outer sweep (18°-32°) for the high-aspect-ratio outer wing.
  - Integrated Nacelle Lofting: The center-body trailing edge extends aft into
    a smooth motor nacelle tail boom (housing the AT4130 motor and 12" prop hub).
    The nacelle is built DIRECTLY into the wing surface loft — zero floating eggs,
    zero unattached parts.
  - Monotonic Taper & Smooth Hermite Blend: Mathematically impossible to produce
    forward-swept kinks, serrations, or simple fat triangles.
"""

import math
from typing import Optional

import aerosandbox as asb
import aerosandbox.numpy as np

from flying_wing_optimizer.config import (
    ATMOSPHERE,
    GEOMETRY,
    MASS,
    MASS_PLACEMENT,
    MOTOR_MOUNT,
    STRUCTURE,
    WINGLET,
)


def get_mh60_airfoil() -> asb.Airfoil:
    """Returns the MH60 reflexed airfoil, repaneled for consistent mesh."""
    return asb.Airfoil("mh60").repanel(n_points_per_side=50)


def get_mh91_airfoil() -> asb.Airfoil:
    """Returns the MH91 reflexed airfoil, repaneled for consistent mesh."""
    return asb.Airfoil("mh91").repanel(n_points_per_side=50)


def check_physical_constraints(
    airplane: asb.Airplane,
    c_root: float,
    c_blend: float,
    y_blend: float,
) -> tuple:
    """Evaluates physical volumetric and clearance constraints for the payload bay,
    avionics/battery compartment, and structural spar channel.

    Returns:
        tuple: (is_valid: bool, reasons: list[str])
    """
    import numpy as onp

    reasons = []
    WALL_TOTAL = 0.003

    root_t_frac = getattr(airplane, "root_thickness_frac", 0.150)
    outer_t_frac = getattr(airplane, "outer_thickness_frac", 0.101)

    # 1. Check A: Nose payload height clearance (50mm at 20% chord)
    nose_height_m = 0.85 * root_t_frac * c_root - WALL_TOTAL
    if nose_height_m < 0.050:
        reasons.append(
            f"Nose payload height clearance insufficient: {nose_height_m:.4f}m < 0.050m"
        )
        return False, reasons

    # 2. Check B: Avionics + Battery volume (25% to 70% chord, center body span) >= 2.5 L
    n_integration_steps = 20
    if y_blend > 0:
        y_steps = onp.linspace(0.0, y_blend, n_integration_steps + 1)
        dy = y_blend / n_integration_steps
    else:
        y_steps = onp.array([0.0])
        dy = 0.0

    total_volume_m3 = 0.0
    for y_val in y_steps:
        frac = y_val / y_blend if y_blend > 0 else 0.0
        local_chord = c_root + (c_blend - c_root) * (3 * frac**2 - 2 * frac**3)
        local_t_frac = root_t_frac + (outer_t_frac - root_t_frac) * frac
        internal_height_m = (local_t_frac * local_chord) - WALL_TOTAL
        if internal_height_m <= 0:
            continue
        usable_width_m = 0.45 * local_chord
        station_area = 0.75 * usable_width_m * internal_height_m
        total_volume_m3 += station_area * dy

    total_volume_liters = total_volume_m3 * 2.0 * 1000.0
    if total_volume_liters < 2.5:
        reasons.append(
            f"Center body internal volume insufficient: {total_volume_liters:.2f}L < 2.50L"
        )
        return False, reasons

    # 3. Check C: Spar channel minimum thickness (>= 10mm) in center body / inner wing
    if hasattr(airplane, "wings") and len(airplane.wings) > 0:
        for xsec in airplane.wings[0].xsecs:
            y_val = float(xsec.xyz_le[1])
            if y_val <= y_blend + 1e-6:
                local_chord = float(xsec.chord)
                frac = y_val / y_blend if y_blend > 0 else 0.0
                local_t_frac = root_t_frac + (outer_t_frac - root_t_frac) * frac
                spar_thickness_m = 0.93 * local_t_frac * local_chord
                if spar_thickness_m < 0.010:
                    reasons.append(
                        f"Spar channel thickness at y={y_val:.3f}m insufficient: {spar_thickness_m:.4f}m < 0.010m"
                    )
                    return False, reasons

    return True, []


def _compute_cg(
    c_root: float,
    mac: float,
    x_le_mac: float,
    battery_mass: float,
    S_ref: float,
) -> float:
    """Compute longitudinal CG from physical mass placements."""
    x_payload = MASS_PLACEMENT.payload_centroid_fraction * c_root
    x_battery = MASS_PLACEMENT.battery_centroid_fraction * c_root
    x_motor = c_root + MOTOR_MOUNT.motor_cg_beyond_te_m  # 0.115m behind TE
    x_avionics = MASS_PLACEMENT.avionics_fraction * c_root
    x_airframe = x_le_mac + 0.40 * mac

    airframe_mass = 1.0 + 1.5 * S_ref
    fixed_mass_dynamic = MASS.payload_mass + airframe_mass + MASS.propulsion_mass + MASS.avionics_mass
    total_mass = fixed_mass_dynamic + battery_mass
    
    numerator = (
        MASS.payload_mass * x_payload
        + battery_mass * x_battery
        + airframe_mass * x_airframe
        + MASS.propulsion_mass * x_motor
        + MASS.avionics_mass * x_avionics
    )
    return numerator / total_mass


def _build_nacelle_fairing(c_root: float) -> asb.Fuselage:
    """Build a Unified Blended Wing Body (BWB) Center-Body Fuselage.

    The fuselage runs continuously from X=0 (nose cap) through the payload
    and battery bay in the center section, smoothly tapering aft into the
    54mm cylindrical motor housing at X=c_root, and terminating at the 30mm
    propeller hub spinner cap at X=c_root+0.152m.

    This ensures the motor mount is a primary structural continuation of the
    aircraft body from the beginning, not an unattached afterthought.
    """
    fuse_xs = [
        0.0,
        0.12 * c_root,
        0.40 * c_root,
        0.75 * c_root,
        1.00 * c_root,
        1.00 * c_root + 0.152,
        1.00 * c_root + 0.190,
    ]

    widths = [0.020, 0.140, 0.160, 0.100, 0.054, 0.030, 0.001]
    heights = [0.020, 0.105, 0.110, 0.080, 0.054, 0.030, 0.001]

    xsecs = []
    for x_pos, w, h in zip(fuse_xs, widths, heights):
        xsecs.append(
            asb.FuselageXSec(
                xyz_c=[x_pos, 0.0, 0.0],
                width=w,
                height=h,
            )
        )

    return asb.Fuselage(name="UnifiedBWBBody", xsecs=xsecs)


def build_wing_geometry(
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
    battery_mass: float = 2.0,
    winglet_sweep_deg: float = 30.0,
    winglet_twist_deg: float = -5.0,
) -> asb.Airplane:
    """Builds a true dual-sweep Blended Wing Body with integrated motor nacelle tail boom."""
    import numpy as onp

    b = GEOMETRY.wingspan_m
    c_tip = c_blend * taper

    # Sanity clamps on geometry bounds
    y_blend = float(np.clip(y_blend, 0.12, b / 2.0 - 0.15))
    c_blend = float(np.clip(c_blend, 0.25, c_root - 0.05))
    c_tip = float(np.clip(c_tip, 0.05, c_blend - 0.05))

    clamped_params = {
        "y_blend": y_blend,
        "c_blend": c_blend,
        "c_tip": c_tip,
        "taper": c_tip / c_blend,
    }

    airfoil_root = get_mh91_airfoil()
    airfoil_main = get_mh60_airfoil()
    root_thickness_frac = airfoil_root.max_thickness()
    outer_thickness_frac = airfoil_main.max_thickness()


    # ── 1. Build smooth 25-station spanwise distribution ─────────────────
    y_stations = onp.linspace(0.0, b / 2.0 - 0.05, 25)

    xsecs = []
    L_nacelle = MOTOR_MOUNT.motor_cg_beyond_te_m + 0.037  # 0.152m total extension

    for y_val in y_stations:
        # Dual leading-edge sweep
        if y_val <= y_blend:
            tan_theta = math.tan(math.radians(sweep_inner))
            x_le_blend = y_blend * tan_theta
            x_te_blend = x_le_blend + c_blend
            
            # Cubic curve for leading edge (zero sweep at root, matching sweep_inner at blend)
            x_le = -tan_theta / (y_blend**2) * (y_val**3) + 2 * tan_theta / y_blend * (y_val**2)
            
            # Linear blend for trailing edge (removes the "buttcrack" notch)
            x_te = c_root + (x_te_blend - c_root) * (y_val / y_blend)
            
            c_base = x_te - x_le
            frac = y_val / y_blend
            twist = theta0 + (theta1 - theta0) * frac
            frac_af = y_val / y_blend
            af = airfoil_root.blend_with_another_airfoil(airfoil_main, blend_fraction=frac_af)
        else:
            x_le_b = y_blend * math.tan(math.radians(sweep_inner))
            x_le = x_le_b + (y_val - y_blend) * math.tan(math.radians(sweep_outer))
            frac = (y_val - y_blend) / (b / 2.0 - 0.05 - y_blend)
            c_base = c_blend + (c_tip - c_blend) * frac
            # Piecewise linear twist interpolation through theta1 -> theta2 -> tip_twist
            if frac < 0.5:
                twist = theta1 + (theta2 - theta1) * (frac / 0.5)
            else:
                twist = theta2 + (tip_twist - theta2) * ((frac - 0.5) / 0.5)
            af = airfoil_main

        xsecs.append(
            asb.WingXSec(
                xyz_le=[x_le, y_val, 0.0],
                chord=c_base,
                twist=twist,
                airfoil=af,
            )
        )

    # ── 2. Blended winglet ───────────────────────────────────────────────
    r_blend = 0.05
    y_start_blend = b / 2.0 - r_blend
    winglet_dihedral_rad = np.radians(WINGLET.dihedral_deg)
    winglet_sweep_rad = np.radians(winglet_sweep_deg)

    blend_angles = np.linspace(0, winglet_dihedral_rad, 4)[1:]

    c_blend_start = xsecs[-1].chord
    x_25_blend_start = xsecs[-1].xyz_le[0] + 0.25 * c_blend_start
    c_winglet = c_blend_start * WINGLET.chord_fraction

    for angle in blend_angles:
        dy_w = r_blend * np.sin(angle)
        dz_w = r_blend * (1 - np.cos(angle))
        y_w = y_start_blend + dy_w
        z_w = dz_w

        span_dist = r_blend * angle
        x_25_w = x_25_blend_start + span_dist * np.tan(winglet_sweep_rad)

        frac_w = angle / winglet_dihedral_rad
        chord_w = c_blend_start * (1 - frac_w) + c_winglet * frac_w
        x_le_w = x_25_w - 0.25 * chord_w
        twist_w = tip_twist * (1 - frac_w) + winglet_twist_deg * frac_w

        xsecs.append(
            asb.WingXSec(
                xyz_le=[x_le_w, y_w, z_w],
                chord=chord_w,
                twist=twist_w,
                airfoil=airfoil_main,
            )
        )

    # Winglet tip
    span_remaining = WINGLET.span - r_blend
    dy_w = span_remaining * np.cos(winglet_dihedral_rad)
    dz_w = span_remaining * np.sin(winglet_dihedral_rad)
    y_w = y_start_blend + r_blend * np.sin(winglet_dihedral_rad) + dy_w
    z_w = r_blend * (1 - np.cos(winglet_dihedral_rad)) + dz_w

    x_25_w = x_25_blend_start + WINGLET.span * np.tan(winglet_sweep_rad)
    chord_tip_wl = c_winglet * 0.7
    
    if chord_tip_wl < 0.06:
        return None, [f"Winglet tip chord too thin: {chord_tip_wl:.3f}m < 0.06m"]
        
    x_le_w = x_25_w - 0.25 * chord_tip_wl

    xsecs.append(
        asb.WingXSec(
            xyz_le=[x_le_w, y_w, z_w],
            chord=chord_tip_wl,
            twist=winglet_twist_deg,
            airfoil=airfoil_main,
        )
    )

    # ── 3. Nacelle fairing ───────────────────────────────────────────────
    nacelle = _build_nacelle_fairing(c_root)

    # ── 4. Build Airplane ────────────────────────────────────────────────
    wing = asb.Wing(name="BWB Wing", symmetric=True, xsecs=xsecs)

    S_ref = wing.area()
    mac = wing.mean_aerodynamic_chord()

    # Numerical integration for true x_le_mac
    xsecs_main = [xs for xs in xsecs if xs.xyz_le[1] <= (b / 2.0 - 0.05)]
    c_vals = onp.array([xs.chord for xs in xsecs_main])
    xle_vals = onp.array([xs.xyz_le[0] for xs in xsecs_main])
    y_vals = onp.array([xs.xyz_le[1] for xs in xsecs_main])
    dy_arr = onp.diff(y_vals)
    c_mid = (c_vals[:-1] + c_vals[1:]) / 2.0
    xle_mid = (xle_vals[:-1] + xle_vals[1:]) / 2.0
    den = onp.sum(c_mid**2 * dy_arr)
    x_le_mac = float(onp.sum(c_mid**2 * xle_mid * dy_arr) / den) if den > 0 else 0.0

    x_cg = _compute_cg(c_root, mac, x_le_mac, battery_mass=battery_mass, S_ref=S_ref)
    cg_mac_fraction = (x_cg - x_le_mac) / mac if mac > 0 else 0.30

    x_motor_start = max(c_root, c_root + MOTOR_MOUNT.motor_cg_beyond_te_m - 0.0275)
    x_motor_end = x_motor_start + 0.055
    x_spinner_tip = x_motor_end + 0.070

    boom_assembly = asb.Fuselage(
        name="MotorBoom",
        xsecs=[
            asb.FuselageXSec(xyz_c=[c_root - 0.05, 0, 0], radius=0.028 / 2),
            asb.FuselageXSec(xyz_c=[x_motor_start, 0, 0], radius=0.028 / 2),
            asb.FuselageXSec(xyz_c=[x_motor_end, 0, 0], radius=0.054 / 2),
            asb.FuselageXSec(xyz_c=[x_spinner_tip, 0, 0], radius=0.015 / 2),
        ]
    )

    airplane = asb.Airplane(
        name="BlendedFlyingWing",
        xyz_ref=[x_cg, 0, 0],
        wings=[wing],
        fuselages=[boom_assembly],
    )

    airplane.mac = mac
    airplane.S_ref = S_ref
    airplane.x_cg = x_cg
    airplane.cg_mac_fraction = cg_mac_fraction
    airplane.x_le_mac = x_le_mac
    airplane.nacelle = nacelle

    airplane.battery_mass = battery_mass
    airframe_mass = 1.0 + 1.5 * S_ref
    fixed_mass_dynamic = MASS.payload_mass + airframe_mass + MASS.propulsion_mass + MASS.avionics_mass
    airplane.total_mass = fixed_mass_dynamic + battery_mass
    airplane.airframe_mass = airframe_mass

    airplane.t_root_m = c_root * root_thickness_frac
    airplane.t_tip_m = c_tip * outer_thickness_frac

    airplane.thickness_ok = airplane.t_tip_m >= STRUCTURE.min_thickness_tip_m
    airplane.payload_bay_height_m = airplane.t_root_m
    airplane.payload_bay_ok = True
    airplane.te_sweep_ok = True

    airplane.airfoil_root = airfoil_root
    airplane.airfoil_main = airfoil_main
    airplane.root_thickness_frac = root_thickness_frac
    airplane.outer_thickness_frac = outer_thickness_frac

    # First validity check: x_le must be monotonically increasing from y=0 to y=y_blend
    prev_x_le = -1e9
    for xs in xsecs:
        if xs.xyz_le[1] <= y_blend + 1e-6:
            if xs.xyz_le[0] < prev_x_le:
                return None, ["Leading edge not monotonic in center body"]
            prev_x_le = xs.xyz_le[0]

    # Second validity check: chord at y=0.15m >= 0.55 * c_root
    y_test = 0.15
    if y_test <= y_blend:
        frac_test = y_test / y_blend
        c_15 = c_root + (c_blend - c_root) * (3 * frac_test ** 2 - 2 * frac_test ** 3)
    else:
        frac_test = (y_test - y_blend) / (b / 2.0 - 0.05 - y_blend)
        c_15 = c_blend + (c_tip - c_blend) * frac_test

    if c_15 < 0.55 * c_root:
        return None, ["Center body too narrow at y=0.15m"]

    # Third validity check: Monotonic Trailing Edge sweep (No W-shaped/bat-wing or forward-swept TE)
    prev_x_te = -1e9
    for xs in xsecs:
        # Ignore vertical winglet tip cross-sections
        if xs.xyz_le[1] >= (b / 2.0 - 0.05):
            break
        x_te = xs.xyz_le[0] + xs.chord
        # 5mm numerical tolerance to avoid rounding chatter
        if x_te < prev_x_te - 0.005:
            return None, [f"Trailing edge sweeps forward at y={xs.xyz_le[1]:.3f}m (W-shape/bat-wing)"]
        prev_x_te = x_te

    # Fourth validity check: Outer Wing Trailing Edge sweep angle (>= 8.0° to ensure a true swept wing, not a flat outer delta)
    x_te_root = xsecs[0].xyz_le[0] + xsecs[0].chord
    idx_blend = 0
    idx_tip_main = len(xsecs) - 1
    for i, xs in enumerate(xsecs):
        if xs.xyz_le[1] >= y_blend - 1e-4 and idx_blend == 0:
            idx_blend = i
        if xs.xyz_le[1] >= (b / 2.0 - 0.05):
            idx_tip_main = i
            break
            
    x_te_blend = xsecs[idx_blend].xyz_le[0] + xsecs[idx_blend].chord
    y_blend_actual = xsecs[idx_blend].xyz_le[1]
    x_te_tip_main = xsecs[idx_tip_main].xyz_le[0] + xsecs[idx_tip_main].chord
    y_tip_main = xsecs[idx_tip_main].xyz_le[1]
    
    dy_outer = y_tip_main - y_blend_actual
    if dy_outer > 0.05:
        te_outer_sweep_deg = math.degrees(math.atan((x_te_tip_main - x_te_blend) / dy_outer))
        if te_outer_sweep_deg < 15.0:
            return None, [f"Outer wing TE sweep too flat ({te_outer_sweep_deg:.1f}° < 15.0°). Rejected flat delta wing planform."]

    # Fifth validity check: Outer Wing Panel Aspect Ratio (>= 3.0 to enforce elongated swept wing panels)
    c_avg_outer = (xsecs[idx_blend].chord + xsecs[idx_tip_main].chord) / 2.0
    if dy_outer > 0.05 and c_avg_outer > 0:
        ar_outer_panel = (2.0 * dy_outer) / c_avg_outer
        if ar_outer_panel < 3.0:
            return None, [f"Outer wing panel AR too low ({ar_outer_panel:.2f} < 3.0). Delta wing planform."]

    # ── Volumetric and manufacturing constraints ─────────────────────────
    vol_valid, vol_reasons = check_physical_constraints(airplane, c_root, c_blend, y_blend)
    if not vol_valid:
        return None, vol_reasons

    return airplane, clamped_params
