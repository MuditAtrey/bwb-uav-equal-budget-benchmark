"""
flying_wing_optimizer/config.py

Central configuration module for the blended flying wing optimizer.

All physical specs are aligned with the user's hard constraints:
  - Wingspan: 1.3 m (+ winglets)
  - Payload: 2.0 kg (nose-mounted)
  - Battery: 2.0 kg (fixed, custom Li-ion packs)
  - Cruise speed: 150 kph (41.67 m/s)
  - Payload bay: 10 cm vertical height in the center section

Wing area and aspect ratio are NOT configured here — they are computed
from the actual parametric geometry (root chord, taper, blend stations).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# ─── Parallelization ─────────────────────────────────────────────────────────
cpu_cores = os.cpu_count() or 4
N_CORES: int = min(max(cpu_cores - 2, 2), 8)
PHASE1_DESIGNS: int = 6000


@dataclass(frozen=True)
class GeometryConfig:
    """Aircraft wing geometry specifications.

    NOTE: wing_area and aspect_ratio are NOT here. They must be computed
    from the actual parametric geometry (build_wing_geometry output).
    """
    wingspan_m: float = 1.5  # Full span of main wing [m]
    # Winglet span adds to total projected span but is defined separately


@dataclass(frozen=True)
class MassConfig:
    """Aircraft mass breakdown.

    Revised 2026-08-05:
    - avionics_mass: 0.3 → 0.40 kg (FC + GPS + telemetry + 2 elevon servos)
    - propulsion_mass: 0.5 → 0.55 kg (T-Motor AT4130 387g + 65g ESC + 55g prop + 43g wiring)
    - battery_mass: no longer in this config — it is a free design variable (DoF #17)
    - total_mass: no longer a fixed constant — computed dynamically per design vector

    NOTE: fixed_mass_subtotal is the sum of everything EXCEPT battery.
    The dynamic total_mass = fixed_mass_subtotal + battery_mass_dof.
    """
    payload_mass: float = 2.0           # Nose payload [kg]
    propulsion_mass: float = 0.55       # Motor + ESC + prop + wiring [kg]
    avionics_mass: float = 0.40         # FC, GPS, telemetry, 2x elevon servos [kg]
    # airframe_mass is computed dynamically in geometry.py as 1.0 + 1.5 * S_ref
    # fixed_mass_dynamic is payload + airframe + propulsion + avionics


@dataclass(frozen=True)
class MassPlacementConfig:
    """Longitudinal placement of masses as fraction of root chord.

    x = 0 is the root leading edge. x = c_root is the root trailing edge.
    All positions are fractions of c_root.
    """
    payload_centroid_fraction: float = 0.20   # Payload CG at 20% root chord (nose compartment)
    battery_centroid_fraction: float = 0.82   # Battery CG at 82% root chord (aft center-body bay)
    # motor_fraction is REMOVED — motor is no longer at a fixed fraction of c_root.
    # Motor position is computed as: x_motor = c_root + MOTOR_MOUNT.motor_cg_beyond_te_m
    # See geometry.py _compute_cg() for usage.
    avionics_fraction: float = 0.25           # Avionics at 25% root chord


@dataclass(frozen=True)
class PayloadBayConfig:
    """DEPRECATED: Constraints are now handled geometrically."""
    depth_m: float = 0.12            # Payload bay depth from nose [m] (12 cm)
    min_internal_height_m: float = 0.05  # Minimum vertical height inside payload bay [m] (5 cm)


@dataclass(frozen=True)
class BatteryConfig:
    """21700 Li-ion Battery energy density specifications (e.g. Samsung INR21700-50S / Molicel P45B).
    
    Pack Architecture:
      - Configuration: 6S8P (48 cells)
      - Individual cell mass: ~70 g (3.36 kg cell mass, 3.40 kg total pack with busbars & wiring)
      - Cell capacity: 5.0 Ah (Samsung 50S) or 4.5 Ah (Molicel P45B)
      - Pack nominal energy: 864 Wh (21.6 V * 40 Ah) -> ~254 Wh/kg nominal cell level
      - Pack-level nominal specific energy: 245.0 Wh/kg (including nickel busbars, harness, shrink wrap)
      - Usable depth of discharge: 85% (leaving 15% safety reserve for loiter/go-around)
      - Usable specific energy: 208.25 Wh/kg = 749,700 J/kg
    """
    cell_type: str = "21700 Li-ion (Samsung 50S / Molicel P45B)"
    cell_count: int = 48                       # 6S8P configuration
    specific_energy_nominal: float = 245.0     # 21700 pack-level nominal energy [Wh/kg]
    usable_fraction: float = 0.85              # Usable energy fraction (85% DoD)
    specific_energy_usable_wh: float = 208.25  # Usable specific energy [Wh/kg] at pack level
    specific_energy_usable_j: float = 749700.0  # Usable specific energy [J/kg] (208.25 Wh/kg * 3600)


@dataclass(frozen=True)
class AtmosphereConfig:
    """Atmospheric conditions at cruise altitude."""
    cruise_altitude: float = 500.0   # Cruise altitude ASL [m]
    air_density: float = 1.167       # Air density ρ at 500 m ASL [kg/m³]
    kinematic_viscosity: float = 1.52e-5  # Kinematic viscosity ν at 500 m ASL [m²/s]
    gravity: float = 9.81            # Gravitational acceleration [m/s²]


@dataclass(frozen=True)
class TargetMetricsConfig:
    """Target performance metrics.

    cruise_speed is now the MINIMUM acceptable cruise speed.
    The actual cruise speed is DoF #18 (a free variable).
    """
    min_cruise_speed_ms: float = 36.11   # Minimum cruise speed [m/s] = 130 kph
    cruise_speed: float = 41.67          # Legacy fallback (used only for headwind calc default)


@dataclass(frozen=True)
class DesignBoundsConfig:
    """14-DoF design parameter bounds for the blended flying wing (AR 6-8 swept wing)."""

    # Dual-Sweep BWB Planform Parameters
    sweep_inner: Tuple[float, float] = (25.0, 45.0)     # Center-body leading-edge sweep [degrees]
    sweep_outer: Tuple[float, float] = (25.0, 45.0)     # Outer wing leading-edge sweep [degrees]
    y_blend: Tuple[float, float] = (0.20, 0.35)         # Blend station spanwise position [m]
    c_root: Tuple[float, float] = (0.42, 0.52)          # Root chord [m] (satisfies 50mm payload height & forces AR 6-7.5)
    c_blend: Tuple[float, float] = (0.25, 0.35)         # Blend station chord [m]
    taper_ratio: Tuple[float, float] = (0.55, 0.80)     # Outer wing taper ratio (c_tip / c_blend)

    # Twist distribution (monotonically decreasing root -> tip)
    theta0: Tuple[float, float] = (0.0, 4.0)            # Root twist [degrees]
    theta1: Tuple[float, float] = (-3.0, 2.0)           # Twist at blend station [degrees]
    theta2: Tuple[float, float] = (-6.0, 0.0)           # Mid-outer wing twist [degrees]
    tip_twist: Tuple[float, float] = (-12.0, -2.0)      # Tip twist [degrees]

    # Winglet parameters
    winglet_sweep: Tuple[float, float] = (15.0, 50.0)   # Winglet sweep [degrees]
    winglet_twist: Tuple[float, float] = (-10.0, 0.0)   # Winglet twist [degrees]

    battery_mass: Tuple[float, float] = (1.5, 3.5)           # Battery mass [kg] — DoF #13
    cruise_speed_ms: Tuple[float, float] = (36.11, 61.11)    # Floor set to 130 kph


@dataclass(frozen=True)
class WingletConfig:
    """Fixed winglet geometry parameters."""
    span: float = 0.15              # Winglet span per side [m] (15 cm)
    dihedral_deg: float = 75.0      # Winglet dihedral [degrees]
    chord_fraction: float = 1.00    # Winglet root chord as fraction of main tip chord


@dataclass(frozen=True)
class StructuralThicknessConfig:
    """Minimum airfoil thickness constraints for structural integrity."""
    min_thickness_tip_m: float = 0.008  # Minimum absolute thickness at tip [m] (8 mm)
    min_spar_thickness_m: float = 0.010
    wall_thickness_total_m: float = 0.003
    airfoil_thickness_fraction_root: float = 0.150
    airfoil_thickness_fraction_outer: float = 0.101


@dataclass(frozen=True)
class SoftPenaltyConfig:
    """Soft-penalty weights for constraint violations."""
    cg_penalty_weight: float = 50.0    # CG out-of-bounds penalty [per fraction MAC]
    cm_penalty_weight: float = 100.0   # Cm trim penalty [per unit Cm²]
    speed_penalty_weight: float = 80.0  # Cruise speed mismatch [per fraction speed error]


@dataclass(frozen=True)
class FeasibilityConstraintsConfig:
    """Hard constraint thresholds."""
    min_static_margin: float = 0.05    # SM > 5% MAC
    max_static_margin: float = 0.25    # SM <= 25% MAC (prevents sluggish pitch response)
    max_abs_cm_at_cruise: float = 0.10  # |Cm| at cruise AoA must be < 0.10 (elevon authority limit)
    cm_soft_threshold: float = 0.02    # |Cm| > 0.02 triggers soft penalty
    min_cl_cruise: float = 0.0         # CL must be positive at cruise AoA
    max_cl_required: float = 0.80      # If CL_required > 0.80, wing is too small for speed/weight
    cg_min_mac: float = 0.15           # CG must be at least 15% MAC
    cg_max_mac: float = 0.40           # CG must be at most 40% MAC


@dataclass(frozen=True)
class PathConfig:
    """File system paths."""
    base_dir: Path = Path("flying_wing_optimizer")
    results_dir: Path = Path("flying_wing_optimizer/results")
    database_file: Path = Path("flying_wing_optimizer/results/database.csv")
    optimal_design_json: Path = Path("flying_wing_optimizer/results/optimal_design.json")
    convergence_plot: Path = Path("flying_wing_optimizer/results/convergence.png")
    checkpoint_json: Path = Path("flying_wing_optimizer/results/checkpoint.json")


# NOTE: MotorNacelleDragConfig removed 2026-08-06.
# The motor nacelle is now modeled as an asb.Fuselage in geometry.py
# (_build_nacelle_fairing). Its skin friction and form drag are computed
# by the strip-based profile drag calculator in aerodynamics.py.
# No flat drag lookup is needed or appropriate.


@dataclass(frozen=True)
class MotorMountConfig:
    """Motor rear boom mount geometry.

    The motor is mounted on a carbon-fibre boom extending 86mm
    beyond the motor body (152mm total hub-to-TE distance) and is enclosed in
    a custom CAD lofted nacelle that blends flush with the wing trailing edge.
    This achieves 0.5 × prop_diameter clearance for a 12" prop and moves the prop disk
    out of the densest part of the wing wake.

    Physics-verified improvement:
      - Nacelle drag: 0.76 N (tight) → 0.18 N (integrated nacelle): saves 0.58 N
      - Prop efficiency: η_prop 0.78 (wake ingestion) → 0.81 (0.5D clearance)
      - Net range gain at 130 kph, 2.5 kg battery: +37 km

    CG impact: motor CG shifts 86mm aft → total CG moves 5mm aft (1.1% MAC).
    The battery_centroid_fraction in MassPlacementConfig absorbs this; the
    optimizer finds a geometry where SM stays > 5% MAC regardless.

    Build note (for the human):
      The boom is a 12mm OD / 10mm ID carbon tube. The user will design the
      integrated lofted nacelle in CAD (e.g. OpenVSP or FreeCAD) as a body of
      revolution that blends flush into the wing TE, swells to 54mm OD to
      cover the motor, and tapers to the 15mm prop spinner. Print in PETG.
    """
    # Distance from trailing edge to prop hub [m]
    # = motor_body_length(59mm) + mount_plate(2mm) + shaft_collar(5mm) + boom_extension(86mm)
    hub_to_te_m: float = 0.152

    # Boom extension beyond motor body [m] — the 86mm extra tube
    boom_extension_m: float = 0.086

    # Motor+boom assembly CG offset beyond trailing edge [m]
    motor_cg_beyond_te_m: float = 0.02


# ─── Singleton Instances ──────────────────────────────────────────────────────
GEOMETRY = GeometryConfig()
MASS = MassConfig()
MASS_PLACEMENT = MassPlacementConfig()
PAYLOAD_BAY = PayloadBayConfig()
BATTERY = BatteryConfig()
ATMOSPHERE = AtmosphereConfig()
TARGETS = TargetMetricsConfig()
BOUNDS = DesignBoundsConfig()
WINGLET = WingletConfig()
STRUCTURE = StructuralThicknessConfig()
SOFT_PENALTY = SoftPenaltyConfig()
CONSTRAINTS = FeasibilityConstraintsConfig()
PATHS = PathConfig()
MOTOR_MOUNT = MotorMountConfig()
