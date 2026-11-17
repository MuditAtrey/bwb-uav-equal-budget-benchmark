"""
flying_wing_optimizer/mission.py

Range calculation using the Electric Breguet Equation.

R = (m_bat / m_tot) * (e_usable * η_total / g) * (L/D)

Rewrite 2026-08-06:
  - η_total is no longer a static constant (0.769).
    Propeller efficiency varies with advance ratio J = V/(n*D).
    For a fixed-pitch APC 12x8E on AT4130 at cruise speeds 130-220 kph:
      130 kph → η_prop ≈ 0.72
      150 kph → η_prop ≈ 0.78
      180 kph → η_prop ≈ 0.80
      220 kph → η_prop ≈ 0.65 (high-J efficiency drop)
    Motor + ESC efficiency is relatively constant at 0.88.
    Combined η_total = η_motor_esc * η_prop(V).
"""

import math
from typing import Any, Dict

from flying_wing_optimizer.config import ATMOSPHERE, BATTERY

# Constants
GRAVITY: float = 9.81
ETA_MOTOR_ESC: float = 0.88  # Motor + ESC efficiency (Turnigy AT4130 + 80A ESC)


def _prop_efficiency(v_cruise_ms: float) -> float:
    """Speed-dependent propeller efficiency for APC 12x8E on AT4130.

    Piecewise linear interpolation of advance ratio J = V / (n * D).
    AT4130 at cruise: ~10,000 RPM = 166.7 rev/s, D = 12" = 0.3048 m.

    Args:
        v_cruise_ms: Cruise speed [m/s].

    Returns:
        η_prop: Propeller efficiency [dimensionless].
    """
    import numpy as np

    n = 166.67       # Approximate cruise RPM in rev/s (10,000 RPM)
    D = 0.3048       # 12" prop diameter [m]
    J = v_cruise_ms / (n * D)  # Advance ratio

    J_vals = [0.55, 0.65, 0.80, 0.95]
    eta_vals = [0.72, 0.78, 0.80, 0.65]

    eta = np.interp(J, J_vals, eta_vals, left=0.0, right=0.65)
    return float(eta)


def calculate_mission_performance(
    L_D: float,
    CL: float,
    S_ref: float,
    battery_mass: float,    # DoF #17 [kg]
    total_mass: float,      # Computed dynamically [kg]
) -> Dict[str, Any]:
    """Calculate mission range for given aerodynamic performance.

    Args:
        L_D: Lift-to-drag ratio at trimmed cruise AoA.
        CL:  Lift coefficient at trimmed cruise AoA.
        S_ref: Actual wing reference area from geometry [m²].
        battery_mass: Battery mass (free variable) [kg].
        total_mass: Total dynamic aircraft mass [kg].

    Returns:
        Dict with m_battery_kg, m_total_kg, range_calm_km,
        range_headwind_km, V_cruise_ms, eta_prop, eta_total.
    """
    m_battery_kg = battery_mass
    m_total_kg = total_mass
    energy_density_j = BATTERY.specific_energy_usable_j  # 749,700 J/kg (208.25 Wh/kg usable for 21700 pack @ 85% DoD)
    v_headwind = 10.0  # Headwind for secondary target [m/s]

    if L_D <= 0 or CL <= 0 or S_ref <= 0:
        return {
            "m_battery_kg": float(m_battery_kg),
            "m_total_kg": float(m_total_kg),
            "range_calm_km": 0.0,
            "range_headwind_km": 0.0,
            "V_cruise_ms": 0.0,
            "eta_prop": 0.0,
            "eta_total": 0.0,
        }

    # Compute actual cruise speed from CL and S_ref
    rho = ATMOSPHERE.air_density
    V_cruise = math.sqrt(
        (2.0 * m_total_kg * GRAVITY) / (rho * S_ref * CL)
    )

    # Speed-dependent propeller efficiency
    eta_prop = _prop_efficiency(V_cruise)
    eta_total = ETA_MOTOR_ESC * eta_prop

    # Breguet range (electric): R = (m_bat/m_tot) * (e * η / g) * L/D
    range_factor = (energy_density_j * eta_total) / GRAVITY
    range_calm_m = (m_battery_kg / m_total_kg) * range_factor * L_D
    range_calm_km = range_calm_m / 1000.0

    # Headwind correction
    headwind_factor = max(0.0, 1.0 - (v_headwind / V_cruise))
    range_headwind_km = range_calm_km * headwind_factor

    return {
        "m_battery_kg": float(m_battery_kg),
        "m_total_kg": float(m_total_kg),
        "range_calm_km": float(range_calm_km),
        "range_headwind_km": float(range_headwind_km),
        "V_cruise_ms": float(V_cruise),
        "eta_prop": float(eta_prop),
        "eta_total": float(eta_total),
    }
