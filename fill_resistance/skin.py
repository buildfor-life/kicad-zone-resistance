"""Skin-effect corrections: frequency-dependent effective sheet
resistance of a copper foil and via-barrel wall.

1D diffusion through the foil thickness (exact): with tau = (1+j)/delta,
the internal impedance per square of a foil of thickness t is

    one-sided field (plane over a return plane):  Zs = tau*rho * coth(tau*t)
    two-sided field (isolated foil):              Zs = tau*rho/2 * coth(tau*t/2)

Both reduce to rho/t at DC and to rho/delta (resp. rho/(2*delta)) at
high frequency. R_AC = Re(Zs) is used as the effective sheet resistance.

HONESTY NOTE (also in the README): only the through-thickness current
crowding is modeled. Lateral redistribution (proximity effect - AC
current following the minimum-inductance path) needs a magneto-
quasistatic solve and is NOT captured; since the resistance-driven
distribution is the minimum-dissipation one, the reported AC resistance
is a rigorous LOWER BOUND at the given frequency.
"""
from __future__ import annotations

import cmath
import math

MU0 = 4e-7 * math.pi


def skin_depth_m(freq_hz: float, rho_ohm_m: float) -> float:
    return math.sqrt(2.0 * rho_ohm_m / (2.0 * math.pi * freq_hz * MU0))


def _coth(x: complex) -> complex:
    return 1.0 / cmath.tanh(x)


def sheet_resistance_ac(thickness_m: float, freq_hz: float,
                        rho_ohm_m: float, sides: int = 1) -> float:
    """Effective sheet resistance [ohm/sq] of a foil at freq_hz.
    sides=1: field on one side (plane facing a return plane, conservative);
    sides=2: symmetric field on both sides (isolated foil)."""
    if freq_hz <= 0.0:
        return rho_ohm_m / thickness_m
    delta = skin_depth_m(freq_hz, rho_ohm_m)
    tau = (1.0 + 1.0j) / delta
    if sides == 2:
        zs = tau * rho_ohm_m / 2.0 * _coth(tau * thickness_m / 2.0)
    else:
        zs = tau * rho_ohm_m * _coth(tau * thickness_m)
    return zs.real


def resistance_factor(thickness_m: float, freq_hz: float,
                      rho_ohm_m: float, sides: int = 1) -> float:
    """R_AC / R_DC of a foil (or barrel wall) of the given thickness."""
    if freq_hz <= 0.0:
        return 1.0
    return (sheet_resistance_ac(thickness_m, freq_hz, rho_ohm_m, sides)
            / (rho_ohm_m / thickness_m))


def parse_frequency(text: str) -> float:
    """'0', '100k', '1.5M', '142500' -> Hz. Empty/invalid -> 0 (DC)."""
    t = text.strip().lower().replace(",", ".").removesuffix("hz").strip()
    if not t:
        return 0.0
    mult = 1.0
    if t.endswith("meg"):
        mult, t = 1e6, t[:-3]
    elif t.endswith("m"):
        mult, t = 1e6, t[:-1]
    elif t.endswith("k"):
        mult, t = 1e3, t[:-1]
    elif t.endswith("g"):
        mult, t = 1e9, t[:-1]
    try:
        return max(0.0, float(t) * mult)
    except ValueError:
        return 0.0
