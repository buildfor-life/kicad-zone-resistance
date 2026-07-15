"""Skin-effect model tests. The single-layer AC solve scales ALL in-plane
conductances identically, so R_AC = R_DC * resistance_factor EXACTLY -
which turns the analytic foil formula into an end-to-end exact test."""
import math

import numpy as np
import pytest

from fill_resistance import raster, skin, solver
from tests.util import NM, make_multilayer, strip_problem

RHO = 1.68e-8


def _solve(problem, h_mm, i_test=1.0, freq=0.0):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, i_test, freq,
                            contact_model="equipotential"), stack


def test_skin_depth_value():
    # copper @ 1 MHz: ~65-66 um
    assert skin.skin_depth_m(1e6, RHO) * 1e6 == pytest.approx(65.2, rel=0.01)


def test_sheet_resistance_dc_limit():
    t = 70e-6
    assert skin.sheet_resistance_ac(t, 0.0, RHO) == pytest.approx(RHO / t)
    # low frequency: within 0.1% of DC
    assert skin.sheet_resistance_ac(t, 100.0, RHO) == pytest.approx(
        RHO / t, rel=1e-3)


def test_sheet_resistance_high_f_limits():
    t = 70e-6
    f = 1e9                                  # delta << t
    delta = skin.skin_depth_m(f, RHO)
    assert skin.sheet_resistance_ac(t, f, RHO, sides=1) == pytest.approx(
        RHO / delta, rel=0.01)
    assert skin.sheet_resistance_ac(t, f, RHO, sides=2) == pytest.approx(
        RHO / (2 * delta), rel=0.01)


def test_resistance_factor_monotonic():
    t = 70e-6
    factors = [skin.resistance_factor(t, f, RHO)
               for f in (0, 1e4, 1e5, 1e6, 1e7, 1e8)]
    assert all(b >= a - 1e-12 for a, b in zip(factors, factors[1:]))
    assert factors[0] == 1.0


def test_parse_frequency():
    assert skin.parse_frequency("") == 0.0
    assert skin.parse_frequency("0") == 0.0
    assert skin.parse_frequency("142k") == 142_000.0
    assert skin.parse_frequency("1.5M") == 1_500_000.0
    assert skin.parse_frequency("2meg") == 2_000_000.0
    assert skin.parse_frequency("100000") == 100_000.0
    assert skin.parse_frequency("100 kHz") == 100_000.0
    with pytest.raises(ValueError):
        skin.parse_frequency("junk")      # must not silently become DC
    with pytest.raises(ValueError):
        skin.parse_frequency("-5k")


def test_single_layer_ac_scales_exactly():
    """Uniform conductance scaling leaves the field shape unchanged:
    R_AC = R_DC * factor to solver precision."""
    p = strip_problem(length=50, width=10, e_len=5)
    f = 2e6                                    # delta=46um < t=70um
    r_dc, _ = _solve(p, 0.5)
    r_ac, _ = _solve(p, 0.5, freq=f)
    factor = skin.resistance_factor(70e-6, f, RHO, sides=1)
    assert factor > 1.2                        # real crowding at 2 MHz
    assert r_ac.R_ohm == pytest.approx(r_dc.R_ohm * factor, rel=1e-9)
    assert r_ac.rs_ratios[0] == pytest.approx(factor, rel=1e-12)
    assert r_ac.skin_depth_um == pytest.approx(
        skin.skin_depth_m(f, RHO) * 1e6, rel=1e-12)


def test_via_chain_ac_exact():
    """1D chain: layers scale by the foil factor, the barrel by the
    plating-wall factor - exact composition."""
    STRIP = [(0, 0), (10, 0), (10, 1), (0, 1)]
    p = make_multilayer(
        [[(STRIP, [])], [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    f = 2e6
    sig_ac = 1.0 / skin.sheet_resistance_ac(70e-6, f, RHO, sides=1)
    via_factor = skin.resistance_factor(18e-6, f, RHO, sides=2)
    r_via_dc = RHO * 1e-3 / (math.pi * 0.3e-3 * 18e-6)
    r_exact = (5 + 4) / sig_ac + r_via_dc * via_factor
    res, _ = _solve(p, 1.0, freq=f)
    assert res.R_ohm == pytest.approx(r_exact, rel=1e-9)


def test_dc_default_unchanged():
    """freq omitted -> identical to the pre-skin behavior."""
    p = strip_problem(length=50, width=10, e_len=5)
    res, _ = _solve(p, 0.5)
    assert res.freq_hz == 0.0
    assert res.skin_depth_um is None
    assert all(r == 1.0 for r in res.rs_ratios)
