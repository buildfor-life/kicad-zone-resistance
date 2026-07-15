"""Multi-layer / via solver tests. The 1-cell-wide strip cases are pure
series chains, so the discrete solution is exact and validates the via
barrel model, layer coupling, and flux integration to solver precision."""
import math

import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.errors import ConnectivityError, ElectrodeError
from tests.util import NM, make_multilayer, sigma_s


def _solve(problem, h_mm, i_test=1.0):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, i_test,
                            contact_model="equipotential"), stack


STRIP = [(0, 0), (10, 0), (10, 1), (0, 1)]     # 10 x 1 mm; h=1 -> 1 row


def _r_via(length_mm, drill_mm=0.3, plating_um=18.0, rho=1.68e-8):
    area = math.pi * (drill_mm * 1e-3) * (plating_um * 1e-6)
    return rho * (length_mm * 1e-3) / area


def test_two_identical_layers_parallel():
    """Both electrodes contact both layers, no vias needed: two identical
    independent sheets in parallel -> exactly half the single-layer R."""
    one = make_multilayer([[(STRIP, [])]], (0, 0, 1, 1), (9, 0, 10, 1))
    two = make_multilayer([[(STRIP, [])], [(STRIP, [])]],
                          (0, 0, 1, 1), (9, 0, 10, 1))
    r1, _ = _solve(one, 1.0)
    r2, _ = _solve(two, 1.0)
    assert r2.R_ohm == pytest.approx(r1.R_ohm / 2, rel=1e-9)


def test_via_chain_1d_exact():
    """e1 on L0 left end, e2 on L1 right end, one through-via at x=5.5:
    R = faces_L0/sigma + R_via + faces_L1/sigma, exact."""
    p = make_multilayer(
        [[(STRIP, [])], [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    res, stack = _solve(p, 1.0)
    # cells at x centers 0.5..9.5 -> cols 0..9 (plus margins); electrode
    # cells: col of 0.5 (e1), col of 9.5 (e2); via cell: col of 5.5
    sig = sigma_s()
    faces_l0 = 5          # cols 0->5: faces between 0|1 .. 4|5
    faces_l1 = 4          # cols 5->9
    r_exact = (faces_l0 + faces_l1) / sig + _r_via(1.0)
    assert res.R_ohm == pytest.approx(r_exact, rel=1e-9)
    assert res.mismatch_rel < 1e-10
    assert len(res.via_reports) == 1
    # the single via carries the full test current
    assert res.via_reports[0].current_a == pytest.approx(1.0, rel=1e-9)
    assert res.via_reports[0].power_w == pytest.approx(_r_via(1.0), rel=1e-9)


def test_parallel_vias_halve_barrel_resistance():
    base = dict(rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
                contact1="L0", contact2="L1", gap_mm=1.0)
    one = make_multilayer([[(STRIP, [])], [(STRIP, [])]],
                          vias_mm=[(5.5, 0.5)], **base)
    two = make_multilayer([[(STRIP, [])], [(STRIP, [])]],
                          vias_mm=[(5.5, 0.5), (5.5, 0.5)], **base)
    r_one, _ = _solve(one, 1.0)
    r_two, _ = _solve(two, 1.0)
    assert (r_one.R_ohm - r_two.R_ohm) == pytest.approx(_r_via(1.0) / 2,
                                                        rel=1e-9)


def test_antipad_bridging():
    """3 layers; the middle layer has an antipad hole at the via, WIDER
    than the barrel connection search (pad footprint + 1 cell), so the
    barrel bridges L0 -> L2 directly with DOUBLE the length."""
    mid_with_hole = [(STRIP, [[(4, 0), (7, 0), (7, 1), (4, 1)]])]
    p = make_multilayer(
        [[(STRIP, [])], mid_with_hole, [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L2",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    res, _ = _solve(p, 1.0)
    sig = sigma_s()
    r_exact = (5 + 4) / sig + _r_via(2.0)      # barrel length 2 mm
    assert res.R_ohm == pytest.approx(r_exact, rel=1e-9)


def test_thermal_gap_via_connects_to_nearby_copper():
    """The cell under the via is not copper (thermal-relief knockout),
    but fill copper within the pad footprint (+1 cell) still reaches the
    barrel: the link lands on the nearest copper cell instead of being
    silently dropped."""
    top_with_gap = [(STRIP, [[(5, 0), (7, 0), (7, 1), (5, 1)]])]
    p = make_multilayer(
        [top_with_gap, [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    res, _ = _solve(p, 1.0)
    sig = sigma_s()
    # L0 attaches at col 4 (nearest copper, 1.0 mm from the barrel),
    # L1 at col 5: 4 faces on L0, the barrel, 4 faces on L1 - exact
    r_exact = (4 + 4) / sig + _r_via(1.0)
    assert res.R_ohm == pytest.approx(r_exact, rel=1e-9)
    assert len(res.via_reports) == 1
    assert res.via_reports[0].current_a == pytest.approx(1.0, rel=1e-9)


def test_dead_barrel_is_warned(capsys):
    """A via isolated from the fill by an antipad wider than the search
    radius on all but one layer carries nothing and is reported."""
    bot_with_hole = [(STRIP, [[(3, 0), (8, 0), (8, 1), (3, 1)]])]
    p = make_multilayer(
        [[(STRIP, [])], bot_with_hole],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L0",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    res, _ = _solve(p, 1.0)
    sig = sigma_s()
    assert res.R_ohm == pytest.approx(9 / sig, rel=1e-9)   # L0 alone
    assert res.via_reports == []
    assert "carry no current" in capsys.readouterr().out


def test_via_short_between_electrodes_raises():
    """Both electrodes over the SAME cell on different layers with a via
    there = direct short, no free copper -> error."""
    p = make_multilayer(
        [[(STRIP, [])], [(STRIP, [])]],
        rect1_mm=(5, 0, 6, 1), rect2_mm=(5, 0, 6, 1),
        contact1="L0", contact2="L1",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    stack = raster.rasterize_stack(p, 1.0 * NM)
    e1, e2 = raster.electrode_masks(stack, p)
    with pytest.raises(ElectrodeError, match="directly connected"):
        solver.run_solve(p, stack, e1, e2, 1.0,
                         contact_model="equipotential")


def test_layers_without_via_disconnected():
    p = make_multilayer(
        [[(STRIP, [])], [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1", gap_mm=1.0)   # no vias
    stack = raster.rasterize_stack(p, 1.0 * NM)
    e1, e2 = raster.electrode_masks(stack, p)
    with pytest.raises(ConnectivityError, match="not connected"):
        solver.run_solve(p, stack, e1, e2, 1.0,
                         contact_model="equipotential")


def test_power_split_layers_and_vias():
    """Power accounting: layer + via powers sum to I^2 R."""
    p = make_multilayer(
        [[(STRIP, [])], [(STRIP, [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1",
        vias_mm=[(5.5, 0.5)], gap_mm=1.0)
    res, _ = _solve(p, 1.0, i_test=10.0)
    assert res.power_balance_rel < 1e-9
    assert res.P_vias == pytest.approx(100 * _r_via(1.0), rel=1e-9)
    sig = sigma_s()
    assert res.P_layers[0] == pytest.approx(100 * 5 / sig, rel=1e-9)
    assert res.P_layers[1] == pytest.approx(100 * 4 / sig, rel=1e-9)


def test_pad_polygon_electrode():
    """A polygon-shaped electrode (pad) restricted to one layer."""
    from fill_resistance.geometry import Electrode, Polygon
    from tests.util import rect_mm, ring_mm
    p = make_multilayer([[(STRIP, [])]], (0, 0, 1, 1), (9, 0, 10, 1))
    # replace terminal 1 with a small polygon pad covering the same cell
    p.electrodes1 = [Electrode(
        rect=rect_mm((0.2, 0.2, 0.8, 0.8)),
        contact="all",
        polygons=[Polygon(outline=ring_mm(
            [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]))],
        label="pad TP1.1",
    )]
    res, _ = _solve(p, 1.0)
    sig = sigma_s()
    assert res.R_ohm == pytest.approx(9 / sig, rel=1e-9)  # cols 0..9
