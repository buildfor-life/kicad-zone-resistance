"""Track (trace) conductor tests: capsule / arc-band outline generation
and solves on rasterized traces. The 1-cell-wide capsule chain is exact;
the arc band is checked against the analytic annular-sector resistance."""
import math

import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.geometry import (Electrode, LayerFill, Polygon, Problem,
                                      arc_band_ring, capsule_ring)
from tests.util import NM, rect_mm, sigma_s

TOL_NM = 10_000


def _track_problem(rings, rect1, rect2, t_um=70.0):
    return Problem(
        board_path="synthetic", net_name="TEST", rho_ohm_m=1.68e-8,
        plating_nm=18_000,
        layers=[LayerFill(layer_name="F.Cu", thickness_nm=int(t_um * 1000),
                          z_nm=0,
                          polygons=[Polygon(outline=r) for r in rings])],
        vias=[],
        electrodes1=[Electrode(rect=rect_mm(rect1))],
        electrodes2=[Electrode(rect=rect_mm(rect2))],
    )


def _solve(problem, h_mm):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, 1.0,
                            contact_model="equipotential"), stack


def test_straight_track_exact_chain():
    """A 1.2 mm wide capsule at h = 1 mm rasterizes to a single-cell-high
    row (the grid origin floats with the polygon bbox, so the count is
    taken from the mask): an N-cell chain solves to exactly (N-1) faces."""
    ring = capsule_ring(1 * NM, NM // 2, 9 * NM, NM // 2,
                        int(1.2 * NM), TOL_NM)
    p = _track_problem([ring], (0, 0, 1.5, 1), (8.5, 0, 10, 1))
    res, stack = _solve(p, 1.0)
    m = stack.masks[0]
    rows = np.flatnonzero(m.any(axis=1))
    assert len(rows) == 1                     # one 1-cell-high chain
    n = int(m.sum())
    assert n >= 8
    assert res.R_ohm == pytest.approx((n - 1) / sigma_s(), rel=1e-9)


def test_capsule_zero_length_is_circle():
    ring = capsule_ring(5 * NM, 5 * NM, 5 * NM, 5 * NM, 2 * NM, TOL_NM)
    d = np.hypot(ring[:, 0] - 5 * NM, ring[:, 1] - 5 * NM)
    assert np.allclose(d, NM, atol=TOL_NM + 2)
    assert len(ring) >= 8


def test_capsule_ring_geometry():
    """Every outline point lies on the capsule boundary: at half-width
    from the centerline segment."""
    ring = capsule_ring(2 * NM, 3 * NM, 17 * NM, 11 * NM,
                        int(1.5 * NM), TOL_NM)
    a = np.array([2 * NM, 3 * NM], dtype=float)
    b = np.array([17 * NM, 11 * NM], dtype=float)
    ab = b - a
    t = np.clip(((ring - a) @ ab) / (ab @ ab), 0.0, 1.0)
    d = np.hypot(*(ring - (a + t[:, None] * ab)).T)
    assert np.allclose(d, 0.75 * NM, atol=TOL_NM + 2)


def test_arc_band_ring_geometry():
    """Arc-band points lie on the annulus walls or on the end caps."""
    start, mid, end = ((10 * NM, 0), (int(10 * NM / math.sqrt(2)),
                                      int(10 * NM / math.sqrt(2))),
                      (0, 10 * NM))
    ring = arc_band_ring(start, mid, end, 1 * NM, TOL_NM).astype(float)
    r = np.hypot(ring[:, 0], ring[:, 1])
    on_annulus = (np.abs(r - 10.5 * NM) < TOL_NM + 2) \
        | (np.abs(r - 9.5 * NM) < TOL_NM + 2)
    d_start = np.hypot(ring[:, 0] - start[0], ring[:, 1] - start[1])
    d_end = np.hypot(ring[:, 0] - end[0], ring[:, 1] - end[1])
    on_caps = (d_start < 0.5 * NM + TOL_NM + 2) | (d_end < 0.5 * NM + TOL_NM + 2)
    assert (on_annulus | on_caps).all()


def test_collinear_arc_degrades_to_capsule():
    cap = capsule_ring(0, 0, 10 * NM, 0, NM, TOL_NM)
    band = arc_band_ring((0, 0), (5 * NM, 0), (10 * NM, 0), NM, TOL_NM)
    assert np.array_equal(cap, band)


def test_arc_track_matches_annular_sector():
    """90 deg arc trace, r = 10 mm, w = 1 mm: R = theta / (sigma *
    ln(r_out/r_in)) between the radial end faces (electrodes cover the
    end caps). The staircase on the curved walls narrows the band, so R
    converges to the analytic value from above as h shrinks."""
    start = (10 * NM, 0)
    mid = (int(round(10 * NM / math.sqrt(2))),
           int(round(10 * NM / math.sqrt(2))))
    end = (0, 10 * NM)
    ring = arc_band_ring(start, mid, end, 1 * NM, TOL_NM)

    def solve_at(h_mm):
        p = _track_problem([ring], rect1=(9.3, -0.8, 10.7, 0.05),
                           rect2=(-0.8, 9.3, 0.05, 10.7))
        res, _ = _solve(p, h_mm)
        return res.R_ohm

    r_exact = (math.pi / 2) / (sigma_s() * math.log(10.5 / 9.5))
    err_coarse = abs(solve_at(0.1) / r_exact - 1)
    err_fine = abs(solve_at(0.05) / r_exact - 1)
    assert err_fine < err_coarse            # converges toward analytic
    assert err_fine < 0.04


def test_track_unions_with_fill():
    """A trace overlapping a plate merges into one conductor: the mask is
    the union, and R drops when the trace bridges a slot."""
    plate = [(0, 0), (20, 0), (20, 10), (0, 10)]
    slot = [(9, 2), (11, 2), (11, 10), (9, 10)]     # slot open to the top
    plate_poly = Polygon(
        outline=np.array([(x * NM, y * NM) for x, y in plate]),
        holes=[np.array([(x * NM, y * NM) for x, y in slot])])
    bridge = capsule_ring(6 * NM, 6 * NM, 14 * NM, 6 * NM,
                          int(1.2 * NM), TOL_NM)

    def problem(polys):
        return Problem(
            board_path="synthetic", net_name="TEST", rho_ohm_m=1.68e-8,
            plating_nm=18_000,
            layers=[LayerFill(layer_name="F.Cu", thickness_nm=70_000,
                              z_nm=0, polygons=polys)],
            vias=[],
            electrodes1=[Electrode(rect=rect_mm((0, 0, 1, 10)))],
            electrodes2=[Electrode(rect=rect_mm((19, 0, 20, 10)))],
        )

    r_plate, s_plate = _solve(problem([plate_poly]), 0.25)
    r_both, s_both = _solve(problem([plate_poly,
                                     Polygon(outline=bridge)]), 0.25)
    assert int(s_both.masks.sum()) > int(s_plate.masks.sum())
    assert r_both.R_ohm < 0.75 * r_plate.R_ohm    # bridge shortens the detour
    assert r_both.power_balance_rel < 1e-9
