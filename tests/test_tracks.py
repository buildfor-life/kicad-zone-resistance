"""Track (trace) conductor tests: capsule / arc-band outline generation,
solves on rasterized traces, and the 1D resistor-chain model for traces
narrower than TRACK_1D_FACTOR grid cells."""
import math

import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.geometry import (Electrode, LayerFill, Polygon, Problem,
                                      TrackSeg, arc_band_ring, capsule_ring,
                                      load_problem, save_problem)
from tests.util import NM, rect_mm, sigma_s

TOL_NM = 10_000
RHO = 1.68e-8
T_M = 70e-6


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


def _seg(points_mm, w_mm, layer="F.Cu") -> TrackSeg:
    pts = (np.asarray(points_mm, dtype=float) * NM).astype(np.int64)
    return TrackSeg(layer_name=layer, points=pts, width_nm=int(w_mm * NM))


def _seg_problem(segs, rect1, rect2, fills_mm=(), t_um=70.0):
    polys = [Polygon(outline=(np.asarray(o, dtype=float) * NM
                              ).astype(np.int64)) for o in fills_mm]
    return Problem(
        board_path="synthetic", net_name="TEST", rho_ohm_m=RHO,
        plating_nm=18_000,
        layers=[LayerFill(layer_name="F.Cu", thickness_nm=int(t_um * 1000),
                          z_nm=0, polygons=polys)],
        vias=[],
        electrodes1=[Electrode(rect=rect_mm(rect1))],
        electrodes2=[Electrode(rect=rect_mm(rect2))],
        tracks=segs,
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


def test_narrow_trace_1d_matches_fine_raster():
    """A 0.2 mm trace at h = 1 mm (1D chain) must agree with the same
    trace finely rasterized at h = 0.05 mm (4 cells wide) and with the
    analytic R between the electrode inner edges."""
    seg = _seg([(1, 0.5), (41, 0.5)], 0.2)
    p = _seg_problem([seg], (0, 0, 2, 1), (40, 0, 42, 1))
    res_1d, stack = _solve(p, 1.0)
    assert stack.chain is not None and stack.chain.any()
    res_fine, stack_f = _solve(_seg_problem([seg], (0, 0, 2, 1),
                                            (40, 0, 42, 1)), 0.05)
    assert stack_f.chain is None or not stack_f.chain.any()
    r_analytic = RHO * 0.038 / (0.2e-3 * T_M)      # between x=2 and x=40
    assert res_fine.R_ohm == pytest.approx(r_analytic, rel=0.03)
    assert res_1d.R_ohm == pytest.approx(res_fine.R_ohm, rel=0.06)


def test_diagonal_narrow_trace_no_staircase():
    """1D links carry the TRUE arc length: a diagonal trace must not be
    inflated by the 4-connected staircase (which would be up to +41%)."""
    seg = _seg([(1, 1), (25, 19)], 0.2)
    p = _seg_problem([seg], (0, 0, 2, 2), (24, 18, 26, 20))
    res, _ = _solve(p, 1.0)
    L = math.hypot(24, 18) * 1e-3                  # 30 mm
    r_full = RHO * L / (0.2e-3 * T_M)
    assert res.R_ohm < 1.05 * r_full               # no staircase inflation
    assert res.R_ohm == pytest.approx(r_full, rel=0.08)


def test_narrow_arc_trace_uses_arc_length():
    """Quarter-circle 0.2 mm trace, r = 10 mm, as a 1D chain: R follows
    the arc length (a chord-based length would read ~10% low)."""
    seg = TrackSeg(layer_name="F.Cu", points=np.array(
        [[10 * NM, 0],
         [int(round(10 * NM / math.sqrt(2))),
          int(round(10 * NM / math.sqrt(2)))],
         [0, 10 * NM]], dtype=np.int64), width_nm=int(0.2 * NM))
    p = _seg_problem([seg], (9, -1, 11, 1), (-1, 9, 1, 11))
    res, _ = _solve(p, 0.5)
    # the electrode rects cover the arc where y < 1 (resp. x < 1), so the
    # free span is theta in [asin(0.1), pi/2 - asin(0.1)]
    th = math.asin(0.1)
    r_arc = RHO * ((math.pi / 2 - 2 * th) * 10e-3) / (0.2e-3 * T_M)
    assert res.R_ohm == pytest.approx(r_arc, rel=0.06)


def test_narrow_trace_bridges_pours():
    """A sub-resolution trace joins two pours: without it they are
    disconnected; with it R is dominated by the trace's gap length."""
    from fill_resistance.errors import ConnectivityError
    pour1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
    pour2 = [(30, 0), (40, 0), (40, 10), (30, 10)]
    rects = ((0, 0, 2, 10), (38, 0, 40, 10))
    bare = _seg_problem([], *rects, fills_mm=(pour1, pour2))
    stack = raster.rasterize_stack(bare, 1.0 * NM)
    e1, e2 = raster.electrode_masks(stack, bare)
    with pytest.raises(ConnectivityError):
        solver.run_solve(bare, stack, e1, e2, 1.0,
                         contact_model="equipotential")

    seg = _seg([(5, 5), (35, 5)], 0.2)
    bridged = _seg_problem([seg], *rects, fills_mm=(pour1, pour2))
    res, _ = _solve(bridged, 1.0)
    r_gap = RHO * 0.020 / (0.2e-3 * T_M)           # 20 mm between pours
    assert res.R_ohm == pytest.approx(r_gap, rel=0.10)
    assert res.power_balance_rel < 1e-9


def test_wide_track_still_rasterized():
    """At or above the width threshold the trace is rasterized normally
    and no chain cells appear."""
    seg = _seg([(1, 2), (19, 2)], 2.0)
    p = _seg_problem([seg], (0, 1, 2, 3), (18, 1, 20, 3))
    res, stack = _solve(p, 0.25)
    assert stack.chain is None or not stack.chain.any()
    assert int(stack.masks.sum()) > 300            # a real 2D band
    assert np.isfinite(res.R_ohm) and res.R_ohm > 0


def test_json_v5_roundtrip_with_tracks(tmp_path):
    seg = _seg([(1, 0.5), (41, 0.5)], 0.2)
    p = _seg_problem([seg], (0, 0, 2, 1), (40, 0, 42, 1))
    f = tmp_path / "d.json"
    save_problem(p, f)
    q = load_problem(f)
    assert len(q.tracks) == 1
    assert q.tracks[0].layer_name == "F.Cu"
    assert q.tracks[0].width_nm == int(0.2 * NM)
    assert np.array_equal(q.tracks[0].points, p.tracks[0].points)
    r_p, _ = _solve(p, 1.0)
    r_q, _ = _solve(q, 1.0)
    assert r_q.R_ohm == pytest.approx(r_p.R_ohm, rel=1e-12)


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


def test_pad_copper_bridges_track_junction():
    """Two traces meet ON an SMD pad, their rounded ends 0.5 mm apart:
    the junction only exists through the pad copper (board_io stamps
    the net's pad shapes onto their layers). Without the pad the net
    is severed - at both track models (rasterized and 1D chain)."""
    from fill_resistance.errors import ConnectivityError
    tabs = [[(0, 4.5), (1, 4.5), (1, 5.5), (0, 5.5)],
            [(19, 4.5), (20, 4.5), (20, 5.5), (19, 5.5)]]
    pad = [(9.25, 4.4), (10.75, 4.4), (10.75, 5.6), (9.25, 5.6)]
    segs = [_seg([(0.5, 5), (9.5, 5)], 0.5),
            _seg([(10.5, 5), (19.5, 5)], 0.5)]
    r1, r2 = (0, 4.5, 1, 5.5), (19, 4.5, 20, 5.5)

    for h in (0.1, 0.25):          # 5 cells: outlines; 2 cells: 1D chains
        res, _ = _solve(_seg_problem(segs, r1, r2, fills_mm=tabs + [pad]), h)
        # ~36 squares of 0.5 mm trace + tabs/pad: sanity-band the value
        assert 0.007 < res.R_ohm < 0.011

        with pytest.raises(ConnectivityError):
            _solve(_seg_problem(segs, r1, r2, fills_mm=tabs), h)
