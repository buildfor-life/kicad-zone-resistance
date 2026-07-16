"""Via ring-copper and drill-mouth (capping) tests. THT pads (kind
'pad') skip both, which doubles as the feature-off reference."""
import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.errors import ConnectivityError
from tests.util import NM, make_multilayer


def _two_layer(width_mm=5.0, drill_mm=0.3, pad_mm=0.6, kind="via",
               capped=True, cap_um=15.0, hole_mm=None,
               cap_max_drill_mm=10.0):
    """10 x width strip on both (outer-named) layers, e1 left on F.Cu,
    e2 right on B.Cu, one via mid-strip. Optionally a circular hole in
    the F.Cu fill around the via (ring-bridging scenario). The cap-drill
    threshold defaults to 10 mm here (= every drill capped) so the tests
    exercise the mouth treatment itself; the threshold has its own test."""
    y = width_mm / 2
    strip = [(0, 0), (10, 0), (10, width_mm), (0, width_mm)]
    holes = []
    if hole_mm is not None:
        ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        holes = [[(5 + hole_mm * np.cos(a), y + hole_mm * np.sin(a))
                  for a in ang]]
    p = make_multilayer(
        [[(strip, holes)], [(strip, [])]],
        rect1_mm=(0, 0, 1, width_mm), rect2_mm=(9, 0, 10, width_mm),
        contact1="F.Cu", contact2="B.Cu",
        vias_mm=[(5, y)], gap_mm=1.0, drill_mm=drill_mm)
    p.layers[0].layer_name = "F.Cu"
    p.layers[1].layer_name = "B.Cu"
    p.vias[0].kind = kind
    p.vias[0].pad_nm = int(pad_mm * NM)
    p.vias_capped = capped
    p.cap_plating_nm = int(cap_um * 1000)
    p.cap_max_drill_nm = int(cap_max_drill_mm * NM)
    return p


def _solve(problem, h_mm):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, 1.0,
                            contact_model="equipotential"), stack


def test_cap_at_foil_thickness_is_identity():
    """cap thickness == foil thickness makes every mouth scale exactly 1,
    so the result equals the feature-off reference (a 'pad'-kind barrel,
    which skips rings and mouths) with the mouth fully inside copper."""
    r_cap, _ = _solve(_two_layer(capped=True, cap_um=70.0), 0.1)
    # a bare 'pad' barrel (solder_filled defaults False) is plating-only,
    # exactly like the via's
    r_ref, _ = _solve(_two_layer(kind="pad"), 0.1)
    assert r_cap.R_ohm == pytest.approx(r_ref.R_ohm, rel=1e-9)


def test_mouth_ordering_solid_capped_uncapped():
    """A large mouth in the current path: R(solid) < R(15 um cap) <
    R(open hole), and the barrel stays connected through the ring."""
    kw = dict(drill_mm=2.0, pad_mm=2.6)
    r_solid, _ = _solve(_two_layer(capped=True, cap_um=70.0, **kw), 0.25)
    r_cap, s_cap = _solve(_two_layer(capped=True, cap_um=15.0, **kw), 0.25)
    r_open, s_open = _solve(_two_layer(capped=False, **kw), 0.25)
    assert r_solid.R_ohm < r_cap.R_ohm < r_open.R_ohm
    assert s_cap.thick_scale is not None
    # open mouths remove the fully covered cells from the copper
    assert int(s_open.masks.sum()) < int(s_cap.masks.sum())
    assert r_open.power_balance_rel < 1e-9


def test_ring_bridges_fill_gap():
    """F.Cu fill has a 1 mm-radius hole around the via; the 2.4 mm pad
    ring bridges it. A 'pad'-kind barrel (no ring) stays disconnected."""
    p = _two_layer(drill_mm=0.3, pad_mm=2.4, hole_mm=1.0)
    res, _ = _solve(p, 0.2)
    assert np.isfinite(res.R_ohm) and res.R_ohm > 0
    assert len(res.via_reports) == 1

    bare = _two_layer(drill_mm=0.3, pad_mm=0.0, kind="pad", hole_mm=1.0)
    stack = raster.rasterize_stack(bare, 0.2 * NM)
    e1, e2 = raster.electrode_masks(stack, bare)
    with pytest.raises(ConnectivityError):
        solver.run_solve(bare, stack, e1, e2, 1.0,
                         contact_model="equipotential")


def test_subcell_mouth_perturbs_gently():
    """A 0.3 mm mouth at 0.5 mm cells must not knock out whole cells:
    the area-weighted scaling changes R only slightly."""
    r_solid, _ = _solve(_two_layer(capped=True, cap_um=70.0), 0.5)
    r_open, s = _solve(_two_layer(capped=False), 0.5)
    assert int(s.masks.sum()) == int(_solve(
        _two_layer(capped=True, cap_um=70.0), 0.5)[1].masks.sum())
    assert r_solid.R_ohm <= r_open.R_ohm <= 1.05 * r_solid.R_ohm


def test_cap_drill_threshold():
    """Drills above cap_max_drill_nm stay open even with capping on: a
    2 mm drill over a 0.5 mm threshold behaves exactly like uncapped,
    while a threshold above the drill restores the cap."""
    kw = dict(drill_mm=2.0, pad_mm=2.6)
    r_big, s_big = _solve(_two_layer(capped=True, cap_max_drill_mm=0.5,
                                     **kw), 0.25)
    r_open, s_open = _solve(_two_layer(capped=False, **kw), 0.25)
    assert r_big.R_ohm == pytest.approx(r_open.R_ohm, rel=1e-12)
    assert int(s_big.masks.sum()) == int(s_open.masks.sum())

    r_cap, _ = _solve(_two_layer(capped=True, cap_max_drill_mm=2.1,
                                 **kw), 0.25)
    assert r_cap.R_ohm < r_open.R_ohm


def test_capping_json_roundtrip(tmp_path):
    from fill_resistance.geometry import load_problem, save_problem
    p = _two_layer(capped=False, cap_um=12.0, cap_max_drill_mm=0.8)
    f = tmp_path / "d.json"
    save_problem(p, f)
    q = load_problem(f)
    assert q.vias_capped is False
    assert q.cap_plating_nm == 12_000
    assert q.cap_max_drill_nm == 800_000
    r_p, _ = _solve(p, 0.25)
    r_q, _ = _solve(q, 0.25)
    assert r_q.R_ohm == pytest.approx(r_p.R_ohm, rel=1e-12)
