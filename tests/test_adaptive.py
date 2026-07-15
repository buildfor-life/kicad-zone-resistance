"""Adaptive solve path (phase 2): full run_solve equivalence against the
uniform grid across the feature set. On piecewise-linear fields (strips)
the leaf system is EXACT, so those compare at solver precision."""
import numpy as np
import pytest

from fill_resistance import config, raster, solver
from fill_resistance.geometry import Electrode, LayerFill, Polygon, Problem, \
    TrackSeg
from tests.test_capping import _two_layer
from tests.util import NM, make_multilayer, make_problem, rect_mm, \
    strip_problem


def _run(problem, h_mm, model="equipotential", adaptive=False,
         monkeypatch=None, parts=False, freq=0.0):
    if monkeypatch is not None:
        monkeypatch.setattr(config, "ADAPTIVE_CELLS", adaptive)
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    kw = {}
    if parts:
        p1, p2 = raster.electrode_partition(stack, problem)
        kw = dict(parts1=p1, parts2=p2)
    return solver.run_solve(problem, stack, e1, e2, 1.0, freq,
                            contact_model=model, **kw)


def test_strip_close_both_models(monkeypatch):
    """Uniform strip, both contact models. Coarse-fine interfaces carry
    a first-order tangential flux error (laterally offset leaf centers),
    so adaptive R sits up to ~2% LOW of the production R - the narrow
    strip is the worst case (transition rings span most of the width)."""
    for model in ("equipotential", "uniform"):
        p = strip_problem(length=50, width=10, e_len=5)
        ref = _run(p, 0.25, model, adaptive=False, monkeypatch=monkeypatch)
        p2 = strip_problem(length=50, width=10, e_len=5)
        ada = _run(p2, 0.25, model, adaptive=True, monkeypatch=monkeypatch)
        assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=0.02), model
        assert ada.R_ohm <= ref.R_ohm * 1.001    # bias is low, not high
        assert ada.n_free < ref.n_free
        assert ada.power_balance_rel < 1e-9


def test_plate_with_holes_close(monkeypatch):
    holes = []
    for i in range(5):
        for j in range(5):
            x, y = 8 * i + 3, 8 * j + 3
            holes.append([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
    outline = [(0, 0), (40, 0), (40, 40), (0, 40)]

    def prob():
        return make_problem([(outline, holes)], rect1_mm=(0, 15, 2, 25),
                            rect2_mm=(38, 15, 40, 25))

    ref = _run(prob(), 0.1, adaptive=False, monkeypatch=monkeypatch)
    ada = _run(prob(), 0.1, adaptive=True, monkeypatch=monkeypatch)
    assert ada.n_free < 0.5 * ref.n_free
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=0.01)


def test_via_chain_exact(monkeypatch):
    """1-cell strips + via: everything is keep-fine or boundary, so the
    adaptive path must reproduce the exact discrete solution."""
    STRIP = [(0, 0), (10, 0), (10, 1), (0, 1)]

    def prob():
        return make_multilayer(
            [[(STRIP, [])], [(STRIP, [])]],
            rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
            contact1="L0", contact2="L1",
            vias_mm=[(5.5, 0.5)], gap_mm=1.0)

    ref = _run(prob(), 1.0, adaptive=False, monkeypatch=monkeypatch)
    ada = _run(prob(), 1.0, adaptive=True, monkeypatch=monkeypatch)
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=1e-9)
    assert len(ada.via_reports) == 1
    assert ada.via_reports[0].current_a == pytest.approx(1.0, rel=1e-9)


def test_1d_trace_bridge(monkeypatch):
    """Sub-resolution trace bridging two pours: chain cells are pinned
    fine, pours coarsen; R matches the uniform grid closely."""
    pour1 = [(0, 0), (30, 0), (30, 30), (0, 30)]
    pour2 = [(50, 0), (80, 0), (80, 30), (50, 30)]

    def prob():
        seg = TrackSeg(layer_name="F.Cu",
                       points=np.array([[15 * NM, 15 * NM],
                                        [65 * NM, 15 * NM]], dtype=np.int64),
                       width_nm=int(0.2 * NM))
        return Problem(
            board_path="synthetic", net_name="TEST", rho_ohm_m=1.68e-8,
            plating_nm=18_000,
            layers=[LayerFill(
                layer_name="F.Cu", thickness_nm=70_000, z_nm=0,
                polygons=[Polygon(outline=(np.array(pour1) * NM
                                           ).astype(np.int64)),
                          Polygon(outline=(np.array(pour2) * NM
                                           ).astype(np.int64))])],
            vias=[],
            electrodes1=[Electrode(rect=rect_mm((0, 10, 2, 20)))],
            electrodes2=[Electrode(rect=rect_mm((78, 10, 80, 20)))],
            tracks=[seg])

    ref = _run(prob(), 0.5, adaptive=False, monkeypatch=monkeypatch)
    ada = _run(prob(), 0.5, adaptive=True, monkeypatch=monkeypatch)
    assert ada.n_free < 0.6 * ref.n_free
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=2e-3)


def test_buildup_close_on_strip(monkeypatch):
    """Half-coverage buildup strip: buildup cells are pinned fine; the
    remaining deviation is the interface flux bias of the plain half."""
    from tests.test_buildup import _with_buildup

    def prob():
        return _with_buildup(strip_problem(length=50, width=10, e_len=5),
                             [[(25, 0), (50, 0), (50, 10), (25, 10)]])

    ref = _run(prob(), 0.5, adaptive=False, monkeypatch=monkeypatch)
    ada = _run(prob(), 0.5, adaptive=True, monkeypatch=monkeypatch)
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=0.02)


def test_capped_via_close(monkeypatch):
    """Ring + thin-cap mouth (thick_scale) under the adaptive grid."""
    ref = _run(_two_layer(drill_mm=2.0, pad_mm=2.6), 0.25,
               adaptive=False, monkeypatch=monkeypatch)
    ada = _run(_two_layer(drill_mm=2.0, pad_mm=2.6), 0.25,
               adaptive=True, monkeypatch=monkeypatch)
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=2e-3)


def test_part_currents_and_ac(monkeypatch):
    """Per-part currents and the AC path work on leaves."""
    p = strip_problem(length=50, width=10, e_len=5)
    ref = _run(p, 0.5, adaptive=False, monkeypatch=monkeypatch,
               parts=True, freq=2e6)
    p2 = strip_problem(length=50, width=10, e_len=5)
    ada = _run(p2, 0.5, adaptive=True, monkeypatch=monkeypatch,
               parts=True, freq=2e6)
    assert ada.R_ohm == pytest.approx(ref.R_ohm, rel=0.02)
    assert ada.part_currents1[0][1] == pytest.approx(
        ref.part_currents1[0][1], rel=1e-9)      # single part = full current
    assert ada.rs_ratios == ref.rs_ratios


def test_max_cell_size_respected(monkeypatch):
    from fill_resistance.adaptive import _max_block
    assert _max_block(100_000.0) == 16      # 2000 um / 100 um cells
    monkeypatch.setattr(config, "ADAPTIVE_MAX_CELL_UM", 250.0)
    assert _max_block(100_000.0) == 2
    monkeypatch.setattr(config, "ADAPTIVE_MAX_CELL_UM", 50.0)
    assert _max_block(100_000.0) == 1       # never below the fine cell