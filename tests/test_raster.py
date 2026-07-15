import numpy as np
import pytest

from fill_resistance import config, raster, solver
from fill_resistance.errors import (ConnectivityError, ElectrodeError,
                                    GridSizeError)
from tests.util import NM, make_problem, strip_problem


def _stack(problem, h_mm):
    return raster.rasterize_stack(problem, h_mm * NM)


def test_exact_cell_count_square_with_hole():
    # 10x10 mm square, centered 4x4 mm hole, h=1 mm: cell centers at
    # half-integers, no boundary ambiguity -> exactly 100 - 16 cells
    p = make_problem(
        [([(0, 0), (10, 0), (10, 10), (0, 10)],
          [[(3, 3), (7, 3), (7, 7), (3, 7)]])],
        rect1_mm=(0, 0, 1, 10), rect2_mm=(9, 0, 10, 10))
    stack = _stack(p, 1.0)
    assert int(stack.masks[0].sum()) == 100 - 16


def test_hybrid_raster_matches_exact_point_test():
    """The PIL-fill + exact-edge-band rasterizer must be cell-for-cell
    identical to a pure center-in-polygon pass, including awkward
    fractional offsets, concave lobes and a hole."""
    from matplotlib.path import Path as MplPath
    ang = np.linspace(0, 2 * np.pi, 257, endpoint=False)
    r = 7.3 + 1.7 * np.sin(5 * ang) + 0.9 * np.cos(9 * ang + 0.4)
    blob = np.stack([20.05 + r * np.cos(ang), 20.13 + r * np.sin(ang)],
                    axis=1)
    hole = np.stack([20.4 + 2.1 * np.cos(ang), 19.8 + 2.2 * np.sin(ang)],
                    axis=1)
    p = make_problem([(blob.tolist(), [hole.tolist()])],
                     rect1_mm=(14, 19, 16, 21), rect2_mm=(24, 19, 26, 21))
    stack = _stack(p, 0.25)

    ny, nx = stack.shape2d
    xg, yg = stack.cell_centers(0, ny, 0, nx)
    pts = np.column_stack([xg.ravel(), yg.ravel()])

    def exact(ring):
        verts = np.vstack([ring, ring[:1]])
        return MplPath(verts, closed=True).contains_points(pts).reshape(
            ny, nx)

    poly = p.layers[0].polygons[0]
    ref = exact(poly.outline) & ~exact(poly.holes[0])
    assert np.array_equal(stack.masks[0], ref)


def test_margin_cells_are_empty():
    p = strip_problem()
    stack = _stack(p, 0.5)
    m = stack.masks[0]
    assert not m[0, :].any() and not m[-1, :].any()
    assert not m[:, 0].any() and not m[:, -1].any()


def test_electrode_masks_and_counts():
    p = strip_problem(length=50, width=10, e_len=5)
    stack = _stack(p, 0.5)
    e1, e2 = raster.electrode_masks(stack, p)
    # electrodes: 5 mm / 0.5 mm = 10 columns x 20 rows on 1 layer
    assert int(e1.sum()) == 200 and int(e2.sum()) == 200


def test_electrode_off_copper_raises():
    p = make_problem([([(0, 0), (10, 0), (10, 10), (0, 10)], [])],
                     rect1_mm=(20, 20, 25, 25), rect2_mm=(8, 0, 10, 10))
    stack = _stack(p, 0.5)
    with pytest.raises(ElectrodeError, match="does not overlap"):
        raster.electrode_masks(stack, p)


def test_touching_electrodes_raise_equipotential():
    p = make_problem([([(0, 0), (10, 0), (10, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 5, 10), rect2_mm=(5, 0, 10, 10))
    stack = _stack(p, 0.5)
    e1, e2 = raster.electrode_masks(stack, p)   # touch is fine at mask level
    with pytest.raises(ElectrodeError, match="touch"):
        solver.run_solve(p, stack, e1, e2, 1.0,
                         contact_model="equipotential")


def test_disconnected_regions_raise():
    p = make_problem(
        [([(0, 0), (10, 0), (10, 10), (0, 10)], []),
         ([(20, 0), (30, 0), (30, 10), (20, 10)], [])],
        rect1_mm=(0, 0, 2, 10), rect2_mm=(28, 0, 30, 10))
    stack = _stack(p, 0.5)
    e1, e2 = raster.electrode_masks(stack, p)
    with pytest.raises(ConnectivityError, match="not connected"):
        solver.run_solve(p, stack, e1, e2, 1.0,
                         contact_model="equipotential")


def test_islands_dropped():
    # island square not touching the main strip disappears from the mask
    p = make_problem(
        [([(0, 0), (50, 0), (50, 10), (0, 10)], []),
         ([(20, 20), (30, 20), (30, 30), (20, 30)], [])],
        rect1_mm=(0, 0, 5, 10), rect2_mm=(45, 0, 50, 10))
    stack = _stack(p, 0.5)
    e1, e2 = raster.electrode_masks(stack, p)
    before = int(stack.masks.sum())
    solver.run_solve(p, stack, e1, e2, 1.0, contact_model="equipotential")
    after = int(stack.masks.sum())
    assert after < before
    assert after == 100 * 20  # only the strip remains


def test_hard_max_cells_guard(monkeypatch):
    monkeypatch.setattr(config, "CELL_UM_OVERRIDE", 1.0)  # 1 um cells
    p = strip_problem()
    with pytest.raises(GridSizeError, match="M cells"):
        raster.choose_cell_size(p.copper_bbox(), len(p.layers))


def test_cell_override_nonpositive_raises(monkeypatch):
    p = strip_problem()
    for bad in (0.0, -50.0):
        monkeypatch.setattr(config, "CELL_UM_OVERRIDE", bad)
        with pytest.raises(GridSizeError, match="positive"):
            raster.choose_cell_size(p.copper_bbox(), len(p.layers))


def test_auto_cell_size_hits_target():
    # large plane: unclamped regime, cell count tracks TARGET_CELLS
    p = strip_problem(length=200, width=100, e_len=5)
    h = raster.choose_cell_size(p.copper_bbox(), 1)
    ncells = (200.0 * NM / h) * (100.0 * NM / h)
    assert 0.5 * config.TARGET_CELLS < ncells < 2.0 * config.TARGET_CELLS
    # small board: MIN_CELL_UM clamp kicks in instead
    q = strip_problem()  # 50x10 mm
    hq = raster.choose_cell_size(q.copper_bbox(), 1)
    assert hq == pytest.approx(config.MIN_CELL_UM * 1000)
