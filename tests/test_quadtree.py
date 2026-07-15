"""Quadtree grid engine tests (phase 1): exact uniform limit against the
production graph and solver, partition/alignment/balance invariants, and
adaptive-vs-fine R agreement."""
import numpy as np
import pytest
from scipy import ndimage

from fill_resistance import quadtree, raster, solver
from tests.util import NM, make_problem, strip_problem


def _plate_with_holes(n=5, size_mm=40.0):
    holes = []
    pitch = size_mm / n
    for i in range(n):
        for j in range(n):
            x, y = pitch * (i + 0.4), pitch * (j + 0.4)
            holes.append([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
    outline = [(0, 0), (size_mm, 0), (size_mm, size_mm), (0, size_mm)]
    return make_problem([(outline, holes)],
                        rect1_mm=(0, 15, 2, 25),
                        rect2_mm=(size_mm - 2, 15, size_mm, 25))


def _solve_on_leaves(problem, stack, e1, e2, grid):
    """Equipotential mini-solve on the leaf graph, using the production
    assembly and linear solver."""
    ia, ib, g = quadtree.leaf_edges(grid, problem.sigma_s(0))
    state = np.ones(grid.n, dtype=np.uint8)
    for e, code in ((e1[0], 2), (e2[0], 3)):
        ids = grid.id_grid[e]
        state[ids[ids >= 0]] = code
    edges = solver.Edges(a=ia, b=ib, w=g,
                         via_index=np.full(len(ia), -1, dtype=np.int32))
    A, rhs, _ = solver._assemble(state, edges, None)
    x, _ = solver.solve_system(A, rhs)
    V = np.zeros(grid.n)
    V[state == 2] = 1.0
    V[state == 1] = x
    Ie = g * (V[ia] - V[ib])
    sa, sb = state[ia], state[ib]
    I1 = float(Ie[sa == 2].sum() - Ie[sb == 2].sum())
    I2 = float(Ie[sb == 3].sum() - Ie[sa == 3].sum())
    return 1.0 / (0.5 * (I1 + I2))


def test_uniform_limit_graph_identical():
    """max_block=1: one leaf per cell, and the edge list matches the
    production in-plane graph exactly (same pairs, conductance sigma)."""
    p = _plate_with_holes()
    stack = raster.rasterize_stack(p, 0.5 * NM)
    grid = quadtree.build_leaves(stack.masks[0], max_block=1)
    assert int(stack.masks.sum()) == grid.n
    assert (grid.size == 1).all()

    ny, nx = stack.shape2d
    flat_of_leaf = grid.y0.astype(np.int64) * nx + grid.x0
    ia, ib, g = quadtree.leaf_edges(grid, p.sigma_s(0))
    ours = np.sort(np.stack([
        np.minimum(flat_of_leaf[ia], flat_of_leaf[ib]),
        np.maximum(flat_of_leaf[ia], flat_of_leaf[ib])], axis=1), axis=0)

    edges = solver.build_edges(stack, p, [p.sigma_s(0)])
    ref = np.sort(np.stack([np.minimum(edges.a, edges.b),
                            np.maximum(edges.a, edges.b)], axis=1), axis=0)
    assert ours.shape == ref.shape
    assert np.array_equal(np.sort(ours.view("i8,i8"), order=["f0", "f1"],
                                  axis=0),
                          np.sort(ref.view("i8,i8"), order=["f0", "f1"],
                                  axis=0))
    assert np.allclose(g, p.sigma_s(0), rtol=0, atol=0)


def test_uniform_limit_R_matches_production():
    p = strip_problem(length=50, width=10, e_len=5)
    stack = raster.rasterize_stack(p, 0.25 * NM)
    e1, e2 = raster.electrode_masks(stack, p)
    ref = solver.run_solve(p, stack, e1, e2, 1.0,
                           contact_model="equipotential")
    stack2 = raster.rasterize_stack(p, 0.25 * NM)
    e1b, e2b = raster.electrode_masks(stack2, p)
    grid = quadtree.build_leaves(stack2.masks[0], max_block=1)
    R = _solve_on_leaves(p, stack2, e1b, e2b, grid)
    assert R == pytest.approx(ref.R_ohm, rel=1e-12)


def test_partition_alignment_and_balance():
    p = _plate_with_holes()
    stack = raster.rasterize_stack(p, 0.1 * NM)
    mask = stack.masks[0]
    grid = quadtree.build_leaves(mask, max_block=32)

    # exact partition of the copper
    assert int((grid.size.astype(np.int64) ** 2).sum()) == int(mask.sum())
    assert (grid.id_grid >= 0).sum() == int(mask.sum())
    assert not (grid.id_grid[~mask] >= 0).any()
    counts = np.bincount(grid.id_grid[grid.id_grid >= 0], minlength=grid.n)
    assert np.array_equal(counts, grid.size.astype(np.int64) ** 2)

    # power-of-two sizes, aligned to their own size
    assert np.array_equal(grid.size & (grid.size - 1),
                          np.zeros_like(grid.size))
    assert (grid.y0 % grid.size == 0).all()
    assert (grid.x0 % grid.size == 0).all()

    # 2:1 balance and real coarsening (max size is geometry-limited by
    # the guard distance, not by max_block, on this feature-dense plate)
    assert quadtree.balanced(grid)
    assert grid.n < 0.5 * int(mask.sum())
    assert grid.size.max() >= 4


def test_boundary_and_keep_fine_stay_fine():
    p = _plate_with_holes()
    stack = raster.rasterize_stack(p, 0.1 * NM)
    mask = stack.masks[0]
    keep = np.zeros_like(mask)
    keep[50:60, 50:60] = True
    grid = quadtree.build_leaves(mask, keep_fine=keep, max_block=32)

    boundary = mask & ndimage.binary_dilation(~mask)
    assert (grid.size[grid.id_grid[boundary]] == 1).all()
    assert (grid.size[grid.id_grid[keep & mask]] == 1).all()


def test_adaptive_R_close_to_fine():
    """Adaptive leaves reproduce the fine-uniform R within 1% on the
    holey plate (features everywhere - the adversarial case)."""
    p = _plate_with_holes()
    stack = raster.rasterize_stack(p, 0.1 * NM)
    e1, e2 = raster.electrode_masks(stack, p)
    ref = solver.run_solve(p, stack, e1, e2, 1.0,
                           contact_model="equipotential")

    stack2 = raster.rasterize_stack(p, 0.1 * NM)
    e1b, e2b = raster.electrode_masks(stack2, p)
    grid = quadtree.build_leaves(stack2.masks[0],
                                 keep_fine=(e1b[0] | e2b[0]))
    R = _solve_on_leaves(p, stack2, e1b, e2b, grid)
    assert grid.n < 0.4 * ref.solve_info.n_unknowns
    assert R == pytest.approx(ref.R_ohm, rel=0.01)
