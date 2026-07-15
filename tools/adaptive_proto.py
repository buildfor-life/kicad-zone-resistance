"""Prototype: quadtree-adaptive grid for the fill-resistance solver.

Coarsens the existing fine raster bottom-up (power-of-two blocks that are
fully copper away from electrodes, erosion-graded per level), builds the
leaf graph fully vectorized via an id-grid (face conductance
g = sigma * overlap / mean-size, which reduces EXACTLY to the production
sigma in the uniform limit), and reuses the production assembly + AMG
solver. Run from the repo root: .venv/Scripts/python tools/adaptive_proto.py

Measured 2026-07-15 (120x120 mm plate, 400 holes, h = 50 um; adversarial:
features everywhere, so geometric refinement has no smooth interior):

    uniform      R = 0.508504 mOhm   5.58M unknowns   27 s   (reference)
    max_block=4  R = 0.506368 mOhm   474k  unknowns    3 s   -0.42%  12x
    max_block=8  R = 0.503386 mOhm   253k  unknowns    1 s   -1.0%   22x
    max_block=16 R = 0.497873 mOhm   213k  unknowns    1 s   -2.1%   26x

    uniform-limit check (max_block=1): rel diff 0.00e+00 vs production.

Coarsening biases R low (coarse cells overestimate conductance where the
field curves); a production version needs true 2:1 balancing + a guard
band, and optionally one residual-driven refine pass, to push the
max_block=4 accuracy to larger blocks. On big-pour boards (smooth
interiors) the unknown ratios are far higher than on this geometry.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fill_resistance import raster, solver
from tests.util import NM, make_problem, strip_problem

MAX_BLOCK = 32          # coarsest leaf = 32 x 32 fine cells


def build_leaves(mask, e1, e2, max_block=MAX_BLOCK):
    """Greedy top-down coarsening. Returns (y0, x0, size) per leaf plus
    an id_grid at fine resolution (-1 = empty)."""
    ny, nx = mask.shape
    pad_y = (-ny) % max_block
    pad_x = (-nx) % max_block
    m = np.pad(mask, ((0, pad_y), (0, pad_x)))
    coars = m & ~np.pad(e1 | e2, ((0, pad_y), (0, pad_x)))
    NY, NX = m.shape

    from scipy import ndimage
    id_grid = np.full((NY, NX), -1, dtype=np.int64)
    covered = np.zeros((NY, NX), dtype=bool)
    y0s, x0s, sizes = [], [], []
    nid = 0
    levels = []
    red = coars.copy()
    s = 1
    while s < max_block:
        red = red.reshape(red.shape[0] // 2, 2, red.shape[1] // 2, 2
                          ).all(axis=(1, 3))
        s *= 2
        # grading: a size-s block must sit in an all-copper 3x3 block
        # neighborhood at its own level, so leaf sizes step down smoothly
        # toward boundaries (guard band + approximate 2:1 balance)
        graded = ndimage.binary_erosion(red, np.ones((3, 3), dtype=bool))
        levels.append((s, graded))
    for s, allc in reversed(levels):
        cov_k = covered.reshape(NY // s, s, NX // s, s).any(axis=(1, 3))
        cand = allc & ~cov_k
        ii, jj = np.nonzero(cand)
        for i_, j_ in zip(ii, jj):
            id_grid[i_ * s:(i_ + 1) * s, j_ * s:(j_ + 1) * s] = nid
            y0s.append(i_ * s)
            x0s.append(j_ * s)
            sizes.append(s)
            nid += 1
        covered |= np.repeat(np.repeat(cand, s, axis=0), s, axis=1)
    fi, fj = np.nonzero(m & ~covered)
    n_fine = len(fi)
    id_grid[fi, fj] = nid + np.arange(n_fine)
    y0s.extend(fi.tolist())
    x0s.extend(fj.tolist())
    sizes.extend([1] * n_fine)
    return (np.array(y0s), np.array(x0s), np.array(sizes),
            id_grid[:ny, :nx])


def leaf_edges(id_grid, sizes, sigma):
    """All leaf-leaf face conductances, vectorized: count shared fine
    faces per leaf pair (= overlap length w), g = sigma * w / mean(sa, sb).
    Uniform limit: w = 1, sizes 1 -> g = sigma (identical to production)."""
    aa, bb, ww = [], [], []
    n = len(sizes)
    for sl_a, sl_b in ((np.s_[:, :-1], np.s_[:, 1:]),
                       (np.s_[:-1, :], np.s_[1:, :])):
        a = id_grid[sl_a].ravel()
        b = id_grid[sl_b].ravel()
        ok = (a >= 0) & (b >= 0) & (a != b)
        key = a[ok] * n + b[ok]
        uniq, counts = np.unique(key, return_counts=True)
        ia = uniq // n
        ib = uniq % n
        g = sigma * counts / (0.5 * (sizes[ia] + sizes[ib]))
        aa.append(ia)
        bb.append(ib)
        ww.append(g)
    return (np.concatenate(aa), np.concatenate(bb), np.concatenate(ww))


def solve_adaptive(problem, h_mm, max_block=MAX_BLOCK):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    t0 = time.perf_counter()
    y0, x0, sizes, id_grid = build_leaves(stack.masks[0], e1[0], e2[0],
                                          max_block)
    a, b, w = leaf_edges(id_grid, sizes, problem.sigma_s(0))
    t_build = time.perf_counter() - t0

    n = len(sizes)
    state = np.ones(n, dtype=np.uint8)
    fine_ids = id_grid[e1[0]]
    state[fine_ids[fine_ids >= 0]] = 2
    fine_ids = id_grid[e2[0]]
    state[fine_ids[fine_ids >= 0]] = 3

    edges = solver.Edges(a=a, b=b, w=w,
                         via_index=np.full(len(a), -1, dtype=np.int32))
    t0 = time.perf_counter()
    A, rhs, idx = solver._assemble(state, edges, None)
    x, info = solver.solve_system(A, rhs)
    t_solve = time.perf_counter() - t0

    V = np.zeros(n)
    V[state == 2] = 1.0
    V[state == 1] = x
    Ie = w * (V[a] - V[b])
    sa, sb = state[a], state[b]
    I1 = float(Ie[sa == 2].sum() - Ie[sb == 2].sum())
    I2 = float(Ie[sb == 3].sum() - Ie[sa == 3].sum())
    R = 1.0 / (0.5 * (I1 + I2))
    mismatch = abs(I1 - I2) / max(abs(I1), abs(I2))
    return R, n, info, t_build, t_solve, mismatch


# --- 1) uniform-limit correctness: max_block=1 must equal production ---
p = strip_problem(length=50, width=10, e_len=5)
stack = raster.rasterize_stack(p, 0.25 * NM)
e1, e2 = raster.electrode_masks(stack, p)
res = solver.run_solve(p, stack, e1, e2, 1.0, contact_model="equipotential")
R_u, n_u, *_ = solve_adaptive(p, 0.25, max_block=1)
print(f"uniform-limit check: production R={res.R_ohm:.12g}, "
      f"prototype R={R_u:.12g}, rel diff {abs(R_u / res.R_ohm - 1):.2e}")

# --- 2) the payoff case: 120x120 plate, 400 holes, h = 50 um ---
holes = []
for i in range(20):
    for j in range(20):
        x, y = 3 + 6 * i, 3 + 6 * j
        holes.append([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
outline = [(0, 0), (120, 0), (120, 120), (0, 120)]
big = make_problem([(outline, holes)],
                   rect1_mm=(0, 55, 2, 65), rect2_mm=(118, 55, 120, 65))

t0 = time.perf_counter()
stack = raster.rasterize_stack(big, 0.05 * NM)
e1, e2 = raster.electrode_masks(stack, big)
ref = solver.run_solve(big, stack, e1, e2, 1.0,
                       contact_model="equipotential")
t_ref = time.perf_counter() - t0
print(f"\nuniform 50um : R = {ref.R_ohm * 1e3:.6g} mOhm, "
      f"{ref.solve_info.n_unknowns} unknowns, {t_ref:.1f} s total "
      f"({ref.solve_info.method})")

for mb in (4, 8, 16, 32):
    R_a, n_a, info, t_b, t_s, mm = solve_adaptive(big, 0.05, max_block=mb)
    print(f"adaptive mb={mb:2d}: R = {R_a * 1e3:.6g} mOhm, "
          f"{info.n_unknowns:8d} unknowns, build {t_b:.1f} s + "
          f"solve {t_s:.1f} s, rel diff {abs(R_a / ref.R_ohm - 1):.2e}, "
          f"ratio {ref.solve_info.n_unknowns / info.n_unknowns:.0f}x")
