"""Benchmark for the adaptive quadtree grid engine
(fill_resistance/quadtree.py, phase 1 of the adaptive-cell work).

    .venv/Scripts/python tools/adaptive_proto.py

Solves a feature-dense 120x120 mm plate (400 holes) at h = 50 um on the
uniform grid and on the balanced quadtree (engine defaults: guard=4,
max_block=32), using the production assembly + solver for both.

Measured 2026-07-15 on this machine:

    uniform          R = 0.508504 mOhm   5.58M unknowns   ~30-40 s
    adaptive guard=4 R = 0.502793 mOhm    823k unknowns    ~8 s   -1.1%
    adaptive guard=8 R = 0.506102 mOhm   1.74M unknowns   ~12 s   -0.47%

    uniform-limit check (max_block=1): rel diff 0.00e+00 vs production.

This is the ADVERSARIAL case (features at 6 mm pitch everywhere, no
smooth interior); big-pour boards coarsen far more aggressively. The
residual bias is the first-order coarse-fine interface flux - phase 4
(gradient-corrected fluxes / solution-adaptive refinement) is the
lever if tighter accuracy per leaf is needed.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fill_resistance import quadtree, raster, solver
from tests.util import NM, make_problem, strip_problem


def solve_adaptive(problem, h_mm, **kw):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    t0 = time.perf_counter()
    grid = quadtree.build_leaves(stack.masks[0],
                                 keep_fine=(e1[0] | e2[0]), **kw)
    ia, ib, g = quadtree.leaf_edges(grid, problem.sigma_s(0))
    t_build = time.perf_counter() - t0

    state = np.ones(grid.n, dtype=np.uint8)
    for e, code in ((e1[0], 2), (e2[0], 3)):
        ids = grid.id_grid[e]
        state[ids[ids >= 0]] = code
    edges = solver.Edges(a=ia, b=ib, w=g,
                         via_index=np.full(len(ia), -1, dtype=np.int32))
    t0 = time.perf_counter()
    A, rhs, _ = solver._assemble(state, edges, None)
    x, info = solver.solve_system(A, rhs)
    t_solve = time.perf_counter() - t0

    V = np.zeros(grid.n)
    V[state == 2] = 1.0
    V[state == 1] = x
    Ie = g * (V[ia] - V[ib])
    sa, sb = state[ia], state[ib]
    I1 = float(Ie[sa == 2].sum() - Ie[sb == 2].sum())
    I2 = float(Ie[sb == 3].sum() - Ie[sa == 3].sum())
    return 1.0 / (0.5 * (I1 + I2)), grid, info, t_build, t_solve


# --- 1) uniform-limit correctness ---
p = strip_problem(length=50, width=10, e_len=5)
stack = raster.rasterize_stack(p, 0.25 * NM)
e1, e2 = raster.electrode_masks(stack, p)
res = solver.run_solve(p, stack, e1, e2, 1.0, contact_model="equipotential")
R_u, *_ = solve_adaptive(p, 0.25, max_block=1)
print(f"uniform-limit check: rel diff {abs(R_u / res.R_ohm - 1):.2e}")

# --- 2) feature-dense plate at h = 50 um ---
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
print(f"uniform : R = {ref.R_ohm * 1e3:.6g} mOhm, "
      f"{ref.solve_info.n_unknowns} unknowns, {t_ref:.1f} s "
      f"({ref.solve_info.method})")

R_a, grid, info, t_b, t_s = solve_adaptive(big, 0.05)
print(f"adaptive: R = {R_a * 1e3:.6g} mOhm, {info.n_unknowns} unknowns, "
      f"build {t_b:.1f} s + solve {t_s:.1f} s ({info.method})")
print(f"rel diff {abs(R_a / ref.R_ohm - 1):.2e}, "
      f"{ref.solve_info.n_unknowns / info.n_unknowns:.1f}x fewer unknowns, "
      f"max leaf {int(grid.size.max())} cells, balanced="
      f"{quadtree.balanced(grid)}")
