"""Adaptive-grid solve path (phase 2): maps the fully rasterized problem
onto per-layer balanced quadtree leaf graphs (quadtree.py), solves with
the production assembly/AMG, and expands every field back to the fine
grid, so plots, reports and dumps are unchanged.

Enabled via config.ADAPTIVE_CELLS (dialog checkbox "adaptive cells").
Every fine cell that carries anything non-uniform - electrodes, 1D
trace-chain cells, solder buildup, via-mouth thickness scaling - is
pinned at the fine size (keep_fine), so all coarser leaves have the
plain layer conductance and the leaf system reduces EXACTLY to the
production system wherever the grid is fine. The minimum element size
is therefore the grid cell size itself; ADAPTIVE_MAX_CELL_UM caps the
coarsest leaf.

ACCURACY: coarse-fine interfaces carry a first-order two-point-flux
error (centers of different-size neighbors are laterally offset), which
biases R LOW by ~0.5-2% depending on geometry - worst where transition
rings span much of the current path (narrow strips), mild on large
pours. Symmetric fine pairs under a coarse face cancel pairwise; the
residue comes from unpaired larger-neighbor faces. Gradient-corrected
interface fluxes (phase 4) are the known cure if tighter accuracy per
leaf is ever needed.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from . import config, quadtree, skin
from . import solver as sv
from .errors import ConnectivityError
from .geometry import Problem
from .raster import RasterStack


def _max_block(h_nm: float) -> int:
    mb = 1
    while mb * 2 * h_nm <= config.ADAPTIVE_MAX_CELL_UM * 1000.0:
        mb *= 2
    return mb


def _nodes_of_cells(grids, offs, li: int, cells2d: np.ndarray) -> np.ndarray:
    """Node ids of the (copper) fine cells selected by a 2D bool mask."""
    ids = grids[li].id_grid[cells2d]
    ids = ids[ids >= 0].astype(np.int64)
    return offs[li] + np.unique(ids)


def run_solve_adaptive(problem: Problem, stack: RasterStack,
                       e1: np.ndarray, e2: np.ndarray, i_test: float,
                       freq_hz: float, contact_model: str,
                       parts1: list | None,
                       parts2: list | None) -> sv.Result:
    timings = {}
    L, ny, nx = stack.masks.shape
    h_m = stack.h_nm * 1e-9
    plane = ny * nx

    sigmas, rs_ratios, via_factor, sigma_buildup = \
        sv._conductance_params(problem, stack, freq_hz)

    # --- leaves per layer -------------------------------------------------
    t0 = time.perf_counter()
    keep = e1 | e2
    if stack.chain is not None:
        keep |= stack.chain
    if stack.buildup is not None:
        keep |= stack.buildup
    if stack.thick_scale is not None:
        keep |= stack.thick_scale != 1.0
    mb = _max_block(stack.h_nm)
    grids = [quadtree.build_leaves(stack.masks[li], keep_fine=keep[li],
                                   max_block=mb,
                                   guard=config.ADAPTIVE_GUARD)
             for li in range(L)]
    offs = np.zeros(L + 1, dtype=np.int64)
    for li in range(L):
        offs[li + 1] = offs[li] + grids[li].n
    N = int(offs[-1])
    n_cells = int(stack.masks.sum())
    print(f"adaptive grid: {N} leaves for {n_cells} copper cells "
          f"({n_cells / max(N, 1):.1f}x, max leaf "
          f"{max(int(g.size.max()) if g.n else 1 for g in grids)} cells)")

    # --- edges: in-plane faces, 1D chain links, barrels -------------------
    aa, bb, ww, vv = [], [], [], []
    sig_leaves, teq_leaves = [], []
    for li in range(L):
        g_ = grids[li]
        sig_leaf = np.full(g_.n, sigmas[li])
        t_m = problem.layers[li].thickness_nm * 1e-9
        s2d = sv._sigma_2d(stack, li, sigmas[li], sigma_buildup)
        fine = g_.size == 1
        if s2d is not None and fine.any():
            sig_leaf[fine] = s2d[g_.y0[fine], g_.x0[fine]]
        # J reference thickness: same convention as the uniform grid
        teq_leaves.append(sig_leaf * problem.rho_ohm_m if s2d is not None
                          else np.full(g_.n, t_m))
        sig_leaves.append(sig_leaf)
        chainleaf = np.zeros(g_.n, dtype=bool)
        if stack.chain is not None and fine.any():
            chainleaf[fine] = stack.chain[li][g_.y0[fine], g_.x0[fine]]
        ia, ib, wl, ax = quadtree.leaf_faces(g_)
        ok = ~(chainleaf[ia] | chainleaf[ib])
        ia, ib, wl, ax = ia[ok], ib[ok], wl[ok], ax[ok]
        gcond = wl / (g_.size[ia] / (2.0 * sig_leaf[ia])
                      + g_.size[ib] / (2.0 * sig_leaf[ib]))
        aa.append(offs[li] + ia)
        bb.append(offs[li] + ib)
        ww.append(gcond)
        vv.append(np.full(len(ia), -1, dtype=np.int32))

    if stack.chain_edges is not None and len(stack.chain_edges[0]):
        ca, cb, cg, cl = stack.chain_edges
        alive = stack.masks.ravel()[ca] & stack.masks.ravel()[cb]
        if alive.any():
            na = np.empty(len(ca), dtype=np.int64)
            nb = np.empty(len(ca), dtype=np.int64)
            for flat, out in ((ca, na), (cb, nb)):
                li_ = flat // plane
                rem = flat - li_ * plane
                for l in range(L):
                    m = li_ == l
                    if m.any():
                        ids = grids[l].id_grid[rem[m] // nx, rem[m] % nx]
                        out[m] = np.where(ids >= 0, offs[l] + ids, -1)
            alive &= (na >= 0) & (nb >= 0)
            fac = np.array([sigmas[l] * problem.rho_ohm_m
                            / (problem.layers[l].thickness_nm * 1e-9)
                            for l in range(L)])
            aa.append(na[alive])
            bb.append(nb[alive])
            ww.append((cg * fac[cl])[alive])
            vv.append(np.full(int(alive.sum()), -1, dtype=np.int32))

    links, dead_barrels = sv._barrel_links(stack, problem)
    for vi, la, ia_, ja_, lb, ib_, jb_, r_dc in links:
        na = offs[la] + grids[la].id_grid[ia_, ja_]
        nb = offs[lb] + grids[lb].id_grid[ib_, jb_]
        aa.append(np.array([na], dtype=np.int64))
        bb.append(np.array([nb], dtype=np.int64))
        ww.append(np.array([1.0 / (r_dc * via_factor)]))
        vv.append(np.array([vi], dtype=np.int32))
    if dead_barrels:
        print(f"warning: {dead_barrels} via/pad barrel(s) found fill "
              f"copper on fewer than 2 layers and carry no current (pad "
              f"copper is not modeled; a finer grid may pick up thermal "
              f"spokes)")

    if not aa:
        raise ConnectivityError("No copper found on the selected layers.")
    edges = sv.Edges(a=np.concatenate(aa), b=np.concatenate(bb),
                     w=np.concatenate(ww), via_index=np.concatenate(vv),
                     dead_barrels=dead_barrels)

    # --- connectivity restriction on the leaf graph -----------------------
    graph = sparse.coo_matrix(
        (np.ones(len(edges.a)), (edges.a, edges.b)), shape=(N, N))
    _, labels = csgraph.connected_components(graph, directed=False)
    e1n = np.zeros(N, dtype=bool)
    e2n = np.zeros(N, dtype=bool)
    for li in range(L):
        e1n[_nodes_of_cells(grids, offs, li, e1[li])] = True
        e2n[_nodes_of_cells(grids, offs, li, e2[li])] = True
    common = np.intersect1d(np.unique(labels[e1n]), np.unique(labels[e2n]))
    if len(common) == 0:
        raise ConnectivityError(
            "The two terminals are not connected by the selected fill "
            "layers (not even through vias). Check the layer selection and "
            "that the fills are up to date."
        )
    if len(common) > 1 and contact_model != "equipotential":
        raise ConnectivityError(
            f"The selected fills form {len(common)} disconnected copper "
            f"groups that each touch both terminals. The uniform-injection "
            f"contact model cannot determine the current split between "
            f"disconnected sheets - switch to the equipotential contact "
            f"model (bonded lug), or include the layers/vias that join "
            f"them."
        )
    keepn = np.isin(labels, common)
    if not keepn.all():
        sel = keepn[edges.a] & keepn[edges.b]
        edges = sv.Edges(a=edges.a[sel], b=edges.b[sel], w=edges.w[sel],
                         via_index=edges.via_index[sel],
                         dead_barrels=dead_barrels)
        for li in range(L):
            ids = grids[li].id_grid
            kept_cells = (ids >= 0) & keepn[offs[li] + np.maximum(ids, 0)]
            stack.masks[li] &= kept_cells
            e1[li] &= kept_cells
            e2[li] &= kept_cells
        if stack.buildup is not None:
            stack.buildup &= stack.masks
        if stack.chain is not None:
            stack.chain &= stack.masks
        e1n &= keepn
        e2n &= keepn
    for label, m in (parts1 or []) + (parts2 or []):
        had = bool(m.any())
        m &= stack.masks
        if had and not m.any():
            print(f"warning: contact part '{label}' only touches copper "
                  f"that is not connected to both terminals - it carries "
                  f"no current")
    timings["edges_s"] = time.perf_counter() - t0

    # --- solve -------------------------------------------------------------
    t0 = time.perf_counter()
    state = np.zeros(N, dtype=np.uint8)
    state[keepn] = 1
    if contact_model == "equipotential":
        state[e1n] = 2
        state[e2n] = 3
        Vflat, R, I1, I2, mismatch, volts_per_amp, info = \
            sv._equipotential_core(state, edges)
    else:
        n1, n2 = int(e1n.sum()), int(e2n.sum())
        inj = np.zeros(N)
        inj[e1n] = 1.0 / n1
        inj[e2n] = -1.0 / n2
        ground = int(np.flatnonzero(e2n)[0])
        state[ground] = 3
        Vflat, R, I1, I2, mismatch, volts_per_amp, info = \
            sv._uniform_core(state, inj, e1n, e2n, edges)
    timings["solve_s"] = time.perf_counter() - t0

    # --- fields on leaves, expanded to the fine grid ------------------------
    t0 = time.perf_counter()
    s = i_test * volts_per_amp

    Pe = edges.w * ((Vflat[edges.a] - Vflat[edges.b]) * s) ** 2
    inplane = edges.via_index < 0
    Pnode = np.zeros(N)
    np.add.at(Pnode, edges.a[inplane], 0.5 * Pe[inplane])
    np.add.at(Pnode, edges.b[inplane], 0.5 * Pe[inplane])
    P_layers = [float(Pnode[offs[li]:offs[li + 1]].sum()) for li in range(L)]
    P_vias = float(Pe[~inplane].sum())
    P_total = i_test ** 2 * R
    balance = abs((sum(P_layers) + P_vias) - P_total) / max(P_total, 1e-300)
    if not np.isfinite(balance) or balance > 1e-3:
        raise sv.SolverError(
            f"Inconsistent solve: R = {R:.6g} ohm with power-balance error "
            f"{balance:.2e} (sum of edge powers vs I^2*R). The result is "
            f"not trustworthy - try the equipotential contact model or a "
            f"different grid size."
        )

    Ie = edges.w * (Vflat[edges.a] - Vflat[edges.b])
    via_reports = []
    if problem.vias:
        vidx = edges.via_index
        for vi in np.unique(vidx[vidx >= 0]):
            sel = vidx == vi
            via = problem.vias[vi]
            via_reports.append(sv.ViaReport(
                x_mm=via.x * 1e-6, y_mm=via.y * 1e-6, kind=via.kind,
                drill_mm=via.drill_nm * 1e-6,
                current_a=float(np.abs(Ie[sel]).max()) * s,
                power_w=float(Pe[sel].sum()),
            ))
        via_reports.sort(key=lambda v: v.current_a, reverse=True)

    def part_currents(parts, e_nodes, n_total_cells):
        out = []
        for label, mask3 in (parts or []):
            n_part = int(mask3.sum())
            if contact_model == "uniform":
                amps = i_test * n_part / max(n_total_cells, 1)
            else:
                pf = np.zeros(N, dtype=bool)
                for li in range(L):
                    pf[_nodes_of_cells(grids, offs, li, mask3[li])] = True
                pf &= e_nodes
                amps = abs(float(Ie[pf[edges.a]].sum()
                                 - Ie[pf[edges.b]].sum())) * s
            out.append((label, amps))
        return out

    part_currents1 = part_currents(parts1, e1n, int(e1.sum()))
    part_currents2 = part_currents(parts2, e2n, int(e2.sum()))

    V3 = np.full((L, ny, nx), np.nan)
    J3 = np.full((L, ny, nx), np.nan)
    Parea = np.full((L, ny, nx), np.nan)
    for li in range(L):
        g_ = grids[li]
        ids = g_.id_grid
        m = stack.masks[li]
        Vl = Vflat[offs[li]:offs[li + 1]]
        V3[li][m] = Vl[ids[m]] * s

        # per-leaf |J| from face currents at unit drive, reconstructed
        # with the same series-half-cell rule (edges were filtered by
        # the restriction, so recompute locally)
        gio = offs[li]
        Ixn = np.zeros(g_.n)
        Iyn = np.zeros(g_.n)
        sig_leaf = sig_leaves[li]
        ia2, ib2, wl2, ax2 = quadtree.leaf_faces(g_)
        chain_ok = np.ones(len(ia2), dtype=bool)
        if stack.chain is not None:
            fine = g_.size == 1
            cl = np.zeros(g_.n, dtype=bool)
            if fine.any():
                cl[fine] = stack.chain[li][g_.y0[fine], g_.x0[fine]]
            chain_ok = ~(cl[ia2] | cl[ib2])
        ia2, ib2, wl2, ax2 = (ia2[chain_ok], ib2[chain_ok], wl2[chain_ok],
                              ax2[chain_ok])
        keep_f = keepn[gio + ia2] & keepn[gio + ib2]
        ia2, ib2, wl2, ax2 = ia2[keep_f], ib2[keep_f], wl2[keep_f], \
            ax2[keep_f]
        g2 = wl2 / (g_.size[ia2] / (2.0 * sig_leaf[ia2])
                    + g_.size[ib2] / (2.0 * sig_leaf[ib2]))
        If = g2 * (Vflat[gio + ia2] - Vflat[gio + ib2])
        for axis, acc in ((0, Ixn), (1, Iyn)):
            selx = ax2 == axis
            np.add.at(acc, ia2[selx], If[selx])
            np.add.at(acc, ib2[selx], If[selx])
        span_m = g_.size.astype(float) * h_m
        with np.errstate(invalid="ignore", divide="ignore"):
            Jl = np.hypot(0.5 * Ixn, 0.5 * Iyn) / (span_m * teq_leaves[li])
        J3[li][m] = Jl[ids[m]] * s

        cellP = Pnode[offs[li]:offs[li + 1]] \
            / (g_.size.astype(float) ** 2 * h_m * h_m)
        Parea[li][m] = cellP[ids[m]]
    timings["postprocess_s"] = time.perf_counter() - t0

    return sv.Result(
        R_ohm=R, i_test=i_test, V=V3, Jmag=J3, Parea=Parea,
        layer_names=list(stack.layer_names),
        P_total=P_total, P_layers=P_layers, P_vias=P_vias,
        power_balance_rel=balance, via_reports=via_reports,
        I1_a=I1, I2_a=I2, mismatch_rel=mismatch,
        n_free=info.n_unknowns, solve_info=info,
        part_currents1=part_currents1, part_currents2=part_currents2,
        contact_model=contact_model,
        freq_hz=freq_hz,
        skin_depth_um=(skin.skin_depth_m(freq_hz, problem.rho_ohm_m) * 1e6
                       if freq_hz > 0 else None),
        rs_ratios=rs_ratios,
        timings=timings,
    )
