"""Coupled multi-layer finite-difference solver.

Each included copper layer is a 2D 5-point sheet with per-layer face
conductance sigma_s = t/rho [S] (square cells: independent of h); via and
plated-through-pad barrels add vertical conductances between the layers
they span AND reach copper on. Per layer the barrel attaches to the cell
under it, or to the nearest copper cell within the pad footprint (+1
cell) - fills joined by thermal-relief spokes still connect. A barrel
passing a (wider) antipad still bridges the layers above/below it with
the full barrel length. At freq > 0 the per-layer sheet conductances and the
barrel walls get the 1D skin-effect correction (see skin.py; AC results
are a rigorous lower bound - lateral redistribution is not modeled).

Two contact models for the terminals (each terminal = merged parts):

- "uniform" (default): a conductor pressed onto the contact area injects
  the current orthogonally with UNIFORM surface density: every contact
  cell sources (sinks) I/N. The in-plane current density ramps across
  the contact instead of being zero. The pure-Neumann system is grounded
  at one V- cell (that cell's sink share is exactly the flux that exits
  through the ground reference, so the solution equals the singular
  system's). R = (<V over V+ cells> - <V over V- cells>) / I; because
  the injection and averaging weights coincide, sum(edge powers) = I^2 R
  holds exactly and remains the consistency check.

- "equipotential": ideal bonded lug; contact cells are Dirichlet
  (V+ = 1 V, V- = 0). R from the exact discrete electrode flux. Touching
  terminals are rejected (a direct face would short the Dirichlet
  regions); with "uniform" contacts touching is physically fine.

The two models bracket a real contact: R_equipotential <= R_real <=
R_uniform. Missing neighbors give no matrix term = insulated boundary.
Current density per layer comes from face currents (np.gradient across
the NaN staircase boundary would pollute the field). Power density per
layer distributes each in-plane edge's dissipation half to each endpoint
cell. All reported fields are rescaled to the test current I_test.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as sla

from . import config, skin
from .errors import ConnectivityError, ElectrodeError, SolverError
from .geometry import Problem
from .raster import RasterStack, electrodes_touch


@dataclass
class SolveInfo:
    method: str                   # "spsolve" | "cg+jacobi"
    n_unknowns: int
    iterations: int | None = None
    residual: float | None = None


@dataclass
class Edges:
    a: np.ndarray                 # int64 flat cell ids
    b: np.ndarray
    w: np.ndarray                 # conductance [S]
    via_index: np.ndarray         # int32; -1 = in-plane edge
    dead_barrels: int = 0         # barrels spanning >=2 layers that found
                                  # fill copper on fewer than 2 of them


@dataclass
class ViaReport:
    x_mm: float
    y_mm: float
    kind: str
    drill_mm: float
    current_a: float              # max barrel-segment current @ I_test
    power_w: float                # total barrel dissipation @ I_test


@dataclass
class Result:
    R_ohm: float
    i_test: float
    V: np.ndarray                 # (L, ny, nx) volts @ I_test, NaN off-copper
    Jmag: np.ndarray              # (L, ny, nx) A/m^2 @ I_test
    Parea: np.ndarray             # (L, ny, nx) W/m^2 @ I_test
    layer_names: list[str]
    P_total: float                # I_test^2 * R
    P_layers: list[float]         # in-plane dissipation per layer @ I_test
    P_vias: float                 # total barrel dissipation @ I_test
    power_balance_rel: float      # |sum(edge powers) - I^2 R| / I^2 R
    via_reports: list[ViaReport]  # sorted by current, descending
    I1_a: float                   # electrode currents (unit drive)
    I2_a: float
    mismatch_rel: float
    n_free: int
    solve_info: SolveInfo
    # per-part terminal currents @ I_test: [(label, amps), ...];
    # computed flux for "equipotential", prescribed area share for "uniform"
    part_currents1: list = field(default_factory=list)
    part_currents2: list = field(default_factory=list)
    contact_model: str = "uniform"
    freq_hz: float = 0.0
    skin_depth_um: float | None = None
    rs_ratios: list[float] = field(default_factory=list)  # R_AC/R_DC per layer
    timings: dict = field(default_factory=dict)


def _shifts2d():
    return [
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ]


def _sigma_2d(stack: RasterStack, li: int, sigma_layer: float,
              sigma_buildup: float) -> np.ndarray | None:
    """Per-cell sheet conductance for one layer, or None if uniform.
    Combines the via-mouth thickness map (cap-thin / partially drilled
    cells) with the solder-buildup addition."""
    have_b = (stack.buildup is not None and sigma_buildup > 0
              and stack.buildup[li].any())
    have_t = (stack.thick_scale is not None
              and bool((stack.thick_scale[li] != 1.0).any()))
    if not have_b and not have_t:
        return None
    s = np.full(stack.shape2d, sigma_layer)
    if have_t:
        s *= stack.thick_scale[li]
    if have_b:
        s[stack.buildup[li]] += sigma_buildup
    return s


def _barrel_links(stack: RasterStack, problem: Problem
                  ) -> tuple[list, int]:
    """Vertical barrel links on the fine grid, shared by the uniform and
    adaptive paths: [(via_index, layer_a, i_a, j_a, layer_b, i_b, j_b,
    r_dc), ...] plus the count of dead barrels (span >= 2 layers but
    reached copper on < 2). Connection cell per layer: the cell under
    the barrel, or the nearest copper cell whose center lies within the
    pad footprint (+1 cell of rasterization slop) - fills joined to the
    barrel by thermal-relief spokes still connect, wider antipads do not
    (the barrel then bridges the layers above/below)."""
    L, ny, nx = stack.masks.shape
    h = stack.h_nm
    links = []
    dead = 0
    for vi, via in enumerate(problem.vias):
        cell = stack.cell_of(via.x, via.y)
        if cell is None:
            continue
        i, j = cell
        span = [li for li, layer in enumerate(problem.layers)
                if via.spans(layer.z_nm)]
        r_nm = max(via.pad_nm, via.drill_nm + 300_000) / 2.0 + h
        win = int(r_nm // h) + 1
        i0, i1 = max(0, i - win), min(ny, i + win + 1)
        j0, j1 = max(0, j - win), min(nx, j + win + 1)
        xs = stack.x0_nm + (np.arange(j0, j1) + 0.5) * h - via.x
        ys = stack.y0_nm + (np.arange(i0, i1) + 0.5) * h - via.y
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        d2 = np.where(d2 <= r_nm * r_nm, d2, np.inf)
        present = []                            # (layer, i, j) per layer
        for li in span:
            if stack.masks[li, i, j]:
                present.append((li, i, j))
                continue
            dc = np.where(stack.masks[li, i0:i1, j0:j1], d2, np.inf)
            ci, cj = np.unravel_index(int(np.argmin(dc)), dc.shape)
            if np.isfinite(dc[ci, cj]):
                present.append((li, i0 + ci, j0 + cj))
        if len(span) >= 2 and len(present) < 2:
            dead += 1
        for (la, ia, ja), (lb, ib, jb) in zip(present[:-1], present[1:]):
            length = problem.layers[lb].z_nm - problem.layers[la].z_nm
            if length <= 0:
                continue
            # THT pads carry a soldered component lead: the hole is
            # solder-filled, the core conducts in parallel with the plating
            r_dc = via.barrel_resistance(
                length, problem.rho_ohm_m, problem.plating_nm,
                solder_rho_ohm_m=(problem.solder_rho_ohm_m
                                  if via.kind == "pad" else None))
            links.append((vi, la, ia, ja, lb, ib, jb, r_dc))
    return links, dead


def build_edges(stack: RasterStack, problem: Problem, sigmas: list[float],
                via_factor: float = 1.0,
                sigma_buildup: float = 0.0) -> Edges:
    """All copper-copper conductances: in-plane faces + via barrels.
    sigmas: effective (possibly AC) sheet conductance per layer;
    via_factor: R_AC/R_DC of the barrel wall; sigma_buildup: extra sheet
    conductance on solder-buildup cells. Faces between cells of unequal
    conductance use the harmonic mean (series half-cells), which reduces
    exactly to sigma for uniform regions."""
    L, ny, nx = stack.masks.shape
    plane = ny * nx
    aa, bb, ww, vv = [], [], [], []

    for li in range(L):
        m = stack.masks[li]
        if stack.chain is not None:
            # chain-only cells connect through their explicit 1D links,
            # never through sheet faces (their copper is narrower than h)
            m = m & ~stack.chain[li]
        sig = sigmas[li]
        scell = _sigma_2d(stack, li, sig, sigma_buildup)
        base = li * plane
        for src, dst in _shifts2d():
            pair = m[src] & m[dst]
            ii, jj = np.nonzero(pair)
            if src[0] == slice(None):            # horizontal: j, j+1
                a = base + ii * nx + jj
                b = a + 1
            else:                                 # vertical: i, i+1
                a = base + ii * nx + jj
                b = a + nx
            aa.append(a.astype(np.int64))
            bb.append(b.astype(np.int64))
            if scell is None:
                ww.append(np.full(len(a), sig))
            else:
                s_a = scell[src][pair]
                s_b = scell[dst][pair]
                ww.append(2.0 * s_a * s_b / (s_a + s_b))
            vv.append(np.full(len(a), -1, dtype=np.int32))

    links, dead_barrels = _barrel_links(stack, problem)
    for vi, la, ia, ja, lb, ib, jb, r_dc in links:
        aa.append(np.array([la * plane + ia * nx + ja], dtype=np.int64))
        bb.append(np.array([lb * plane + ib * nx + jb], dtype=np.int64))
        ww.append(np.array([1.0 / (r_dc * via_factor)]))
        vv.append(np.array([vi], dtype=np.int32))

    if stack.chain_edges is not None and len(stack.chain_edges[0]):
        ca, cb, cg, cl, _ = stack.chain_edges
        alive = stack.masks.ravel()[ca] & stack.masks.ravel()[cb]
        if alive.any():
            # skin correction: scale like the layer's sheet conductance
            fac = np.array([sigmas[l] * problem.rho_ohm_m
                            / (problem.layers[l].thickness_nm * 1e-9)
                            for l in range(L)])
            aa.append(ca[alive])
            bb.append(cb[alive])
            ww.append((cg * fac[cl])[alive])
            vv.append(np.full(int(alive.sum()), -1, dtype=np.int32))

    if not aa:
        raise ConnectivityError("No copper found on the selected layers.")
    return Edges(a=np.concatenate(aa), b=np.concatenate(bb),
                 w=np.concatenate(ww), via_index=np.concatenate(vv),
                 dead_barrels=dead_barrels)


def connected_restrict(stack: RasterStack, e1: np.ndarray, e2: np.ndarray,
                       edges: Edges) -> tuple[bool, int]:
    """Keep only components (through-plane AND through-via) touching both
    terminals. Mutates stack.masks / e1 / e2. Returns (changed,
    n_components): whether anything was dropped (caller must rebuild
    edges) and how many disjoint copper groups survive."""
    n = stack.masks.size
    graph = sparse.coo_matrix(
        (np.ones(len(edges.a)), (edges.a, edges.b)), shape=(n, n))
    _, labels = csgraph.connected_components(graph, directed=False)
    labels3 = labels.reshape(stack.masks.shape)
    common = np.intersect1d(np.unique(labels3[e1]), np.unique(labels3[e2]))
    if len(common) == 0:
        raise ConnectivityError(
            "The two terminals are not connected by the selected fill "
            "layers (not even through vias). Check the layer selection and "
            "that the fills are up to date."
        )
    keep = np.isin(labels3, common) & stack.masks
    changed = bool((stack.masks & ~keep).any())
    stack.masks &= keep
    e1 &= keep
    e2 &= keep
    return changed, len(common)


def _assemble(state: np.ndarray, edges: Edges, rhs_extra: np.ndarray | None):
    """Weighted-Laplacian assembly with Dirichlet elimination.
    state: 0 off, 1 free, 2 Dirichlet@1V, 3 Dirichlet@0V.
    rhs_extra: per-flat-cell current injection [A] added for free cells."""
    n = state.size
    sa, sb = state[edges.a], state[edges.b]
    short = ((sa == 2) & (sb == 3)) | ((sa == 3) & (sb == 2))
    if short.any():
        n_via = int((edges.via_index[short] >= 0).sum())
        raise ElectrodeError(
            f"The terminals are directly connected by {int(short.sum())} "
            f"conductance(s) ({n_via} via barrel(s)) without any free copper "
            f"in between - move the contacts apart."
        )

    free = state == 1
    n_free = int(free.sum())
    if n_free == 0:
        raise ElectrodeError(
            "No free copper cells remain between the terminals - the "
            "contacts cover the whole fill at this grid resolution."
        )
    idx = np.full(n, -1, dtype=np.int64)
    idx[free] = np.arange(n_free)

    diag = np.zeros(n_free)
    rhs = np.zeros(n_free)
    fa, fb = sa == 1, sb == 1
    np.add.at(diag, idx[edges.a[fa]], edges.w[fa])
    np.add.at(diag, idx[edges.b[fb]], edges.w[fb])
    r1a = fa & (sb == 2)
    r1b = fb & (sa == 2)
    np.add.at(rhs, idx[edges.a[r1a]], edges.w[r1a])
    np.add.at(rhs, idx[edges.b[r1b]], edges.w[r1b])
    if rhs_extra is not None:
        rhs += rhs_extra[free]

    ff = fa & fb
    rows = np.concatenate([idx[edges.a[ff]], idx[edges.b[ff]],
                           np.arange(n_free)])
    cols = np.concatenate([idx[edges.b[ff]], idx[edges.a[ff]],
                           np.arange(n_free)])
    vals = np.concatenate([-edges.w[ff], -edges.w[ff], diag])
    A = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(n_free, n_free)).tocsr()
    return A, rhs, idx


def solve_system(A: sparse.csr_matrix, b: np.ndarray) -> tuple[np.ndarray, SolveInfo]:
    n = A.shape[0]
    if n <= config.SPSOLVE_MAX_UNKNOWNS:
        x = sla.spsolve(A.tocsc(), b)
        return x, SolveInfo(method="spsolve", n_unknowns=n)
    try:
        return _solve_amg(A, b)
    except ImportError:
        print("note: pyamg not installed - falling back to Jacobi-CG "
              "(much slower on large grids)")
        return _solve_cg_jacobi(A, b)


class PreparedSolver:
    """Factor/set up once, solve several right-hand sides with the SAME
    matrix (deferred-correction passes): the direct path keeps the LU,
    the iterative path keeps the AMG hierarchy."""

    def __init__(self, A: sparse.csr_matrix):
        self.n = A.shape[0]
        self._A = A.tocsr()
        self._lu = None
        self._ml = None
        if self.n <= config.SPSOLVE_MAX_UNKNOWNS:
            self._lu = sla.splu(A.tocsc())
            self.method = "spsolve"
        else:
            try:
                import pyamg
                self._ml = pyamg.smoothed_aggregation_solver(self._A,
                                                             max_coarse=500)
                self.method = "amg+cg"
            except ImportError:
                print("note: pyamg not installed - falling back to "
                      "Jacobi-CG (much slower on large grids)")
                self.method = "cg+jacobi"

    def solve(self, b: np.ndarray) -> tuple[np.ndarray, SolveInfo]:
        if self._lu is not None:
            return self._lu.solve(b), SolveInfo(method="spsolve",
                                                n_unknowns=self.n)
        if self._ml is not None:
            residuals: list[float] = []
            x = self._ml.solve(b, tol=config.AMG_TOL, maxiter=300,
                               accel="cg", residuals=residuals)
            res = float(np.linalg.norm(b - self._A @ x)
                        / max(np.linalg.norm(b), 1e-300))
            if not np.isfinite(res) or res > 1e-6:
                raise SolverError(
                    f"AMG-CG did not converge (residual {res:.2e}). Try a "
                    f"different grid size, or force the direct solver by "
                    f"raising SPSOLVE_MAX_UNKNOWNS in config.py."
                )
            return x, SolveInfo(method="amg+cg", n_unknowns=self.n,
                                iterations=max(len(residuals) - 1, 0),
                                residual=res)
        return _solve_cg_jacobi(self._A, b)


def _solve_amg(A: sparse.csr_matrix, b: np.ndarray) -> tuple[np.ndarray, SolveInfo]:
    """CG preconditioned with smoothed-aggregation AMG: near-linear
    scaling on these 2D Laplacians and a fraction of spsolve's memory."""
    import pyamg

    n = A.shape[0]
    ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=500)
    residuals: list[float] = []
    x = ml.solve(b, tol=config.AMG_TOL, maxiter=300, accel="cg",
                 residuals=residuals)
    res = float(np.linalg.norm(b - A @ x) / max(np.linalg.norm(b), 1e-300))
    if not np.isfinite(res) or res > 1e-6:
        raise SolverError(
            f"AMG-CG did not converge (residual {res:.2e}). Try a "
            f"different grid size, or force the direct solver by raising "
            f"SPSOLVE_MAX_UNKNOWNS in config.py."
        )
    return x, SolveInfo(method="amg+cg", n_unknowns=n,
                        iterations=max(len(residuals) - 1, 0), residual=res)


def _solve_cg_jacobi(A: sparse.csr_matrix, b: np.ndarray) -> tuple[np.ndarray, SolveInfo]:
    # The matrix is SPD, so CG is guaranteed to converge. Jacobi is the
    # only preconditioner in scipy that keeps the preconditioned operator
    # SPD without a factorization that can break down at this scale.
    n = A.shape[0]
    d = A.diagonal()
    M = sla.LinearOperator((n, n), lambda v: v / d)
    iters = 0

    def count(_):
        nonlocal iters
        iters += 1

    try:
        x, code = sla.cg(A, b, M=M, rtol=config.CG_TOL,
                         maxiter=config.CG_MAXITER, callback=count)
    except TypeError:  # scipy < 1.12 uses tol=
        x, code = sla.cg(A, b, M=M, tol=config.CG_TOL,
                         maxiter=config.CG_MAXITER, callback=count)
    if code != 0:
        raise SolverError(
            f"CG did not converge in {config.CG_MAXITER} iterations "
            f"(code {code}). Try a coarser grid or raise CG_MAXITER."
        )
    res = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
    return x, SolveInfo(method="cg+jacobi", n_unknowns=n, iterations=iters,
                        residual=res)


def _face_current_density(V2: np.ndarray, mask2: np.ndarray, sigma: float,
                          h_m: float, t_m: float,
                          sig2d: np.ndarray | None = None,
                          rho: float | None = None) -> np.ndarray:
    """|J| (A/m^2) for one layer from face currents; V2 in volts.
    J is referenced to the conductance-equivalent copper thickness
    t_eq = sigma_cell * rho: the geometric t for plain DC copper, the
    skin-reduced conducting cross-section at AC. With a per-cell
    conductance map (buildup), face currents use the harmonic mean."""
    ny, nx = mask2.shape
    face_x = mask2[:, :-1] & mask2[:, 1:]
    face_y = mask2[:-1, :] & mask2[1:, :]
    if sig2d is None:
        wx = wy = sigma
        teq = np.full((ny, nx), sigma * rho if rho is not None else t_m)
    else:
        wx = 2.0 * sig2d[:, :-1] * sig2d[:, 1:] / (sig2d[:, :-1] + sig2d[:, 1:])
        wy = 2.0 * sig2d[:-1, :] * sig2d[1:, :] / (sig2d[:-1, :] + sig2d[1:, :])
        teq = sig2d * rho
    with np.errstate(invalid="ignore"):
        Ix = np.where(face_x, (V2[:, :-1] - V2[:, 1:]) * wx, 0.0)
        Iy = np.where(face_y, (V2[:-1, :] - V2[1:, :]) * wy, 0.0)
    IxP = np.zeros((ny, nx + 1))
    IxP[:, 1:nx] = Ix
    IyP = np.zeros((ny + 1, nx))
    IyP[1:ny, :] = Iy
    Jx = 0.5 * (IxP[:, :-1] + IxP[:, 1:])
    Jy = 0.5 * (IyP[:-1, :] + IyP[1:, :])
    Jmag = np.hypot(Jx, Jy) / (h_m * teq)
    Jmag[~mask2] = np.nan
    return Jmag


def overlay_chain_density(stack: RasterStack, rho: float, V3: np.ndarray,
                          J3: np.ndarray) -> None:
    """Fill chain (sub-resolution trace) cells of J3 with the true 1D
    link current density |dV| / (rho * dl), referenced to the
    conduction-equivalent trace cross-section: the AC scaling of the
    link conductance and of the cross-section cancel, so the expression
    holds at any frequency. V3/J3 are the display-scaled (L, ny, nx)
    maps; chain cells carry the max density of their attached links."""
    if stack.chain is None or stack.chain_edges is None \
            or not len(stack.chain_edges[0]):
        return
    ca, cb, _, _, cdl = stack.chain_edges
    mflat = stack.masks.reshape(-1)
    alive = mflat[ca] & mflat[cb]
    if not alive.any():
        return
    V3f = np.nan_to_num(V3.reshape(-1))
    Jl = np.abs(V3f[ca] - V3f[cb]) / (rho * cdl)
    Jc = np.zeros(mflat.size)
    np.maximum.at(Jc, ca[alive], Jl[alive])
    np.maximum.at(Jc, cb[alive], Jl[alive])
    fill = stack.chain & stack.masks
    J3[fill] = Jc.reshape(J3.shape)[fill]


def _equipotential_core(state: np.ndarray, edges: Edges):
    """Dirichlet solve on any node space (fine cells or leaves): state
    codes 0 off / 1 free / 2 V+ / 3 V-. Returns (Vflat_unit, R, I1, I2,
    mismatch, volts_per_amp, info); fields at 1 V drive."""
    A, rhs, _ = _assemble(state, edges, None)
    x, info = solve_system(A, rhs)

    Vflat = np.zeros(state.size)
    Vflat[state == 2] = 1.0
    Vflat[state == 1] = x

    Ie = edges.w * (Vflat[edges.a] - Vflat[edges.b])
    sa, sb = state[edges.a], state[edges.b]
    I1 = float(Ie[sa == 2].sum() - Ie[sb == 2].sum())
    I2 = float(Ie[sb == 3].sum() - Ie[sa == 3].sum())
    mismatch = abs(I1 - I2) / max(abs(I1), abs(I2), 1e-300)
    R = 1.0 / (0.5 * (I1 + I2))
    return Vflat, R, I1, I2, mismatch, R, info


def _uniform_core(state: np.ndarray, inj: np.ndarray, e1f: np.ndarray,
                  e2f: np.ndarray, edges: Edges):
    """Uniform-injection solve on any node space. state must carry the
    single ground node (code 3); inj the per-node current shares."""
    A, rhs, _ = _assemble(state, edges, inj)
    x, info = solve_system(A, rhs)

    Vflat = np.zeros(state.size)
    Vflat[state == 1] = x

    v_plus = float(Vflat[e1f].mean())
    v_minus = float(Vflat[e2f].mean())
    R = (v_plus - v_minus) / 1.0
    Vflat = Vflat - v_minus                 # display reference: <V-> = 0

    # quality: KCL residual of the solved system
    res = info.residual
    if res is None:
        res = float(np.linalg.norm(A @ x - rhs)
                    / max(np.linalg.norm(rhs), 1e-300))
    return Vflat, R, 1.0, 1.0, res, 1.0, info


def _solve_equipotential(stack, e1, e2, edges):
    """Dirichlet terminals at 1 V / 0 V on the uniform grid. Fields at
    1 V drive; scale by i_test * R to get volts at I_test."""
    if (layer := electrodes_touch(stack, e1, e2)) is not None:
        raise ElectrodeError(
            f"The terminals touch on {layer}. With the equipotential "
            f"contact model at least one cell of copper must separate "
            f"them; the uniform-injection model allows touching contacts."
        )
    state = np.zeros(stack.masks.size, dtype=np.uint8)
    state[stack.masks.ravel()] = 1
    state[e1.ravel()] = 2
    state[e2.ravel()] = 3
    return _equipotential_core(state, edges)


def _solve_uniform(stack, e1, e2, edges):
    """Uniform orthogonal injection on the uniform grid: every contact
    cell sources (sinks) 1 A / N, grounded at one V- cell (its sink
    share is exactly the flux that exits through the reference, so the
    grounded solution equals the pure-Neumann one). Fields at 1 A."""
    n = stack.masks.size
    e1f, e2f = e1.ravel(), e2.ravel()
    n1, n2 = int(e1f.sum()), int(e2f.sum())

    inj = np.zeros(n)
    inj[e1f] = 1.0 / n1
    inj[e2f] = -1.0 / n2

    state = np.zeros(n, dtype=np.uint8)
    state[stack.masks.ravel()] = 1
    ground = int(np.flatnonzero(e2f)[0])
    state[ground] = 3
    return _uniform_core(state, inj, e1f, e2f, edges)


def _part_currents(parts, Ie, edges, e_flat, scale,
                   i_test, contact_model, n_terminal_cells):
    """Current through each contact part @ I_test. Equipotential: exact
    discrete flux out of the part's cells (same-terminal internal edges
    carry zero, opposite-terminal edges are forbidden). Uniform: the
    injection is prescribed, so a part carries exactly its cell share."""
    out = []
    for label, mask3 in parts:
        pf = mask3.ravel() & e_flat
        n = int(pf.sum())
        if contact_model == "uniform":
            amps = i_test * n / max(n_terminal_cells, 1)
        else:
            ina = pf[edges.a]
            inb = pf[edges.b]
            amps = abs(float(Ie[ina].sum() - Ie[inb].sum())) * scale
        out.append((label, amps))
    return out


def _conductance_params(problem: Problem, stack: RasterStack,
                        freq_hz: float):
    """Effective (possibly AC) sheet conductances per layer, Rs ratios,
    barrel factor and buildup conductance - shared by the uniform-grid
    and adaptive solve paths."""
    L = stack.nlayers
    sigmas = [
        1.0 / skin.sheet_resistance_ac(
            problem.layers[li].thickness_nm * 1e-9, freq_hz,
            problem.rho_ohm_m, config.SKIN_SIDES)
        for li in range(L)
    ]
    rs_ratios = [
        skin.resistance_factor(problem.layers[li].thickness_nm * 1e-9,
                               freq_hz, problem.rho_ohm_m, config.SKIN_SIDES)
        for li in range(L)
    ]
    via_factor = skin.resistance_factor(problem.plating_nm * 1e-9, freq_hz,
                                        problem.rho_ohm_m, sides=2)
    sigma_buildup = 0.0
    if problem.buildups and stack.buildup is not None \
            and stack.buildup.any():
        sigma_buildup = 1.0 / skin.sheet_resistance_ac(
            problem.solder_thickness_nm * 1e-9, freq_hz,
            problem.solder_rho_ohm_m, config.SKIN_SIDES)
        if problem.extra_cu_nm > 0:
            sigma_buildup += 1.0 / skin.sheet_resistance_ac(
                problem.extra_cu_nm * 1e-9, freq_hz, problem.rho_ohm_m,
                config.SKIN_SIDES)
        eq_um = sigma_buildup * problem.rho_ohm_m * 1e6
        print(f"solder buildup: {problem.solder_thickness_nm / 1000:.0f} um "
              f"solder + {problem.extra_cu_nm / 1000:.0f} um Cu on "
              f"{int(stack.buildup.sum())} cells "
              f"(= {eq_um:.1f} um equivalent copper)")
    if freq_hz > 0:
        depth = skin.skin_depth_m(freq_hz, problem.rho_ohm_m)
        print(f"AC @ {freq_hz:g} Hz: skin depth {depth * 1e6:.0f} um, "
              f"per-layer Rs ratio "
              f"{', '.join(f'{r:.2f}' for r in rs_ratios)}, "
              f"via factor {via_factor:.2f}")
    return sigmas, rs_ratios, via_factor, sigma_buildup


def run_solve(problem: Problem, stack: RasterStack, e1: np.ndarray,
              e2: np.ndarray, i_test: float, freq_hz: float = 0.0,
              contact_model: str | None = None,
              parts1: list | None = None,
              parts2: list | None = None) -> Result:
    if contact_model is None:
        contact_model = config.CONTACT_MODEL
    if config.ADAPTIVE_CELLS:
        from . import adaptive
        return adaptive.run_solve_adaptive(problem, stack, e1, e2, i_test,
                                           freq_hz, contact_model,
                                           parts1, parts2)

    timings = {}
    L, ny, nx = stack.masks.shape
    h_m = stack.h_nm * 1e-9

    sigmas, rs_ratios, via_factor, sigma_buildup = \
        _conductance_params(problem, stack, freq_hz)

    t0 = time.perf_counter()
    edges = build_edges(stack, problem, sigmas, via_factor, sigma_buildup)
    changed, n_groups = connected_restrict(stack, e1, e2, edges)
    if changed:
        edges = build_edges(stack, problem, sigmas, via_factor, sigma_buildup)
    if edges.dead_barrels:
        print(f"warning: {edges.dead_barrels} via/pad barrel(s) found fill "
              f"copper on fewer than 2 layers and carry no current (pad "
              f"copper is not modeled; a finer grid may pick up thermal "
              f"spokes)")
    if n_groups > 1 and contact_model != "equipotential":
        raise ConnectivityError(
            f"The selected fills form {n_groups} disconnected copper groups "
            f"that each touch both terminals. The uniform-injection contact "
            f"model cannot determine the current split between disconnected "
            f"sheets - switch to the equipotential contact model (bonded "
            f"lug), or include the layers/vias that join them."
        )
    if stack.buildup is not None:
        stack.buildup &= stack.masks
    if stack.chain is not None:
        stack.chain &= stack.masks
    for label, m in (parts1 or []) + (parts2 or []):
        had = bool(m.any())
        m &= stack.masks                     # follow the component restriction
        if had and not m.any():
            print(f"warning: contact part '{label}' only touches copper "
                  f"that is not connected to both terminals - it carries "
                  f"no current")
    timings["edges_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if contact_model == "equipotential":
        Vflat, R, I1, I2, mismatch, volts_per_amp, info = \
            _solve_equipotential(stack, e1, e2, edges)
    else:
        Vflat, R, I1, I2, mismatch, volts_per_amp, info = \
            _solve_uniform(stack, e1, e2, edges)
    timings["solve_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    s = i_test * volts_per_amp              # unit-drive volts -> volts @ I_test

    # per-edge power @ I_test; distribute in-plane power to endpoint cells
    Pe = edges.w * ((Vflat[edges.a] - Vflat[edges.b]) * s) ** 2
    inplane = edges.via_index < 0
    Pflat = np.zeros(Vflat.size)
    np.add.at(Pflat, edges.a[inplane], 0.5 * Pe[inplane])
    np.add.at(Pflat, edges.b[inplane], 0.5 * Pe[inplane])
    Parea = Pflat.reshape(L, ny, nx) / (h_m * h_m)
    Parea[~stack.masks] = np.nan
    plane = ny * nx
    P_layers = [float(Pflat[li * plane:(li + 1) * plane].sum())
                for li in range(L)]
    P_vias = float(Pe[~inplane].sum())
    P_total = i_test ** 2 * R
    balance = abs((sum(P_layers) + P_vias) - P_total) / max(P_total, 1e-300)
    if not np.isfinite(balance) or balance > 1e-3:
        raise SolverError(
            f"Inconsistent solve: R = {R:.6g} ohm with power-balance error "
            f"{balance:.2e} (sum of edge powers vs I^2*R). The result is "
            f"not trustworthy - try the equipotential contact model or a "
            f"different grid size."
        )

    # via reports: max segment current + total power per via
    Ie = edges.w * (Vflat[edges.a] - Vflat[edges.b])   # amps at unit drive
    via_reports = []
    if problem.vias:
        vidx = edges.via_index
        for vi in np.unique(vidx[vidx >= 0]):
            sel = vidx == vi
            via = problem.vias[vi]
            via_reports.append(ViaReport(
                x_mm=via.x * 1e-6, y_mm=via.y * 1e-6, kind=via.kind,
                drill_mm=via.drill_nm * 1e-6,
                current_a=float(np.abs(Ie[sel]).max()) * s,
                power_w=float(Pe[sel].sum()),
            ))
        via_reports.sort(key=lambda v: v.current_a, reverse=True)

    # per-injection-area currents
    part_currents1 = _part_currents(
        parts1 or [], Ie, edges, e1.ravel(), s, i_test,
        contact_model, int(e1.sum()))
    part_currents2 = _part_currents(
        parts2 or [], Ie, edges, e2.ravel(), s, i_test,
        contact_model, int(e2.sum()))

    # embedded potential + per-layer current density @ I_test; chain
    # cells have no sheet faces in the model, so keep them out of the
    # face computation and overlay their true 1D link density instead
    V3 = np.full((L, ny, nx), np.nan)
    V3[stack.masks] = Vflat.reshape(L, ny, nx)[stack.masks] * s
    sheet = stack.masks if stack.chain is None \
        else stack.masks & ~stack.chain
    J3 = np.stack([
        _face_current_density(
            np.nan_to_num(V3[li]), sheet[li], sigmas[li],
            h_m, problem.layers[li].thickness_nm * 1e-9,
            sig2d=_sigma_2d(stack, li, sigmas[li], sigma_buildup),
            rho=problem.rho_ohm_m)
        for li in range(L)
    ])
    overlay_chain_density(stack, problem.rho_ohm_m, V3, J3)
    timings["postprocess_s"] = time.perf_counter() - t0

    return Result(
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
