"""Solder/mask-opening buildup tests. Uniform-coverage and split-coverage
strips are 1D-exact, validating the harmonic-mean face weights and the
parallel-sheet conductance model to solver precision."""
import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.geometry import (Polygon, SurfaceBuildup, load_problem,
                                      save_problem)
from tests.util import NM, make_problem, ring_mm, sigma_s, strip_problem

RHO_CU = 1.68e-8
RHO_SN = 1.32e-7


def _with_buildup(p, polys_mm, layer="F.Cu", solder_um=50.0, extra_um=0.0):
    p.buildups = [SurfaceBuildup(
        layer_name=layer,
        polygons=[Polygon(outline=ring_mm(pts)) for pts in polys_mm])]
    p.solder_thickness_nm = int(solder_um * 1000)
    p.solder_rho_ohm_m = RHO_SN
    p.extra_cu_nm = int(extra_um * 1000)
    return p


def _solve(problem, h_mm):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, 1.0,
                            contact_model="equipotential"), stack


def _sigma_buildup(solder_um=50.0, extra_um=0.0):
    return solder_um * 1e-6 / RHO_SN + extra_um * 1e-6 / RHO_CU


def test_full_coverage_exact():
    """Buildup over the whole strip: uniform parallel sheet, exact."""
    plain = strip_problem(length=50, width=10, e_len=5)
    covered = _with_buildup(strip_problem(length=50, width=10, e_len=5),
                            [[(0, 0), (50, 0), (50, 10), (0, 10)]])
    r0, _ = _solve(plain, 0.5)
    r1, stack = _solve(covered, 0.5)
    sig = sigma_s() + _sigma_buildup()
    assert r1.R_ohm == pytest.approx(81 / 20 / sig, rel=1e-9)
    assert r1.R_ohm < r0.R_ohm
    assert stack.buildup is not None and stack.buildup.any()


def test_half_coverage_series_exact():
    """Buildup over the right half: plain faces + one harmonic-mean
    interface face + buildup faces in series, exact."""
    p = _with_buildup(strip_problem(length=50, width=10, e_len=5),
                      [[(25, 0), (50, 0), (50, 10), (25, 10)]])
    res, _ = _solve(p, 0.5)
    s1 = sigma_s()
    s2 = s1 + _sigma_buildup()
    r_row = 40 / s1 + (s1 + s2) / (2 * s1 * s2) + 40 / s2
    assert res.R_ohm == pytest.approx(r_row / 20, rel=1e-9)


def test_buildup_only_over_hole_is_inert():
    """Solder wets copper only: an opening over a hole changes nothing."""
    holed = [([(0, 0), (50, 0), (50, 10), (0, 10)],
              [[(20, 2), (30, 2), (30, 8), (20, 8)]])]
    plain = make_problem(holed, rect1_mm=(0, 0, 5, 10),
                         rect2_mm=(45, 0, 50, 10))
    masked = _with_buildup(
        make_problem(holed, rect1_mm=(0, 0, 5, 10),
                     rect2_mm=(45, 0, 50, 10)),
        [[(21, 3), (29, 3), (29, 7), (21, 7)]])   # strictly inside the hole
    r0, _ = _solve(plain, 0.5)
    r1, _ = _solve(masked, 0.5)
    assert r1.R_ohm == pytest.approx(r0.R_ohm, rel=1e-12)


def test_extra_copper_helps_more_than_solder():
    base = strip_problem(length=50, width=10, e_len=5)
    solder_only = _with_buildup(strip_problem(length=50, width=10, e_len=5),
                                [[(0, 0), (50, 0), (50, 10), (0, 10)]])
    with_cu = _with_buildup(strip_problem(length=50, width=10, e_len=5),
                            [[(0, 0), (50, 0), (50, 10), (0, 10)]],
                            extra_um=70.0)
    r0, _ = _solve(base, 0.5)
    r1, _ = _solve(solder_only, 0.5)
    r2, _ = _solve(with_cu, 0.5)
    assert r2.R_ohm < r1.R_ohm < r0.R_ohm
    # 50 um SAC solder ~ 6.4 um Cu: expect a modest (<15%) improvement
    assert r1.R_ohm > 0.85 * r0.R_ohm
    # +70 um Cu roughly halves R (2x thickness + solder)
    assert r2.R_ohm < 0.55 * r0.R_ohm


def test_json_v4_roundtrip(tmp_path):
    p = _with_buildup(strip_problem(), [[(0, 0), (50, 0), (50, 10), (0, 10)]],
                      extra_um=35.0)
    f = tmp_path / "d.json"
    save_problem(p, f)
    q = load_problem(f)
    assert len(q.buildups) == 1 and q.buildups[0].layer_name == "F.Cu"
    assert q.solder_thickness_nm == 50_000
    assert q.extra_cu_nm == 35_000
    assert q.solder_rho_ohm_m == pytest.approx(RHO_SN)
    r_p, _ = _solve(p, 0.5)
    r_q, _ = _solve(q, 0.5)
    assert r_q.R_ohm == pytest.approx(r_p.R_ohm, rel=1e-12)
