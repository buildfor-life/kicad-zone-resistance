import numpy as np
import pytest

from fill_resistance import config, raster, solver
from tests.util import NM, make_problem, strip_problem


def _solve(problem, h_mm, i_test=1.0):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, i_test,
                            contact_model="equipotential"), stack


def test_uniform_strip_exact_discrete():
    """Full-width electrodes on a uniform strip: every row is an identical
    series chain, so the discrete solution is exact:
    R = (n_free_columns + 1) / (n_rows * sigma_s)."""
    p = strip_problem(length=50, width=10, e_len=5)
    res, stack = _solve(p, 0.5)
    n_free_cols = int(round((50 - 2 * 5) / 0.5))   # 80
    n_rows = int(round(10 / 0.5))                  # 20
    r_exact = (n_free_cols + 1) / n_rows / p.sigma_s(0)
    assert res.R_ohm == pytest.approx(r_exact, rel=1e-9)
    assert res.mismatch_rel < 1e-10
    r_cont = p.rho_ohm_m * 0.0405 / (0.010 * 70e-6)
    assert res.R_ohm == pytest.approx(r_cont, rel=1e-6)


def test_strip_R_independent_of_h():
    p = strip_problem(length=50, width=10, e_len=5)
    for h in (1.0, 0.5, 0.25):
        res, _ = _solve(p, h)
        m = int(round(40 / h))
        rows = int(round(10 / h))
        assert res.R_ohm == pytest.approx((m + 1) / rows / p.sigma_s(0),
                                          rel=1e-9)


def test_partial_electrode_constriction():
    full = strip_problem(length=50, width=10, e_len=5)
    partial = make_problem(
        [([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
        rect1_mm=(0, 4, 5, 6),
        rect2_mm=(45, 4, 50, 6))
    r_full, _ = _solve(full, 0.25)
    r_a, _ = _solve(partial, 0.25)
    r_b, _ = _solve(partial, 0.125)
    assert r_a.R_ohm > r_full.R_ohm * 1.05
    assert abs(r_a.R_ohm - r_b.R_ohm) < 0.01 * r_b.R_ohm


def test_l_shape_corner_squares():
    """Right-angle bend of equal-width arms: corner square counts as
    ~0.559 squares (conformal-mapping result), within 5% at a fine grid."""
    w = 10.0
    outline = [(0, 0), (30, 0), (30, 30), (20, 30), (20, 10), (0, 10)]
    p = make_problem([(outline, [])],
                     rect1_mm=(0, 0, 2, 10),
                     rect2_mm=(20, 28, 30, 30))
    res, _ = _solve(p, 0.125)
    a_sq = (20 - 2) / w
    b_sq = (28 - 10) / w
    r_expect = (a_sq + b_sq + 0.559) / p.sigma_s(0)
    assert res.R_ohm == pytest.approx(r_expect, rel=0.05)


def test_hole_increases_resistance_and_converges():
    solid = make_problem([([(0, 0), (40, 0), (40, 20), (0, 20)], [])],
                         rect1_mm=(0, 0, 2, 20), rect2_mm=(38, 0, 40, 20))
    holed = make_problem(
        [([(0, 0), (40, 0), (40, 20), (0, 20)],
          [[(15, 5), (25, 5), (25, 15), (15, 15)]])],
        rect1_mm=(0, 0, 2, 20), rect2_mm=(38, 0, 40, 20))
    r_solid, _ = _solve(solid, 0.25)
    r_a, _ = _solve(holed, 0.25)
    r_b, _ = _solve(holed, 0.125)
    assert r_a.R_ohm > r_solid.R_ohm * 1.1
    assert abs(r_a.R_ohm - r_b.R_ohm) < 0.01 * r_b.R_ohm


def test_cg_path_matches_direct(monkeypatch):
    p = strip_problem(length=50, width=10, e_len=5)
    r_direct, _ = _solve(p, 0.25)
    monkeypatch.setattr(config, "SPSOLVE_MAX_UNKNOWNS", 0)
    r_cg, _ = _solve(p, 0.25)
    assert r_cg.solve_info.method == "cg+jacobi"
    assert r_cg.R_ohm == pytest.approx(r_direct.R_ohm, rel=1e-6)
    assert r_cg.mismatch_rel < 1e-5


def test_current_density_and_potential_scale():
    """Uniform strip at 1 A: |J| in the free region equals I/(W t); the
    potential span equals R * I."""
    p = strip_problem(length=50, width=10, e_len=5)
    res, stack = _solve(p, 0.5)
    j_expect = 1.0 / (0.010 * 70e-6)
    ny, nx = stack.shape2d
    assert res.Jmag[0, ny // 2, nx // 2] == pytest.approx(j_expect, rel=1e-6)
    assert np.nanmax(res.V) == pytest.approx(res.R_ohm, rel=1e-9)


def test_test_current_scaling():
    """V and J scale linearly with I_test, power quadratically; R fixed."""
    p = strip_problem(length=50, width=10, e_len=5)
    r1, _ = _solve(p, 0.5, i_test=1.0)
    r10, _ = _solve(p, 0.5, i_test=10.0)
    assert r10.R_ohm == pytest.approx(r1.R_ohm, rel=1e-12)
    assert np.nanmax(r10.V) == pytest.approx(10 * np.nanmax(r1.V), rel=1e-9)
    assert np.nanmax(r10.Jmag) == pytest.approx(10 * np.nanmax(r1.Jmag),
                                                rel=1e-9)
    assert r10.P_total == pytest.approx(100 * r1.P_total, rel=1e-9)


def test_nonpositive_test_current_rejected():
    """i_test <= 0 would divide by zero in the percentage reporting;
    it must be rejected up front with a clean user-facing message."""
    from fill_resistance import pipeline
    from fill_resistance.errors import UserFacingError
    p = strip_problem()
    for bad in (0.0, -1.0):
        with pytest.raises(UserFacingError, match="Test current"):
            pipeline.run(p, None, show=False, i_test=bad)


def test_power_identity():
    """Sum of edge powers equals I^2 R exactly for the direct solve."""
    p = make_problem(
        [([(0, 0), (40, 0), (40, 20), (0, 20)],
          [[(15, 5), (25, 5), (25, 15), (15, 15)]])],
        rect1_mm=(0, 0, 2, 20), rect2_mm=(38, 0, 40, 20))
    res, _ = _solve(p, 0.25, i_test=40.0)
    assert res.power_balance_rel < 1e-9
    assert res.P_total == pytest.approx(40.0 ** 2 * res.R_ohm, rel=1e-12)
    assert res.P_vias == 0.0
