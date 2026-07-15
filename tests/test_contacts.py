"""Contact-model (uniform vs equipotential) and multi-part terminal tests."""
import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.errors import ConnectivityError, ElectrodeError
from fill_resistance.geometry import Electrode
from tests.util import (NM, make_multilayer, make_problem, rect_mm, sigma_s,
                        strip_problem)


def _solve(problem, h_mm, model, i_test=1.0):
    stack = raster.rasterize_stack(problem, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    parts1, parts2 = raster.electrode_partition(stack, problem)
    return solver.run_solve(problem, stack, e1, e2, i_test,
                            contact_model=model,
                            parts1=parts1, parts2=parts2), stack


def _uniform_1d_reference(m_cols, n1, n2, sig, rows):
    """Independent 1D reference: uniform injection over the first n1
    columns, extraction over the last n2, unit total current. R from
    mean potentials, per row conductance sig, `rows` parallel rows."""
    inj = np.zeros(m_cols)
    inj[:n1] += 1.0 / n1
    inj[m_cols - n2:] -= 1.0 / n2
    face_current = np.cumsum(inj)[:-1]          # current through face k,k+1
    v = np.zeros(m_cols)
    v[1:] = -np.cumsum(face_current) / sig      # per single row of cells
    v_plus = v[:n1].mean()
    v_minus = v[m_cols - n2:].mean()
    return (v_plus - v_minus) / 1.0 / rows      # rows in parallel


def test_uniform_strip_exact_1d():
    """Full-width contacts on a uniform strip: rows are identical 1D
    chains; compare with an independent 1D computation, exact."""
    p = strip_problem(length=50, width=10, e_len=5)
    res, _ = _solve(p, 0.5, "uniform")
    sig = sigma_s()
    r_ref = _uniform_1d_reference(m_cols=100, n1=10, n2=10, sig=sig, rows=20)
    assert res.R_ohm == pytest.approx(r_ref, rel=1e-9)
    assert res.contact_model == "uniform"
    assert res.power_balance_rel < 1e-9         # P = b^T V = I^2 R identity


def test_uniform_higher_than_equipotential():
    p = strip_problem(length=50, width=10, e_len=5)
    r_uni, _ = _solve(p, 0.5, "uniform")
    r_equ, _ = _solve(p, 0.5, "equipotential")
    assert r_uni.R_ohm > r_equ.R_ohm


def test_uniform_current_density_ramps_inside_contact():
    """Inside the V+ contact, |J| must ramp: ~0 at the outer edge,
    ~full sheet current at the inner (leading) edge; the equipotential
    model shows ~0 throughout the contact interior."""
    p = strip_problem(length=50, width=10, e_len=5)
    res_u, stack = _solve(p, 0.5, "uniform")
    ny, nx = stack.shape2d
    row = ny // 2
    # contact columns are the first 10 copper columns (margin = 2)
    j_outer = res_u.Jmag[0, row, 2]              # first contact column
    j_inner = res_u.Jmag[0, row, 11]             # last contact column
    j_free = res_u.Jmag[0, row, nx // 2]         # mid strip = I/(W t)
    assert j_inner > 0.8 * j_free                # ramped up to ~full
    assert j_outer < 0.2 * j_free                # near zero at outer edge
    assert j_inner > 5 * max(j_outer, 1e-30)

    res_e, _ = _solve(p, 0.5, "equipotential")
    j_center_e = res_e.Jmag[0, row, 6]           # deep inside Dirichlet region
    assert j_center_e < 0.05 * j_free


def test_multipart_terminal_equals_single_rect():
    """V+ split into two half-height rectangles == one full rectangle,
    for both contact models (exact)."""
    whole = strip_problem(length=50, width=10, e_len=5)
    split = strip_problem(length=50, width=10, e_len=5)
    split.electrodes1 = [
        Electrode(rect=rect_mm((0, 0, 5, 5))),
        Electrode(rect=rect_mm((0, 5, 5, 10))),
    ]
    for model in ("uniform", "equipotential"):
        r_whole, _ = _solve(whole, 0.5, model)
        r_split, _ = _solve(split, 0.5, model)
        assert r_split.R_ohm == pytest.approx(r_whole.R_ohm, rel=1e-9), model


def test_multipart_asymmetric_parts():
    """Two separated V+ parts feeding one V-: sane R, balance holds."""
    p = make_problem([([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 2, 3), rect2_mm=(45, 0, 50, 10))
    p.electrodes1 = [
        Electrode(rect=rect_mm((0, 0, 2, 3)), label="top lug"),
        Electrode(rect=rect_mm((0, 7, 2, 10)), label="bottom lug"),
    ]
    single, _ = _solve(
        make_problem([([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 2, 3), rect2_mm=(45, 0, 50, 10)),
        0.25, "uniform")
    multi, _ = _solve(p, 0.25, "uniform")
    assert multi.R_ohm < single.R_ohm            # more contact area helps
    assert multi.power_balance_rel < 1e-9


def test_part_off_copper_raises_with_label():
    p = strip_problem(length=50, width=10, e_len=5)
    p.electrodes1 = [
        Electrode(rect=rect_mm((0, 0, 5, 10))),
        Electrode(rect=rect_mm((100, 100, 105, 105)), label="stray part"),
    ]
    stack = raster.rasterize_stack(p, 0.5 * NM)
    with pytest.raises(ElectrodeError, match="stray part"):
        raster.electrode_masks(stack, p)


def test_injection_area_currents_equipotential_flux():
    """Two V+ lugs at different distances: the nearer one carries more;
    the flux split sums exactly to the test current."""
    p = make_problem([([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 2, 10), rect2_mm=(48, 0, 50, 10))
    p.electrodes1 = [
        Electrode(rect=rect_mm((0, 4, 2, 6)), label="far lug"),
        Electrode(rect=rect_mm((10, 4, 12, 6)), label="near lug"),
    ]
    res, _ = _solve(p, 0.25, "equipotential", i_test=10.0)
    pc = dict(res.part_currents1)
    assert pc["near lug"] > pc["far lug"]
    assert pc["near lug"] + pc["far lug"] == pytest.approx(10.0, rel=1e-9)
    # V- side: single part carries everything
    assert res.part_currents2[0][1] == pytest.approx(10.0, rel=1e-9)


def test_injection_area_currents_uniform_area_share():
    """Uniform model: each injection area carries exactly its cell share."""
    p = make_problem([([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 2, 10), rect2_mm=(48, 0, 50, 10))
    p.electrodes1 = [
        Electrode(rect=rect_mm((0, 0, 2, 6)), label="big"),    # 2x6 mm
        Electrode(rect=rect_mm((0, 6, 2, 9)), label="small"),  # 2x3 mm
    ]
    res, _ = _solve(p, 0.25, "uniform", i_test=9.0)
    pc = dict(res.part_currents1)
    # cell counts: 8x24 = 192 and 8x12 = 96 at h=0.25 -> shares 2/3, 1/3
    assert pc["big"] == pytest.approx(6.0, rel=1e-12)
    assert pc["small"] == pytest.approx(3.0, rel=1e-12)


def test_injection_area_partition_first_wins():
    """Overlapping parts: shared cells attributed to the first part, so
    the shares still sum to the terminal current."""
    p = make_problem([([(0, 0), (50, 0), (50, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 2, 10), rect2_mm=(48, 0, 50, 10))
    p.electrodes1 = [
        Electrode(rect=rect_mm((0, 0, 2, 6)), label="first"),
        Electrode(rect=rect_mm((0, 4, 2, 10)), label="second"),  # overlaps
    ]
    res, _ = _solve(p, 0.25, "uniform", i_test=1.0)
    total = sum(a for _, a in res.part_currents1)
    assert total == pytest.approx(1.0, rel=1e-12)


def test_uniform_multicomponent_raises():
    """Two disconnected sheets that each touch both terminals: the
    uniform model would build a singular system (one ground cell, pure-
    Neumann second component) and previously returned garbage silently
    (e.g. negative gigaohms). It must refuse; equipotential handles it."""
    strip1 = [(0, 0), (10, 0), (10, 1), (0, 1)]
    strip2 = [(0, 0), (10, 0), (10, 2), (0, 2)]   # asymmetric shares
    p = make_multilayer([[(strip1, [])], [(strip2, [])]],
                        (0, 0, 1, 2), (9, 0, 10, 1))  # contact 'all', no vias
    with pytest.raises(ConnectivityError, match="disconnected"):
        _solve(p, 1.0, "uniform")
    res, _ = _solve(p, 1.0, "equipotential")
    assert np.isfinite(res.R_ohm) and res.R_ohm > 0
    assert res.power_balance_rel < 1e-9


def test_touching_ok_uniform_error_equipotential():
    p = make_problem([([(0, 0), (10, 0), (10, 10), (0, 10)], [])],
                     rect1_mm=(0, 0, 5, 10), rect2_mm=(5, 0, 10, 10))
    res, _ = _solve(p, 0.5, "uniform")           # touching is fine here
    assert res.R_ohm > 0
    p2 = make_problem([([(0, 0), (10, 0), (10, 10), (0, 10)], [])],
                      rect1_mm=(0, 0, 5, 10), rect2_mm=(5, 0, 10, 10))
    with pytest.raises(ElectrodeError, match="touch"):
        _solve(p2, 0.5, "equipotential")