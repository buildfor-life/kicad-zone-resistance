"""Barrel (via / through-hole pad) contact tests: current enters at the
drill-wall ring, not the pad face, and soldered THT joints carry a
solder-filled hole plus an average-thickness solder coat on the pad."""
import math

import numpy as np
import pytest

from fill_resistance import raster, solver
from fill_resistance.geometry import (Electrode, Polygon, ViaLink,
                                      contact_solder_buildups, load_problem,
                                      problem_from_json, problem_to_json,
                                      save_problem, tht_joint_buildups)
from tests.util import NM, make_problem, rect_mm, ring_mm

PLATE20 = [(0, 0), (20, 0), (20, 20), (0, 20)]


def _barrel(x_mm, y_mm, drill_mm, pad_mm=0.0, solder=False, polygons=None):
    r = max(pad_mm, drill_mm) / 2
    return Electrode(
        rect=rect_mm((x_mm - r, y_mm - r, x_mm + r, y_mm + r)),
        contact="all", label=f"via({x_mm},{y_mm})",
        drill_nm=int(drill_mm * NM), pad_nm=int(pad_mm * NM),
        center=(int(x_mm * NM), int(y_mm * NM)), solder=solder,
        polygons=polygons)


def _disc(x_mm, y_mm, r_mm, n=64) -> Polygon:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(outline=ring_mm(
        [(x_mm + r_mm * np.cos(a), y_mm + r_mm * np.sin(a)) for a in ang]))


def _solve(p, h_mm, model="equipotential"):
    stack = raster.rasterize_stack(p, h_mm * NM)
    e1, e2 = raster.electrode_masks(stack, p)
    return solver.run_solve(p, stack, e1, e2, 1.0, contact_model=model), stack


def test_ring_cells_at_drill_wall():
    """The contact cells of a barrel electrode form a ring at the drill
    wall (one-cell tolerance), not the pad face."""
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=1.6)]
    stack = raster.rasterize_stack(p, 0.1 * NM)
    e1, _ = raster.electrode_masks(stack, p)
    ii, jj = np.nonzero(e1[0])
    xs = stack.x0_nm + (jj + 0.5) * stack.h_nm - 10 * NM
    ys = stack.y0_nm + (ii + 0.5) * stack.h_nm - 10 * NM
    d = np.hypot(xs, ys)
    assert len(ii) >= 8
    assert (np.abs(d - 0.5 * NM) <= stack.h_nm + 1).all()
    # far fewer cells than the full 1.6 mm pad disc
    assert len(ii) < 0.5 * math.pi * (0.8 * NM / stack.h_nm) ** 2


def test_two_barrel_contacts_match_acosh():
    """Two equipotential circular contacts of radius a, centers d apart,
    on a large sheet: R = rho/(pi t) * acosh(d / 2a). The barrel-ring
    contact must reproduce the analytic spreading resistance."""
    t_um, rho = 70.0, 1.68e-8
    plate = [(0, 0), (80, 0), (80, 60), (0, 60)]
    p = make_problem([(plate, [])], rect1_mm=(0, 0, 1, 1),
                     rect2_mm=(79, 59, 80, 60), t_um=t_um, rho=rho)
    p.electrodes1 = [_barrel(30, 30, drill_mm=2.0)]
    p.electrodes2 = [_barrel(50, 30, drill_mm=2.0)]
    res, _ = _solve(p, 0.15)
    r_ref = rho / (math.pi * t_um * 1e-6) * math.acosh(20e-3 / (2 * 1e-3))
    assert res.R_ohm == pytest.approx(r_ref, rel=0.08)


def test_barrel_includes_pad_spreading_resistance():
    """Injecting at the barrel wall (0.5 mm ring) sees the spreading
    resistance the whole-pad-face contact (2.4 mm equipotential disc)
    short-circuits: R_barrel > R_pad_face."""
    p1 = make_problem([(PLATE20, [])],
                      rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p1.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=2.4)]
    r_barrel, _ = _solve(p1, 0.1)

    p2 = make_problem([(PLATE20, [])],
                      rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p2.electrodes1 = [Electrode(rect=rect_mm((8.8, 8.8, 11.2, 11.2)),
                                contact="all", label="pad face",
                                polygons=[_disc(10, 10, 1.2)])]
    r_face, _ = _solve(p2, 0.1)
    assert r_barrel.R_ohm > r_face.R_ohm * 1.05


def test_ring_fallback_nearest_copper():
    """Antipad bigger than the drill: no copper at the wall ring, the
    contact falls back to the nearest copper ring inside the pad
    footprint (e.g. thermal-spoke tips / hole edge)."""
    hole = [(10 + 1.2 * np.cos(a), 10 + 1.2 * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
    p = make_problem([(PLATE20, [hole])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.electrodes1 = [_barrel(10, 10, drill_mm=0.6, pad_mm=4.0)]
    res, stack = _solve(p, 0.1)
    e1, _ = raster.electrode_masks(stack, p)
    ii, jj = np.nonzero(e1[0])
    d = np.hypot(stack.x0_nm + (jj + 0.5) * stack.h_nm - 10 * NM,
                 stack.y0_nm + (ii + 0.5) * stack.h_nm - 10 * NM)
    assert len(ii) >= 8
    assert (d >= 1.2 * NM - stack.h_nm).all()
    assert (d <= 1.2 * NM + 2.5 * stack.h_nm).all()
    assert np.isfinite(res.R_ohm) and res.R_ohm > 0


def test_solder_filled_barrel_resistance():
    """THT joints: the solder core conducts in parallel with the plating.
    Exact parallel-area formula, and a sanity ratio for a 1 mm drill."""
    v = ViaLink(x=0, y=0, drill_nm=1_000_000, z_top_nm=-1, z_bot_nm=1)
    rho, sn = 1.68e-8, 1.32e-7
    r_plain = v.barrel_resistance(1_600_000, rho, 18_000)
    r_fill = v.barrel_resistance(1_600_000, rho, 18_000,
                                 solder_rho_ohm_m=sn)
    ga = math.pi * 1e-3 * 18e-6 / rho
    ga += math.pi * (0.5e-3 - 18e-6) ** 2 / sn
    assert r_fill == pytest.approx(1.6e-3 / ga, rel=1e-12)
    assert 1.5 < r_plain / r_fill < 4.0


def test_contact_solder_coat():
    """A soldered THT contact adds an average-thickness solder buildup
    over the pad face on its SOLDER side only (opposite the component),
    lowering the spreading resistance vs the bare barrel contact."""
    def prob():
        p = make_problem([(PLATE20, [])],
                         rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
        p.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=2.4,
                                 solder=True, polygons=[_disc(10, 10, 1.2)])]
        p.electrodes1[0].protrusion_side = "F.Cu"
        return p

    # solder side not among the included layers -> no coat there
    q = prob()
    q.electrodes1[0].protrusion_side = "B.Cu"
    assert contact_solder_buildups(q) == []

    p = prob()
    assert contact_solder_buildups(p) == ["F.Cu"]
    assert len(p.buildups) == 1 and p.buildups[0].layer_name == "F.Cu"
    r_coat, stack = _solve(p, 0.1)
    assert stack.buildup is not None and stack.buildup.any()

    r_bare, _ = _solve(prob(), 0.1)          # helper not called: no coat
    assert r_coat.R_ohm < r_bare.R_ohm


def test_lead_fillet_profile():
    """The protruding-lead solder cone paints thick_scale with the exact
    per-cell formula: 1 + H*clip((rb-r)/(rb-ra), 0, 1)*(rho_cu/rho_sn)/t
    on copper of the protrusion side; nothing elsewhere."""
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=2.4, solder=True)]
    p.electrodes1[0].protrusion_side = "F.Cu"
    stack = raster.rasterize_stack(p, 0.1 * NM)
    assert stack.thick_scale is not None
    ny, nx = stack.shape2d
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.hypot(stack.x0_nm + (jj + 0.5) * stack.h_nm - 10 * NM,
                 stack.y0_nm + (ii + 0.5) * stack.h_nm - 10 * NM)
    ra, rb, H = 0.5 * NM, 1.2 * NM, p.tht_protrusion_nm
    t_eq = H * np.clip((rb - r) / (rb - ra), 0, 1) \
        * (p.rho_ohm_m / p.solder_rho_ohm_m)
    expect = np.where(stack.masks[0],
                      1.0 + t_eq / p.layers[0].thickness_nm, 1.0)
    assert np.allclose(stack.thick_scale[0], expect, rtol=1e-12)
    # 1.5 mm of solder at the wall ~ 191 um copper: factor ~ 3.7 on 70 um
    assert stack.thick_scale[0].max() > 3.0

    p.electrodes1[0].protrusion_side = None    # e.g. via contact: no cone
    s2 = raster.rasterize_stack(p, 0.1 * NM)
    assert s2.thick_scale is None


def test_lead_fillet_lowers_resistance(monkeypatch):
    """The cone shorts the joint vicinity: R(with cone) < R(coat-less
    bare barrel); the adaptive grid pins the cone cells fine and
    matches the uniform grid."""
    def prob(protrude=True):
        p = make_problem([(PLATE20, [])],
                         rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
        p.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=2.4,
                                 solder=True)]
        p.electrodes1[0].protrusion_side = "F.Cu"
        if not protrude:
            p.tht_protrusion_nm = 0
        return p

    r_cone, _ = _solve(prob(), 0.1)
    r_bare, _ = _solve(prob(protrude=False), 0.1)
    assert r_cone.R_ohm < r_bare.R_ohm

    from fill_resistance import config
    monkeypatch.setattr(config, "ADAPTIVE_CELLS", True)
    r_ada, _ = _solve(prob(), 0.1)
    assert r_ada.R_ohm == pytest.approx(r_cone.R_ohm, rel=2e-3)


def _pad_link(populated=True):
    return ViaLink(x=10 * NM, y=10 * NM, drill_nm=1_000_000, z_top_nm=-1,
                   z_bot_nm=1, kind="pad", pad_nm=2_400_000,
                   solder_filled=populated,
                   protrusion_side="F.Cu" if populated else None)


def test_stitching_pad_joint():
    """A populated THT pad on the net (not a contact) gets the full
    joint: solder-side coat, cone, and a conducting (plugged) mouth;
    a DNP pad gets an open hole and nothing else."""
    def prob(populated=True):
        p = make_problem([(PLATE20, [])],
                         rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
        p.vias = [_pad_link(populated)]
        return p

    p = prob()
    assert tht_joint_buildups(p) == ["F.Cu"]
    assert len(p.buildups) == 1
    r_joint, stack = _solve(p, 0.1)
    assert stack.thick_scale is not None and stack.thick_scale.max() > 3.0
    assert stack.buildup is not None and stack.buildup.any()
    assert stack.masks[0][stack.cell_of(10 * NM, 10 * NM)]   # plugged mouth

    q = prob(populated=False)
    assert tht_joint_buildups(q) == []
    r_bare, s2 = _solve(q, 0.1)
    assert s2.buildup is None
    assert not s2.masks[0][s2.cell_of(10 * NM, 10 * NM)]     # DNP: open hole
    assert r_joint.R_ohm < r_bare.R_ohm


def test_cone_not_doubled_at_contact():
    """A contact THT pad also appears in the net's pad list (ViaLink):
    the cone and coat must be applied once, not squared/stacked."""
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.electrodes1 = [_barrel(10, 10, drill_mm=1.0, pad_mm=2.4, solder=True,
                             polygons=[_disc(10, 10, 1.2)])]
    p.electrodes1[0].protrusion_side = "F.Cu"
    p.vias = [_pad_link()]
    assert contact_solder_buildups(p) == ["F.Cu"]
    assert tht_joint_buildups(p) == []       # contact center is skipped
    stack = raster.rasterize_stack(p, 0.1 * NM)
    wall = 1.0 + p.tht_protrusion_nm \
        * (p.rho_ohm_m / p.solder_rho_ohm_m) / p.layers[0].thickness_nm
    assert stack.thick_scale.max() == pytest.approx(wall, rel=1e-12)


def test_lead_in_barrel_resistance():
    """Populated hole: plating || lead cylinder || solder annulus, with
    the lead clipped to the plating bore."""
    v = ViaLink(x=0, y=0, drill_nm=1_000_000, z_top_nm=-1, z_bot_nm=1)
    rho, sn = 1.68e-8, 1.32e-7
    r_solder = v.barrel_resistance(1_600_000, rho, 18_000,
                                   solder_rho_ohm_m=sn)
    r_lead = v.barrel_resistance(1_600_000, rho, 18_000,
                                 solder_rho_ohm_m=sn,
                                 lead_nm=750_000, lead_rho_ohm_m=rho)
    rl, rc = 0.375e-3, 0.5e-3 - 18e-6
    ga = math.pi * 1e-3 * 18e-6 / rho
    ga += math.pi * rl ** 2 / rho + math.pi * (rc ** 2 - rl ** 2) / sn
    assert r_lead == pytest.approx(1.6e-3 / ga, rel=1e-12)
    assert r_lead < r_solder
    # a lead wider than the bore is clipped to it
    r_big = v.barrel_resistance(1_600_000, rho, 18_000,
                                solder_rho_ohm_m=sn,
                                lead_nm=2_000_000, lead_rho_ohm_m=rho)
    ga2 = math.pi * 1e-3 * 18e-6 / rho + math.pi * rc ** 2 / rho
    assert r_big == pytest.approx(1.6e-3 / ga2, rel=1e-12)


def test_oblong_pad_cone_uses_inscribed_dim():
    """Oblong pads: the cone tapers to the inscribed circle (pad_min),
    never past it, so the long pad axis is not overstated sideways."""
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.vias = [_pad_link()]
    p.vias[0].pad_min_nm = 1_600_000          # 2.4 mm max, 1.6 mm min
    stack = raster.rasterize_stack(p, 0.1 * NM)
    ii, jj = np.nonzero(stack.thick_scale[0] != 1.0)
    d = np.hypot(stack.x0_nm + (jj + 0.5) * stack.h_nm - 10 * NM,
                 stack.y0_nm + (ii + 0.5) * stack.h_nm - 10 * NM)
    assert len(d) and d.max() < 0.8 * NM


def test_stitching_coat_exact_shape():
    """When KiCad supplies the exact pad polygon, the coat uses it
    instead of the pad-diameter disc (oblong pads stay honest)."""
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.vias = [_pad_link()]
    shape = _disc(10, 10, 0.9)
    assert tht_joint_buildups(p, {(10 * NM, 10 * NM): [shape]}) == ["F.Cu"]
    assert p.buildups[0].polygons[0] is shape


def test_vialink_solder_json():
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.vias = [_pad_link()]
    d = problem_to_json(p)
    q = problem_from_json(d)
    assert q.vias[0].solder_filled is True
    assert q.vias[0].protrusion_side == "F.Cu"
    # legacy dumps without the flag: THT pads counted as solder-filled,
    # vias as plating-only
    del d["vias"][0]["solder_filled"], d["vias"][0]["protrusion_side"]
    q = problem_from_json(d)
    assert q.vias[0].solder_filled is True
    assert q.vias[0].protrusion_side is None
    d["vias"][0]["kind"] = "via"
    assert problem_from_json(d).vias[0].solder_filled is False


def test_barrel_electrode_json_roundtrip(tmp_path):
    p = make_problem([(PLATE20, [])],
                     rect1_mm=(0, 0, 1, 20), rect2_mm=(19, 0, 20, 20))
    p.electrodes1 = [_barrel(10, 10, drill_mm=0.6, pad_mm=1.2, solder=True,
                             polygons=[_disc(10, 10, 0.6)])]
    p.electrodes1[0].barrel_z = (-1, 1_600_001)
    p.electrodes1[0].protrusion_side = "B.Cu"
    p.tht_protrusion_nm = 1_200_000
    f = tmp_path / "d.json"
    save_problem(p, f)
    q = load_problem(f)
    e = q.electrodes1[0]
    assert e.drill_nm == 600_000 and e.pad_nm == 1_200_000
    assert e.center == (10 * NM, 10 * NM)
    assert e.barrel_z == (-1, 1_600_001)
    assert e.solder is True and len(e.polygons) == 1
    assert e.protrusion_side == "B.Cu"
    assert q.tht_protrusion_nm == 1_200_000
