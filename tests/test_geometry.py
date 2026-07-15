import json
import math

import numpy as np

from fill_resistance.geometry import (arc_points, linearize_ring,
                                      load_problem, problem_from_json,
                                      save_problem)
from tests.util import make_multilayer, strip_problem


def test_arc_points_quarter_circle():
    r = 10_000_000  # 10 mm in nm
    start = (r, 0)
    mid = (int(r / math.sqrt(2)), int(r / math.sqrt(2)))
    end = (0, r)
    tol = 50_000  # 50 um
    pts = arc_points(start, mid, end, tol)
    assert len(pts) >= 3
    radii = np.hypot(pts[:, 0].astype(float), pts[:, 1].astype(float))
    assert np.allclose(radii, r, rtol=1e-6)
    allpts = np.vstack([pts, [end]]).astype(float)
    for a, b in zip(allpts[:-1], allpts[1:]):
        half_chord = np.hypot(*(b - a)) / 2
        sagitta = r - math.sqrt(max(r**2 - half_chord**2, 0.0))
        assert sagitta <= tol * 1.01


def test_arc_points_collinear_degrades_to_segment():
    pts = arc_points((0, 0), (5_000_000, 0), (10_000_000, 0), 1000)
    assert len(pts) == 1
    assert tuple(pts[0]) == (0, 0)


def test_linearize_ring_mixed_nodes_and_closure():
    nodes = [
        ("pt", (0, 0)),
        ("pt", (10, 0)),
        ("arc", ((10, 0), (17, 7), (10, 14))),
        ("pt", (0, 14)),
        ("pt", (0, 0)),
    ]
    ring = linearize_ring(nodes, tol_nm=1)
    assert (ring[0] != ring[-1]).any()
    assert len(ring) > 4


def test_problem_json_roundtrip_v2(tmp_path):
    p = make_multilayer(
        [[([(0, 0), (10, 0), (10, 1), (0, 1)], [])],
         [([(0, 0), (10, 0), (10, 1), (0, 1)], [])]],
        rect1_mm=(0, 0, 1, 1), rect2_mm=(9, 0, 10, 1),
        contact1="L0", contact2="L1", vias_mm=[(5.5, 0.5)])
    p.vias[0].pad_nm = 600_000
    f = tmp_path / "dump.json"
    save_problem(p, f)
    q = load_problem(f)
    assert q.layer_names == ["L0", "L1"]
    assert q.electrodes1[0].contact == "L0"
    assert q.electrodes2[0].contact == "L1"
    assert len(q.vias) == 1 and q.vias[0].drill_nm == p.vias[0].drill_nm
    assert q.vias[0].pad_nm == 600_000
    assert q.plating_nm == p.plating_nm
    assert np.array_equal(q.layers[0].polygons[0].outline,
                          p.layers[0].polygons[0].outline)
    assert abs(q.sigma_s(0) - p.sigma_s(0)) < 1e-12 * p.sigma_s(0)


def test_v1_schema_still_loads():
    p = strip_problem()
    v1 = {
        "schema_version": 1,
        "board_path": "old",
        "layer_name": "F.Cu",
        "net_name": "GND",
        "thickness_nm": 70000,
        "thickness_source": "stackup",
        "rho_ohm_m": 1.68e-8,
        "rect1": {"x0": 0, "y0": 0, "x1": 1000, "y1": 1000,
                  "layer_name": "User.1"},
        "rect2": {"x0": 5000, "y0": 0, "x1": 6000, "y1": 1000,
                  "layer_name": "User.1"},
        "polygons": [{"outline": p.layers[0].polygons[0].outline.tolist(),
                      "holes": []}],
    }
    q = problem_from_json(json.loads(json.dumps(v1)))
    assert q.layer_names == ["F.Cu"]
    assert q.vias == []
    assert q.electrodes1[0].contact == "all"
    assert q.layers[0].thickness_nm == 70000
