"""Low-current copper marking: threshold mask -> polygons in board nm.

The kipy pushing side is exercised only against a live KiCad (as for
the overlays); the proto assembly of a single polygon is testable
offline and covered here.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fill_resistance import trim


def _stack(names=("F.Cu",), h_nm=100_000, x0=0, y0=0):
    return SimpleNamespace(layer_names=list(names), h_nm=h_nm,
                           x0_nm=x0, y0_nm=y0)


def test_low_current_mask_threshold():
    J = np.full((1, 4, 4), np.nan)
    J[0, :2, :] = 1.0                  # 8 cells carrying little
    J[0, 2, :2] = 100.0                # 2 hot cells; mean = 20.8
    mask, thr = trim.low_current_mask(J, pct=10.0)
    assert thr == pytest.approx(2.08)
    assert mask[0, :2, :].all()
    assert not mask[0, 2, :2].any()
    assert not mask[0, 3, :].any()     # NaN = no copper, never marked


def test_low_current_mask_absolute():
    J = np.full((1, 3, 3), np.nan)
    J[0, 0, :] = 0.5e6                 # 0.5 A/mm2 in A/m2
    J[0, 1, :] = 2.0e6                 # 2 A/mm2
    mask, thr = trim.low_current_mask(J, abs_a_mm2=1.0)
    assert thr == pytest.approx(1.0e6)
    assert mask[0, 0, :].all()
    assert not mask[0, 1, :].any()


def test_low_current_mask_needs_exactly_one_threshold():
    J = np.ones((1, 2, 2))
    with pytest.raises(ValueError):
        trim.low_current_mask(J)
    with pytest.raises(ValueError):
        trim.low_current_mask(J, pct=10.0, abs_a_mm2=1.0)


def test_mask_rectangle_polygon():
    m = np.zeros((20, 30), dtype=bool)
    m[5:15, 4:9] = True
    polys = trim.mask_to_polygons(m, x0_nm=0, y0_nm=0, h_nm=1000,
                                  min_area_mm2=0.0)
    assert len(polys) == 1
    p = polys[0]
    assert p.holes == []
    xs, ys = p.outline[:, 0], p.outline[:, 1]
    # the boundary runs on the cell edges of the marked block
    assert xs.min() == 4000 and xs.max() == 9000
    assert ys.min() == 5000 and ys.max() == 15000
    # RDP collapsed the straight runs: 2 bevel points per corner plus at
    # most one leftover at the ring seam (first/last are fixed anchors)
    assert len(p.outline) <= 9


def test_mask_with_hole():
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 2:18] = True
    m[8:12, 8:12] = False
    polys = trim.mask_to_polygons(m, 0, 0, 1000, min_area_mm2=0.0)
    assert len(polys) == 1
    assert len(polys[0].holes) == 1


def test_mask_touching_grid_edge_closes():
    # the padding ring must close regions that touch the raster edge
    # exactly on the raster boundary
    m = np.ones((5, 8), dtype=bool)
    polys = trim.mask_to_polygons(m, 0, 0, 1000, min_area_mm2=0.0)
    assert len(polys) == 1
    xs, ys = polys[0].outline[:, 0], polys[0].outline[:, 1]
    assert xs.min() == 0 and xs.max() == 8000
    assert ys.min() == 0 and ys.max() == 5000


def test_min_area_drops_specks():
    m = np.zeros((10, 10), dtype=bool)
    m[5, 5] = True                     # one 100 um cell = 0.01 mm2
    assert trim.mask_to_polygons(m, 0, 0, 100_000, min_area_mm2=0.5) == []
    assert len(trim.mask_to_polygons(m, 0, 0, 100_000,
                                     min_area_mm2=0.0)) == 1


def test_compute_and_json(tmp_path):
    J = np.full((2, 10, 10), np.nan)
    J[0, :, :] = 10.0
    J[0, :, :5] = 0.01                 # half of the top layer nearly dead
    J[1, :, :] = 10.0
    stack = _stack(names=["F.Cu", "B.Cu"], h_nm=1_000_000)
    tr = trim.compute(SimpleNamespace(Jmag=J), stack, pct=10.0)
    assert tr.mode == "pct" and tr.value == 10.0
    assert [lt.layer for lt in tr.layers] == ["F.Cu", "B.Cu"]
    assert tr.layers[0].polygons and not tr.layers[1].polygons
    assert tr.layers[0].marked_mm2 == pytest.approx(50.0)
    assert tr.layers[0].copper_mm2 == pytest.approx(100.0)
    # mean = (50*0.01 + 150*10) / 200 = 7.5025 A/m2, threshold 10% of it
    assert tr.threshold_a_mm2 == pytest.approx(0.75025e-6)

    p = trim.write_json(tmp_path, tr)
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["layers"][0]["marked_mm2"] == pytest.approx(50.0)
    ring = doc["layers"][0]["polygons"][0]["outline_mm"]
    assert all(0 <= x <= 5.5 and 0 <= y <= 10.0 for x, y in ring)
    assert "F.Cu" in trim.summary_line(tr)
    assert "% of mean" in trim.summary_line(tr)


def test_compute_absolute_mode(tmp_path):
    J = np.full((1, 10, 10), np.nan)
    J[0, :, :] = 10.0e6                # 10 A/mm2
    J[0, :, :5] = 0.1e6                # 0.1 A/mm2: below 1 A/mm2
    stack = _stack(names=["F.Cu"], h_nm=1_000_000)
    tr = trim.compute(SimpleNamespace(Jmag=J), stack, abs_a_mm2=1.0)
    assert tr.mode == "abs" and tr.value == 1.0
    assert tr.threshold_a_mm2 == pytest.approx(1.0)
    assert tr.layers[0].marked_mm2 == pytest.approx(50.0)
    line = trim.summary_line(tr)
    assert "|J| < 1 A/mm2" in line and "% of mean" not in line
    doc = json.loads(trim.write_json(tmp_path, tr)
                     .read_text(encoding="utf-8"))
    assert doc["threshold_mode"] == "absolute"
    assert doc["threshold_value"] == 1.0


def test_trim_shape_proto():
    from kipy.util.board_layer import layer_from_canonical_name

    from fill_resistance import board_io

    tp = trim.TrimPolygon(
        outline=np.array([[0, 0], [10000, 0], [10000, 5000], [0, 5000]],
                         dtype=np.int64),
        holes=[np.array([[2000, 1000], [3000, 1000], [3000, 2000]],
                        dtype=np.int64)])
    layer = layer_from_canonical_name("User.5")
    proto = board_io._trim_shape(tp, layer, lock=False).proto
    poly = proto.shape.polygon.polygons[0]
    assert len(poly.outline.nodes) == 4 and poly.outline.closed
    assert len(poly.holes) == 1 and len(poly.holes[0].nodes) == 3
    assert poly.holes[0].closed
    assert proto.layer == layer
    from kipy.proto.common.types.base_types_pb2 import GraphicFillType
    assert (proto.shape.attributes.fill.fill_type
            == GraphicFillType.GFT_FILLED)
