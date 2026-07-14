"""Synthetic Problem builders for tests. Dimensions in mm here, nm inside."""
import numpy as np

from fill_resistance.geometry import (Electrode, LayerFill, Polygon, Problem,
                                      Rect, ViaLink)

NM = 1_000_000  # nm per mm


def ring_mm(points_mm) -> np.ndarray:
    return (np.asarray(points_mm, dtype=float) * NM).astype(np.int64)


def _polys(polygons_mm) -> list[Polygon]:
    return [
        Polygon(outline=ring_mm(outline), holes=[ring_mm(h) for h in holes])
        for outline, holes in polygons_mm
    ]


def rect_mm(r, layer="User.1") -> Rect:
    return Rect.normalized(int(r[0] * NM), int(r[1] * NM),
                           int(r[2] * NM), int(r[3] * NM), layer)


def make_multilayer(layers_mm, rect1_mm, rect2_mm, contact1="all",
                    contact2="all", vias_mm=(), t_um=70.0, gap_mm=1.0,
                    drill_mm=0.3, plating_um=18.0, rho=1.68e-8) -> Problem:
    """layers_mm: one list of (outline_pts, [holes]) per layer; layer i is
    named 'L{i}' at z = i * gap_mm. vias_mm: (x, y) through-barrels."""
    nlayers = len(layers_mm)
    return Problem(
        board_path="synthetic",
        net_name="TEST",
        rho_ohm_m=rho,
        plating_nm=int(plating_um * 1000),
        layers=[
            LayerFill(layer_name=f"L{i}", thickness_nm=int(t_um * 1000),
                      z_nm=int(i * gap_mm * NM), polygons=_polys(polys_mm))
            for i, polys_mm in enumerate(layers_mm)
        ],
        vias=[
            ViaLink(x=int(x * NM), y=int(y * NM),
                    drill_nm=int(drill_mm * NM),
                    z_top_nm=-1, z_bot_nm=int((nlayers - 1) * gap_mm * NM) + 1)
            for x, y in vias_mm
        ],
        electrodes1=[Electrode(rect=rect_mm(rect1_mm), contact=contact1)],
        electrodes2=[Electrode(rect=rect_mm(rect2_mm), contact=contact2)],
        thickness_source="override",
    )


def make_problem(polygons_mm, rect1_mm, rect2_mm, t_um=70.0,
                 rho=1.68e-8) -> Problem:
    """Single-layer problem (the v1 test surface)."""
    p = make_multilayer([polygons_mm], rect1_mm, rect2_mm, t_um=t_um, rho=rho)
    p.layers[0].layer_name = "F.Cu"
    return p


def strip_problem(length=50.0, width=10.0, e_len=5.0, t_um=70.0):
    """Uniform strip with full-width electrodes at both ends."""
    outline = [(0, 0), (length, 0), (length, width), (0, width)]
    return make_problem(
        [(outline, [])],
        rect1_mm=(0, 0, e_len, width),
        rect2_mm=(length - e_len, 0, length, width),
        t_um=t_um,
    )


def sigma_s(t_um=70.0, rho=1.68e-8) -> float:
    return t_um * 1e-6 / rho
