"""Low-current copper marking (EXPERIMENTAL): polygons around the copper
that carries almost no current at the solved operating point.

The mask is |J| < threshold, the threshold given as a percentage of the
MEAN |J| over the copper cells of every solved layer (mean, not max:
|J| spikes at contact corners would dwarf a max-relative threshold).
Cell mask -> polygons via the 0.5 contour of the binary field
(contourpy, matplotlib's own contour engine - already installed in
every plugin venv), simplified with Douglas-Peucker so the staircase
bevels collapse but one-cell-wide strips survive.

The marked copper is a SUGGESTION, not a safe cut list: it carries
little current BECAUSE the rest carries it - removing copper
redistributes the current and raises |J| everywhere else. Re-run after
any change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

JSON_NAME = "low_current_copper.json"


@dataclass
class TrimPolygon:
    outline: np.ndarray             # (N, 2) int64 board nm, unclosed ring
    holes: list[np.ndarray]         # same format


@dataclass
class LayerTrim:
    layer: str                      # copper layer name
    polygons: list[TrimPolygon]
    marked_mm2: float               # below-threshold copper area
    copper_mm2: float               # total copper area of the layer


@dataclass
class TrimResult:
    mode: str                       # "pct" (of the mean |J|) or "abs"
    value: float                    # as entered: % or A/mm2
    threshold_a_mm2: float          # the absolute threshold this run used
    layers: list[LayerTrim]         # stackup order, top first


def low_current_mask(Jmag: np.ndarray, pct: float | None = None,
                     abs_a_mm2: float | None = None
                     ) -> tuple[np.ndarray, float]:
    """(L, ny, nx) |J| in A/m2 with NaN outside copper -> boolean mask of
    the copper cells below the threshold, plus the absolute threshold
    (A/m2). Exactly one of the two threshold forms:

    pct - % of the mean |J| over ALL layers' copper. Global on purpose:
    a layer that carries little current overall is exactly the copper
    the mask should show, not a reason to lower its own threshold.
    abs_a_mm2 - absolute A/mm2. |J| scales with the test current, so
    this applies at the chosen operating point.
    """
    if (pct is None) == (abs_a_mm2 is None):
        raise ValueError("exactly one of pct / abs_a_mm2 must be given")
    copper = np.isfinite(Jmag)
    if not copper.any():
        raise ValueError("no copper cells in the solved field")
    if pct is not None:
        thr = float(np.nanmean(Jmag)) * pct / 100.0
    else:
        thr = abs_a_mm2 * 1e6                  # A/mm2 -> A/m2
    below = np.zeros(Jmag.shape, dtype=bool)
    below[copper] = Jmag[copper] < thr
    return below, thr


def _rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    """Iterative Douglas-Peucker; the first and last point always stay."""
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        seg = pts[i1] - pts[i0]
        rel = pts[i0 + 1:i1] - pts[i0]
        length = float(np.hypot(seg[0], seg[1]))
        if length == 0.0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / length
        k = int(np.argmax(d))
        if d[k] > tol:
            j = i0 + 1 + k
            keep[j] = True
            stack.append((i0, j))
            stack.append((j, i1))
    return pts[keep]


def _ring_area_nm2(ring: np.ndarray) -> float:
    x = ring[:, 0].astype(np.float64)
    y = ring[:, 1].astype(np.float64)
    return abs(float(np.dot(x, np.roll(y, -1))
                     - np.dot(y, np.roll(x, -1)))) / 2.0


def mask_to_polygons(mask2: np.ndarray, x0_nm: float, y0_nm: float,
                     h_nm: float, min_area_mm2: float) -> list[TrimPolygon]:
    """Boolean cell mask -> TrimPolygons in board nm. The boundary runs
    along cell edges, corners cut at 45 degrees by the marching-squares
    interpolation - half a cell, below the model's own resolution."""
    if not mask2.any():
        return []
    import contourpy

    # a ring of 0-cells so regions touching the grid edge close exactly
    # on the raster boundary
    z = np.pad(mask2.astype(np.float32), 1)
    xs = x0_nm + (np.arange(z.shape[1], dtype=np.float64) - 0.5) * h_nm
    ys = y0_nm + (np.arange(z.shape[0], dtype=np.float64) - 0.5) * h_nm
    gen = contourpy.contour_generator(
        x=xs, y=ys, z=z, fill_type=contourpy.FillType.OuterOffset)
    points_list, offsets_list = gen.filled(0.5, 1.5)

    tol = 0.4 * h_nm    # > 0.354h kills the staircase bevels, < 0.5h
                        # keeps the half-width of a one-cell-wide strip
    out: list[TrimPolygon] = []
    for pts, offs in zip(points_list, offsets_list):
        rings = []
        for i in range(len(offs) - 1):
            ring = pts[offs[i]:offs[i + 1] - 1]   # drop closing duplicate
            rings.append(np.rint(_rdp(ring, tol)).astype(np.int64))
        if _ring_area_nm2(rings[0]) < min_area_mm2 * 1e12:
            continue                              # speck: nothing to reclaim
        out.append(TrimPolygon(outline=rings[0], holes=rings[1:]))
    return out


def compute(result, stack, pct: float | None = None,
            abs_a_mm2: float | None = None) -> TrimResult:
    """Threshold the solved |J| (exactly one of pct / abs_a_mm2, see
    low_current_mask) and vectorize the below-threshold copper of every
    layer; areas are cell counts (exact for the model)."""
    below, thr = low_current_mask(result.Jmag, pct=pct, abs_a_mm2=abs_a_mm2)
    cell_mm2 = (stack.h_nm * 1e-6) ** 2
    layers = []
    for li, name in enumerate(stack.layer_names):
        polys = mask_to_polygons(below[li], stack.x0_nm, stack.y0_nm,
                                 stack.h_nm, config.TRIM_MIN_AREA_MM2)
        layers.append(LayerTrim(
            layer=name, polygons=polys,
            marked_mm2=float(below[li].sum()) * cell_mm2,
            copper_mm2=float(np.isfinite(result.Jmag[li]).sum()) * cell_mm2))
    return TrimResult(mode=("pct" if pct is not None else "abs"),
                      value=(pct if pct is not None else abs_a_mm2),
                      threshold_a_mm2=thr * 1e-6, layers=layers)


def summary_line(trim: TrimResult) -> str:
    parts = []
    for lt in trim.layers:
        pct = (f" ({100.0 * lt.marked_mm2 / lt.copper_mm2:.0f}%)"
               if lt.copper_mm2 else "")
        parts.append(f"{lt.layer} {lt.marked_mm2:.1f} mm2{pct}")
    head = (f"|J| < {trim.value:g}% of mean = {trim.threshold_a_mm2:.3g}"
            if trim.mode == "pct" else f"|J| < {trim.threshold_a_mm2:g}")
    return f"low-current copper ({head} A/mm2): " + "; ".join(parts)


def write_json(outdir: Path, trim: TrimResult) -> Path:
    def ring_mm(ring: np.ndarray) -> list:
        return [[round(x * 1e-6, 4), round(y * 1e-6, 4)]
                for x, y in ring.tolist()]

    p = Path(outdir) / JSON_NAME
    doc = {
        "threshold_mode": ("pct_of_mean_J" if trim.mode == "pct"
                           else "absolute"),
        "threshold_value": trim.value,
        "threshold_a_per_mm2": trim.threshold_a_mm2,
        "note": ("marked = copper below the threshold at the solved "
                 "operating point; removing copper redistributes the "
                 "current and raises |J| elsewhere - re-run after changes"),
        "layers": [{
            "layer": lt.layer,
            "marked_mm2": round(lt.marked_mm2, 3),
            "copper_mm2": round(lt.copper_mm2, 3),
            "polygons": [{"outline_mm": ring_mm(tp.outline),
                          "holes_mm": [ring_mm(h) for h in tp.holes]}
                         for tp in lt.polygons],
        } for lt in trim.layers],
    }
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p
