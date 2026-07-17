"""Rendering for the experimental in-KiCad result overlays: a solved
field (|J|) as an RGBA PNG, one pixel per grid cell, transparent where
there is no copper. The pushing side (ReferenceImages via the IPC API)
lives in board_io; this module stays KiCad-free so it is testable
headless.
"""
from __future__ import annotations

import io

import numpy as np

from . import config

# the colormap's near-black bottom must stay distinguishable from
# KiCad's dark canvas (matplotlib figures sit on a light background
# instead), so the log scale starts this far up the colormap
FLOOR = 0.18


def heatmap_png(data3: np.ndarray, li: int, alpha: int | None = None,
                bleed: bool = True) -> bytes:
    """One layer of a field (e.g. |J|, NaN = no copper) as opaque-over-
    copper RGBA PNG bytes. Color scale matches the plugin's log figure
    (global vmax across layers). `bleed` extends the edge color one
    pixel outward at half opacity: the raster mask covers cells whose
    CENTER is inside the copper, so without it the overlay stops half a
    cell short of the outline KiCad draws."""
    import matplotlib
    from PIL import Image
    from scipy import ndimage

    if alpha is None:
        alpha = config.OVERLAY_ALPHA
    if not np.isfinite(data3).any():
        raise ValueError("field is empty - nothing to overlay")
    vmax = float(np.nanmax(data3))
    if vmax <= 0:
        raise ValueError("field is empty - nothing to overlay")
    vmin = vmax / config.CURRENT_DYNAMIC_RANGE
    d = np.clip(data3[li], vmin, vmax)
    if config.LOG_CURRENT_SCALE:
        u = (np.log(d) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    else:
        u = d / vmax
    u = FLOOR + (1.0 - FLOOR) * u
    cmap = matplotlib.colormaps[config.CMAP_CURRENT]
    rgba = (cmap(np.nan_to_num(u)) * 255).astype(np.uint8)
    copper = ~np.isnan(data3[li])
    rgba[..., 3] = np.where(copper, alpha, 0)

    if bleed and copper.any() and not copper.all():
        ring = ndimage.binary_dilation(
            copper, structure=np.ones((3, 3), dtype=bool)) & ~copper
        iy, ix = ndimage.distance_transform_edt(
            ~copper, return_distances=False, return_indices=True)
        rgba[ring, :3] = rgba[iy[ring], ix[ring], :3]
        rgba[ring, 3] = alpha // 2

    buf = io.BytesIO()
    # no dpi metadata: KiCad assumes its 300 PPI default, which the
    # pusher's scale computation relies on
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()
