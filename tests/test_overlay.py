"""In-KiCad overlay rendering (fill_resistance.overlay): copper-shaped
RGBA heatmaps with a visibility floor and a soft edge bleed. The kipy
pushing side is exercised only against a live KiCad (tools/)."""
import io

import numpy as np
import pytest
from PIL import Image

from fill_resistance import config, overlay


def _field(ny=20, nx=30):
    """Two-layer |J| field: copper disc on layer 0, NaN elsewhere."""
    data = np.full((2, ny, nx), np.nan)
    yy, xx = np.mgrid[:ny, :nx]
    disc = (yy - ny / 2) ** 2 + (xx - nx / 2) ** 2 <= 8 ** 2
    data[0][disc] = 1.0 + xx[disc]              # spans the log range
    data[1][disc] = 1e-12                       # below the global log floor
    return data, disc


def test_heatmap_png_shape_and_alpha():
    data, disc = _field()
    img = Image.open(io.BytesIO(overlay.heatmap_png(data, 0, bleed=False)))
    assert img.size == (30, 20)
    rgba = np.asarray(img)
    assert (rgba[..., 3][disc] == config.OVERLAY_ALPHA).all()
    assert (rgba[..., 3][~disc] == 0).all()


def test_heatmap_floor_not_black():
    """The coldest copper must stay distinguishable from a dark canvas:
    the colormap starts FLOOR up, never at its near-black bottom."""
    data, disc = _field()
    rgba = np.asarray(Image.open(io.BytesIO(
        overlay.heatmap_png(data, 1, bleed=False))))   # layer 1: all-cold
    floor = np.array(__import__("matplotlib").colormaps[
        config.CMAP_CURRENT](overlay.FLOOR)[:3]) * 255
    assert np.abs(rgba[..., :3][disc] - floor).max() <= 1
    assert rgba[..., :3][disc].sum(axis=-1).min() > 30   # not near-black


def test_heatmap_bleed_ring():
    """bleed=True: one pixel of half-alpha edge color outside the copper
    (the mask stops half a cell short of the drawn outline)."""
    from scipy import ndimage
    data, disc = _field()
    rgba = np.asarray(Image.open(io.BytesIO(overlay.heatmap_png(data, 0))))
    ring = ndimage.binary_dilation(
        disc, structure=np.ones((3, 3), dtype=bool)) & ~disc
    assert (rgba[..., 3][ring] == config.OVERLAY_ALPHA // 2).all()
    outside = ~disc & ~ring
    assert (rgba[..., 3][outside] == 0).all()
    assert (rgba[..., 3][disc] == config.OVERLAY_ALPHA).all()


def test_heatmap_empty_field():
    with pytest.raises(ValueError):
        overlay.heatmap_png(np.full((1, 4, 4), np.nan), 0)
