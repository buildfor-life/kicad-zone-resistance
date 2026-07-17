"""Route-A experiment: push a bitmap overlay into the open KiCad board as
a locked ReferenceImage on a User layer via the IPC API.

Pushes a fiducial test pattern (corner + center crosshairs, 10 mm grid,
translucent gradient) sized to the board outline so alignment and scale
can be verified by eye in the editor. Re-running replaces the previous
overlay. Requires KiCad >= 10.0.1 (ReferenceImage over the API).

    python tools/kicad_overlay_test.py [--layer Cmts.User] [--remove]
    python tools/kicad_overlay_test.py --image heat.png --bbox x0,y0,x1,y1
                                       (mm; push an arbitrary PNG instead)

The overlay is editor-only: reference images never plot to gerbers.
Delete it any time by selecting it in KiCad (it sits on the chosen
layer) or with --remove. The layer must be enabled in Board Setup:
User.1..User.45 usually are NOT (KiCad refuses the item with 'no
overlapping layers with the board'); Cmts.User/Eco1.User always exist.
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kipy.board_types import ReferenceImage
from kipy.geometry import Vector2
from kipy.util.board_layer import canonical_name, layer_from_canonical_name

from fill_resistance.board_io import (OVERLAY_PIX_NM as PIX_NM,
                                      _create_reference_image, connect,
                                      remove_overlays)

NM = 1_000_000


def board_bbox_nm(board):
    """Union bbox of the Edge.Cuts shapes (fallback: all pads)."""
    items = [s for s in board.get_shapes()
             if canonical_name(s.layer) == "Edge.Cuts"]
    if not items:
        items = list(board.get_pads())
    if not items:
        raise SystemExit("board has no Edge.Cuts shapes and no pads")
    x0 = y0 = None
    x1 = y1 = None
    for it in items:
        box = board.get_item_bounding_box(it)
        if box is None:
            continue
        lo_x, lo_y = box.pos.x, box.pos.y
        hi_x, hi_y = lo_x + box.size.x, lo_y + box.size.y
        x0 = lo_x if x0 is None else min(x0, lo_x)
        y0 = lo_y if y0 is None else min(y0, lo_y)
        x1 = hi_x if x1 is None else max(x1, hi_x)
        y1 = hi_y if y1 is None else max(y1, hi_y)
    return x0, y0, x1, y1


def fiducial_png(w_nm: float, h_nm: float, px_per_mm: float = 16.0):
    """RGBA test pattern: translucent gradient, 10 mm grid, opaque
    crosshairs at the four corners and the center."""
    import numpy as np
    from PIL import Image

    w_px = max(2, round(w_nm / NM * px_per_mm))
    h_px = max(2, round(h_nm / NM * px_per_mm))
    xx = np.linspace(0.0, 1.0, w_px)[None, :]
    yy = np.linspace(0.0, 1.0, h_px)[:, None]
    rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    rgba[..., 0] = (255 * xx).astype(np.uint8)          # red ramp ->
    rgba[..., 2] = (255 * yy).astype(np.uint8)          # blue ramp v
    rgba[..., 1] = 60
    rgba[..., 3] = 70                                   # mostly see-through

    step = round(10.0 * px_per_mm)                      # 10 mm grid
    for x in range(0, w_px, step):
        rgba[:, x:x + 2, :3] = 255
        rgba[:, x:x + 2, 3] = 150
    for y in range(0, h_px, step):
        rgba[y:y + 2, :, :3] = 255
        rgba[y:y + 2, :, 3] = 150

    def cross(cx, cy, arm=round(3 * px_per_mm)):
        x_lo, x_hi = max(0, cx - arm), min(w_px, cx + arm + 1)
        y_lo, y_hi = max(0, cy - arm), min(h_px, cy + arm + 1)
        cy2 = np.clip(cy, 0, h_px - 2)
        cx2 = np.clip(cx, 0, w_px - 2)
        rgba[cy2:cy2 + 2, x_lo:x_hi] = (255, 0, 0, 255)
        rgba[y_lo:y_hi, cx2:cx2 + 2] = (255, 0, 0, 255)

    for cx in (0, w_px - 1):
        for cy in (0, h_px - 1):
            cross(cx, cy)
    cross(w_px // 2, h_px // 2)

    buf = io.BytesIO()
    # no dpi= : without a density chunk KiCad assumes the 300 PPI default
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue(), w_px, h_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", default="Cmts.User",
                    help="destination layer (default Cmts.User; must be "
                         "enabled in Board Setup)")
    ap.add_argument("--remove", action="store_true",
                    help="only remove existing overlays on the layer")
    ap.add_argument("--image", help="push this PNG instead of the pattern")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 [mm] for --image")
    args = ap.parse_args()

    _, board = connect()
    layer = layer_from_canonical_name(args.layer)

    n = remove_overlays(board, layer)
    if n:
        print(f"removed {n} previous overlay(s) on {args.layer}")
    if args.remove:
        return

    if args.image:
        if not args.bbox:
            raise SystemExit("--image needs --bbox x0,y0,x1,y1 [mm]")
        x0, y0, x1, y1 = (float(v) * NM for v in args.bbox.split(","))
        png = Path(args.image).read_bytes()
        from PIL import Image
        w_px, h_px = Image.open(io.BytesIO(png)).size
    else:
        x0, y0, x1, y1 = board_bbox_nm(board)
        png, w_px, h_px = fiducial_png(x1 - x0, y1 - y0)

    scale = (x1 - x0) / (w_px * PIX_NM)

    ref = ReferenceImage()
    ref.layer = layer
    ref.position = Vector2.from_xy(round((x0 + x1) / 2), round((y0 + y1) / 2))
    ref.image_scale = scale
    ref.image_data = png
    ref.locked = False          # unlocked: easy to delete; reruns replace
    _create_reference_image(board, ref)

    got = [r for r in board.get_reference_images() if r.layer == layer]
    print(f"pushed {len(png) / 1024:.0f} kB PNG ({w_px}x{h_px} px) onto "
          f"{args.layer}: {(x1 - x0) / NM:.2f} x {(y1 - y0) / NM:.2f} mm at "
          f"({x0 / NM:.2f}, {y0 / NM:.2f}) mm, scale {scale:.4f}")
    for r in got:
        print(f"readback: {r!r}")
    print(f"-> enable layer '{args.layer}' in the Appearance panel; the "
          f"red crosshairs must sit on the board bbox corners/center and "
          f"the white grid must be 10 mm. Remove with --remove.")


if __name__ == "__main__":
    main()
