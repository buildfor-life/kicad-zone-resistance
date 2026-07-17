"""Standalone runner for the EXPERIMENTAL in-KiCad result overlays
(also available as a dialog checkbox in the plugin): solve the open
board headlessly and push per-layer |J| heatmaps as unlocked
ReferenceImages, transparent outside copper.

    python tools/kicad_heatmap_overlay.py --net VOUT+ --amps 45
        -> all included copper layers onto config.OVERLAY_LAYERS
           (User.9..User.12, stackup order, top first)
    python tools/kicad_heatmap_overlay.py --net X --source B.Cu --dest Eco1.User
        -> a single layer wherever you want

Needs KiCad >= 10.0.1 with the board open, electrode markers or a
selection as in a normal plugin run, and the destination layers enabled
in Board Setup. Re-running replaces the previous overlays. Remove with
tools/kicad_overlay_test.py --remove --layer <dest>.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kipy.board_types import ReferenceImage
from kipy.geometry import Vector2
from kipy.util.board_layer import layer_from_canonical_name

from fill_resistance import config, raster, solver
from fill_resistance import board_io as bio
from fill_resistance.overlay import heatmap_png


def extract_problem(board, net_arg=None):
    """Same flow as `python -m fill_resistance.board_io` (dump path)."""
    # a clicked overlay must not switch the electrode scan into
    # selection mode - reference images can never be contacts
    sel = list(board.get_selection())
    if sel and all(isinstance(s, ReferenceImage) for s in sel):
        board.clear_selection()
    stackup = bio.get_stackup_info(board)
    es1, es2, net_hint = bio.get_electrodes(board, stackup)
    if bio.any_zone_unfilled(board):
        bio.refill(board)
    fills = bio.gather_net_fills(board)
    tracks = bio.gather_net_tracks(board) if config.INCLUDE_TRACKS else {}
    copper = bio.merge_copper(fills, bio.tracks_as_polygons(tracks))
    nets = bio.nets_overlapping(copper, es1, es2)
    if net_arg:
        net = net_arg
    elif net_hint in nets:
        net = net_hint
    elif len(nets) == 1:
        net = nets[0]
    else:
        raise SystemExit(f"candidate nets: {nets}; pass one with --net")
    # marker rectangles may exist for SEVERAL nets (board-wide scan):
    # keep only the parts overlapping the chosen net's copper
    per_layer = copper.get(net, {})
    def on_net(e):
        return any(bio._rect_overlaps(e.rect, polys)
                   for polys in per_layer.values())
    es1, es2 = [e for e in es1 if on_net(e)], [e for e in es2 if on_net(e)]
    if not es1 or not es2:
        raise SystemExit(f"no V+/V- marker overlaps {net} copper")
    print(f"{len(es1)} V+ / {len(es2)} V- marker(s) on {net}")
    return bio.build_problem(board, net, list(per_layer), es1, es2,
                             stackup, fills, tracks=tracks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="single copper layer to overlay (default: ALL "
                         "included layers onto config.OVERLAY_LAYERS)")
    ap.add_argument("--dest", default=None,
                    help="destination layer for --source (default User.9; "
                         "must be enabled in Board Setup)")
    ap.add_argument("--net", default=None, help="net name (default: auto)")
    ap.add_argument("--amps", type=float, default=None,
                    help="test current [A] (default: config)")
    ap.add_argument("--lock", action="store_true",
                    help="lock the overlays (default unlocked: easier to "
                         "delete; reruns replace them either way)")
    ap.add_argument("--alpha", type=int, default=None,
                    help="overlay opacity over copper, 0-255 (default "
                         "config.OVERLAY_ALPHA)")
    args = ap.parse_args()
    if args.alpha is not None:
        config.OVERLAY_ALPHA = args.alpha

    _, board = bio.connect()
    problem = extract_problem(board, args.net)

    h = raster.choose_cell_size(problem.copper_bbox(), len(problem.layers))
    print(f"rasterizing at {h / 1000:.1f} um ...")
    stack = raster.rasterize_stack(problem, h)
    # board-wide marker scan: drop parts that land on no copper of THIS
    # net (markers belonging to other nets' analyses)
    for name in ("electrodes1", "electrodes2"):
        parts = getattr(problem, name)
        keep = [e for e in parts
                if raster._part_mask3d(stack, problem, e).any()]
        if len(keep) != len(parts):
            print(f"ignoring {len(parts) - len(keep)} marker(s) off-net "
                  f"({name[-1] == '1' and 'V+' or 'V-'})")
        if not keep:
            raise SystemExit(f"no {name} marker lands on this net's copper")
        setattr(problem, name, keep)
    e1, e2 = raster.electrode_masks(stack, problem)
    i_test = args.amps if args.amps is not None else config.TEST_CURRENT_A
    print(f"solving @ {i_test:g} A DC ...")
    result = solver.run_solve(problem, stack, e1, e2, i_test)
    print(f"R = {result.R_ohm * 1e3:.4f} mOhm, P = {result.P_total:.3f} W "
          f"@ {i_test:g} A")

    if args.source is None:
        bio.push_result_overlays(board, stack, result, lock=args.lock)
        return

    names = stack.layer_names
    if args.source not in names:
        raise SystemExit(f"layer {args.source} not in solve ({names})")
    png = heatmap_png(result.Jmag * 1e-6, names.index(args.source))
    ny, nx = stack.shape2d
    w_nm, h_nm = nx * stack.h_nm, ny * stack.h_nm
    dest_name = args.dest or "User.9"
    dest = layer_from_canonical_name(dest_name)
    n = bio.remove_overlays(board, dest)
    ref = ReferenceImage()
    ref.layer = dest
    ref.position = Vector2.from_xy(round(stack.x0_nm + w_nm / 2),
                                   round(stack.y0_nm + h_nm / 2))
    ref.image_scale = w_nm / (nx * bio.OVERLAY_PIX_NM)
    ref.image_data = png
    ref.locked = args.lock
    bio._create_reference_image(board, ref)
    print(f"{args.source} -> {dest_name} ({nx}x{ny} px, "
          f"{len(png) / 1024:.0f} kB" + (f", replaced {n}" if n else "") + ")")


if __name__ == "__main__":
    main()
