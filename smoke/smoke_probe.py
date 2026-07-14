"""M1 smoke probe: verify the IPC API surface against a live KiCad.

Run from the dev venv while KiCad is open with the board loaded:
    .venv\\Scripts\\python.exe smoke\\smoke_probe.py

Deliberately uses kipy directly (not board_io) so it works even if
board_io has a bug. Prints: version, board path, selection contents,
shapes summary, zones + filled polygon stats, stackup table.
"""
import sys
import traceback

from kipy import KiCad
from kipy.board_types import BoardRectangle, BoardShape
from kipy.proto.board.board_pb2 import BoardStackupLayerType
from kipy.proto.board.board_types_pb2 import ZoneType
from kipy.util.board_layer import canonical_name


def mm(nm):
    return nm / 1e6


def main():
    print("== connect ==")
    kicad = KiCad()
    kicad.ping()
    print("version:", kicad.get_version())
    try:
        print("check_version:", kicad.check_version())
    except Exception as e:
        print("check_version raised:", e)

    board = kicad.get_board()
    print("board.name:", board.name)
    print("board_filename:", getattr(board.document, "board_filename", "?"))

    print("\n== selection ==")
    sel = list(board.get_selection())
    print(f"{len(sel)} item(s) selected")
    for item in sel:
        line = f"  {type(item).__name__}"
        if isinstance(item, BoardRectangle):
            tl, br = item.top_left, item.bottom_right
            line += (f"  layer={canonical_name(item.layer)}"
                     f"  tl=({mm(tl.x):.2f}, {mm(tl.y):.2f})mm"
                     f"  br=({mm(br.x):.2f}, {mm(br.y):.2f})mm")
        elif isinstance(item, BoardShape):
            line += f"  layer={canonical_name(item.layer)}"
        print(line)

    print("\n== shapes (board-wide rectangles) ==")
    shapes = list(board.get_shapes())
    rect_shapes = [s for s in shapes if isinstance(s, BoardRectangle)]
    print(f"{len(shapes)} shapes total, {len(rect_shapes)} rectangles")
    for s in rect_shapes[:20]:
        tl, br = s.top_left, s.bottom_right
        print(f"  rect on {canonical_name(s.layer)}: "
              f"({mm(tl.x):.2f}, {mm(tl.y):.2f}) - ({mm(br.x):.2f}, {mm(br.y):.2f}) mm")

    print("\n== zones ==")
    for zone in board.get_zones():
        ztype = ZoneType.Name(zone.type)
        net = zone.net.name if zone.net is not None else "<none>"
        layers = [canonical_name(l) for l in zone.layers]
        print(f"  zone '{zone.name}' type={ztype} net={net} "
              f"layers={layers} filled={zone.filled}")
        try:
            for layer, polys in zone.filled_polygons.items():
                narcs = 0
                nnodes = 0
                nholes = 0
                for p in polys:
                    nnodes += len(p.outline.nodes)
                    nholes += len(p.holes)
                    narcs += sum(1 for n in p.outline.nodes if n.has_arc)
                print(f"    fill on {canonical_name(layer)}: {len(polys)} "
                      f"poly(s), {nnodes} outline nodes, {nholes} holes, "
                      f"{narcs} arc nodes")
        except Exception:
            print("    filled_polygons FAILED:")
            traceback.print_exc()

    print("\n== stackup ==")
    try:
        for sl in board.get_stackup().layers:
            tname = BoardStackupLayerType.Name(sl.type)
            lname = canonical_name(sl.layer) if sl.type == \
                BoardStackupLayerType.BSLT_COPPER else "-"
            print(f"  {tname:18s} layer={lname:8s} thickness={sl.thickness} nm"
                  f"  enabled={sl.enabled}")
    except Exception:
        traceback.print_exc()

    print("\nsmoke probe DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
