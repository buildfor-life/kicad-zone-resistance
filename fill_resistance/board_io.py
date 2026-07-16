"""All KiCad IPC access. This is the ONLY module that imports kipy;
everything downstream works on plain geometry dataclasses.

Run `python -m fill_resistance.board_io dump.json [net]` against a live
KiCad to extract without the dialog (all layers of the net, defaults).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kipy import KiCad
from kipy.board import Board
from kipy.board_types import ArcTrack, BoardRectangle, Pad, Via
from kipy.proto.board.board_pb2 import BoardStackupLayerType
from kipy.proto.board.board_types_pb2 import ZoneType
from kipy.util.board_layer import (canonical_name, is_copper_layer,
                                   layer_from_canonical_name)

import numpy as np

from . import config
from .errors import ApiVersionError, CandidateError, SelectionError
from .geometry import (Electrode, LayerFill, Polygon, Problem, Rect,
                       SurfaceBuildup, TrackSeg, ViaLink,
                       contact_solder_buildups, linearize_ring)

MASK_TO_COPPER = {"F.Mask": "F.Cu", "B.Mask": "B.Cu"}

# zone fills are polygonal in practice; tolerance only guards arc nodes
ARC_TOL_NM = 10_000


def connect() -> tuple[KiCad, Board]:
    try:
        kicad = KiCad()
        kicad.ping()
    except Exception as e:
        raise ApiVersionError(
            f"Could not connect to KiCad's IPC API: {e}\n"
            f"Is KiCad running with the API server enabled "
            f"(Preferences > Plugins > Enable KiCad API)?"
        )
    try:
        print(f"connected to KiCad {kicad.get_version()}")
    except Exception:
        pass
    try:
        board = kicad.get_board()
    except Exception as e:
        raise SelectionError(
            f"Could not get the open board from KiCad: {e}\n"
            f"Open the PCB in the board editor and run again."
        )
    return kicad, board


def board_dir(board: Board) -> Path:
    # document.board_filename is a bare file name (no directory) in
    # KiCad 10.0.1; the project path is the reliable location
    try:
        path = board.get_project().path
        if path and Path(path).is_dir():
            return Path(path)
    except Exception:
        pass
    try:
        filename = getattr(board.document, "board_filename", "") or ""
        if Path(filename).is_absolute():
            return Path(filename).parent
    except Exception:
        pass
    return Path.cwd()


# --- stackup geometry --------------------------------------------------------

@dataclass
class StackupInfo:
    names: list[str]                      # copper layers, top to bottom
    thickness_nm: dict[str, int]
    z_nm: dict[str, int]                  # copper center depth
    z_bot_nm: int                         # total stack thickness


def get_stackup_info(board: Board) -> StackupInfo:
    names: list[str] = []
    thickness: dict[str, int] = {}
    z_center: dict[str, int] = {}
    z = 0
    for sl in board.get_stackup().layers:
        t = int(sl.thickness or 0)
        if sl.type == BoardStackupLayerType.BSLT_COPPER:
            name = canonical_name(sl.layer)
            if t <= 0:
                t = int(config.FALLBACK_THICKNESS_UM * 1000)
                print(f"warning: stackup gives no thickness for {name}; "
                      f"assuming {config.FALLBACK_THICKNESS_UM} um")
            names.append(name)
            thickness[name] = t
            z_center[name] = z + t // 2
        z += t
    if not names:
        raise CandidateError(
            "Could not read any copper layer from the board stackup."
        )
    return StackupInfo(names=names, thickness_nm=thickness, z_nm=z_center,
                       z_bot_nm=z)


# --- electrodes from selection ----------------------------------------------

def _box2_to_rect(box, layer_name: str) -> Rect:
    try:
        pos, size = box.pos, box.size
        return Rect.normalized(pos.x, pos.y, pos.x + size.x, pos.y + size.y,
                               layer_name)
    except AttributeError:
        c, s = box.center, box.size
        return Rect.normalized(c.x - s.x // 2, c.y - s.y // 2,
                               c.x + s.x // 2, c.y + s.y // 2, layer_name)


def _convert_poly(poly_with_holes) -> Polygon:
    def ring(polyline):
        nodes = []
        for node in polyline.nodes:
            if node.has_point:
                nodes.append(("pt", (node.point.x, node.point.y)))
            elif node.has_arc:
                arc = node.arc
                nodes.append(("arc", ((arc.start.x, arc.start.y),
                                      (arc.mid.x, arc.mid.y),
                                      (arc.end.x, arc.end.y))))
        return linearize_ring(nodes, ARC_TOL_NM)

    return Polygon(outline=ring(poly_with_holes.outline),
                   holes=[ring(h) for h in poly_with_holes.holes])


def _pad_drill_nm(pad_or_via) -> int:
    try:
        return int(pad_or_via.padstack.drill.diameter.x)
    except Exception:
        return 0


def _pad_default_contact(pad: Pad) -> str:
    if _pad_drill_nm(pad) > 0:
        return "all"                       # through-hole: contacts the stack
    try:
        copper = [canonical_name(l) for l in pad.padstack.layers
                  if is_copper_layer(l)]
        if len(copper) == 1:
            return copper[0]               # SMD: its own layer
    except Exception:
        pass
    return "all"


def _pad_polygons(board: Board, pad: Pad, contact: str) -> list[Polygon] | None:
    layer_ids = []
    if contact != "all":
        try:
            layer_ids.append(layer_from_canonical_name(contact))
        except Exception:
            pass
    for name in ("F.Cu", "B.Cu"):
        try:
            layer_ids.append(layer_from_canonical_name(name))
        except Exception:
            pass
    for lid in layer_ids:
        try:
            shape = board.get_pad_shapes_as_polygons(pad, layer=lid)
            if shape is not None:
                return [_convert_poly(shape)]
        except Exception:
            continue
    return None


def _tht_protrusion_side(pad: Pad, footprints) -> str:
    """Outer layer where the clipped THT lead protrudes (tent + solder
    cone): the side OPPOSITE the component. Footprint pads are stored
    with absolute positions, so the owning footprint is matched by pad
    number + position. Unknown owner -> assume the component sits on
    F.Cu (lead tents on B.Cu)."""
    try:
        for fp in footprints or []:
            for fpad in fp.definition.pads:
                if fpad.number == pad.number \
                        and fpad.position.x == pad.position.x \
                        and fpad.position.y == pad.position.y:
                    side = canonical_name(fp.layer)
                    return "F.Cu" if side == "B.Cu" else "B.Cu"
    except Exception:
        pass
    print(f"note: no footprint found for pad {pad.number} - assuming its "
          f"lead protrudes on B.Cu")
    return "B.Cu"


def _to_electrode(board: Board, item, stackup: StackupInfo | None = None,
                  footprints=None) -> Electrode:
    if isinstance(item, BoardRectangle):
        tl, br = item.top_left, item.bottom_right
        rect = Rect.normalized(tl.x, tl.y, br.x, br.y,
                               canonical_name(item.layer))
        cx = (rect.x0 + rect.x1) / 2e6
        cy = (rect.y0 + rect.y1) / 2e6
        return Electrode(rect=rect, contact="all",
                         label=f"rect({cx:.1f},{cy:.1f})")
    if isinstance(item, Via):
        via: Via = item
        x, y = via.position.x, via.position.y
        drill = int(via.drill_diameter or 0) or _pad_drill_nm(via)
        if drill <= 0:
            raise SelectionError(
                f"Selected via at ({x / 1e6:.2f}, {y / 1e6:.2f}) mm has no "
                f"drill diameter - cannot use it as a contact.")
        pad_nm = _padstack_pad_nm(via)
        r = max(pad_nm, drill) // 2
        rect = Rect.normalized(x - r, y - r, x + r, y + r, "via")
        return Electrode(
            rect=rect, contact="all", label=f"via({x / 1e6:.1f},{y / 1e6:.1f})",
            drill_nm=drill, pad_nm=pad_nm, center=(x, y),
            barrel_z=(_padstack_span(via.padstack, stackup)
                      if stackup is not None else None))
    # Pad
    pad: Pad = item
    contact = _pad_default_contact(pad)
    net = pad.net.name if pad.net is not None else "?"
    label = f"pad {pad.number}@{net}"
    box = board.get_item_bounding_box(pad)
    if box is None:
        raise SelectionError(f"Could not get the bounding box of {label}.")
    rect = _box2_to_rect(box, "pad")
    drill = _pad_drill_nm(pad)
    return Electrode(rect=rect, contact=contact,
                     polygons=_pad_polygons(board, pad, contact), label=label,
                     # through-hole pad: current enters at the soldered
                     # barrel; the joint is solder-filled + pad-coated,
                     # with a solder cone around the protruding lead
                     drill_nm=drill, pad_nm=_padstack_pad_nm(pad),
                     center=(pad.position.x, pad.position.y),
                     solder=drill > 0,
                     protrusion_side=(_tht_protrusion_side(pad, footprints)
                                      if drill > 0 else None))


def _net_hint_of(items: list) -> str | None:
    for item in items:
        if item.net is not None:
            return item.net.name
    return None


def get_electrodes(board: Board, stackup: StackupInfo | None = None
                   ) -> tuple[list[Electrode], list[Electrode], str | None]:
    """Terminals from the selection. Each terminal may have MULTIPLE parts
    (all merged into one externally-bonded contact):

    - rectangles on ELECTRODE_POS_LAYER -> V+ parts, on ELECTRODE_NEG_LAYER
      -> V- parts; selected pads/vias fill a side that has no rectangles;
    - no marker rectangles selected: legacy mode, exactly 2 items
      (rects/pads/vias, any layer) -> one part each;
    - empty selection: board-wide scan of both marker layers.

    Selected vias and through-hole pads become BARREL contacts: current
    enters at the drill-wall ring (the soldered lead/wire), not the pad
    face. Draw a marker rectangle over the pad instead to model a probe
    pressed onto the pad face.
    """
    pos_l = config.ELECTRODE_POS_LAYER
    neg_l = config.ELECTRODE_NEG_LAYER
    scheme = (f"Draw V+ rectangle(s) on {pos_l} and V- rectangle(s) on "
              f"{neg_l} (axis-aligned), and/or select pads/vias for a side "
              f"without rectangles.")

    selection = list(board.get_selection())
    rects = [s for s in selection if isinstance(s, BoardRectangle)]
    pads = [s for s in selection if isinstance(s, (Pad, Via))]
    # protrusion-side lookup needs the owning footprints (THT pads only)
    footprints = (board.get_footprints()
                  if any(isinstance(s, Pad) and _pad_drill_nm(s) > 0
                         for s in pads) else None)

    if not selection:
        allr = [s for s in board.get_shapes() if isinstance(s, BoardRectangle)]
        pos = [r for r in allr if canonical_name(r.layer) == pos_l]
        neg = [r for r in allr if canonical_name(r.layer) == neg_l]
        if pos and neg:
            print(f"selection empty - using {len(pos)} rectangle(s) on "
                  f"{pos_l} as V+ and {len(neg)} on {neg_l} as V-")
            return ([_to_electrode(board, r) for r in pos],
                    [_to_electrode(board, r) for r in neg], None)
        raise SelectionError(
            f"Nothing selected, and the board-wide scan found "
            f"{len(pos)} rectangle(s) on {pos_l} / {len(neg)} on {neg_l} "
            f"(need at least one on each).\n{scheme}"
        )

    pos = [r for r in rects if canonical_name(r.layer) == pos_l]
    neg = [r for r in rects if canonical_name(r.layer) == neg_l]
    other = [r for r in rects if canonical_name(r.layer) not in (pos_l, neg_l)]

    if pos or neg:
        if other:
            raise SelectionError(
                f"{len(other)} selected rectangle(s) are on neither marker "
                f"layer ({pos_l} = V+, {neg_l} = V-). {scheme}"
            )
        es1 = [_to_electrode(board, r) for r in pos]
        es2 = [_to_electrode(board, r) for r in neg]
        if pads and es1 and es2:
            raise SelectionError(
                f"Cannot assign the {len(pads)} selected pad(s)/via(s): both "
                f"marker layers already provide rectangles. Use pads/vias "
                f"only for a side that has none."
            )
        if pads:
            pad_parts = [_to_electrode(board, p, stackup, footprints)
                         for p in pads]
            if not es1:
                es1 = pad_parts
            else:
                es2 = pad_parts
        if es1 and es2:
            return es1, es2, _net_hint_of(pads)
        raise SelectionError(
            f"Only one terminal defined: V+ has {len(es1)} and V- has "
            f"{len(es2)} contact(s). {scheme}"
        )

    items = rects + pads
    if len(items) == 2:
        return ([_to_electrode(board, items[0], stackup, footprints)],
                [_to_electrode(board, items[1], stackup, footprints)],
                _net_hint_of(pads))
    raise SelectionError(
        f"The selection has {len(rects)} rectangle(s) (none on the marker "
        f"layers) and {len(pads)} pad(s)/via(s); without marker layers "
        f"exactly 2 contacts are needed.\n{scheme}"
    )


# --- fills -------------------------------------------------------------------

def gather_net_fills(board: Board) -> dict[str, dict[str, list[Polygon]]]:
    """net -> layer_name -> merged fill polygons (non-empty only)."""
    fills: dict[str, dict[str, list[Polygon]]] = {}
    for zone in board.get_zones():
        # teardrop fills are conducting copper too, but KiCad types them
        # ZT_TEARDROP instead of ZT_COPPER
        if zone.type not in (ZoneType.ZT_COPPER, ZoneType.ZT_TEARDROP):
            continue
        net = zone.net.name if zone.net is not None else "<no net>"
        for layer, polys in zone.filled_polygons.items():
            if not is_copper_layer(layer) or not polys:
                continue
            fills.setdefault(net, {}).setdefault(
                canonical_name(layer), []).extend(
                _convert_poly(p) for p in polys)
    return fills


def gather_net_tracks(board: Board) -> dict[str, dict[str, list[TrackSeg]]]:
    """net -> layer -> TrackSeg (centerline + width). Traces conduct
    together with the zone fills; the raster decides per run whether a
    trace is rasterized from its outline or becomes a 1D chain."""
    out: dict[str, dict[str, list[TrackSeg]]] = {}
    for t in board.get_tracks():
        if not is_copper_layer(t.layer):
            continue
        width = int(t.width or 0)
        if width <= 0:
            continue
        if isinstance(t, ArcTrack):
            pts = np.array([[t.start.x, t.start.y], [t.mid.x, t.mid.y],
                            [t.end.x, t.end.y]], dtype=np.int64)
        else:
            pts = np.array([[t.start.x, t.start.y], [t.end.x, t.end.y]],
                           dtype=np.int64)
        net = t.net.name if t.net is not None else "<no net>"
        layer = canonical_name(t.layer)
        out.setdefault(net, {}).setdefault(layer, []).append(
            TrackSeg(layer_name=layer, points=pts, width_nm=width))
    return out


def tracks_as_polygons(tracks: dict) -> dict:
    """net -> layer -> outline polygons of the tracks (for the bbox-based
    candidate detection; the Problem keeps the TrackSegs themselves)."""
    return {
        net: {layer: [Polygon(outline=seg.outline(ARC_TOL_NM))
                      for seg in segs]
              for layer, segs in per_layer.items()}
        for net, per_layer in tracks.items()
    }


def merge_copper(fills: dict, tracks: dict) -> dict:
    """net -> layer -> fill + track polygons, for candidate detection
    and the dialog's layer lists (build_problem merges the same way)."""
    out: dict[str, dict[str, list[Polygon]]] = {}
    for src in (fills, tracks):
        for net, per_layer in src.items():
            for layer, polys in per_layer.items():
                out.setdefault(net, {}).setdefault(layer, []).extend(polys)
    return out


def _rect_overlaps(rect: Rect, polygons: list[Polygon]) -> bool:
    for p in polygons:
        px0, py0 = p.outline.min(axis=0)
        px1, py1 = p.outline.max(axis=0)
        if rect.x0 <= px1 and rect.x1 >= px0 and rect.y0 <= py1 and rect.y1 >= py0:
            return True
    return False


def nets_overlapping(fills: dict, es1: list[Electrode],
                     es2: list[Electrode]) -> list[str]:
    """Nets whose fills overlap both terminals (any part, any layer each -
    the connection may go through vias). Permissive bbox prefilter."""
    out = []
    for net, per_layer in fills.items():
        hit1 = any(_rect_overlaps(e.rect, polys) for e in es1
                   for polys in per_layer.values())
        hit2 = any(_rect_overlaps(e.rect, polys) for e in es2
                   for polys in per_layer.values())
        if hit1 and hit2:
            out.append(net)
    return sorted(out)


def gather_mask_buildups(board: Board) -> dict[str, list[Polygon]]:
    """Zones on F.Mask/B.Mask (mask openings) -> fill polygons keyed by
    the outer copper layer they expose."""
    out: dict[str, list[Polygon]] = {}
    for zone in board.get_zones():
        try:
            filled = zone.filled_polygons
        except Exception:
            continue
        for layer, polys in filled.items():
            copper = MASK_TO_COPPER.get(canonical_name(layer))
            if copper and polys:
                out.setdefault(copper, []).extend(
                    _convert_poly(p) for p in polys)
    return out


def any_zone_unfilled(board: Board) -> bool:
    return any(z.type in (ZoneType.ZT_COPPER, ZoneType.ZT_TEARDROP)
               and not z.filled for z in board.get_zones())


def refill(board: Board) -> None:
    print("refilling zones - this modifies the open document ...")
    board.refill_zones(block=True)


# --- barrels -----------------------------------------------------------------

def _padstack_pad_nm(item) -> int:
    """Largest copper pad diameter of a via/pad padstack; 0 if unknown.
    Used to bound the barrel-to-fill connection search in the solver."""
    try:
        sizes = [max(int(l.size.x), int(l.size.y))
                 for l in item.padstack.copper_layers]
        return max(sizes) if sizes else 0
    except Exception:
        return 0


def _padstack_span(padstack, stackup: StackupInfo) -> tuple[int, int]:
    """(z_top, z_bot) of the barrel; falls back to the full stack."""
    try:
        copper = [canonical_name(l) for l in padstack.layers
                  if is_copper_layer(l)]
        zs = [stackup.z_nm[c] for c in copper if c in stackup.z_nm]
        if len(zs) >= 2:
            return min(zs) - 1, max(zs) + 1
    except Exception:
        pass
    return -1, stackup.z_bot_nm + 1


def gather_barrels(board: Board, net_name: str,
                   stackup: StackupInfo) -> list[ViaLink]:
    barrels = []
    for via in board.get_vias():
        if via.net is None or via.net.name != net_name:
            continue
        drill = int(via.drill_diameter or 0) or _pad_drill_nm(via)
        if drill <= 0:
            continue
        z_top, z_bot = _padstack_span(via.padstack, stackup)
        barrels.append(ViaLink(x=via.position.x, y=via.position.y,
                               drill_nm=drill, z_top_nm=z_top,
                               z_bot_nm=z_bot, kind="via",
                               pad_nm=_padstack_pad_nm(via)))
    if config.INCLUDE_TH_PADS:
        for pad in board.get_pads():
            if pad.net is None or pad.net.name != net_name:
                continue
            drill = _pad_drill_nm(pad)
            if drill <= 0:
                continue
            barrels.append(ViaLink(x=pad.position.x, y=pad.position.y,
                                   drill_nm=drill, z_top_nm=-1,
                                   z_bot_nm=stackup.z_bot_nm + 1, kind="pad",
                                   pad_nm=_padstack_pad_nm(pad)))
    return barrels


# --- top level ----------------------------------------------------------------

def build_problem(board: Board, net: str, layer_names: list[str],
                  es1: list[Electrode], es2: list[Electrode],
                  stackup: StackupInfo, fills: dict,
                  buildups: dict[str, list[Polygon]] | None = None,
                  extra_cu_um: float | None = None,
                  tracks: dict | None = None,
                  vias_capped: bool | None = None,
                  cap_max_drill_mm: float | None = None) -> Problem:
    per_layer = fills.get(net, {})
    per_layer_tracks = (tracks or {}).get(net, {})
    layers = []
    segs: list[TrackSeg] = []
    for name in stackup.names:                 # keep stackup order
        if name not in layer_names:
            continue
        polys = list(per_layer.get(name, []))
        layer_segs = per_layer_tracks.get(name, [])
        if not polys and not layer_segs:
            print(f"note: net {net} has no copper on {name} - layer skipped")
            continue
        if config.COPPER_THICKNESS_UM is not None:
            t = int(config.COPPER_THICKNESS_UM * 1000)
        else:
            t = stackup.thickness_nm[name]
        layers.append(LayerFill(layer_name=name, thickness_nm=t,
                                z_nm=stackup.z_nm[name], polygons=polys))
        segs.extend(layer_segs)
    if not layers:
        raise CandidateError(
            f"Net {net} has no fill on any of the selected layers "
            f"({', '.join(layer_names)})."
        )
    vias = gather_barrels(board, net, stackup) if len(layers) > 1 else []
    included = {l.layer_name for l in layers}
    buildup_list = [
        SurfaceBuildup(layer_name=name, polygons=polys)
        for name, polys in (buildups or {}).items() if name in included
    ]
    print(f"net {net}: {len(layers)} layer(s) "
          f"({', '.join(l.layer_name for l in layers)}), "
          f"{len(segs)} track(s), {len(vias)} via/pad barrel(s)"
          + (f", solder buildup on "
             f"{', '.join(b.layer_name for b in buildup_list)}"
             if buildup_list else ""))
    problem = Problem(
        board_path=board.name or "",
        net_name=net,
        rho_ohm_m=config.RHO_CU_OHM_M,
        plating_nm=int(config.VIA_PLATING_UM * 1000),
        layers=layers,
        vias=vias,
        electrodes1=es1,
        electrodes2=es2,
        thickness_source=("override" if config.COPPER_THICKNESS_UM is not None
                          else "stackup"),
        buildups=buildup_list,
        solder_thickness_nm=int(config.SOLDER_THICKNESS_UM * 1000),
        solder_rho_ohm_m=config.SOLDER_RHO_OHM_M,
        extra_cu_nm=int((extra_cu_um if extra_cu_um is not None
                         else config.BUILDUP_EXTRA_CU_UM) * 1000),
        tracks=segs,
        vias_capped=(vias_capped if vias_capped is not None
                     else config.VIAS_CAPPED),
        cap_plating_nm=int(config.CAP_PLATING_UM * 1000),
        cap_max_drill_nm=int((cap_max_drill_mm if cap_max_drill_mm is not None
                              else config.CAP_MAX_DRILL_MM) * 1e6),
        tht_protrusion_nm=int(config.THT_LEAD_PROTRUSION_MM * 1e6),
    )
    solder_layers = contact_solder_buildups(problem)
    if solder_layers:
        sides = sorted({e.protrusion_side
                        for e in problem.electrodes1 + problem.electrodes2
                        if e.solder and e.protrusion_side})
        cone = (f", {config.THT_LEAD_PROTRUSION_MM:g} mm lead + solder cone "
                f"on {', '.join(sides)}"
                if sides and problem.tht_protrusion_nm > 0 else "")
        print(f"THT contact(s): solder-filled hole + "
              f"{config.SOLDER_THICKNESS_UM:g} um average solder coat on the "
              f"pad face ({', '.join(solder_layers)}){cone}")
    return problem


if __name__ == "__main__":
    import sys

    from .geometry import save_problem

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("geometry_dump.json")
    _, board = connect()
    stackup = get_stackup_info(board)
    es1, es2, net_hint = get_electrodes(board, stackup)
    if any_zone_unfilled(board):
        refill(board)
    fills = gather_net_fills(board)
    tracks = gather_net_tracks(board) if config.INCLUDE_TRACKS else {}
    copper = merge_copper(fills, tracks_as_polygons(tracks))
    nets = nets_overlapping(copper, es1, es2)
    if len(sys.argv) > 2:
        net = sys.argv[2]
    elif net_hint in nets:
        net = net_hint
    elif len(nets) == 1:
        net = nets[0]
    else:
        print(f"candidate nets: {nets}; pass one as second argument")
        sys.exit(1)
    problem = build_problem(board, net, list(copper.get(net, {})), es1, es2,
                            stackup, fills, tracks=tracks)
    save_problem(problem, out)
    print(f"wrote {out}")
