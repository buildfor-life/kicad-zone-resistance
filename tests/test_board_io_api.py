"""board_io's kipy-facing paths, against a fake board.

Real protobuf messages, a fake transport. These cover what a live KiCad
would otherwise be needed for: the overlay push (kipy's
Board.remove_items discards the DeleteItemsResponse, so board_io talks
to the proto layer directly and these pin the status handling that
depends on) and per-layer pad copper selection.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from kipy.proto.common.commands.editor_commands_pb2 import (
    CreateItemsResponse, DeleteItemsResponse, ItemDeletionStatus)
from kipy.proto.common.types.base_types_pb2 import KIID
from kipy.util.board_layer import layer_from_canonical_name

from fill_resistance import board_io, config


class _Ref:
    """Stand-in for a reference image already on the board (kipy board
    items carry a KIID message, not a bare id)."""
    def __init__(self, layer_name, ident):
        self.layer = layer_from_canonical_name(layer_name)
        self.id = KIID(value=f"00000000-0000-0000-0000-{ident:012d}")


class _FakeKiCad:
    def __init__(self, delete_status=ItemDeletionStatus.IDS_OK):
        self.delete_status = delete_status
        self.deleted = []          # layers we were asked to clear
        self.created = []          # ReferenceImages we were asked to add

    def send(self, cmd, response_type):
        if response_type is DeleteItemsResponse:
            resp = DeleteItemsResponse()
            for _ in cmd.item_ids:
                resp.deleted_items.add().status = self.delete_status
            self.deleted.append(len(cmd.item_ids))
            return resp
        if response_type is CreateItemsResponse:
            resp = CreateItemsResponse()
            resp.created_items.add().status.code = 1      # ISC_OK
            self.created.append(cmd)
            return resp
        raise AssertionError(f"unexpected command {type(cmd).__name__}")


class _FakeBoard:
    def __init__(self, existing=(), delete_status=ItemDeletionStatus.IDS_OK):
        self._kicad = _FakeKiCad(delete_status)
        self._refs = list(existing)
        self.commits = []
        self.pushed = []
        self.dropped = []

    # kipy Board surface board_io actually uses
    @property
    def _doc(self):
        from kipy.proto.common.types.base_types_pb2 import DocumentSpecifier
        return DocumentSpecifier()

    def get_reference_images(self):
        return list(self._refs)

    def begin_commit(self):
        self.commits.append("open")
        return object()

    def push_commit(self, commit, message=""):
        self.pushed.append(message)

    def drop_commit(self, commit):
        self.dropped.append(commit)


class _Stack:
    layer_names = ["F.Cu", "B.Cu"]
    shape2d = (12, 16)
    h_nm = 100_000
    x0_nm = 0
    y0_nm = 0


class _Result:
    def __init__(self, nlayers=2, ny=12, nx=16):
        self.Jmag = np.full((nlayers, ny, nx), 1e6)


def test_remove_overlays_counts_deleted():
    layer = layer_from_canonical_name("User.9")
    board = _FakeBoard(existing=[_Ref("User.9", 1), _Ref("User.9", 2),
                                 _Ref("User.10", 3)])
    assert board_io.remove_overlays(board, layer) == 2   # not the User.10 one


def test_remove_overlays_no_images_is_a_noop():
    board = _FakeBoard()
    assert board_io.remove_overlays(
        board, layer_from_canonical_name("User.9")) == 0
    assert board._kicad.deleted == []      # no DeleteItems sent at all


def test_locked_overlay_raises_instead_of_stacking():
    """A locked image comes back IDS_IMMUTABLE while the overall request
    still reports OK. Unchecked, the caller would add a second image on
    top of the one it believed it had replaced."""
    board = _FakeBoard(existing=[_Ref("User.9", 1)],
                       delete_status=ItemDeletionStatus.IDS_IMMUTABLE)
    with pytest.raises(RuntimeError, match="could not be removed"):
        board_io.remove_overlays(board, layer_from_canonical_name("User.9"))


def test_already_gone_overlay_is_not_an_error():
    board = _FakeBoard(existing=[_Ref("User.9", 1)],
                       delete_status=ItemDeletionStatus.IDS_NONEXISTENT)
    assert board_io.remove_overlays(
        board, layer_from_canonical_name("User.9")) == 1


def test_push_clears_slots_this_run_does_not_write(monkeypatch):
    """A 2-layer run after a 4-layer run must not leave the previous
    solve's heatmap sitting on User.11/User.12."""
    stale = [_Ref(n, i) for i, n in enumerate(config.OVERLAY_LAYERS)]
    board = _FakeBoard(existing=stale)
    board_io.push_result_overlays(board, _Stack(), _Result())

    written = {c.items[0].type_url for c in board._kicad.created}
    assert len(board._kicad.created) == 2          # F.Cu, B.Cu -> 2 slots
    assert written                                  # images really created
    # 2 written slots cleared + 2 unwritten slots cleared = 4 delete calls
    assert len(board._kicad.deleted) == 4


def test_push_is_one_undo_step():
    board = _FakeBoard()
    board_io.push_result_overlays(board, _Stack(), _Result())
    assert board.commits and board.pushed and not board.dropped


def _square(side):
    """Minimal duck-typed PolygonWithHoles: an origin square."""
    pts = [(0, 0), (side, 0), (side, side), (0, side)]
    return NS(outline=NS(nodes=[NS(has_point=True, has_arc=False,
                                   point=NS(x=x, y=y)) for x, y in pts]),
              holes=[])


class _PadBoard:
    """F.Cu carries a small pad, B.Cu a deliberately larger one - KiCad
    allows a different pad size per copper layer."""
    def __init__(self):
        self.f = layer_from_canonical_name("F.Cu")
        self.b = layer_from_canonical_name("B.Cu")
        self.asked = []

    def get_pad_shapes_as_polygons(self, pad, layer):
        self.asked.append(layer)
        return {self.f: _square(1000), self.b: _square(5000)}.get(layer)


def _width(polys):
    xs = [p[0] for p in polys[0].outline]
    return max(xs) - min(xs)


def test_tht_pad_copper_comes_from_the_solder_side():
    """The solder coat is sized from this shape, so a B.Cu-protruding
    joint must not be measured with F.Cu's (here smaller) pad."""
    board = _PadBoard()
    polys = board_io._pad_polygons(board, pad=None, contact="all",
                                   prefer="B.Cu")
    assert _width(polys) == 5000
    assert board.asked[0] == board.b          # probed before the F.Cu default


def test_pad_copper_falls_back_when_no_side_is_known():
    board = _PadBoard()
    polys = board_io._pad_polygons(board, pad=None, contact="all")
    assert _width(polys) == 1000              # F.Cu, the documented fallback


def test_explicit_contact_layer_still_wins():
    board = _PadBoard()
    polys = board_io._pad_polygons(board, pad=None, contact="B.Cu",
                                   prefer="F.Cu")
    assert _width(polys) == 5000


def test_push_drops_the_commit_if_it_cannot_finish(monkeypatch):
    board = _FakeBoard()
    monkeypatch.setattr(board_io.config, "OVERLAY_LAYERS", ("User.9",))

    def boom(*a, **k):
        raise RuntimeError("transport died")
    monkeypatch.setattr(board, "push_commit", boom)

    with pytest.raises(RuntimeError):
        board_io.push_result_overlays(board, _Stack(), _Result())
    assert board.dropped
