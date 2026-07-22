"""Top-level orchestration for the KiCad-launched action.

Flow: connect -> read the two selected contacts (rectangles/pads) ->
gather fills -> selection dialog (net, layers, contacts, current, cell)
-> extract vias -> solve -> figures + report.

Every failure is reported twice: on stdout (lands in the KiCad status-bar
warning list) and as a matplotlib error figure, so it cannot be missed.
"""
from __future__ import annotations

import sys
import traceback

from . import config, pipeline, report
from .errors import CandidateError, UserFacingError


def _fail(message: str, outdir) -> None:
    print(f"ERROR: {message}")
    try:
        if outdir is None:
            # A failure before the run has an output directory (a broken
            # plugin environment throws on import) would otherwise save
            # no PNG - and with no GUI toolkit, plots falls back to
            # opening the saved PNGs, so the figure would never be shown
            # either. Exactly the case the docstring promises to cover.
            import tempfile
            from pathlib import Path
            outdir = Path(tempfile.gettempdir()) / "fill-resistance-error"
        from . import plots
        fig = plots.fig_error(message)
        plots.save_and_show([(fig, "error")], outdir)
    except Exception:                     # reporting must not mask the fault
        traceback.print_exc()
    sys.exit(1)


def main() -> None:
    outdir = None
    try:
        from kipy.errors import ApiError

        from . import board_io, dialog
        try:
            kicad, board = board_io.connect()
            stackup = board_io.get_stackup_info(board)
            es1, es2, net_hint = board_io.get_electrodes(board, stackup)
            if board_io.any_zone_unfilled(board) or config.ALWAYS_REFILL:
                board_io.refill(board)
            fills = board_io.gather_net_fills(board)
            tracks = board_io.gather_net_tracks(board)
            copper = board_io.merge_copper(
                fills, board_io.tracks_as_polygons(tracks))
            candidate_nets = board_io.nets_overlapping(copper, es1, es2)
            buildups = board_io.gather_mask_buildups(board)
        except ApiError as e:
            raise UserFacingError(
                f"KiCad API error: {e}\nIf KiCad is showing a dialog, close "
                f"it and run again."
            )

        if not candidate_nets:
            raise CandidateError(
                "No copper (zone fill or trace) overlaps both contacts. "
                "Check that both sit over copper of the same net and that "
                "the fills are up to date (press B in the board editor)."
            )

        def group_label(parts):
            names = [p.label for p in parts[:3]]
            more = f" +{len(parts) - 3}" if len(parts) > 3 else ""
            return f"{len(parts)}× " + ", ".join(names) + more

        def group_contact(parts):
            contacts = {p.contact for p in parts}
            return contacts.pop() if len(contacts) == 1 else "auto"

        default_net = (net_hint if net_hint in candidate_nets
                       else candidate_nets[0])
        selection = dialog.ask(
            candidates={n: list(copper[n].keys()) for n in candidate_nets},
            layer_order=stackup.names,
            default_net=default_net,
            e1_label=group_label(es1), e2_label=group_label(es2),
            contact1=group_contact(es1), contact2=group_contact(es2),
            buildup_layers=sorted(buildups.keys()),
        )
        if selection is None:
            print("cancelled")
            return

        if selection.contact1 != "auto":
            for e in es1:
                e.contact = selection.contact1
        if selection.contact2 != "auto":
            for e in es2:
                e.contact = selection.contact2
        if selection.cell_um is not None:
            config.CELL_UM_OVERRIDE = selection.cell_um
        config.ADAPTIVE_CELLS = selection.adaptive

        try:
            problem = board_io.build_problem(
                board, selection.net, selection.layers, es1, es2, stackup,
                fills,
                buildups=(buildups if selection.include_buildup else None),
                extra_cu_um=selection.extra_cu_um,
                tracks=(tracks if selection.include_tracks else None),
                vias_capped=selection.vias_capped,
                cap_max_drill_mm=selection.cap_max_drill_mm)
            outdir = report.make_output_dir(board_io.board_dir(board))
        except ApiError as e:
            raise UserFacingError(f"KiCad API error: {e}")

        report.write_geometry_dump(outdir, problem)
        overlay_cb = None
        if selection.push_overlays:
            def overlay_cb(stack, result):
                board_io.push_result_overlays(board, stack, result)
        pipeline.run(problem, outdir, show=True, i_test=selection.current_a,
                     freq_hz=selection.freq_hz,
                     contact_model=selection.contact_model,
                     overlay=overlay_cb)
    except UserFacingError as e:
        _fail(str(e), outdir)
    except Exception:
        _fail(traceback.format_exc(), outdir)


if __name__ == "__main__":
    main()
