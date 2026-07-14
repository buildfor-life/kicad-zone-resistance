"""Offline runner: solve a geometry_dump.json without KiCad.

    python -m fill_resistance.standalone dump.json [--current 40]
        [--cell-um 50] [--layers F.Cu,In1.Cu] [--no-show] [--out DIR]
        [--force-iterative]

This is the dev loop and the convergence-study tool (KiCad 10 has no
headless API server, so the plugin path always needs the GUI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, pipeline
from .errors import UserFacingError
from .geometry import load_problem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path, help="geometry_dump.json from a plugin run")
    ap.add_argument("--current", type=float, default=None,
                    help="test current [A] (default: config TEST_CURRENT_A)")
    ap.add_argument("--freq", type=str, default="0",
                    help="frequency, e.g. 142k or 1.5M (default: DC). "
                         "AC results are a lower bound (skin per foil only)")
    ap.add_argument("--cell-um", type=float, default=None,
                    help="force grid cell size [um]")
    ap.add_argument("--layers", type=str, default=None,
                    help="comma-separated subset of layers to include")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: next to the dump)")
    ap.add_argument("--no-show", action="store_true",
                    help="save PNGs only, no windows")
    ap.add_argument("--contact-model", choices=["uniform", "equipotential"],
                    default=None, help="contact model (default: config)")
    ap.add_argument("--strip-buildup", action="store_true",
                    help="ignore solder buildup stored in the dump")
    ap.add_argument("--extra-cu-um", type=float, default=None,
                    help="override the added copper in mask openings [um]")
    ap.add_argument("--force-iterative", action="store_true",
                    help="use CG (Jacobi) regardless of problem size")
    args = ap.parse_args(argv)

    if args.cell_um is not None:
        config.CELL_UM_OVERRIDE = args.cell_um
    if args.no_show:
        config.INTERACTIVE = False
    if args.force_iterative:
        config.SPSOLVE_MAX_UNKNOWNS = 0

    problem = load_problem(args.dump)
    if args.strip_buildup:
        problem.buildups = []
    if args.extra_cu_um is not None:
        problem.extra_cu_nm = int(args.extra_cu_um * 1000)
    if args.layers:
        keep = [s.strip() for s in args.layers.split(",")]
        problem.layers = [l for l in problem.layers if l.layer_name in keep]
        if not problem.layers:
            print(f"ERROR: no layer of the dump matches --layers {args.layers}",
                  file=sys.stderr)
            return 1

    from .skin import parse_frequency
    outdir = args.out if args.out is not None else args.dump.parent
    try:
        pipeline.run(problem, outdir, show=not args.no_show,
                     i_test=args.current, freq_hz=parse_frequency(args.freq),
                     contact_model=args.contact_model)
    except UserFacingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
