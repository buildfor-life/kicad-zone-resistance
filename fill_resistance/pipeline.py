"""Geometry-in -> results-out pipeline shared by the KiCad entrypoint and
the offline standalone runner."""
from __future__ import annotations

from pathlib import Path

from . import config, plots, raster, report, solver
from .errors import UserFacingError
from .geometry import Problem
from .solver import Result


def run(problem: Problem, outdir: Path | None, show: bool = True,
        i_test: float | None = None, freq_hz: float = 0.0,
        contact_model: str | None = None, overlay=None) -> Result:
    """overlay: optional callback(stack, result) run after the solve
    (EXPERIMENTAL in-KiCad overlays); its failures are non-fatal."""
    if i_test is None:
        i_test = config.TEST_CURRENT_A
    if i_test <= 0:
        raise UserFacingError(f"Test current must be > 0 A (got {i_test:g}).")
    h = raster.choose_cell_size(problem.copper_bbox(), len(problem.layers))
    print(f"rasterizing {len(problem.layers)} layer(s) at cell size "
          f"{h / 1000:.1f} um ...")
    stack = raster.rasterize_stack(problem, h)
    print(f"grid {stack.shape2d[1]}x{stack.shape2d[0]}x{stack.nlayers}, "
          f"{int(stack.masks.sum())} copper cells, {len(problem.vias)} "
          f"via/pad barrel(s)")

    e1, e2 = raster.electrode_masks(stack, problem)
    parts1, parts2 = raster.electrode_partition(stack, problem)

    print(f"solving @ {i_test:g} A"
          + (f", {freq_hz:g} Hz" if freq_hz > 0 else " DC") + " ...")
    result = solver.run_solve(problem, stack, e1, e2, i_test, freq_hz,
                              contact_model, parts1, parts2)
    for prefix, pcs in (("P", result.part_currents1),
                        ("N", result.part_currents2)):
        for i, (label, amps) in enumerate(pcs):
            print(f"  {prefix}{i + 1} ({label}): {amps:.4g} A "
                  f"({100 * amps / i_test:.1f}%)")

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        report.write_summary(outdir, problem, stack, result)
    print(report.result_line(result, problem, stack))

    if overlay is not None:
        try:
            overlay(stack, result)
        except Exception as e:
            print(f"overlay push failed: {e}")

    figs = [
        (plots.fig_raster(stack, e1, e2, problem, result), "1_raster_map"),
        (plots.fig_potential(result, stack, e1, e2, problem), "2_potential"),
        (plots.fig_current(result, stack, e1, e2, problem),
         "3_current_density"),
        (plots.fig_power(result, stack, e1, e2, problem), "4_power_density"),
    ]
    plots.save_and_show(figs, outdir, show=show)
    return result
