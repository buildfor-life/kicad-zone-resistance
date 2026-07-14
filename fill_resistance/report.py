"""Output directory, summary.txt, geometry dump, stdout one-liner."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .geometry import Problem, save_problem
from .raster import RasterStack
from .solver import Result


def make_output_dir(board_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(board_dir) / config.OUTPUT_DIRNAME / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_geometry_dump(outdir: Path, problem: Problem) -> Path:
    p = outdir / "geometry_dump.json"
    save_problem(problem, p)
    return p


def result_line(result: Result, problem: Problem, stack: RasterStack) -> str:
    ny, nx = stack.shape2d
    ac = (f" @ {result.freq_hz / 1e3:g} kHz (lower bound)"
          if result.freq_hz > 0 else "")
    return (f"R = {result.R_ohm * 1000:.4g} mOhm{ac}, "
            f"P = {result.P_total:.4g} W @ {result.i_test:g} A "
            f"(net {problem.net_name}, {'+'.join(stack.layer_names)}, "
            f"grid {nx}x{ny}x{stack.nlayers}, cell {stack.h_nm / 1000:.0f} um)")


def _electrode_line(e) -> str:
    r = e.rect
    return (f"{e.label:12s} contact={e.contact:8s} "
            f"x [{r.x0 / 1e6:.2f}, {r.x1 / 1e6:.2f}] "
            f"y [{r.y0 / 1e6:.2f}, {r.y1 / 1e6:.2f}] mm")


def write_summary(outdir: Path, problem: Problem, stack: RasterStack,
                  result: Result) -> Path:
    ny, nx = stack.shape2d
    info = result.solve_info
    lines = [
        "fill_resistance summary",
        "=======================",
        f"board:             {problem.board_path}",
        f"net:               {problem.net_name}",
        f"test current:      {result.i_test:g} A",
        f"resistivity:       {problem.rho_ohm_m:.3e} ohm*m",
        f"via plating:       {problem.plating_nm / 1000:.0f} um",
        "",
        (f"frequency:         "
         + (f"{result.freq_hz:g} Hz (skin depth {result.skin_depth_um:.0f} um)"
            if result.freq_hz > 0 else "DC")),
        f"RESISTANCE:        {result.R_ohm * 1000:.6g} mOhm"
        + (" (AC LOWER BOUND: lateral/proximity redistribution not modeled)"
           if result.freq_hz > 0 else ""),
        f"VOLTAGE DROP:      {result.R_ohm * result.i_test * 1000:.4g} mV "
        f"@ {result.i_test:g} A",
        f"TOTAL POWER:       {result.P_total:.6g} W @ {result.i_test:g} A",
        f"  in vias:         {result.P_vias:.4g} W",
        f"  power balance:   {result.power_balance_rel:.2e} (consistency)",
        "",
        "layers (top to bottom):",
    ]
    if problem.buildups and stack.buildup is not None:
        eq_um = (problem.solder_thickness_nm / 1000
                 * problem.rho_ohm_m / problem.solder_rho_ohm_m
                 + problem.extra_cu_nm / 1000)
        cell_mm2 = (stack.h_nm * 1e-6) ** 2
        per_layer = {name: float(stack.buildup[li].sum()) * cell_mm2
                     for li, name in enumerate(stack.layer_names)
                     if stack.buildup[li].any()}
        areas = ", ".join(f"{n}: {a:.0f} mm^2" for n, a in per_layer.items())
        lines.insert(-1, f"solder buildup:    "
                         f"{problem.solder_thickness_nm / 1000:.0f} um solder"
                         + (f" + {problem.extra_cu_nm / 1000:.0f} um Cu"
                            if problem.extra_cu_nm else "")
                         + f" = {eq_um:.1f} um equivalent Cu  ({areas})")
    for li, layer in enumerate(problem.layers):
        ac = (f"  Rs_AC/Rs_DC={result.rs_ratios[li]:.2f}"
              if result.freq_hz > 0 else "")
        lines.append(
            f"  {layer.layer_name:8s} t={layer.thickness_nm / 1000:5.1f} um "
            f"z={layer.z_nm / 1000:7.1f} um  "
            f"P={result.P_layers[li]:.4g} W  "
            f"maxJ={float(np.nanmax(result.Jmag[li])) * 1e-6 if np.isfinite(result.Jmag[li]).any() else 0:.4g} A/mm^2"
            + ac
        )
    lines += [
        "",
        f"grid:              {nx} x {ny} x {stack.nlayers} cells @ "
        f"{stack.h_nm / 1000:.1f} um",
        f"copper cells:      {int(stack.masks.sum())}",
        f"free unknowns:     {result.n_free}",
        f"solver:            {info.method}"
        + (f", {info.iterations} iters, residual {info.residual:.2e}"
           if info.iterations is not None else ""),
        f"I1/I2 @ 1V:        {result.I1_a:.9g} / {result.I2_a:.9g} A "
        f"(mismatch {result.mismatch_rel:.2e})",
        f"timings [s]:       "
        f"{', '.join(f'{k}={v:.2f}' for k, v in result.timings.items())}",
        "",
        f"contact model:     {result.contact_model}"
        + ("  (uniform orthogonal injection; R is the upper contact bound)"
           if result.contact_model == "uniform" else "  (ideal bonded lug)"),
        f"terminals:",
        f"  V+ ({len(problem.electrodes1)} injection area(s)):",
        *(f"    {_electrode_line(e)}" for e in problem.electrodes1),
        f"  V- ({len(problem.electrodes2)} injection area(s)):",
        *(f"    {_electrode_line(e)}" for e in problem.electrodes2),
    ]
    if result.part_currents1 or result.part_currents2:
        how = ("prescribed by area share (uniform model)"
               if result.contact_model == "uniform"
               else "computed flux (equipotential model)")
        lines += ["", f"current per injection area @ {result.i_test:g} A "
                      f"({how}):"]
        for sign, pcs in (("+", result.part_currents1),
                          ("-", result.part_currents2)):
            for i, (label, amps) in enumerate(pcs):
                tag = f"{'P' if sign == '+' else 'N'}{i + 1}"
                lines.append(f"  {tag:4s} {label:24s} {amps:9.4g} A  "
                             f"({100 * amps / result.i_test:5.1f}%)")
    if result.via_reports:
        n_shown = min(10, len(result.via_reports))
        lines += [
            "",
            f"vias/pads carrying current (top {n_shown} of "
            f"{len(result.via_reports)}, @ {result.i_test:g} A):",
            "  x [mm]   y [mm]   kind  drill   I [A]     P [W]",
        ]
        for v in result.via_reports[:n_shown]:
            lines.append(
                f"  {v.x_mm:8.2f} {v.y_mm:8.2f} {v.kind:5s} "
                f"{v.drill_mm:5.2f}  {v.current_a:8.4g}  {v.power_w:.4g}"
            )
    p = outdir / "summary.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
