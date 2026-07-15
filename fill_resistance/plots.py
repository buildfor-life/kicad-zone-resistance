"""Figures: per-layer rasterized maps, potential, current density, power
density, and the error figure. PNGs are saved BEFORE any window opens.

Backend: interactive if a GUI toolkit exists (tkinter, else Qt), else Agg
with os.startfile on the saved PNGs so results are never silent.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np


def _pick_backend():
    """matplotlib.use() is lazy and 'succeeds' for backends whose GUI
    toolkit is missing (KiCad's Python has no tkinter), so probe the
    toolkits explicitly."""
    try:
        import tkinter  # noqa: F401
        return "TkAgg"
    except Exception:
        pass
    for qt in ("PySide6", "PyQt6", "PyQt5", "PySide2"):
        try:
            __import__(qt)
            return "QtAgg" if qt in ("PySide6", "PyQt6") else "Qt5Agg"
        except Exception:
            continue
    return None


INTERACTIVE_BACKEND = _pick_backend()
matplotlib.use(INTERACTIVE_BACKEND or "Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config  # noqa: E402

_BG = "#f5f3f0"
_COPPER = "#c98b4e"
_E1_COLOR = "#c8385a"
_E2_COLOR = "#2f6fb0"
_VIA_COLOR = "#2d6b45"
_SOLDER = "#9aa3ad"      # tin-gray: solder buildup areas
_INK = "#3a3a3a"
_GRID_INK = "#b8b4ae"


def _fmt_si(value: float, unit: str) -> str:
    for scale, prefix in ((1.0, ""), (1e-3, "m"), (1e-6, "µ")):
        if abs(value) >= scale:
            return f"{value / scale:.4g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


def _suptitle(problem, stack, result=None) -> str:
    ny, nx = stack.shape2d
    parts = []
    if result is not None:
        parts.append(f"R = {result.R_ohm * 1000:.4g} mΩ")
        parts.append(f"P = {_fmt_si(result.P_total, 'W')} @ "
                     f"{result.i_test:g} A")
        if result.freq_hz > 0:
            parts.append(f"f = {result.freq_hz / 1e3:g} kHz "
                         f"(δ={result.skin_depth_um:.0f} µm, lower bound)")
    parts.append(problem.net_name)
    parts.append(f"{nx}×{ny}×{stack.nlayers} @ {stack.h_nm / 1000:.0f} µm")
    return "  |  ".join(parts)


def _layer_fig(stack, window_title: str):
    L = stack.nlayers
    ny, nx = stack.shape2d
    aspect = ny / nx
    w = 9.5
    row_h = min(max(w * aspect * 0.9 + 0.6, 1.8), 8.5 / L)
    # constrained layout re-runs on every draw, so the figure reflows
    # when the user resizes the window (tight_layout is one-shot)
    fig, axes = plt.subplots(L, 1, figsize=(w, row_h * L + 1.4),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    axes = axes[:, 0]
    if INTERACTIVE_BACKEND:
        fig.canvas.manager.set_window_title(window_title)
    for ax, name in zip(axes, stack.layer_names):
        ax.set_ylabel(f"{name}\ny [mm]", fontsize=8)
        ax.tick_params(colors=_INK, labelsize=8)
        for s in ax.spines.values():
            s.set_color(_GRID_INK)
    axes[-1].set_xlabel("x [mm]")
    return fig, axes


def _electrode_labels(ax, stack, e1_l, e2_l):
    """Label each connected contact part (multi-part terminals get one
    label per island, largest first, up to 4)."""
    from scipy import ndimage
    for e, label, color in ((e1_l, "V+", _E1_COLOR), (e2_l, "V−", _E2_COLOR)):
        if not e.any():
            continue
        labels, n = ndimage.label(e)
        sizes = ndimage.sum_labels(np.ones_like(labels), labels,
                                   range(1, n + 1))
        order = np.argsort(sizes)[::-1][:4] + 1
        for comp in order:
            ii, jj = np.nonzero(labels == comp)
            cx = (stack.x0_nm + (jj.mean() + 0.5) * stack.h_nm) * 1e-6
            cy = (stack.y0_nm + (ii.mean() + 0.5) * stack.h_nm) * 1e-6
            ax.annotate(label, (cx, cy), xytext=(0, 0),
                        textcoords="offset points", color="white",
                        fontsize=9, fontweight="bold", ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc=color,
                                  ec="none", alpha=0.9))


def _via_markers(ax, problem, layer):
    xs = [v.x * 1e-6 for v in problem.vias if v.spans(layer.z_nm)]
    ys = [v.y * 1e-6 for v in problem.vias if v.spans(layer.z_nm)]
    if xs:
        ax.plot(xs, ys, ".", ms=2.5, color=_VIA_COLOR, alpha=0.7)


def area_tag(sign: str, index: int) -> str:
    """Short injection-area tag: P1, P2, ... for V+; N1, N2, ... for V-."""
    return f"{'P' if sign == '+' else 'N'}{index + 1}"


def _injection_area_labels(ax, li, layer_name, problem, result):
    """Mark every injection area with its short tag (currents live in
    the legend)."""
    groups = ((problem.electrodes1, "+", _E1_COLOR),
              (problem.electrodes2, "-", _E2_COLOR))
    for parts, sign, color in groups:
        for i, el in enumerate(parts):
            if el.contact != "all" and el.contact != layer_name:
                continue
            cx = (el.rect.x0 + el.rect.x1) / 2e6
            cy = (el.rect.y0 + el.rect.y1) / 2e6
            ax.annotate(area_tag(sign, i), (cx, cy), xytext=(0, 0),
                        textcoords="offset points", color="white",
                        fontsize=8, fontweight="bold", ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc=color,
                                  ec="none", alpha=0.9))


def fig_raster(stack, e1, e2, problem, result=None):
    fig, axes = _layer_fig(stack, "Fill Resistance - rasterized map")
    cmap = ListedColormap([_BG, _COPPER, _E1_COLOR, _E2_COLOR, _SOLDER])
    has_buildup = stack.buildup is not None and stack.buildup.any()
    for li, ax in enumerate(axes):
        codes = np.zeros(stack.shape2d, dtype=np.uint8)
        codes[stack.masks[li]] = 1
        if has_buildup:
            codes[stack.buildup[li]] = 4
        codes[e1[li]] = 2
        codes[e2[li]] = 3
        ax.imshow(codes, cmap=cmap, vmin=0, vmax=4, origin="upper",
                  extent=stack.extent_mm(), interpolation="nearest")
        _via_markers(ax, problem, problem.layers[li])
        if result is not None and (result.part_currents1
                                   or result.part_currents2):
            _injection_area_labels(ax, li, stack.layer_names[li], problem,
                                   result)
        else:
            _electrode_labels(ax, stack, e1[li], e2[li])
    handles = [Patch(fc=_COPPER, label="copper"),
               Patch(fc=_VIA_COLOR, label="vias")]
    if has_buildup:
        handles.append(Patch(
            fc=_SOLDER,
            label=f"solder buildup "
                  f"({problem.solder_thickness_nm / 1000:.0f} µm"
                  + (f" + {problem.extra_cu_nm / 1000:.0f} µm Cu"
                     if problem.extra_cu_nm else "") + ")"))
    if result is not None and (result.part_currents1
                               or result.part_currents2):
        entries = ([("+", _E1_COLOR, i, amps)
                    for i, (_, amps) in enumerate(result.part_currents1)]
                   + [("-", _E2_COLOR, i, amps)
                      for i, (_, amps) in enumerate(result.part_currents2)])
        shown = entries[:14]
        for sign, color, i, amps in shown:
            handles.append(Patch(
                fc=color,
                label=f"{area_tag(sign, i)}: {amps:.3g} A "
                      f"({100 * amps / result.i_test:.0f}%)"))
        if len(entries) > len(shown):
            handles.append(Patch(fc="#00000000",
                                 label=f"... +{len(entries) - len(shown)} "
                                       f"more in summary.txt"))
    else:
        handles += [Patch(fc=_E1_COLOR, label="V+"),
                    Patch(fc=_E2_COLOR, label="V−")]
    axes[0].legend(handles=handles, loc="upper right", fontsize=7,
                   framealpha=0.9)
    fig.suptitle("Rasterized fill + electrodes  |  "
                 + _suptitle(problem, stack, result), fontsize=10, color=_INK)
    return fig


def fig_potential(result, stack, e1, e2, problem):
    fig, axes = _layer_fig(stack, "Fill Resistance - potential")
    vmax = float(np.nanmax(result.V))
    # uniform model: <V-> = 0 is the reference, individual V- cells can
    # sit slightly below it - keep them in range instead of clipping
    vmin = min(0.0, float(np.nanmin(result.V)))
    unit, scale = ("mV", 1e3) if vmax < 0.1 else ("V", 1.0)
    cmap = matplotlib.colormaps[config.CMAP_POTENTIAL].copy()
    cmap.set_bad(_BG)
    im = None
    for li, ax in enumerate(axes):
        vs = result.V[li] * scale
        im = ax.imshow(vs, cmap=cmap, vmin=vmin * scale, vmax=vmax * scale,
                       origin="upper", extent=stack.extent_mm(),
                       interpolation="nearest")
        if np.isfinite(vs).sum() > 4:
            ext = stack.extent_mm()
            ny, nx = stack.shape2d
            xs = np.linspace(ext[0], ext[1], nx, endpoint=False)
            xs += (xs[1] - xs[0]) / 2
            ys = np.linspace(ext[3], ext[2], ny, endpoint=False)
            ys += (ys[1] - ys[0]) / 2
            with np.errstate(invalid="ignore"):
                ax.contour(xs, ys, vs, levels=15, colors="white",
                           linewidths=0.4, alpha=0.5)
        _electrode_labels(ax, stack, e1[li], e2[li])
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label(f"potential [{unit}] @ {result.i_test:g} A", fontsize=9)
    fig.suptitle("Potential  |  " + _suptitle(problem, stack, result),
                 fontsize=10, color=_INK)
    return fig


def _field_fig(result, stack, e1, e2, problem, data3, cmap_name, dyn_range,
               label, title, window):
    """Shared per-layer LogNorm field figure (current, power)."""
    fig, axes = _layer_fig(stack, window)
    vmax = float(np.nanmax(data3))
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad(_BG)
    if config.LOG_CURRENT_SCALE and vmax > 0:
        norm = LogNorm(vmin=vmax / dyn_range, vmax=vmax)
    else:
        norm = None
    im = None
    for li, ax in enumerate(axes):
        d = data3[li]
        shown = np.clip(d, vmax / dyn_range, None) if norm is not None else d
        im = ax.imshow(shown, cmap=cmap, norm=norm, origin="upper",
                       extent=stack.extent_mm(), interpolation="nearest")
        _electrode_labels(ax, stack, e1[li], e2[li])
    if vmax > 0:
        li, i, j = np.unravel_index(np.nanargmax(data3), data3.shape)
        mx = (stack.x0_nm + (j + 0.5) * stack.h_nm) * 1e-6
        my = (stack.y0_nm + (i + 0.5) * stack.h_nm) * 1e-6
        axes[li].plot(mx, my, "o", ms=9, mfc="none", mec="white", mew=1.4)
        axes[li].annotate(f"max {vmax:.3g}", (mx, my), xytext=(10, -10),
                          textcoords="offset points", color="white",
                          fontsize=8,
                          bbox=dict(boxstyle="round,pad=0.2", fc="#00000088",
                                    ec="none"))
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label(label, fontsize=9)
    fig.suptitle(title + "  |  " + _suptitle(problem, stack, result),
                 fontsize=10, color=_INK)
    return fig, axes


def fig_current(result, stack, e1, e2, problem):
    fig, axes = _field_fig(
        result, stack, e1, e2, problem, result.Jmag * 1e-6,
        config.CMAP_CURRENT, config.CURRENT_DYNAMIC_RANGE,
        f"|J| [A/mm²] @ {result.i_test:g} A",
        "Current density (log)", "Fill Resistance - current density")
    # mark the hottest via
    if result.via_reports:
        v = result.via_reports[0]
        for ax in axes:
            ax.plot(v.x_mm, v.y_mm, "s", ms=7, mfc="none", mec="#7fe0a8",
                    mew=1.2)
        axes[0].annotate(
            f"hottest via {v.current_a:.3g} A", (v.x_mm, v.y_mm),
            xytext=(10, 10), textcoords="offset points", color="white",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="#2d6b45", ec="none"))
    return fig


def fig_power(result, stack, e1, e2, problem):
    # W/m^2 -> W/mm^2
    fig, axes = _field_fig(
        result, stack, e1, e2, problem, result.Parea * 1e-6,
        config.CMAP_POWER, config.POWER_DYNAMIC_RANGE,
        f"p [W/mm²] @ {result.i_test:g} A",
        "Power density (log)", "Fill Resistance - power density")
    for li, ax in enumerate(axes):
        ax.set_title(f"P({stack.layer_names[li]}) = "
                     f"{_fmt_si(result.P_layers[li], 'W')}",
                     fontsize=8, color=_INK, loc="right", pad=2)
    return fig


def fig_error(message: str):
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.axis("off")
    ax.set_title("Fill Resistance — ERROR", color="#b02a2a",
                 fontsize=14, fontweight="bold", loc="left")
    wrapped = "\n".join(
        textwrap.fill(line, width=90) for line in message.splitlines()
    )
    ax.text(0.0, 0.95, wrapped, family="monospace", fontsize=9,
            va="top", ha="left", color=_INK, transform=ax.transAxes)
    return fig


def _resolve_label_overlaps(fig):
    """Measure every annotation's rendered box and greedily push
    overlapping labels upward until nothing collides. Runs on the real
    renderer, so it handles any font/DPI."""
    from matplotlib.text import Annotation
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    for ax in fig.axes:
        anns = [c for c in ax.get_children() if isinstance(c, Annotation)]
        placed = []
        for a in sorted(anns, key=lambda t: t.get_window_extent(renderer).x0):
            try:
                bb = a.get_window_extent(renderer)
            except Exception:
                continue
            guard = 50
            while guard > 0:
                hit = next((p for p in placed if bb.overlaps(p)), None)
                if hit is None:
                    break
                push_px = (hit.y1 - bb.y0) + 3.0
                dx, dy = a.xyann
                a.xyann = (dx, dy + push_px * 72.0 / fig.dpi)
                bb = a.get_window_extent(renderer)
                guard -= 1
            placed.append(bb)


def _fit_to_screen(fig) -> None:
    """Best effort: shrink the window so it fits the screen. Runs AFTER
    the PNGs are saved (so their size is unaffected); the constrained
    layout reflows the content on every subsequent resize."""
    try:
        win = fig.canvas.manager.window
        if hasattr(win, "screen"):                      # Qt
            avail = win.screen().availableGeometry()
            sw, sh = avail.width(), avail.height()
        elif hasattr(win, "winfo_screenwidth"):         # Tk
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        else:
            return
        w, h = fig.get_size_inches()
        scale = min(0.9 * sw / (w * fig.dpi), 0.85 * sh / (h * fig.dpi), 1.0)
        if scale < 1.0:
            fig.set_size_inches(w * scale, h * scale, forward=True)
    except Exception:
        pass


def _raise_windows():
    """Best effort: bring plot windows in front of KiCad (windows spawned
    by a background process tend to open behind)."""
    for num in plt.get_fignums():
        try:
            win = plt.figure(num).canvas.manager.window
            if hasattr(win, "attributes"):          # Tk
                win.attributes("-topmost", True)
                win.after(300, lambda w=win: w.attributes("-topmost", False))
            else:                                    # Qt
                win.raise_()
                win.activateWindow()
        except Exception:
            pass


def save_and_show(figs_named: list[tuple], outdir: Path | None,
                  show: bool = True) -> list[Path]:
    """figs_named: [(figure, basename), ...]. Saves first, then shows."""
    saved = []
    for fig, _ in figs_named:
        _resolve_label_overlaps(fig)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        for fig, name in figs_named:
            p = outdir / f"{name}.png"
            fig.savefig(p, dpi=config.DPI, facecolor="white",
                        bbox_inches="tight")
            saved.append(p)
            print(f"saved {p}")
    if show and config.INTERACTIVE:
        if INTERACTIVE_BACKEND:
            for fig, _ in figs_named:
                _fit_to_screen(fig)
            _raise_windows()
            plt.show()
        else:
            for p in saved:
                _open_in_viewer(p)
    plt.close("all")
    return saved


def _open_in_viewer(path: Path) -> None:
    """Open a saved PNG in the OS default viewer (no-GUI-backend
    fallback so results are never silent)."""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass
