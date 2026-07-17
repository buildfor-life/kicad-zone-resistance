"""Generate the README model figures in docs/img/.

    .venv\\Scripts\\python.exe tools\\gen_readme_figs.py

Real solver output wherever possible: the demo-board maps (raster map
with the adaptive mesh, current density) and the contact-model
comparison come straight from the plugin's own pipeline on small
synthetic boards; only the hole-anatomy cross-section is drawn by hand.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fill_resistance import config  # noqa: E402

config.INTERACTIVE = False

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

from fill_resistance import plots, raster, solver  # noqa: E402
from fill_resistance.geometry import (Electrode, LayerFill, Polygon,  # noqa: E402
                                      Problem, Rect, ViaLink,
                                      contact_solder_buildups)

import matplotlib.pyplot as plt  # noqa: E402

plt.switch_backend("Agg")
plots.INTERACTIVE_BACKEND = None      # save-only: no window panels

NM = 1_000_000
OUT = ROOT / "docs" / "img"

COPPER = plots._COPPER
SOLDER = plots._SOLDER
LEAD = "#5c6570"
CORE = "#ccd6b3"                      # FR-4
FILLER = "#eae5dc"                    # non-conductive via fill
INK = plots._INK


def _poly(pts_mm, holes_mm=()) -> Polygon:
    ring = lambda pts: np.asarray(  # noqa: E731
        [[int(x * NM), int(y * NM)] for x, y in pts], dtype=np.int64)
    return Polygon(outline=ring(pts_mm),
                   holes=[ring(h) for h in holes_mm])


def _rect(x0, y0, x1, y1, layer="User.1") -> Rect:
    return Rect.normalized(int(x0 * NM), int(y0 * NM),
                           int(x1 * NM), int(y1 * NM), layer)


def _disc(x_mm, y_mm, r_mm, n=64) -> Polygon:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return _poly([(x_mm + r_mm * np.cos(a), y_mm + r_mm * np.sin(a))
                  for a in ang])


def _solve(problem, h_mm, i_test=10.0, model=None):
    stack = raster.rasterize_stack(problem, int(h_mm * NM))
    e1, e2 = raster.electrode_masks(stack, problem)
    p1, p2 = raster.electrode_partition(stack, problem)
    res = solver.run_solve(problem, stack, e1, e2, i_test,
                           contact_model=model, parts1=p1, parts2=p2)
    return res, stack, e1, e2


# --- demo board: 2 layers, notched F.Cu pour, half-size B.Cu pour, ----
# --- a soldered THT-pad contact and a stitching-via field -------------

def demo_problem() -> Problem:
    # F.Cu: full pour with a notch from the top edge down to y=6 - the
    # current from the left contact must squeeze through the channel
    top = _poly([(0, 0), (40, 0), (40, 22), (13, 22), (13, 6),
                 (10, 6), (10, 22), (0, 22)])
    # B.Cu: pour on the right half only - vias must carry the transfer
    bot = _poly([(18, 0), (40, 0), (40, 22), (18, 22)])
    z_bot = int(1.6 * NM)
    vias = [ViaLink(x=int(x * NM), y=int(y * NM), drill_nm=300_000,
                    z_top_nm=-1, z_bot_nm=z_bot + 1, pad_nm=600_000)
            for x in (20.5, 24.5, 28.5, 32.5, 36) for y in (3, 7, 11, 15, 19)]
    tht = Electrode(
        rect=_rect(3.4, 9.9, 5.6, 12.1), contact="F.Cu",
        label="THT pad (4.5, 11)", drill_nm=1_000_000,
        pad_nm=int(2.2 * NM), pad_min_nm=int(2.2 * NM),
        center=(int(4.5 * NM), int(11 * NM)), solder=True,
        protrusion_side="F.Cu", polygons=[_disc(4.5, 11, 1.1)])
    lug = Electrode(rect=_rect(37.5, 3, 39.5, 19, "User.2"),
                    contact="B.Cu", label="lug")
    p = Problem(
        board_path="synthetic", net_name="DEMO",
        rho_ohm_m=1.68e-8, plating_nm=18_000,
        layers=[LayerFill("F.Cu", 70_000, 0, [top]),
                LayerFill("B.Cu", 70_000, z_bot, [bot])],
        vias=vias, electrodes1=[tht], electrodes2=[lug],
        thickness_source="override")
    contact_solder_buildups(p)
    return p


def gen_demo_maps():
    p = demo_problem()
    res, stack, e1, e2 = _solve(p, 0.05)
    figs = [
        (plots.fig_raster(stack, e1, e2, p, res), "demo-raster"),
        (plots.fig_current(res, stack, e1, e2, p), "demo-current"),
        (plots.fig_potential(res, stack, e1, e2, p), "demo-potential"),
    ]
    plots.save_and_show(figs, OUT, show=False)


# --- contact models: equipotential vs uniform injection ---------------

def gen_contact_models():
    plate = [(0, 0), (24, 0), (24, 18), (0, 18)]
    r_ohm, zooms = {}, {}
    # uniform grid: the coarse adaptive leaves would pixelate the |J| zoom
    config.ADAPTIVE_CELLS = False
    for model in ("equipotential", "uniform"):
        p = Problem(
            board_path="synthetic", net_name="DEMO",
            rho_ohm_m=1.68e-8, plating_nm=18_000,
            layers=[LayerFill("F.Cu", 70_000, 0, [_poly(plate)])],
            vias=[],
            electrodes1=[Electrode(rect=_rect(4.5, 7.5, 7.5, 10.5))],
            electrodes2=[Electrode(rect=_rect(22, 1, 23.5, 17, "User.2"))],
            thickness_source="override")
        res, stack, _, _ = _solve(p, 0.05, model=model)
        h_mm = stack.h_nm / NM
        j = res.Jmag[0] * 1e-6                       # A/mm^2
        x0, y0 = stack.x0_nm / NM, stack.y0_nm / NM
        c0, c1 = int((2 - x0) / h_mm), int((13 - x0) / h_mm)
        r0, r1 = int((3 - y0) / h_mm), int((15 - y0) / h_mm)
        zooms[model] = (j[r0:r1, c0:c1],
                        (x0 + c0 * h_mm, x0 + c1 * h_mm,
                         y0 + r1 * h_mm, y0 + r0 * h_mm))
        r_ohm[model] = res.R_ohm
    config.ADAPTIVE_CELLS = True

    vmax = float(np.percentile(
        zooms["equipotential"][0][np.isfinite(zooms["equipotential"][0])],
        99.0))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True,
                             layout="constrained")
    titles = {"equipotential": "equipotential (ideal bonded lug):\n"
                               "|J| crowds at the contact edges",
              "uniform": "uniform injection (pressed conductor):\n"
                         "|J| ramps across the contact"}
    for ax, model in zip(axes, ("equipotential", "uniform")):
        data, extent = zooms[model]
        cmap = matplotlib.colormaps[config.CMAP_CURRENT].copy()
        cmap.set_bad(plots._BG)
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, origin="upper",
                       extent=extent, interpolation="nearest")
        ax.add_patch(plt.Rectangle((4.5, 7.5), 3, 3, fill=False,
                                   ec="white", ls="--", lw=1.0))
        ax.set_title(f"{titles[model]}\nR = {r_ohm[model] * 1e3:.3f} mΩ",
                     fontsize=9, color=INK)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.tick_params(labelsize=8, colors=INK)
    axes[0].set_ylabel("y [mm]", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.85)
    cb.set_label("|J| [A/mm²] @ 10 A", fontsize=9)
    fig.suptitle("The two contact models bracket a real contact:  "
                 "R$_{equipotential}$ ≤ R$_{real}$ ≤ R$_{uniform}$",
                 fontsize=10, color=INK)
    fig.savefig(OUT / "contact-models.png", dpi=config.DPI,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / 'contact-models.png'}")


# --- hole anatomy: hand-drawn cross-section of the four hole types ----

CORE_T = 1.6          # substrate thickness [drawing units ~ mm]
FOIL_T = 0.18         # foil thickness, exaggerated
PLATE_W = 0.12        # barrel plating, exaggerated
CAP_T = 0.07          # via cap
COAT_T = 0.1          # pad-face solder coat
Y_TOP = CORE_T + FOIL_T


def _board_segment(ax, x0, x1):
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, CORE_T, fc=CORE, ec="none"))
    for y in (CORE_T, -FOIL_T):
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, FOIL_T,
                                   fc=COPPER, ec="none"))


def _barrel(ax, xc, drill):
    for s in (-1, 1):
        x = xc + s * drill / 2 - (PLATE_W if s > 0 else 0)
        ax.add_patch(plt.Rectangle((x, -FOIL_T), PLATE_W,
                                   CORE_T + 2 * FOIL_T, fc=COPPER,
                                   ec="none"))


def _label(ax, text, xy, xytext, ha="left"):
    ax.annotate(text, xy, xytext=xytext, fontsize=7.5, color=INK, ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7,
                                shrinkA=2, shrinkB=1))


def gen_hole_anatomy():
    fig, ax = plt.subplots(figsize=(12.5, 5.2), layout="constrained")
    holes = [(3.0, 0.7), (10.0, 1.6), (17.5, 1.6), (25.0, 1.6)]
    edges = [0.0]
    for xc, d in holes:
        edges += [xc - d / 2, xc + d / 2]
    edges.append(28.5)
    for x0, x1 in zip(edges[::2], edges[1::2]):
        _board_segment(ax, x0, x1)
    for xc, d in holes:
        _barrel(ax, xc, d)

    # 1: small via, filled + capped
    xc, d = holes[0]
    ax.add_patch(plt.Rectangle((xc - d / 2 + PLATE_W, -FOIL_T),
                               d - 2 * PLATE_W, CORE_T + 2 * FOIL_T,
                               fc=FILLER, ec="none"))
    for y in (Y_TOP, -FOIL_T - CAP_T):
        ax.add_patch(plt.Rectangle((xc - d / 2 - 0.12, y), d + 0.24, CAP_T,
                                   fc=COPPER, ec="none"))
    _label(ax, "cap, CAP_PLATING_UM (15 µm)\non both outer mouths",
           (xc, Y_TOP + CAP_T), (xc, 3.3), ha="center")
    _label(ax, "non-conductive fill", (xc, 0.8), (5.6, -0.9))

    # 2: big via, mouth open
    xc, d = holes[1]
    _label(ax, "open mouth: covered cells\nremoved, sub-cell mouths\n"
               "scale the sheet conductance",
           (xc, Y_TOP - FOIL_T / 2), (xc, 3.2), ha="center")

    # 3: populated THT pad - full solder joint
    xc, d = holes[2]
    lead_w = d - 0.5                      # drill - clearance, exaggerated
    pad_r = 1.7
    prot = 1.5
    sn_edge = "#7d8791"                   # delineate solder sub-shapes
    # solder fill between plating and lead
    for s in (-1, 1):
        x0 = xc + s * lead_w / 2 if s > 0 else xc - d / 2 + PLATE_W
        ax.add_patch(plt.Rectangle((x0, -FOIL_T),
                                   d / 2 - PLATE_W - lead_w / 2,
                                   CORE_T + 2 * FOIL_T, fc=SOLDER,
                                   ec="none"))
    # pad-face coat, solder side only
    ax.add_patch(plt.Rectangle((xc - pad_r, Y_TOP), 2 * pad_r, COAT_T,
                               fc=SOLDER, ec=sn_edge, lw=0.5))
    # solder cone: protrusion height at the wall -> 0 at the pad edge
    for s in (-1, 1):
        wall = xc + s * lead_w / 2
        ax.add_patch(plt.Polygon(
            [(wall, Y_TOP + prot), (wall, Y_TOP + COAT_T),
             (xc + s * pad_r, Y_TOP + COAT_T)],
            closed=True, fc=SOLDER, ec=sn_edge, lw=0.5))
    # lead: through the hole, protruding on top, component below
    ax.add_patch(plt.Rectangle((xc - lead_w / 2, -2.05), lead_w,
                               2.05 + Y_TOP + prot, fc=LEAD, ec="none"))
    ax.add_patch(plt.Rectangle((xc - 1.5, -2.75), 3.0, 0.7,
                               fc="#8a8f96", ec="none"))
    ax.text(xc, -2.4, "component", fontsize=7.5, color="white",
            ha="center", va="center")
    _label(ax, "clipped lead protrudes\nTHT_LEAD_PROTRUSION_MM (1.5 mm)",
           (xc + lead_w / 2, Y_TOP + prot - 0.2), (xc + 3.4, 4.15))
    _label(ax, "solder cone: full height at the\nwall, tapers to 0 at the "
               "pad edge", (xc - (lead_w / 2 + pad_r) / 2, Y_TOP + 0.7),
           (13.6, 4.2), ha="center")
    _label(ax, "pad-face solder coat (50 µm),\nSOLDER side only",
           (xc - pad_r + 0.2, Y_TOP + COAT_T / 2), (12.9, 2.35),
           ha="center")
    _label(ax, "solder-filled hole: lead ∥ solder ∥ plating\n"
               "lead ⌀ = drill − THT_LEAD_CLEARANCE_MM",
           (xc - d / 2 + PLATE_W + 0.07, 0.5), (12.3, -1.5), ha="center")
    _label(ax, "component side:\npad face stays bare",
           (xc + pad_r - 0.3, -FOIL_T), (xc + 4.0, -1.05))

    # 4: DNP THT pad
    xc, d = holes[3]
    _label(ax, "open hole on every layer,\nplating-only barrel, no joint",
           (xc, 0.8), (xc + 1.3, -2.45), ha="center")

    for (xc, _), title in zip(holes, (
            "via ≤ cap-drill\n(capped)", "via > cap-drill\n(open)",
            "THT pad, populated\n(read from KiCad)", "THT pad, DNP")):
        ax.text(xc, 5.6, title, fontsize=9, color=INK, ha="center",
                va="top", fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               (COPPER, SOLDER, LEAD, CORE, FILLER)]
    fig.legend(handles, ("copper (foil / plating / pad)", "solder",
                         "component lead", "FR-4", "non-conductive fill"),
               loc="outside right center", fontsize=8, framealpha=0.95)
    ax.set_title("How drilled holes are modeled — cross-section "
                 "(vertical scale exaggerated)", fontsize=11, color=INK)
    ax.set_xlim(-0.3, 29.0)
    ax.set_ylim(-3.1, 5.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "hole-model.png", dpi=config.DPI,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / 'hole-model.png'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen_hole_anatomy()
    gen_contact_models()
    gen_demo_maps()
