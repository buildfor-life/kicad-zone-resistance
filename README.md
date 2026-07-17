# Fill Resistance — KiCad 10 plugin

Computes the **DC or AC resistance of copper zone fills and traces**
between two contacts, **single- or multi-layer**: the chosen net's fills
(teardrops included) and tracks on the selected copper layers are
solved as coupled finite-difference sheets linked by the net's **via
and through-hole-pad barrels** (18 µm plating, configurable). At a user-set **frequency** the exact 1D foil/barrel
skin-effect correction is applied (AC results are a rigorous lower
bound — see *Model & limits*). Shows per-layer rasterized maps,
potential, current density, and **power density**, reports **per-via
currents** (via ampacity!) and total dissipation at a **selectable test
current**. PNGs + a text summary are saved per run.

![Current density on a two-layer demo net](docs/img/demo-current.png)
*Real output on a synthetic two-layer net: current from a soldered
THT-pad contact (V+, injected at the drill-wall ring) squeezes past a
notch in the F.Cu pour, transfers through the stitching-via field into
the B.Cu pour and leaves at the V− lug — per-via currents and the
hottest via are reported.*

Uses the KiCad **IPC API** (`kicad-python` / `kipy`), not the deprecated
SWIG API. Requires KiCad **10.0.1+**.

## Setup (one-time)

1. **Enable the API server**: KiCad → Preferences → Plugins → check
   *Enable KiCad API*.
2. **Check the interpreter path** on the same page: should point at the
   KiCad 10 Python, e.g. `C:\Program Files\KiCad\10.0\bin\pythonw.exe`
   on Windows or `/usr/bin/python3` on Linux (after a 9→10 upgrade it
   can point at KiCad 9).
3. **Deploy** (dev checkout; end users install the PCM zip instead, see
   *Packaging*):
   ```powershell
   powershell -ExecutionPolicy Bypass -File deploy.ps1        # junction (dev)
   powershell -ExecutionPolicy Bypass -File deploy.ps1 -Mode Copy
   ```
   Linux / macOS (also works on Windows with developer mode):
   ```bash
   python3 tools/deploy.py            # symlink (dev)
   python3 tools/deploy.py --copy
   ```
4. **Restart KiCad**; first load builds the plugin venv (numpy, scipy,
   matplotlib, PySide6 — takes minutes; the Ω button appears when done).
   If stuck: Preferences → Plugins → *Recreate Plugin Environment*.

## Usage

1. Mark the current-injection terminals. Each terminal may have
   **multiple parts** (all merged into one externally-bonded contact):
   - **V+ rectangles on `User.1`**, **V− rectangles on `User.2`**
     (marker layers, configurable via `ELECTRODE_POS_LAYER` /
     `ELECTRODE_NEG_LAYER`), any number per side, axis-aligned;
   - **pads and vias** (SMD pad: real copper shape on its own layer;
     through-hole pads and vias become **barrel contacts** — the current
     enters at the drill wall on every spanned layer, see below) —
     selected pads/vias fill a side that has no rectangles;
   - legacy: exactly 2 selected contacts with no marker rectangles still
     works; empty selection scans the whole board's marker layers.
2. **Select the contacts**, click the **Fill Resistance** Ω button.
3. In the **dialog**, pick the net (defaults to the selected pad's net),
   check the **layers** to include, set each contact's layer scope
   ("All selected layers" = bolted-lug/through contact), the **test
   current**, and optionally a grid cell size. Multiple layers are coupled
   through the net's via/pad barrels automatically.
4. Read R / voltage drop / total power in the figure titles and status
   bar. Outputs land in `<board dir>\fill_res_results\<timestamp>\`:
   per-layer `1_raster_map` / `2_potential` / `3_current_density` /
   `4_power_density` PNGs, `summary.txt` (incl. the busiest vias with
   per-via current and dissipation, and the **current through each
   injection area** — computed flux with the equipotential model,
   prescribed area share with the uniform model), `geometry_dump.json`.

## Model & limits

- Sheet model per layer: R□ = ρ/t, ρ = 1.68e-8 Ωm (20 °C), t from the
  board's physical stackup. Layer z-positions from the stackup drive the
  barrel lengths.
- Via/pad barrels: thin-wall annulus, R = ρ·L/(π·d·t_plating),
  `VIA_PLATING_UM = 18` in `fill_resistance/config.py`. Vias are always
  plated. Each via also contributes its **ring/pad copper** (a
  full-thickness disc of the pad diameter on every spanned layer) and
  its **drill mouth**, area-weighted per cell: with the **"vias filled +
  capped" checkbox** (default on, `VIAS_CAPPED`) the mouth carries a
  thin copper cap (`CAP_PLATING_UM = 15`, fab spec) on the **outer**
  layers and is an open hole on inner layers; unchecked, mouths are open
  holes everywhere. The fab caps only small vias: drills above the
  dialog's **"capped up to drill"** threshold (default
  `CAP_MAX_DRILL_MM = 0.5`) keep open mouths even with capping
  selected. Layer-to-layer the cap never matters at DC (it is in
  parallel with the annular-ring contact, not in series) — the checkbox
  only affects in-plane conduction across outer-layer mouths. Sub-cell
  mouths scale their cells' sheet conductance by the true covered
  fraction (4×4 supersampling), so coarse grids see the correct small
  perturbation instead of a whole-cell hole. Barrels are gathered in
  **single-layer runs too** (drill mouths perforate a lone plane).
  **THT pads are fully modeled**: their exact copper shapes (incl.
  oblong pads, fetched from KiCad; the outer shape stands in for inner
  rings) are stamped onto every included layer, and every **populated**
  pad carries its full **soldered joint** on its SOLDER side (opposite
  the component; the component-side pad face stays bare): the hole
  holds the **component lead** (a cylinder of drill −
  `THT_LEAD_CLEARANCE_MM`, resistivity `THT_LEAD_RHO_OHM_M`, copper by
  default — raise it for brass/steel leads) **plus solder** in the
  remaining annulus, both in parallel with the plating; the mouth
  copper stays conducting (it stands in for the plug — conservative,
  the plug is worth far more than the foil); the pad face gets the
  average-thickness solder coat (exact pad shape) and the
  protruding-lead cone (see barrel contacts below; on oblong pads the
  cone tapers within the inscribed circle). Whether a hole is a via or
  a THT pad, the owning footprint's side, and its **Do not populate**
  flag are all read from KiCad — **DNP pads** get an **open hole** and
  a plating-only barrel, no joint. At f > 0 the thickness scaling is
  applied multiplicatively to the skin-corrected sheet conductance
  (approximation). Per layer a barrel attaches to
  the fill cell under it, or to the nearest copper cell within the pad
  footprint plus one grid cell — fills joined by **thermal-relief
  spokes** still connect; wider antipads do not, and the barrel bridges
  the layers above/below with the full barrel length. Barrels that reach
  fill on fewer than two layers carry no current and are reported.

  ![How drilled holes are modeled — cross-section](docs/img/hole-model.png)
  *The four hole types: capped small via, open large via, populated THT
  pad with its full solder joint (lead ∥ solder ∥ plating in the hole,
  one-sided pad coat, protruding-lead solder cone), and a DNP THT pad.*

- The net's **traces** (straight and arc tracks, exact outline polygons
  incl. rounded ends) conduct together with the fills — dialog checkbox,
  on by default (`INCLUDE_TRACKS`). Traces narrower than
  `TRACK_1D_FACTOR` (3) grid cells are modeled as exact **1D resistor
  chains** along their centerline — true arc length per link, so their
  series resistance carries no discretization error and no cell-size
  tuning is needed for thin traces. 1D-modeled traces show potential,
  power density, and |J| (the true in-trace density from the link
  currents, |ΔV|/(ρ·Δl)). THT pad copper is part of the conductor
  (exact shapes, see above); **SMD** pad copper other than the
  selected contacts is still **not**.
- **Solder buildup on mask openings** (dialog checkbox, **off by
  default**; `INCLUDE_MASK_BUILDUP`): zones drawn on `F.Mask`/`B.Mask`
  are treated as mask openings that collect `SOLDER_THICKNESS_UM`
  (50 µm) of solder on the exposed pour, plus an optional user-defined
  added copper thickness (dialog field, e.g. a soldered busbar/wire).
  The sheet conductance there becomes t_Cu/ρ_Cu + t_solder/ρ_solder +
  t_extra/ρ_Cu (SAC305 ρ = 1.32e-7 Ωm: 50 µm solder ≈ 6.4 µm copper);
  interface faces use harmonic-mean conductances. Buildup areas render
  tin-gray on the raster map; |J| in them is referenced to the
  conductance-equivalent copper thickness.
- **Barrel contacts**: a selected **via or through-hole pad** injects at
  the **drill-wall ring** on every layer the barrel spans — the current
  physically enters through the lead/wire soldered into the hole, so
  the spreading resistance across the pad and surrounding pour is part
  of the result (both contact models; verified against
  R = ρ/(π·t)·acosh(d/2a) for two circular contacts on a sheet). A
  soldered **THT joint** additionally assumes the **hole is filled with
  solder** (core in parallel with the plating) and the **pad face on
  the solder side carries an average-thickness solder coat**
  (`SOLDER_THICKNESS_UM`, 50 µm) over the modeled copper under the pad
  shape — the solder side is the side opposite the component (taken
  from the owning footprint; assumed `B.Cu` if it cannot be found),
  and the component-side pad face stays bare. There the **clipped
  lead protrudes** `THT_LEAD_PROTRUSION_MM` (1.5 mm, 0 = off)
  and a **solder cone** wraps
  it: full protrusion height at the drill wall, tapering linearly to
  zero at the pad edge, applied as extra conduction-equivalent copper
  per cell. The tall solder column at the wall pulls the joint
  vicinity to lead potential — equivalent to extending the barrel wall
  vertically — while the taper carries the radial spreading. To model
  a probe pressed onto the pad face instead, draw a marker rectangle
  over the pad.
- **Contact models** (dialog / `CONTACT_MODEL`): default **uniform
  injection** — a conductor pressed on top feeds the current orthogonally
  with uniform surface density, so |J| ramps across the contact area
  (R = ΔV̄/I from area-averaged terminal potentials); or
  **equipotential** — ideal bonded lug (Dirichlet). The two bracket a
  real contact: R_equipotential ≤ R_real ≤ R_uniform. If the selected
  fills form several disconnected copper groups that each touch both
  terminals (e.g. planes joined only through the bolted lugs), only the
  equipotential model is well-defined; the uniform model stops with an
  error instead of prescribing an arbitrary split.

  ![The two contact models bracket a real contact](docs/img/contact-models.png)
  *|J| around the same 3×3 mm contact under both models: the ideal
  bonded lug crowds the current at the contact edges (no in-sheet
  current inside an equipotential region), the pressed conductor ramps
  it across the contact area.*
- Fields are reported at the dialog's test current; power scales with I².
- **Skin effect (f > 0)**: per-layer effective sheet resistance from the
  exact 1D foil-diffusion solution `Zs = τρ·coth(τt)`, `τ = (1+j)/δ`
  (`SKIN_SIDES = 1` in config: plane facing a return plane; `2` =
  isolated foil), and the analogous correction for the 18 µm barrel wall.
  Enter one frequency per run (e.g. a switching harmonic, with its RMS
  amplitude as the test current) — suffixes `k`/`M` accepted.
  **Caveat:** only through-thickness crowding is modeled. Lateral
  (proximity-effect) redistribution needs a magneto-quasistatic solver
  and is not captured — since the resistance-driven distribution is the
  minimum-dissipation one, AC results are a rigorous **lower bound**.
  Rule of thumb for 70 µm foil: skin is negligible below ~300 kHz
  (δ = 173 µm at 142 kHz), ~+11 % at 1 MHz. At f > 0 the |J| maps are
  referenced to the skin-reduced conduction-equivalent thickness
  t/(R_AC/R_DC) — the density in the copper that actually conducts —
  not the geometric foil thickness.
- 5-point FDM per layer on an auto-sized shared grid (~2 M fine cells
  with the uniform grid; ~8 M with the adaptive grid, whose unknown
  count no longer scales with them). Direct sparse solve up to 500 k
  unknowns, AMG-preconditioned CG (pyamg) above — Jacobi-CG if pyamg is
  missing. Discretization error typically ≲ 2 % at defaults — halve the
  cell size and compare to judge convergence.
- **Adaptive cells** (dialog checkbox, **on by default**;
  `ADAPTIVE_CELLS`):
  solves on a 2:1-balanced quadtree — fine cells at copper boundaries,
  electrodes, traces, via mouths and buildup, blocks up to
  `ADAPTIVE_MAX_CELL_UM` (2 mm) in plane interiors (`ADAPTIVE_GUARD`
  sets the clearance a block needs to grow). The **minimum element size
  is the grid cell size itself** (auto / dialog / `CELL_UM_OVERRIDE`);
  the uniform limit reproduces the normal grid exactly. Large
  speed/memory wins on big pours. The raw coarse–fine interface flux
  bias (~0.5–2 % low) is removed by a **deferred-correction pass**
  (`ADAPTIVE_CORRECTION_PASSES`, default 1: reconstruct leaf gradients,
  move the tangential term to the RHS, re-solve on the reused
  factorization/AMG hierarchy) — measured residual deviation from the
  uniform grid ≲ 0.03 %, with the power-balance identity intact. All
  fields are expanded back to the fine grid for the maps and reports.

  ![Rasterized map with the adaptive mesh overlay](docs/img/demo-raster.png)
  *The raster map of the demo net: quadtree leaves drawn on the copper
  (fine at boundaries, electrodes, via mouths and pads; coarse blocks
  in plane interiors), the tin-gray solder coat of the THT-pad contact
  P1, and the via field with its pad copper.*

## Offline / development

Every run writes `geometry_dump.json`; re-solve without KiCad:

```powershell
.venv\Scripts\python.exe -m fill_resistance.standalone dump.json `
    [--current 40] [--cell-um 50] [--layers F.Cu,In1.Cu] [--no-show] `
    [--out DIR] [--force-iterative]
```

Dev environment, tests, headless extraction (Windows shown; on
Linux/macOS use `.venv/bin/python`):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe kicad-python numpy scipy pyamg matplotlib pytest
.venv\Scripts\python.exe -m pytest tests -q          # incl. exact analytic cases
.venv\Scripts\python.exe tools\api_probe.py          # IPC API probe vs live KiCad
.venv\Scripts\python.exe -m fill_resistance.board_io dump.json [NET]  # extract only
```

## Packaging / publishing

`python tools/build_package.py` builds the PCM addon zip in `dist/`
(installable right away via Plugin and Content Manager → *Install from
File*) plus `dist/metadata-registry.json` with the SHA-256 and sizes
filled in. To publish: upload the zip to a release, set `download_url`
(and the `homepage` resource in `metadata.json`), then submit the
registry copy as `packages/th.co.b4l.fill-resistance/metadata.json` in a
merge request to <https://gitlab.com/kicad/addons/metadata>. Icons are
regenerated with `python tools/gen_icons.py`; the README figures in
`docs/img/` with `.venv\Scripts\python.exe tools\gen_readme_figs.py`
(real solver output on small synthetic boards, plus the hand-drawn
hole cross-section).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Troubleshooting

- **No toolbar button**: venv still building (wait), or build failed →
  *Recreate Plugin Environment*; check the interpreter path (setup 2).
- **"Could not connect to KiCad's IPC API"**: API server not enabled, or
  KiCad not running (no headless mode in KiCad 10).
- **"KiCad is busy"**: a modal dialog is open in KiCad — close it, rerun.
- **Windows don't appear**: they may open behind KiCad (raised
  best-effort); PNGs are always saved regardless.
- **Result seems too low/high**: remember the model is fills + barrels
  only, with ideal contacts; measure electrode-to-electrode.
