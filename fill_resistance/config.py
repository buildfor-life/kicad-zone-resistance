"""All tunable constants. v1 has no GUI dialog: edit here, re-run.

A future version may read overrides from <project>/fill_res_config.json.
"""

# --- Grid sizing ---
# Benchmarked on the VOUT+ plane (147x59 mm): R changes < 0.3% from
# 150 um cells down to 50 um; 1.7M unknowns direct-solve in ~17 s
# (raster ~20 s). Accuracy is feature-limited (slots/necks narrower than
# one cell), not plane-limited - override CELL_UM_OVERRIDE for boards
# with sub-cell slots.
TARGET_CELLS = 2_000_000        # auto cell size aims for roughly this many cells
HARD_MAX_CELLS = 16_000_000     # abort above this (see GridSizeError message)
MIN_CELL_UM = 25.0              # clamp for auto cell size
MAX_CELL_UM = 500.0
CELL_UM_OVERRIDE: float | None = None   # force a cell size, bypasses auto (not HARD_MAX)
MARGIN_CELLS = 2                # empty guard cells around the copper bbox

# --- Physics ---
RHO_CU_OHM_M = 1.68e-8          # copper resistivity at 20 degC
COPPER_THICKNESS_UM: float | None = None  # None -> stackup, fallback 35.0 with warning
FALLBACK_THICKNESS_UM = 35.0
TEST_CURRENT_A = 1.0            # default injected current (dialog/CLI-selectable)
VIA_PLATING_UM = 18.0           # barrel plating thickness (always plated).
                                # Capped vs uncapped vias do not change the
                                # layer-to-layer DC path: the >=5um cap sits
                                # over the hole mouth in parallel with the
                                # annular-ring contact, not in series.
INCLUDE_TH_PADS = True          # plated through-hole pads stitch layers too
SKIN_SIDES = 1                  # skin-effect field config: 1 = plane facing a
                                # return plane (conservative), 2 = isolated foil

# --- Solder / mask-opening buildup ---
INCLUDE_MASK_BUILDUP = False    # OFF by default; dialog-toggleable. Zones on
                                # F.Mask/B.Mask = mask openings that collect
                                # solder on the pour underneath
SOLDER_THICKNESS_UM = 50.0      # solder height over opened copper
SOLDER_RHO_OHM_M = 1.32e-7      # SAC305, ~7.9x copper
BUILDUP_EXTRA_CU_UM = 0.0       # optional user-added copper (busbar/wire
                                # soldered into the opening); dialog-settable

# --- Zone / layer selection ---
LAYER_HINT: str | None = None   # e.g. "F.Cu" to disambiguate candidate fills
ELECTRODE_POS_LAYER = "User.1"  # rectangles on this layer mark V+ contact parts
ELECTRODE_NEG_LAYER = "User.2"  # rectangles on this layer mark V- contact parts
ALWAYS_REFILL = False           # refill zones even if KiCad says they are filled

# --- Solver ---
CONTACT_MODEL = "uniform"       # "uniform": conductor pressed on top injects
                                # orthogonally with uniform surface density
                                # (J ramps across the contact); "equipotential":
                                # ideal bonded lug (Dirichlet). The two bracket
                                # a real contact: R_equi <= R_real <= R_uniform.
SPSOLVE_MAX_UNKNOWNS = 2_500_000  # above this, use CG (Jacobi) instead of
                                  # direct solve (measured: direct is ~14x
                                  # faster at 1.7M unknowns, ~3 GB peak)
CG_TOL = 1e-8
CG_MAXITER = 50_000             # CG iterations are cheap; large grids need many

# --- Geometry ---
ARC_TOL_FRACTION = 0.5          # arc sagitta tolerance as a fraction of cell size

# --- Plots / output ---
CMAP_POTENTIAL = "viridis"
CMAP_CURRENT = "inferno"
CMAP_POWER = "magma"
POWER_DYNAMIC_RANGE = 1e4       # LogNorm span for the power map (power ~ J^2)
LOG_CURRENT_SCALE = True
CURRENT_DYNAMIC_RANGE = 1e3     # LogNorm vmin = vmax / this
DPI = 150
INTERACTIVE = True              # False -> save PNGs only, never open windows
OUTPUT_DIRNAME = "fill_res_results"
