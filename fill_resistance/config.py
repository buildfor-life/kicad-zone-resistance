"""All tunable constants. v1 has no GUI dialog: edit here, re-run.

A future version may read overrides from <project>/fill_res_config.json.
"""

# --- Grid sizing ---
# Benchmarked on the VOUT+ plane (147x59 mm): R changes < 0.3% from
# 150 um cells down to 50 um. Accuracy is feature-limited (slots/necks
# narrower than one cell), not plane-limited - override CELL_UM_OVERRIDE
# for boards with sub-cell slots.
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
VIA_PLATING_UM = 18.0           # barrel plating thickness (always plated)
VIAS_CAPPED = True              # filled + capped vias (dialog checkbox):
                                # outer-layer mouths carry a CAP_PLATING_UM
                                # thin copper cap, inner-layer mouths are
                                # holes. False = open mouths on all layers.
                                # Ring/pad copper of vias is modeled either
                                # way; THT-pad copper/drills are not.
CAP_PLATING_UM = 15.0           # cap plating thickness (fab spec)
CAP_MAX_DRILL_MM = 0.5          # fab caps only small vias: drills above this
                                # stay open even with VIAS_CAPPED
                                # (dialog-settable)
INCLUDE_TH_PADS = True          # plated through-hole pads stitch layers too;
                                # their holes are modeled solder-filled (a
                                # soldered component lead), so the solder core
                                # conducts in parallel with the plating
THT_LEAD_PROTRUSION_MM = 1.5    # clipped THT lead protrusion on the side
                                # opposite the component: a solder cone of
                                # this height at the drill wall (tapering to
                                # zero at the pad edge) wraps the lead of
                                # every populated THT pad. 0 = no cones
THT_LEAD_CLEARANCE_MM = 0.25    # hole diameter minus lead diameter (fab
                                # rule): a lead cylinder of drill - this
                                # conducts inside every solder-filled hole
THT_LEAD_RHO_OHM_M = 1.68e-8    # lead material resistivity: copper leads/
                                # wires; brass ~6.4e-8, phosphor bronze
                                # ~1.1e-7, copper-clad steel higher - raise
                                # this if your components use such leads
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
INCLUDE_TRACKS = True           # the net's traces (straight + arc tracks)
                                # conduct together with the zone fills;
                                # dialog-toggleable
TRACK_1D_FACTOR = 3.0           # traces narrower than this many grid cells
                                # become exact 1D resistor chains along their
                                # centerline instead of rasterized outlines
                                # (no discretization error in the trace R;
                                # 0 = always rasterize)
LAYER_HINT: str | None = None   # e.g. "F.Cu" to disambiguate candidate fills
ELECTRODE_POS_LAYER = "User.1"  # rectangles on this layer mark V+ contact parts
ELECTRODE_NEG_LAYER = "User.2"  # rectangles on this layer mark V- contact parts
ALWAYS_REFILL = False           # refill zones even if KiCad says they are filled

# --- Adaptive grid ---
ADAPTIVE_CELLS = True           # solve on a 2:1-balanced quadtree: fine at
                                # copper boundaries/electrodes/features,
                                # coarse plane interiors (dialog-toggleable).
                                # With the deferred-correction pass the
                                # deviation from the uniform grid is <0.03%
                                # measured; untick for the reference grid
TARGET_CELLS_ADAPTIVE = 8_000_000  # auto cell-size budget with the adaptive
                                # grid: unknowns no longer scale with the
                                # fine cell count, so the auto sizer picks
                                # a ~2x finer h (memory-bound: masks/fields)
ADAPTIVE_MAX_CELL_UM = 1000.0   # coarsest leaf edge length (also the
                                # granularity of the potential/field maps
                                # on plane interiors). The MINIMUM element
                                # size is the grid cell size itself (auto /
                                # dialog / CELL_UM_OVERRIDE). The guard
                                # distance caps leaf growth near features
ADAPTIVE_GUARD = 4              # a leaf of size s needs >= GUARD*s cells of
                                # clearance to the nearest feature
ADAPTIVE_CORRECTION_PASSES = 1  # deferred-correction re-solves fixing the
                                # coarse-fine interface flux bias (same
                                # matrix, reused factorization/AMG). 1 pass
                                # cuts the raw ~0.5-2% low bias to <0.03%
                                # measured; 0 disables

# --- Solver ---
CONTACT_MODEL = "uniform"       # "uniform": conductor pressed on top injects
                                # orthogonally with uniform surface density
                                # (J ramps across the contact); "equipotential":
                                # ideal bonded lug (Dirichlet). The two bracket
                                # a real contact: R_equi <= R_real <= R_uniform.
SPSOLVE_MAX_UNKNOWNS = 500_000  # above this, AMG-preconditioned CG (pyamg;
                                # Jacobi-CG if pyamg is missing). Direct is
                                # exact and fastest for small grids; measured
                                # at 1.4M unknowns: spsolve 13 s / ~3 GB,
                                # AMG-CG 6 s at a fraction of the memory
AMG_TOL = 1e-10                 # relative residual of the AMG-CG solve
CG_TOL = 1e-8                   # Jacobi-CG fallback (no pyamg)
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
