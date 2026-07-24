"""fill_resistance.__version__ is the runtime source of the version:
it must match the packaging metadata (which is not deployed with the
plugin) and show up in summary.txt."""
import json
import re
from pathlib import Path

import fill_resistance
from fill_resistance import raster, report, solver
from tests.util import NM, strip_problem

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_packaging_metadata():
    meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    assert meta["versions"][0]["version"] == fill_resistance.__version__

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert m is not None
    assert m.group(1) == fill_resistance.__version__


def test_summary_shows_version(tmp_path):
    problem = strip_problem()
    stack = raster.rasterize_stack(problem, 1.0 * NM)
    e1, e2 = raster.electrode_masks(stack, problem)
    result = solver.run_solve(problem, stack, e1, e2, 1.0,
                              contact_model="equipotential")
    text = report.write_summary(tmp_path, problem, stack,
                                result).read_text(encoding="utf-8")
    assert fill_resistance.__version__ in text.splitlines()[0]
