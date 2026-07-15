import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _uniform_grid_default(monkeypatch):
    """The exact-value test suite is the UNIFORM reference grid; the
    adaptive default (config.ADAPTIVE_CELLS = True) is pinned off here.
    Adaptive tests opt back in per test via monkeypatch."""
    from fill_resistance import config
    monkeypatch.setattr(config, "ADAPTIVE_CELLS", False)
