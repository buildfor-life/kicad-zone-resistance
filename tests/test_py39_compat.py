"""Python 3.9 compatibility tripwire.

KiCad's macOS builds bundle Python 3.9 and build the plugin venv with
it (README: Platform notes), while the dev environment runs a current
Python - so nothing else in the suite notices a construct that only
breaks on 3.9. The first real Mac run died at import: a module-level
`float | None` annotation in config.py, evaluated at runtime because
the file lacked the future import (PEP 604 unions need Python 3.10
unless annotations are deferred).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = sorted((ROOT / "fill_resistance").glob("*.py"))
SHIPPED.append(ROOT / "fill_res_action.py")


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.ImportFrom)
               and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names)
               for node in tree.body)


def _uses_annotations(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                return True
            a = node.args
            args = (a.posonlyargs + a.args + a.kwonlyargs
                    + ([a.vararg] if a.vararg else [])
                    + ([a.kwarg] if a.kwarg else []))
            if any(arg.annotation is not None for arg in args):
                return True
    return False


def test_annotated_modules_defer_annotations():
    offenders = []
    for path in SHIPPED:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _uses_annotations(tree) and not _has_future_annotations(tree):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} use annotations without 'from __future__ import "
        f"annotations': they are evaluated at import time and PEP 604 "
        f"unions crash on KiCad's macOS Python 3.9.")
