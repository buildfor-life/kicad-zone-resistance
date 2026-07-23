"""Regressions from the first macOS field test.

Two failures that only a non-Windows KiCad could produce: the board
directory was read-only (demo project opened straight from the mounted
installer image), and matplotlib picked TkAgg - macOS' bundled Python
ships tkinter, unlike KiCad's Windows Python - then refused to create
any figure because the PySide6 dialog already had a Qt event loop in
the process.
"""
import pathlib

from fill_resistance import plots, report


def test_backend_prefers_qt_over_tk():
    # The dev environment has both toolkits installed, so this asserts
    # the preference order for real: Qt must win, because the selection
    # dialog / progress window make the process a Qt process before the
    # first figure exists.
    assert plots._pick_backend() in ("QtAgg", "Qt5Agg")


def test_output_dir_falls_back_when_board_dir_unwritable(
        tmp_path, monkeypatch, capsys):
    board_dir = tmp_path / "board"
    board_dir.mkdir()
    real_mkdir = pathlib.Path.mkdir

    def deny_under_board(self, *args, **kwargs):
        if str(self).startswith(str(board_dir)):
            raise OSError(30, "Read-only file system", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", deny_under_board)
    out = report.make_output_dir(board_dir)
    assert out.is_dir()
    assert not str(out).startswith(str(board_dir))
    assert board_dir.name in out.name      # traceable back to the board
    assert "not writable" in capsys.readouterr().out


def test_output_dir_normal_case_unchanged(tmp_path):
    out = report.make_output_dir(tmp_path)
    assert out.is_dir()
    assert out.parent.parent == tmp_path
