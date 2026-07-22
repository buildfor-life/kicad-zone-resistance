"""Busy window for the stretch between the dialog closing and the
figures appearing.

The solve is seconds to minutes on a real board, and until now nothing
was on screen for it: the dialog vanished on OK and the plugin looked
like it had done nothing. This puts a small always-on-top window up for
that stretch - current stage, elapsed time, and a Cancel button.

The state is module-level rather than an object threaded through the
call chain: the linear solve is where the time actually goes, and it
calls tick() from inside a scipy/pyamg iteration callback several
frames deep. Inactive until start() succeeds, so every call is a no-op
for the standalone runner and the tests.

Qt only repaints when the event loop runs, and the solve owns the
thread, so tick() pumps events itself. That is also where a click on
Cancel is noticed - it raises Cancelled at the next tick.
"""
from __future__ import annotations

import time

_win = None
_label = None
_text = ""
_t0 = 0.0
_last = 0.0
_cancelled = False

TICK_INTERVAL_S = 0.05          # ~20 fps: enough to look alive, cheap


class Cancelled(Exception):
    """The user closed the progress window. Not a failure - the caller
    reports it like a cancelled dialog, with no error figure."""


def start(title: str = "Fill Resistance") -> bool:
    """Show the window. False (and inert) if Qt is unavailable."""
    global _win, _label, _t0, _last, _cancelled, _text
    if _win is not None:
        return True
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QApplication, QDialog,
                                       QDialogButtonBox, QLabel,
                                       QProgressBar, QVBoxLayout)
    except Exception:
        return False
    try:
        app = QApplication.instance() or QApplication([])
        win = QDialog()
        win.setWindowTitle(title)
        win.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        # no close button: closing is Cancel, and Cancel is the only way
        # to stop a solve that owns the thread
        win.setWindowFlag(Qt.WindowCloseButtonHint, False)

        label = QLabel("starting ...")
        bar = QProgressBar()
        bar.setRange(0, 0)                  # indeterminate: no total to show
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(bar)
        layout.addWidget(buttons)
        win.setLayout(layout)

        buttons.rejected.connect(_cancel)
        win.rejected.connect(_cancel)
        win.setMinimumWidth(340)
        win.show()
        win.raise_()
        win.activateWindow()
        app.processEvents()
    except Exception:
        return False

    _win, _label, _t0, _last, _cancelled, _text = win, label, \
        time.monotonic(), 0.0, False, ""
    return True


def _cancel() -> None:
    global _cancelled
    _cancelled = True


def stage(text: str, echo: bool = True) -> None:
    """Name the phase now running. Always repaints - stages are rare.

    echo=False for phases that already print their own line (saving a
    PNG prints the path), so the window updates without doubling stdout.
    """
    global _text
    _text = text
    if echo:
        print(text)
    if _win is not None:
        _refresh()


def tick() -> None:
    """Called from inside the solve. Throttled, so it is safe to call
    every iteration."""
    global _last
    if _win is None:
        return
    now = time.monotonic()
    if now - _last < TICK_INTERVAL_S:
        return
    _last = now
    _refresh()


def _refresh() -> None:
    from PySide6.QtWidgets import QApplication

    elapsed = time.monotonic() - _t0
    if _label is not None:
        _label.setText(f"{_text}\n{elapsed:.0f} s elapsed")
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    if _cancelled:
        raise Cancelled()


def done() -> None:
    """Take the window down. Idempotent - callers use it in a finally."""
    global _win, _label, _text, _cancelled
    win, _win, _label, _text = _win, None, None, ""
    _cancelled = False
    if win is None:
        return
    try:
        win.close()
        win.deleteLater()
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass
