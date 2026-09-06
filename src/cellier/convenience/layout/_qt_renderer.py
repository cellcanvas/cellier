"""Qt renderer -- the view layer for Layout specs on the Qt backend."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.layout._walk import render_layout

if TYPE_CHECKING:
    from cellier.convenience.layout._spec import Layout


class _CellierMainWindow:
    """Mixin giving a ``QMainWindow`` the teardown ``_RenderView`` provides.

    The anywidget path hands the caller a ``DisplayHandle`` whose ``close()``
    unsubscribes every control it built.  The Qt path hands back a window and
    nothing else, so its controls stayed subscribed to the bus for as long as
    the controller lived -- **and kept being delivered events** -- even after
    the window was closed.  Measured: building and dropping a Qt viewer left
    ~30 widgets and its controller alive per cycle, growing linearly.

    Closing the window now closes those controls, which is what makes them
    emit ``closed`` and the controller drop their subscriptions.  It
    deliberately does not close the *controller*: a window is a view, and the
    viewer may outlive it.  Releasing the canvases and the controller is
    ``CellierController.close()``, exactly as on the anywidget side.
    """

    def _cellier_init(self) -> None:
        self._cellier_closeables: list = []
        self._cellier_torn_down = False

    def _cellier_teardown(self) -> None:
        if self._cellier_torn_down:
            return
        self._cellier_torn_down = True
        for obj in self._cellier_closeables:
            close = getattr(obj, "close", None)
            if close is None:
                continue
            with suppress(Exception):
                close()
        self._cellier_closeables.clear()


def make_window(QMainWindow):
    """Build the ``QMainWindow`` subclass, at call time so Qt stays optional."""

    class CellierMainWindow(_CellierMainWindow, QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self._cellier_init()

        def closeEvent(self, event) -> None:
            self._cellier_teardown()
            super().closeEvent(event)

    return CellierMainWindow


def render_qt(layout: Layout, viewer: object) -> object:
    """Render a Layout spec to a ``QMainWindow``.

    A wrapper: the walk is shared with every other backend
    (``convenience.layout._walk.render_layout``) and everything Qt-specific
    about it lives in :class:`~cellier.convenience._hosts.QtLayoutHost`.

    Parameters
    ----------
    layout : Layout
        The layout spec to render.
    viewer :
        The viewer whose recorded controls configs the docks are built from.

    Returns
    -------
    QMainWindow
    """
    return render_layout(layout, viewer, QtLayoutHost()).root


def _wrap_dock_widget(widget: object, position: str) -> object:
    """Center *widget* in a stretch container sized for *position*.

    Top/bottom docks: horizontal container (stretch | widget | stretch).
    Left/right docks: vertical container (stretch / widget / stretch).
    """
    from PySide6 import QtWidgets

    container = QtWidgets.QWidget()
    if position in ("top", "bottom"):
        box = QtWidgets.QHBoxLayout(container)
    else:
        box = QtWidgets.QVBoxLayout(container)
    box.setContentsMargins(4, 4, 4, 4)
    box.addStretch()
    box.addWidget(widget)
    box.addStretch()
    return container
