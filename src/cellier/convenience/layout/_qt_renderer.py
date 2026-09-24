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
    """Place *widget* in a stretch container sized for *position*.

    Top/bottom docks: horizontal container (stretch | widget | stretch), so
    the controls are centred along the dock.  Left/right docks: vertical
    container (widget / stretch), so the controls start at the top and the
    spare height goes below them.  The stretch has a stretch factor and the
    widget does not, so all of that height goes to the stretch even when the
    widget's size policy asks to grow (a ``dock_panel`` column does).
    """
    from PySide6 import QtWidgets

    container = QtWidgets.QWidget()
    if position in ("top", "bottom"):
        box = QtWidgets.QHBoxLayout(container)
        box.setContentsMargins(4, 4, 4, 4)
        box.addStretch()
        box.addWidget(widget)
        box.addStretch()
        return container
    box = QtWidgets.QVBoxLayout(container)
    box.setContentsMargins(4, 4, 4, 4)
    box.addWidget(widget)
    box.addStretch(1)
    return container


#: The least height a scrolling side dock asks for, in logical pixels: enough
#: to show that there is something to scroll, never the height of its content.
DOCK_SCROLL_MIN_HEIGHT = 120


def _scroll_dock_widget(content: object) -> object:
    """Put a side dock's *content* in a vertical-only scroll area.

    Without it a dock's minimum height is the sum of its rows, and a
    ``QMainWindow`` is never shorter than its docks: a long controls column
    made the window taller than the screen, taking the canvas (and the dims
    sliders under it) past the bottom edge.  In the scroll area the dock
    asks for :data:`DOCK_SCROLL_MIN_HEIGHT` and scrolls the rest.

    The width does not scroll: the area is as wide as *content* needs plus
    the scroll bar, so the bar never covers the controls.  What it needs is
    the larger of its explicit ``minimumWidth`` (the dock floor ``assemble``
    sets) and its minimum size hint.  Qt's own rule lets an explicit minimum
    win over the hint, which kept the dock at the floor while its controls
    needed more; with no horizontal scroll bar they were then squeezed below
    their minimum and clipped at the right edge.

    A combo box, spin box or slider under the pointer takes wheel events
    only while it has keyboard focus; otherwise the wheel scrolls the dock
    (see ``_WheelGuard``).
    """
    from qtpy.QtCore import QEvent, QObject, QSize, Qt
    from qtpy.QtWidgets import (
        QAbstractSpinBox,
        QComboBox,
        QFrame,
        QScrollArea,
        QSlider,
        QStyle,
        QWidget,
    )

    guarded = (QAbstractSpinBox, QComboBox, QSlider)

    class _WheelGuard(QObject):
        """Send the wheel over an unfocused value control to the dock.

        Installed on every widget in the dock, and on each widget added
        later: a section rebuilt when a visual is added brings new
        controls.  A filter has to sit on the control itself, because the
        control consumes the wheel before any parent sees it.
        """

        def __init__(self, area) -> None:
            super().__init__(area)
            self._area = area

        def guard(self, widget) -> None:
            widget.installEventFilter(self)
            for child in widget.findChildren(QWidget):
                child.installEventFilter(self)

        def eventFilter(self, obj, event) -> bool:
            kind = event.type()
            if kind == QEvent.Type.ChildAdded:
                child = event.child()
                if child.isWidgetType():
                    self.guard(child)
            elif (
                kind == QEvent.Type.Wheel
                and isinstance(obj, guarded)
                and not obj.hasFocus()
            ):
                self._area.scroll_by_wheel(event)
                return True
            return False

    class _DockScrollArea(QScrollArea):
        def __init__(self) -> None:
            super().__init__()
            self.setWidgetResizable(True)
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._guard = _WheelGuard(self)

        def set_content(self, widget) -> None:
            self.setWidget(widget)
            self._guard.guard(widget)

        def scroll_by_wheel(self, event) -> None:
            from qtpy.QtWidgets import QApplication

            QApplication.sendEvent(self.verticalScrollBar(), event)

        def _bar_width(self) -> int:
            style = self.style()
            if style.styleHint(QStyle.StyleHint.SH_ScrollBar_Transient, None, self):
                return 0  # an overlay bar takes no room from the content
            return style.pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent, None, self)

        def minimumSizeHint(self) -> QSize:
            inner = self.widget()
            width = 0
            if inner is not None:
                # The larger of the floor and what the controls need: an
                # explicit minimum alone would outvote the hint (see above).
                width = max(inner.minimumWidth(), inner.minimumSizeHint().width())
            width += self._bar_width() + 2 * self.frameWidth()
            return QSize(width, DOCK_SCROLL_MIN_HEIGHT)

        def sizeHint(self) -> QSize:
            hint = super().sizeHint()
            return QSize(self.minimumSizeHint().width(), hint.height())

    area = _DockScrollArea()
    area.set_content(content)
    return area
