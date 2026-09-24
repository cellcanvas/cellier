"""Side docks scroll vertically, so a dock never sets the window's height.

A dock's minimum height used to be the sum of its rows, and a ``QMainWindow``
is never shorter than its docks: a long controls column pushed the window
past the bottom of the screen, taking the dims sliders with it.  On
anywidget the dock scrolls within the height of the center column.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellier.convenience import Viewer
from cellier.convenience._hosts import JupyterHost, MarimoHost
from cellier.convenience.layout import Layout, RenderControls
from cellier.scene.dims import spatial_axes

_DOCK = RenderControls(sections=("temporal",))
_TALL = 2000


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def _leaf(widget):
    leaf = SimpleNamespace(widget=widget)
    leaf.compose = lambda host, _leaf=leaf: host.leaf(_leaf)
    return leaf


def _tall_column(n_rows=40):
    """A dock column far taller than any screen, with value controls in it."""
    from PySide6 import QtWidgets

    column = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(column)
    for row in range(n_rows):
        combo = QtWidgets.QComboBox()
        combo.addItems(["a", "b", "c"])
        form.addRow(f"row {row}", combo)
    column.setMinimumHeight(_TALL)
    return column


def _window(qtbot, column):
    from PySide6 import QtWidgets

    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    layout = Layout(center=_leaf(QtWidgets.QLabel("canvas")), right_dock=_DOCK)
    window = render_qt(layout, viewer)
    qtbot.addWidget(window)
    # Swap the right dock's content for the tall column, through the same
    # wrappers assemble() uses.
    from cellier.convenience.layout._qt_renderer import (
        _scroll_dock_widget,
        _wrap_dock_widget,
    )

    [right] = window.findChildren(QtWidgets.QDockWidget)
    content = _wrap_dock_widget(column, "right")
    content.setMinimumWidth(260)
    right.setWidget(_scroll_dock_widget(content))
    return window, right


def test_the_window_is_not_as_tall_as_the_dock(qtbot):
    window, dock = _window(qtbot, _tall_column())

    assert dock.minimumSizeHint().height() < 400
    assert window.minimumSizeHint().height() < 600
    window.resize(900, 600)
    window.show()
    qtbot.waitExposed(window)
    assert window.height() == 600
    assert dock.widget().verticalScrollBar().maximum() > 0


def test_the_real_dock_scrolls(qtbot):
    """assemble() itself wraps both side docks, not the top or bottom."""
    from PySide6 import QtWidgets

    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    layout = Layout(
        center=_leaf(QtWidgets.QLabel()),
        left_dock=_DOCK,
        right_dock=_DOCK,
        bottom_dock=_DOCK,
    )
    window = render_qt(layout, viewer)
    qtbot.addWidget(window)
    kinds = {
        dock.windowTitle().lower(): type(dock.widget()).__name__
        for dock in window.findChildren(QtWidgets.QDockWidget)
    }
    assert kinds["left"] == kinds["right"] == "_DockScrollArea"
    assert kinds["bottom"] != "_DockScrollArea"


def test_the_width_leaves_room_for_the_scroll_bar(qtbot):
    _win, dock = _window(qtbot, _tall_column())
    area = dock.widget()

    # The dock floor (260) is the content's width; the bar is added to it.
    assert area.widget().minimumWidth() == 260
    assert area.minimumSizeHint().width() == 260 + area._bar_width()


def _wheel(widget, dy=-120):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    centre = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        centre,
        QPointF(widget.mapToGlobal(centre.toPoint())),
        QPoint(0, 0),
        QPoint(0, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_the_wheel_over_an_unfocused_control_scrolls_the_dock(qtbot):
    from PySide6 import QtWidgets

    column = _tall_column()
    window, dock = _window(qtbot, column)
    window.resize(900, 600)
    window.show()
    qtbot.waitExposed(window)
    bar = dock.widget().verticalScrollBar()
    combo = column.findChildren(QtWidgets.QComboBox)[0]
    assert not combo.hasFocus()

    _wheel(combo)

    assert combo.currentIndex() == 0
    assert bar.value() > 0


def test_a_focused_control_takes_the_wheel(qtbot):
    from PySide6 import QtWidgets

    column = _tall_column()
    window, _dock = _window(qtbot, column)
    window.resize(900, 600)
    window.show()
    qtbot.waitExposed(window)
    spin = QtWidgets.QSpinBox()
    column.layout().insertRow(0, "spin", spin)
    window.activateWindow()
    # Only the active window has a focus widget, and activating one takes a
    # window manager: under a bare Xvfb (Linux CI) the window never becomes
    # active, so the spin box cannot take focus.
    try:
        qtbot.waitUntil(
            lambda: QtWidgets.QApplication.activeWindow() is window, timeout=2000
        )
    except qtbot.TimeoutError:
        pytest.skip("no window manager to activate the window")
    spin.setFocus()
    qtbot.waitUntil(spin.hasFocus)

    _wheel(spin, dy=120)

    assert spin.value() == 1


def test_a_control_added_later_is_guarded(qtbot):
    from PySide6 import QtWidgets

    column = _tall_column()
    window, dock = _window(qtbot, column)
    window.resize(900, 600)
    window.show()
    qtbot.waitExposed(window)
    # Built without a parent, then placed: what a rebuilt section does.
    late = QtWidgets.QWidget()
    inner = QtWidgets.QHBoxLayout(late)
    spin = QtWidgets.QSpinBox()
    inner.addWidget(spin)
    column.layout().insertRow(0, "late", late)
    qtbot.wait(10)
    bar = dock.widget().verticalScrollBar()

    _wheel(spin)

    assert spin.value() == 0
    assert bar.value() > 0


# ---------------------------------------------------------------------------
# anywidget hosts
# ---------------------------------------------------------------------------


def test_jupyter_wraps_each_side_dock_in_a_scroll_box():
    from cellier.gui.anywidget import AnywidgetBox

    center, left, right, top = (AnywidgetBox() for _ in range(4))

    root = JupyterHost().assemble(
        center,
        {"left": left, "right": right, "top": top},
        [],
        dock_min_widths={"left": None, "right": 340},
    )

    top_item, middle = root.children
    assert top_item is top  # top and bottom do not scroll
    left_item, middle_center, right_item = middle.children
    assert middle_center is center
    for item, dock, width in ((left_item, left, 260), (right_item, right, 340)):
        assert item.scroll
        assert item.min_width == width
        assert list(item.children) == [dock]


def test_marimo_wraps_each_side_dock_in_scroll_styles():
    host = MarimoHost.__new__(MarimoHost)  # bypass __init__ (no real marimo import)
    host._mo = SimpleNamespace(
        hstack=lambda items, align=None, **kw: ("hstack", list(items), align, kw),
        vstack=lambda items, align=None, **kw: ("vstack", list(items)),
        style=lambda item, style: ("style", item, style),
    )

    root = host.assemble("center", {"right": "dock"}, [], dock_min_widths={})

    _hstack, [center, outer], align, _kwargs = root
    assert center == "center"
    assert align == "stretch"
    _style, inner, outer_style = outer
    _style, dock, inner_style = inner
    assert dock == "dock"
    assert inner_style["overflow-y"] == "auto"
    assert "min-width" not in outer_style


# ---------------------------------------------------------------------------
# Top alignment: spare height goes below a short dock's controls
# ---------------------------------------------------------------------------


def test_short_side_docks_start_at_the_top(qtbot):
    """A ``dock_panel`` column (which asks to grow) and a plain stack both sit
    at the top margin at their natural height, not centred."""
    from PySide6 import QtWidgets

    from cellier.convenience.layout import VStack
    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    layout = Layout(
        center=_leaf(QtWidgets.QLabel()),
        left_dock=_DOCK,  # a dock_panel column, Expanding
        right_dock=VStack(items=[_DOCK]),  # a plain stack around it
    )
    window = render_qt(layout, viewer)
    qtbot.addWidget(window)
    window.resize(900, 1400)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(10)  # let the dock layouts run

    for dock in window.findChildren(QtWidgets.QDockWidget):
        area = dock.widget()
        box = area.widget().layout()
        column, below = box.itemAt(0), box.itemAt(1)
        # The controls come first, with nothing above them but the margin...
        assert column.widget() is not None
        assert column.geometry().top() < 20
        # ...and the spare height, most of this tall dock, is all below.
        assert below.spacerItem() is not None
        assert below.geometry().top() >= column.geometry().bottom()
        assert below.geometry().height() > area.viewport().height() / 2


def test_top_and_bottom_docks_stay_centred(qtbot):
    from PySide6 import QtWidgets

    from cellier.convenience.layout._qt_renderer import _wrap_dock_widget

    container = _wrap_dock_widget(QtWidgets.QLabel("x"), "bottom")
    box = container.layout()
    assert box.itemAt(0).spacerItem() is not None
    assert box.itemAt(2).spacerItem() is not None
