"""``Layout(left_dock_min_width=..., right_dock_min_width=...)``.

A side dock's minimum width: the narrowest it may be.  On Qt the dock
separator can still drag it wider; on anywidget, which has no splitter, it is
the dock's floor.  Unset, Qt keeps its 260 px default and anywidget leaves the
dock content-sized.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellier.convenience import Viewer
from cellier.convenience._hosts import JupyterHost, MarimoHost, QtLayoutHost
from cellier.convenience.layout import Layout, RenderControls
from cellier.scene.dims import spatial_axes

_DOCK = RenderControls(sections=("temporal",))


def _qt_leaf():
    """A center leaf: ``compose(host)`` plus the ``.widget`` the host unwraps."""
    from PySide6 import QtWidgets

    leaf = SimpleNamespace(widget=QtWidgets.QLabel())
    leaf.compose = lambda host, _leaf=leaf: host.leaf(_leaf)
    return leaf


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def test_widths_default_to_none():
    layout = Layout(center=object(), right_dock=_DOCK)

    assert layout.dock_min_widths() == {"left": None, "right": None}


@pytest.mark.parametrize("width", [True, 340.0, "340"])
def test_a_width_must_be_an_int(width):
    with pytest.raises(TypeError, match="right_dock_min_width must be an int"):
        Layout(center=object(), right_dock=_DOCK, right_dock_min_width=width)


@pytest.mark.parametrize("width", [0, -10])
def test_a_width_must_be_positive(width):
    with pytest.raises(ValueError, match="left_dock_min_width must be positive"):
        Layout(center=object(), left_dock=_DOCK, left_dock_min_width=width)


def test_a_width_needs_a_dock_on_its_side():
    with pytest.raises(ValueError, match="no left_dock"):
        Layout(center=object(), right_dock=_DOCK, left_dock_min_width=300)


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def _qt_window(qtbot, **layout_kwargs):
    from PySide6 import QtWidgets

    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    window = render_qt(Layout(center=_qt_leaf(), **layout_kwargs), viewer)
    qtbot.addWidget(window)
    docks = {
        dock.windowTitle().lower(): dock
        for dock in window.findChildren(QtWidgets.QDockWidget)
    }
    return window, docks


def test_qt_dock_keeps_the_default_floor(qtbot):
    _window, docks = _qt_window(qtbot, right_dock=_DOCK)

    assert docks["right"].widget().minimumWidth() == QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH


def test_qt_dock_takes_the_layout_width(qtbot):
    _window, docks = _qt_window(
        qtbot, left_dock=_DOCK, right_dock=_DOCK, right_dock_min_width=340
    )

    assert docks["right"].widget().minimumWidth() == 340
    # The other side keeps the default.
    assert docks["left"].widget().minimumWidth() == QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH


def test_qt_width_below_the_default_takes_effect(qtbot):
    """The 260 px default is a dock floor a caller replaces, not a column
    floor that would outvote a smaller width from inside the dock."""
    window, docks = _qt_window(qtbot, right_dock=_DOCK, right_dock_min_width=150)
    window.show()
    qtbot.waitExposed(window)

    assert docks["right"].minimumSizeHint().width() < (
        QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH
    )


def test_qt_dock_drags_wider_but_not_narrower(qtbot):
    from PySide6.QtCore import Qt

    window, docks = _qt_window(qtbot, right_dock=_DOCK, right_dock_min_width=340)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    content = docks["right"].widget()

    window.resizeDocks([docks["right"]], [100], Qt.Orientation.Horizontal)
    qtbot.wait(10)
    assert content.width() >= 340

    window.resizeDocks([docks["right"]], [600], Qt.Orientation.Horizontal)
    qtbot.wait(10)
    assert content.width() > 340


# ---------------------------------------------------------------------------
# anywidget hosts
# ---------------------------------------------------------------------------


def test_jupyter_floors_a_side_dock_in_a_box():
    from cellier.gui.anywidget import AnywidgetBox

    center, left, right = AnywidgetBox(), AnywidgetBox(), AnywidgetBox()

    root = JupyterHost().assemble(
        center,
        {"left": left, "right": right},
        [],
        dock_min_widths={"left": None, "right": 340},
    )

    left_item, middle_center, right_item = root.children
    assert left_item is left  # no width, no wrapper
    assert middle_center is center
    assert right_item.min_width == 340
    assert list(right_item.children) == [right]


def test_marimo_floors_a_side_dock_with_a_style():
    host = MarimoHost.__new__(MarimoHost)  # bypass __init__ (no real marimo import)
    host._mo = SimpleNamespace(
        hstack=lambda items, align=None, **kw: ("hstack", list(items)),
        vstack=lambda items, align=None, **kw: ("vstack", list(items)),
        style=lambda item, style: ("style", item, style),
    )

    root = host.assemble(
        "center", {"right": "dock"}, [], dock_min_widths={"right": 340}
    )

    assert root == ("hstack", ["center", ("style", "dock", {"min-width": "340px"})])
