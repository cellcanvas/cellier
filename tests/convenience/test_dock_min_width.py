"""``Layout(left_dock_min_width=..., right_dock_min_width=...)``.

A side dock's minimum width: the narrowest it may be, on both toolkits.  It
must be at least ``DOCK_MIN_WIDTH`` (260 px), which is also the default.  It
is a floor: a dock whose controls need more is as wide as they need.  On Qt
the dock separator can still drag it wider.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cellier.convenience import Viewer
from cellier.convenience._hosts import JupyterHost, MarimoHost, QtLayoutHost
from cellier.convenience.layout import Layout, RenderControls
from cellier.convenience.layout._spec import DOCK_MIN_WIDTH
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


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("width", [-10, 0, 150, DOCK_MIN_WIDTH - 1])
def test_a_width_below_the_floor_raises(side, width):
    with pytest.raises(
        ValueError, match=f"{side}_dock_min_width must be at least 260 px"
    ):
        Layout(
            center=object(),
            **{f"{side}_dock": _DOCK, f"{side}_dock_min_width": width},
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_floor_itself_is_accepted(side):
    layout = Layout(
        center=object(),
        **{f"{side}_dock": _DOCK, f"{side}_dock_min_width": DOCK_MIN_WIDTH},
    )
    assert layout.dock_min_widths()[side] == DOCK_MIN_WIDTH


def test_the_hosts_share_the_floor():
    assert QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH == DOCK_MIN_WIDTH == 260


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


def _content(dock):
    """The floored column inside a side dock's scroll area."""
    return dock.widget().widget()


def test_qt_dock_keeps_the_default_floor(qtbot):
    _window, docks = _qt_window(qtbot, right_dock=_DOCK)

    assert (
        _content(docks["right"]).minimumWidth() == QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH
    )


def test_qt_dock_takes_the_layout_width(qtbot):
    _window, docks = _qt_window(
        qtbot, left_dock=_DOCK, right_dock=_DOCK, right_dock_min_width=340
    )

    assert _content(docks["right"]).minimumWidth() == 340
    # The other side keeps the default.
    assert _content(docks["left"]).minimumWidth() == QtLayoutHost.DEFAULT_DOCK_MIN_WIDTH


@pytest.mark.parametrize("side", ["left", "right"])
def test_qt_dock_grows_past_its_floor_to_fit_its_controls(qtbot, side):
    """Controls that need more than the floor widen the dock, not clip.

    An explicit ``minimumWidth`` used to outvote the content's size hint, so
    the dock stayed at the floor and, with no horizontal scroll bar, the
    controls were squeezed and clipped at the right edge.
    """
    from PySide6 import QtWidgets

    from cellier.convenience.layout._qt_renderer import _scroll_dock_widget

    content = QtWidgets.QWidget()
    content.setMinimumWidth(DOCK_MIN_WIDTH)  # the floor ``assemble`` sets
    wide = QtWidgets.QWidget()
    wide.setMinimumWidth(500)
    QtWidgets.QVBoxLayout(content).addWidget(wide)
    area = _scroll_dock_widget(content)
    qtbot.addWidget(area)

    assert area.minimumSizeHint().width() >= 500

    # Narrow content keeps the floor.
    narrow = QtWidgets.QWidget()
    narrow.setMinimumWidth(DOCK_MIN_WIDTH)
    narrow_area = _scroll_dock_widget(narrow)
    qtbot.addWidget(narrow_area)
    assert DOCK_MIN_WIDTH <= narrow_area.minimumSizeHint().width() < 500


def test_qt_dock_drags_wider_but_not_narrower(qtbot):
    from PySide6.QtCore import Qt

    window, docks = _qt_window(qtbot, right_dock=_DOCK, right_dock_min_width=340)
    window.resize(1200, 800)
    window.show()
    qtbot.waitExposed(window)
    content = _content(docks["right"])

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

    # Both side docks sit in their scroll box; the floor rides on it, and a
    # dock with no width gets the shared default.
    left_item, middle_center, right_item = root.children
    assert left_item.min_width == DOCK_MIN_WIDTH
    assert list(left_item.children) == [left]
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

    _hstack, [center, (_s, (_s2, dock, inner_style), _outer)] = root
    assert (center, dock) == ("center", "dock")
    assert inner_style["min-width"] == "340px"


def test_marimo_floors_a_dock_with_no_width_at_the_default():
    host = MarimoHost.__new__(MarimoHost)  # bypass __init__ (no real marimo import)
    host._mo = SimpleNamespace(
        hstack=lambda items, align=None, **kw: ("hstack", list(items)),
        vstack=lambda items, align=None, **kw: ("vstack", list(items)),
        style=lambda item, style: ("style", item, style),
    )

    root = host.assemble("center", {"left": "dock"}, [], dock_min_widths={})

    _hstack, [(_s, (_s2, dock, inner_style), _outer), center] = root
    assert (center, dock) == ("center", "dock")
    assert inner_style["min-width"] == f"{DOCK_MIN_WIDTH}px"
