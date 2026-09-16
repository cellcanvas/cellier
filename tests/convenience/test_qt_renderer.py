"""Tests for ``cellier.convenience.layout._qt_renderer``.

Closes the Qt-renderer parity hole (the anywidget renderer is covered by
``tests/v2/test_anywidget.py``). The appearance-controls builder is exercised
against a real ``Viewer`` + controller; the center/window builders use fake
leaf widgets (any object exposing ``.widget``) so no render surface is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.convenience import Viewer
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui import (
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
from cellier.convenience.layout._qt_renderer import (
    _wrap_dock_widget,
    render_qt,
)
from cellier.convenience.layout._spec import (
    AppearanceControls,
    Grid,
    HStack,
    Layout,
    VStack,
)
from cellier.convenience.layout._walk import render_center, render_dock
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


def _leaf():
    """A center leaf: ``compose(host)`` plus the ``.widget`` the host unwraps."""
    from PySide6 import QtWidgets

    leaf = SimpleNamespace(widget=QtWidgets.QLabel())
    leaf.compose = lambda host, _leaf=leaf: host.leaf(_leaf)
    return leaf


def _control_names(container):
    """What the panel calls each control it holds.

    A set, because these tests assert presence rather than order (that is
    ``test_qt_appearance_acceptance``'s job).
    """
    from tests.convenience._qt_acceptance import control_labels

    return set(control_labels(container))


# ---------------------------------------------------------------------------
# _render_appearance_controls_qt
# ---------------------------------------------------------------------------


def test_appearance_controls_builds_colormap_and_clim_groups(qtbot, image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["color_map", "clim"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert container is not None
    assert "Image" in _control_names(container)


def test_appearance_controls_explicit_clim_range(qtbot, image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(
            appearance=["clim"], clim_range=(0.0, 5.0)
        ),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert container is not None
    assert "Image" in _control_names(container)


def test_appearance_controls_multiscale_render_and_lod(qtbot, multiscale_image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=["render_mode", "lod_bias"]),
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert container is not None
    assert {"Image", "LOD bias"} <= _control_names(container)


def test_appearance_controls_placeholder_without_configs(qtbot, image_store):
    """A dock with nothing to drive still renders, so a later add can fill it."""
    from qtpy.QtWidgets import QLabel

    from cellier.convenience.layout._controls_dock import APPEARANCE_PLACEHOLDER

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    labels = [label.text() for label in container.findChildren(QLabel)]
    assert APPEARANCE_PLACEHOLDER in labels


# ---------------------------------------------------------------------------
# render_center, driven through QtLayoutHost
# ---------------------------------------------------------------------------


def test_render_center_leaf_returns_inner_widget(qtbot):
    leaf = _leaf()
    assert render_center(leaf, QtLayoutHost(), []) is leaf.widget


def test_render_center_hstack(qtbot):
    from PySide6 import QtWidgets

    container = render_center(HStack(items=[_leaf()]), QtLayoutHost(), [])
    assert isinstance(container.layout(), QtWidgets.QHBoxLayout)


def test_render_center_vstack(qtbot):
    from PySide6 import QtWidgets

    container = render_center(VStack(items=[_leaf()]), QtLayoutHost(), [])
    assert isinstance(container.layout(), QtWidgets.QVBoxLayout)


def test_render_center_grid(qtbot):
    from PySide6 import QtWidgets

    container = render_center(Grid(cells=[[_leaf(), None]]), QtLayoutHost(), [])
    assert isinstance(container.layout(), QtWidgets.QGridLayout)


def test_render_center_unrenderable_raises(qtbot):
    with pytest.raises(TypeError, match="Cannot render"):
        render_center(object(), QtLayoutHost(), [])


# ---------------------------------------------------------------------------
# _wrap_dock_widget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position, layout_cls",
    [("top", "QHBoxLayout"), ("bottom", "QHBoxLayout"), ("left", "QVBoxLayout")],
)
def test_wrap_dock_widget_orientation(qtbot, position, layout_cls):
    from PySide6 import QtWidgets

    container = _wrap_dock_widget(QtWidgets.QLabel(), position)
    assert type(container.layout()).__name__ == layout_cls


# ---------------------------------------------------------------------------
# _render_dock_qt
# ---------------------------------------------------------------------------


def test_render_dock_none_returns_none(qtbot, image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    assert render_dock(None, viewer, QtLayoutHost(), []) is None


def test_render_dock_stack_of_appearance(qtbot, image_store):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["color_map"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    rendered = render_dock(
        VStack(items=[AppearanceControls()]), viewer, QtLayoutHost(), []
    )

    assert rendered is not None


# ---------------------------------------------------------------------------
# render_qt
# ---------------------------------------------------------------------------


def test_render_qt_builds_window_with_dock(qtbot, image_store):
    from PySide6 import QtWidgets

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["color_map", "clim"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    leaf = _leaf()
    layout = Layout(center=leaf, right_dock=AppearanceControls())

    window = render_qt(layout, viewer)

    assert isinstance(window, QtWidgets.QMainWindow)
    assert window.centralWidget() is leaf.widget
    assert len(window.findChildren(QtWidgets.QDockWidget)) == 1
