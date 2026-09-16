"""Tests for the Qt path of ``cellier.convenience.gui._ortho``.

The anywidget path is covered by ``tests/v2/test_anywidget.py``; this covers
``build_ortho_grid_widget(gui="qt")`` -> ``_build_qt_ortho_grid``, the
``OrthoCanvasWidgets`` teardown, and the gui-resolution ``ValueError`` guards.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.convenience import (
    OrthoViewer,
    axis_values_from_ortho,
)
from cellier.convenience.gui import (
    OrthoCanvasWidgets,
    build_ortho_grid_widget,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance

_PANELS = {"xy", "xz", "yz", "vol"}


def _ortho_with_image(image_store, gui="qt"):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui=gui)
    ortho.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    return ortho


def test_build_qt_ortho_grid_has_four_panels(qtbot, image_store):
    ortho = _ortho_with_image(image_store)
    ranges = axis_values_from_ortho(ortho)

    widgets = build_ortho_grid_widget(ortho, ranges, gui="qt")

    assert isinstance(widgets, OrthoCanvasWidgets)
    assert set(widgets.canvases) == _PANELS
    assert widgets.widget is not None


def test_ortho_grid_defaults_gui_from_viewer(qtbot, image_store):
    ortho = _ortho_with_image(image_store)
    ranges = axis_values_from_ortho(ortho)

    # gui=None -> resolves to ortho.gui ("qt").
    widgets = build_ortho_grid_widget(ortho, ranges)
    assert isinstance(widgets, OrthoCanvasWidgets)


def test_ortho_canvas_widgets_close(qtbot, image_store):
    ortho = _ortho_with_image(image_store)
    ranges = axis_values_from_ortho(ortho)
    widgets = build_ortho_grid_widget(ortho, ranges, gui="qt")

    # Teardown unsubscribes every panel; should not raise.
    widgets.close()


def test_gui_conflict_raises(qtbot, image_store):
    ortho = _ortho_with_image(image_store, gui="qt")
    ranges = axis_values_from_ortho(ortho)

    with pytest.raises(ValueError, match="conflicts with"):
        build_ortho_grid_widget(ortho, ranges, gui="anywidget")
