"""Dims widgets show exactly ``scene.slider_axes`` minus the displayed axes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from cmap import Colormap

from cellier.convenience import Viewer
from cellier.convenience.gui import build_canvas_widget
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.gui._axis_values import ContinuousAxisValues
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageAppearance, InMemoryImageSingleAppearance

if TYPE_CHECKING:
    from cellier.events import DimsUpdateEvent

#: World ``(t, w, y, x)``: the 3D ``(w, y, x)`` image broadcasts over ``t``.
_WORLD = [("t", "time"), *spatial_axes("w", "y", "x")]

_AXIS_VALUES = {axis: ContinuousAxisValues(min=0.0, max=3.0) for axis in range(4)}


def _viewer(gui: str) -> Viewer:
    viewer = Viewer(_WORLD, dim="2d", gui=gui)
    store = ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32))
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map=Colormap("gray")),
    )
    return viewer


def _visible_sliders(gui: str, panel) -> set[int]:
    if gui == "qt":
        layout = panel.widget.layout()
        return {axis for axis, row in panel._rows.items() if layout.isRowVisible(row)}
    displayed = {int(a) for a in panel.displayed_axes}
    slider_axes = panel.slider_axes
    return {
        int(axis)
        for axis in panel.axis_values
        if int(axis) not in displayed
        and (slider_axes is None or int(axis) in slider_axes)
    }


@pytest.fixture(params=["qt", "anywidget"])
def gui(request, qtbot):
    if request.param == "qt":
        pytest.importorskip("qtpy")
    else:
        pytest.importorskip("anywidget")
    return request.param


def test_sliders_are_the_slider_axes_minus_the_displayed_axes(gui):
    viewer = _viewer(gui)
    canvas = build_canvas_widget(viewer, _AXIS_VALUES)
    panel = canvas.dims_control

    assert viewer.scene.slider_axes == (1, 2, 3)
    # t is broadcast (no slider); y and x are displayed.
    assert _visible_sliders(gui, panel) == {1}


def test_an_override_reaches_the_widget(gui):
    viewer = _viewer(gui)
    canvas = build_canvas_widget(viewer, _AXIS_VALUES)
    panel = canvas.dims_control

    viewer.controller.set_slider_override(viewer.scene.id, 0, True)
    assert _visible_sliders(gui, panel) == {0, 1}

    viewer.controller.set_slider_override(viewer.scene.id, 1, False)
    assert _visible_sliders(gui, panel) == {0}


async def test_the_toggle_emits_no_slice_indices(gui):
    viewer = _viewer(gui)
    canvas = build_canvas_widget(viewer, _AXIS_VALUES)
    panel = canvas.dims_control
    emitted: list[DimsUpdateEvent] = []
    panel.changed.connect(emitted.append)

    if gui == "qt":
        panel._on_toggle_click()
    else:
        panel._clicks += 1

    assert emitted[-1].displayed_axes == (1, 2, 3)
    assert emitted[-1].slice_indices is None
    assert viewer.scene.dims.selection.displayed_axes == (1, 2, 3)


def test_a_refused_toggle_leaves_the_widget_on_its_previous_mode(gui, monkeypatch):
    viewer = _viewer(gui)
    canvas = build_canvas_widget(viewer, _AXIS_VALUES)
    panel = canvas.dims_control

    def _refuse(*_args, **_kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(viewer.controller, "update_displayed_axes", _refuse)

    with pytest.raises(Exception, match="refused"):
        if gui == "qt":
            panel._on_toggle_click()
        else:
            panel._clicks += 1

    assert viewer.scene.dims.selection.displayed_axes == (2, 3)
    if gui == "qt":
        assert panel._displayed_axes == (2, 3)
        assert panel._toggle_button.text() == "Switch to 3D"
    else:
        assert list(panel.displayed_axes) == [2, 3]
        assert panel.label == "Switch to 3D"
