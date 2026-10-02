"""The half-thickness boxes of ``AnywidgetDimsPanel`` (model-sync pattern)."""

from __future__ import annotations

import pytest

pytest.importorskip("anywidget")

from cellier.controller import CellierController
from cellier.gui._axis_values import ContinuousAxisValues
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import Axis


def _panel(axes=None, thickness=None):
    controller = CellierController()
    cs = world_coordinate_system(axes or spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    if thickness:
        controller.update_thickness(scene.id, thickness)
    panel = AnywidgetDimsPanel.from_scene(
        scene,
        {axis: ContinuousAxisValues(min=0, max=9) for axis in range(scene.dims.ndim)},
    )
    controller.connect_widget(panel, subscription_specs=panel.subscription_specs())
    return controller, scene, panel


def test_the_panel_starts_from_the_scenes_thickness():
    _controller, _scene, panel = _panel(thickness={0: 1.5})
    assert panel.thickness_axes == [0, 1, 2]
    assert panel.thickness == {"0": 1.5, "1": 0.0, "2": 0.0}


def test_a_channel_axis_gets_no_box():
    _controller, _scene, panel = _panel(
        (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x"))
    )
    assert panel.thickness_axes == [1, 2, 3]
    assert set(panel.thickness) == {"1", "2", "3"}


def test_a_write_from_the_browser_reaches_the_scene():
    """What ``dims_panel.js`` does on a box's ``change``: one trait write."""
    _controller, scene, panel = _panel()
    panel.thickness = {**panel.thickness, "0": 2.0}
    assert scene.dims.selection.thickness == {0: 2.0}


def test_a_write_changes_only_the_axes_that_moved():
    _controller, scene, panel = _panel(
        (Axis(name="t", axis_type="time"), *spatial_axes("z", "y", "x")),
        thickness={0: 3.0},
    )
    panel.thickness = {**panel.thickness, "1": 1.0}
    assert scene.dims.selection.thickness == {0: 3.0, 1: 1.0}


def test_the_panel_follows_the_model():
    controller, scene, panel = _panel()
    controller.update_thickness(scene.id, {1: 4.0})
    assert panel.thickness == {"0": 0.0, "1": 4.0, "2": 0.0}
    controller.update_thickness(scene.id, {})
    assert panel.thickness == {"0": 0.0, "1": 0.0, "2": 0.0}


def test_the_panels_own_echo_is_not_submitted_again():
    controller, scene, panel = _panel()
    seen = []
    panel.changed.connect(seen.append)
    controller.update_thickness(scene.id, {0: 1.0})
    assert seen == []
