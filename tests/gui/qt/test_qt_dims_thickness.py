"""The "+/-" half-thickness box on each slider row of ``QtDimsControl``."""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui._dims import thickness_axes
from cellier.gui.qt._scene import QtDimsControl
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import Axis


def _scene(axes=None):
    controller = CellierController()
    cs = world_coordinate_system(axes or spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    return controller, scene


def _control(controller, scene, qtbot, **kwargs) -> QtDimsControl:
    selection = scene.dims.selection
    control = QtDimsControl(
        scene_id=scene.id,
        axis_values={
            axis: ContinuousAxisValues(min=0, max=9, decimals=1, step_size=0.5)
            for axis in range(scene.dims.ndim)
        },
        axis_labels=dict(enumerate(scene.dims.axis_labels)),
        initial_slice_indices=dict(selection.slice_indices),
        initial_displayed_axes=selection.displayed_axes,
        **kwargs,
    )
    qtbot.addWidget(control.widget)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())
    return control


def test_every_axis_gets_a_box_starting_at_the_scenes_thickness(qtbot):
    controller, scene = _scene()
    control = _control(controller, scene, qtbot, initial_thickness={0: 1.5})
    assert set(control._thickness_boxes) == {0, 1, 2}
    box = control._thickness_boxes[0]
    assert box.value() == 1.5
    assert box.minimum() == 0.0
    # The axis's own precision and step.
    assert box.decimals() == 1
    assert box.singleStep() == 0.5
    assert control._thickness_boxes[1].value() == 0.0


def test_the_box_writes_the_scenes_thickness(qtbot):
    controller, scene = _scene()
    control = _control(controller, scene, qtbot)
    control._thickness_boxes[0].setValue(2.0)
    assert scene.dims.selection.thickness == {0: 2.0}


def test_a_box_changes_only_its_own_axis(qtbot):
    """The controller replaces the whole mapping; the box names one axis."""
    controller, scene = _scene(
        (Axis(name="t", axis_type="time"), *spatial_axes("z", "y", "x"))
    )
    controller.update_thickness(scene.id, {0: 3.0})
    control = _control(controller, scene, qtbot, initial_thickness={0: 3.0})
    control._thickness_boxes[1].setValue(1.0)
    assert scene.dims.selection.thickness == {0: 3.0, 1: 1.0}


def test_the_box_follows_the_model(qtbot):
    controller, scene = _scene()
    control = _control(controller, scene, qtbot)
    controller.update_thickness(scene.id, {0: 4.0})
    assert control._thickness_boxes[0].value() == 4.0
    # An axis dropped from the mapping is a plane again.
    controller.update_thickness(scene.id, {})
    assert control._thickness_boxes[0].value() == 0.0


def test_the_box_cannot_go_negative(qtbot):
    controller, scene = _scene()
    control = _control(controller, scene, qtbot)
    control._thickness_boxes[0].setValue(-1.0)
    assert control._thickness_boxes[0].value() == 0.0
    assert scene.dims.selection.thickness.get(0, 0.0) == 0.0


def test_only_the_named_axes_get_a_box(qtbot):
    controller, scene = _scene()
    control = _control(controller, scene, qtbot, thickness_axes=(0,))
    assert set(control._thickness_boxes) == {0}


def test_a_discrete_row_gets_a_box_too(qtbot):
    controller, scene = _scene()
    selection = scene.dims.selection
    control = QtDimsControl(
        scene_id=scene.id,
        axis_values={0: DiscreteAxisValues(values=(0.0, 1.0, 2.0))},
        axis_labels={0: "z"},
        initial_slice_indices=dict(selection.slice_indices),
        initial_displayed_axes=selection.displayed_axes,
    )
    qtbot.addWidget(control.widget)
    controller.connect_widget(control, subscription_specs=control.subscription_specs())
    control._thickness_boxes[0].setValue(1.0)
    assert scene.dims.selection.thickness == {0: 1.0}


def test_a_scene_gives_no_box_to_a_channel_axis(qtbot):
    """``QtCanvasWidget.from_scene_and_canvas`` passes these axes."""
    controller, scene = _scene(
        (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x"))
    )
    assert thickness_axes(scene) == (1, 2, 3)
    control = _control(controller, scene, qtbot, thickness_axes=thickness_axes(scene))
    assert set(control._thickness_boxes) == {1, 2, 3}
