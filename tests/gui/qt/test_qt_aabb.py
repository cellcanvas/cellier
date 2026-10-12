"""Tests for the Qt ``QtAABBWidget`` AABB-controls widget."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from cellier.controller import CellierController
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.gui.qt.visuals._aabb import QtAABBWidget
from cellier.scene.dims import (
    spatial_axes,
    world_coordinate_system,
)


def _make_controller_with_visual():
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("y", "x"), name="world")
    scene = controller.add_scene(
        displayed_axes=("y", "x"), coordinate_system=cs, name="main"
    )
    data = np.zeros((16, 16), dtype=np.int32)
    store = LabelMemoryStore(data=data)
    visual = controller.add_labels(data=store, scene_id=scene.id)
    return controller, visual


def test_instantiate_smoke(qtbot):
    widget = QtAABBWidget(
        visual_id=object(), initial_enabled=True, initial_line_width=3.0
    )
    qtbot.addWidget(widget.widget)
    assert widget._enabled_check.isChecked() is True
    assert widget._line_width_spin.value() == pytest.approx(3.0)


def test_checkbox_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_visual()
    widget = QtAABBWidget(visual_id=visual.id, initial_enabled=False)
    qtbot.addWidget(widget.widget)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    assert visual.aabb.enabled is False
    widget._enabled_check.setChecked(True)
    assert visual.aabb.enabled is True


def test_line_width_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_visual()
    widget = QtAABBWidget(visual_id=visual.id)
    qtbot.addWidget(widget.widget)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    widget._line_width_spin.setValue(5.5)
    assert visual.aabb.line_width == pytest.approx(5.5)


def test_model_push_updates_control_without_reemit(qtbot):
    controller, visual = _make_controller_with_visual()
    widget = QtAABBWidget(visual_id=visual.id)
    qtbot.addWidget(widget.widget)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    emitted = []
    widget.changed.connect(emitted.append)

    controller.update_aabb_field(visual.id, "line_width", 7.0)

    assert widget._line_width_spin.value() == pytest.approx(7.0)
    assert emitted == []


def test_model_push_updates_swatch_color(qtbot):
    controller, visual = _make_controller_with_visual()
    widget = QtAABBWidget(visual_id=visual.id)
    qtbot.addWidget(widget.widget)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())

    controller.update_aabb_field(visual.id, "color", "#123456")

    assert widget._current_color == "#123456"
    assert "#123456" in widget._color_swatch.styleSheet()
