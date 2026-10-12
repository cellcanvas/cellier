"""End-to-end tests for the scene background: model -> controller -> render."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.events import BackgroundChangedEvent, BackgroundUpdateEvent
from cellier.scene._background import (
    DEFAULT_BOTTOM_COLOR,
    DEFAULT_TOP_COLOR,
    BackgroundAppearance,
)


@pytest.fixture
def controller():
    return CellierController()


def _material(controller: CellierController, scene_id):
    return controller._render_manager._scenes[scene_id]._background_material


def _bottom_top(controller: CellierController, scene_id):
    material = _material(controller, scene_id)
    return (
        np.asarray(material.color_bottom_left),
        np.asarray(material.color_top_left),
    )


def test_scene_starts_with_the_model_default(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    bottom, top = _bottom_top(controller, scene.id)

    np.testing.assert_allclose(bottom, DEFAULT_BOTTOM_COLOR, atol=1e-6)
    np.testing.assert_allclose(top, DEFAULT_TOP_COLOR, atol=1e-6)


def test_add_scene_accepts_a_background(controller):
    scene = controller.add_scene(
        name="main",
        displayed_axes=("y", "x"),
        background=BackgroundAppearance(mode="uniform", color=(1.0, 0.0, 0.0, 1.0)),
    )
    bottom, top = _bottom_top(controller, scene.id)

    np.testing.assert_allclose(bottom, (1.0, 0.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(top, (1.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_field_change_reaches_the_render_layer(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))

    scene.background.top_color = (1.0, 0.0, 0.0, 1.0)

    _, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(top, (1.0, 0.0, 0.0, 1.0), atol=1e-6)


def test_mode_change_reaches_the_render_layer(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    scene.background.color = (0.0, 0.0, 1.0, 1.0)

    scene.background.mode = "uniform"

    bottom, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(bottom, (0.0, 0.0, 1.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(top, (0.0, 0.0, 1.0, 1.0), atol=1e-6)


def test_visibility_change_reaches_the_render_layer(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))

    scene.background.visible = False

    assert controller._render_manager._scenes[scene.id].background.visible is False


def test_field_change_emits_the_bus_event(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    scene.background.top_color = (1.0, 0.0, 0.0, 1.0)

    assert len(seen) == 1
    assert seen[0].scene_id == scene.id
    assert seen[0].field_name == "top_color"
    assert seen[0].new_value == (1.0, 0.0, 0.0, 1.0)
    assert seen[0].background is scene.background
    assert seen[0].source_id == controller._id


def test_events_are_filtered_by_scene(controller):
    scene_a = controller.add_scene(name="a", displayed_axes=("y", "x"))
    scene_b = controller.add_scene(name="b", displayed_axes=("y", "x"))
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, entity_id=scene_a.id, owner_id=uuid4()
    )

    scene_b.background.top_color = (1.0, 0.0, 0.0, 1.0)
    assert seen == []

    scene_a.background.top_color = (0.0, 1.0, 0.0, 1.0)
    assert len(seen) == 1


def test_update_background_field_threads_the_source_id(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    widget_id = uuid4()
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    controller.update_background_field(
        scene.id, "bottom_color", (0.5, 0.5, 0.5, 1.0), source_id=widget_id
    )

    assert seen[-1].source_id == widget_id
    assert scene.background.bottom_color == (0.5, 0.5, 0.5, 1.0)
    bottom, _ = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(bottom, (0.5, 0.5, 0.5, 1.0), atol=1e-6)


def test_source_id_override_does_not_leak(controller):
    """A later change with no source_id falls back to the controller's ID."""
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    controller.update_background_field(
        scene.id, "bottom_color", (0.5, 0.5, 0.5, 1.0), source_id=uuid4()
    )
    scene.background.top_color = (0.25, 0.25, 0.25, 1.0)

    assert seen[-1].source_id == controller._id


def test_incoming_bus_drives_the_background(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    widget_id = uuid4()
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    controller.incoming_events.emit(
        BackgroundUpdateEvent(
            source_id=widget_id,
            scene_id=scene.id,
            field="top_color",
            value=(0.1, 0.2, 0.3, 1.0),
        )
    )

    assert scene.background.top_color == (0.1, 0.2, 0.3, 1.0)
    assert seen[-1].source_id == widget_id
    _, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(top, (0.1, 0.2, 0.3, 1.0), atol=1e-6)


def test_replacing_the_model_rewires_the_bridge(controller):
    """A wholesale replacement pushes once, then the new model stays live."""
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    original = scene.background
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    scene.background = BackgroundAppearance(
        bottom_color=(0.0, 1.0, 0.0, 1.0), top_color=(0.0, 0.0, 0.0, 1.0)
    )

    assert len(seen) == 1
    assert seen[-1].field_name is None
    bottom, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(bottom, (0.0, 1.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(top, (0.0, 0.0, 0.0, 1.0), atol=1e-6)

    # The new model drives the render layer ...
    scene.background.top_color = (1.0, 1.0, 0.0, 1.0)
    _, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(top, (1.0, 1.0, 0.0, 1.0), atol=1e-6)

    # ... and the replaced one no longer does.
    original.top_color = (1.0, 0.0, 1.0, 1.0)
    _, top = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(top, (1.0, 1.0, 0.0, 1.0), atol=1e-6)


def test_set_background_stamps_a_source_id(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    widget_id = uuid4()
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    controller.set_background(
        scene.id,
        BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 1.0, 1.0)),
        source_id=widget_id,
    )

    assert len(seen) == 1
    assert seen[-1].source_id == widget_id
    bottom, _ = _bottom_top(controller, scene.id)
    np.testing.assert_allclose(bottom, (0.0, 0.0, 1.0, 1.0), atol=1e-6)


def test_removing_a_scene_disconnects_the_bridge(controller):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    background = scene.background
    seen: list[BackgroundChangedEvent] = []
    controller._outgoing_events.subscribe(
        BackgroundChangedEvent, seen.append, owner_id=uuid4()
    )

    controller.remove_scene(scene.id)
    background.top_color = (1.0, 0.0, 0.0, 1.0)

    assert seen == []
    assert scene.id not in controller._scene_background_bridges


def test_background_survives_a_model_roundtrip(controller, tmp_path):
    scene = controller.add_scene(name="main", displayed_axes=("y", "x"))
    scene.background.mode = "uniform"
    scene.background.color = (0.25, 0.5, 0.75, 1.0)

    path = tmp_path / "viewer.json"
    controller.to_file(path)
    restored = CellierController.from_file(path)

    restored_scene = restored._model.scenes[scene.id]
    assert restored_scene.background.mode == "uniform"
    assert restored_scene.background.color == (0.25, 0.5, 0.75, 1.0)
    bottom, _ = _bottom_top(restored, scene.id)
    np.testing.assert_allclose(bottom, (0.25, 0.5, 0.75, 1.0), atol=1e-6)

    # A restored scene is wired for runtime updates like any other.
    restored_scene.background.color = (1.0, 0.0, 0.0, 1.0)
    bottom, _ = _bottom_top(restored, scene.id)
    np.testing.assert_allclose(bottom, (1.0, 0.0, 0.0, 1.0), atol=1e-6)

    restored_scene.background = BackgroundAppearance(
        bottom_color=(0.0, 1.0, 0.0, 1.0), top_color=(0.0, 0.0, 0.0, 1.0)
    )
    bottom, top = _bottom_top(restored, scene.id)
    np.testing.assert_allclose(bottom, (0.0, 1.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(top, (0.0, 0.0, 0.0, 1.0), atol=1e-6)
