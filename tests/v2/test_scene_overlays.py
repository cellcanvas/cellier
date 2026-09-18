"""Scene overlays end to end: model -> controller -> render layer.

Covers the world bounds a ``SceneBoundingBox`` is built from, every rebuild
trigger in ``plans/scene_overlay_implementation.md``, the overlay event pair,
removal and teardown, and restoring overlays from a serialized model.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest
from transformnd.transforms.affine import Affine

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events import (
    CanvasAddedEvent,
    OverlayChangedEvent,
    OverlayUpdateEvent,
)
from cellier.scene._bounds import scene_world_bounds, visual_world_bounds
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    CenteredAxes2D,
    SceneBoundingBox,
    SceneBoundingBoxAppearance,
)
from cellier.visuals._image_memory import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
)


def _appearance() -> InMemoryImageAppearance:
    return InMemoryImageAppearance()


def _scene(controller: CellierController, dim: str = "3d", world=None):
    return controller.add_scene(
        coordinate_system=world or spatial_axes("z", "y", "x"), dim=dim
    )


def _image(controller, scene, shape=(4, 6, 8), **kwargs):
    store = ImageMemoryStore(data=np.zeros(shape, dtype=np.float32))
    return controller.add_image(
        store,
        scene.id,
        appearance=_appearance(),
        single=InMemoryImageSingleAppearance(clim=(0.0, 1.0)),
        **kwargs,
    )


def _gfx(controller, overlay):
    return controller._overlays[overlay.id].gfx


def _box_extent(controller, overlay) -> tuple[np.ndarray, np.ndarray]:
    """The drawn box's pygfx-space (min, max) corners."""
    positions = np.asarray(_gfx(controller, overlay).node.geometry.positions.data)
    return positions.min(axis=0), positions.max(axis=0)


def _record(controller) -> list:
    events: list = []
    controller._outgoing_events.subscribe(OverlayChangedEvent, events.append)
    return events


@pytest.fixture
def controller():
    return CellierController(gui="offscreen")


# ---------------------------------------------------------------------------
# World bounds
# ---------------------------------------------------------------------------


def test_bounds_use_the_edge_convention_and_the_transform(controller):
    scene = _scene(controller)
    store = ImageMemoryStore(data=np.zeros((4, 6, 8), dtype=np.float32))
    transform = controller.data_to_world(
        scene.id, store, scale=(2.0, 1.0, 0.5), translation=(10.0, 0.0, -1.0)
    )
    controller.add_image(
        store,
        scene.id,
        appearance=_appearance(),
        single=InMemoryImageSingleAppearance(clim=(0.0, 1.0)),
        transform=transform,
    )

    low, high = scene_world_bounds(scene, controller.get_data_store)

    np.testing.assert_allclose(low, (10.0 - 1.0, -0.5, -1.0 - 0.25))
    np.testing.assert_allclose(high, (10.0 + 7.0, 5.5, -1.0 + 3.75))


def test_bounds_take_the_union_of_every_visual_hidden_included(controller):
    scene = _scene(controller)
    _image(controller, scene)
    points = PointsMemoryStore(
        positions=np.array([[-3.0, 1.0, 2.0], [20.0, 2.0, 3.0]], dtype=np.float32)
    )
    visual = controller.add_points(points, scene.id)
    controller.set_visual_visible(visual.id, False)

    low, high = scene_world_bounds(scene, controller.get_data_store)

    np.testing.assert_allclose(low, (-3.0, -0.5, -0.5))
    np.testing.assert_allclose(high, (20.0, 5.5, 7.5))


def test_bounds_of_an_empty_scene_are_none(controller):
    scene = _scene(controller)
    assert scene_world_bounds(scene, controller.get_data_store) is None


def test_every_corner_is_mapped_so_a_rotation_is_exact():
    """Two opposite corners would understate a rotated box; all four do not."""
    from cellier.transform import CoordinateSystem

    data = CoordinateSystem(name="data", axes=spatial_axes("y", "x"))
    world = CoordinateSystem(name="world", axes=spatial_axes("y", "x"))
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform = AffineTransform(
        input_coordinate_system=data.id,
        output_coordinate_system=world.id,
        transform=Affine(matrix),
    )

    low, high = visual_world_bounds(transform, ((0.0, 1.0), (0.0, 1.0)), world)

    # The unit square rotated by 45 degrees spans x in [-sqrt(2)/2, sqrt(2)/2]
    # along the first axis and [0, sqrt(2)] along the second.
    np.testing.assert_allclose(low, (-s, 0.0), atol=1e-12)
    np.testing.assert_allclose(high, (s, c + s), atol=1e-12)


def test_a_broadcast_axis_contributes_nothing(controller):
    """A broadcast axis maps to its translation, which is not an extent."""
    world = [("c", "channel"), *spatial_axes("z", "y", "x")]
    scene = _scene(controller, world=world)
    image = _image(controller, scene)
    store = controller.get_data_store(UUID(image.data_store_id))
    data = store.data_coordinate_system
    world_system = scene.dims.world_coordinate_system
    image.transform = AffineTransform.from_axis_map(
        data,
        world_system,
        axis_map={
            data.axes[0].id: world_system.axes[1].id,
            data.axes[1].id: world_system.axes[2].id,
            data.axes[2].id: world_system.axes[3].id,
        },
        broadcast_output_axes=(world_system.axes[0].id,),
    )

    low, high = scene_world_bounds(scene, controller.get_data_store)

    assert np.isnan(low[0]) and np.isnan(high[0])
    np.testing.assert_allclose(low[1:], (-0.5, -0.5, -0.5))
    np.testing.assert_allclose(high[1:], (3.5, 5.5, 7.5))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_a_3d_box_has_twelve_edges_in_pygfx_order(controller):
    scene = _scene(controller)
    _image(controller, scene)

    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    node = _gfx(controller, box).node
    assert node.geometry.positions.data.shape == (24, 3)
    low, high = _box_extent(controller, box)
    # World (z, y, x) = (4, 6, 8) voxels, drawn as pygfx (x, y, z).
    np.testing.assert_allclose(low, (-0.5, -0.5, -0.5))
    np.testing.assert_allclose(high, (7.5, 5.5, 3.5))
    assert node.visible
    assert node.material.depth_test
    assert not node.material.depth_write
    assert not node.material.pick_write
    assert node.render_order == 1


def test_a_box_added_before_any_visual_appears_with_the_first(controller):
    scene = _scene(controller)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    assert not _gfx(controller, box).node.visible

    _image(controller, scene)

    assert _gfx(controller, box).node.visible
    np.testing.assert_allclose(_box_extent(controller, box)[1], (7.5, 5.5, 3.5))


def test_removing_a_visual_shrinks_the_box(controller):
    scene = _scene(controller)
    _image(controller, scene)
    points = controller.add_points(
        PointsMemoryStore(
            positions=np.array([[0.0, 0.0, 0.0], [30.0, 2.0, 3.0]], dtype=np.float32)
        ),
        scene.id,
    )
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    assert _box_extent(controller, box)[1][2] == pytest.approx(30.0)

    controller.remove_visual(points.id)

    assert _box_extent(controller, box)[1][2] == pytest.approx(3.5)


def test_replacing_a_transform_moves_the_box(controller):
    scene = _scene(controller)
    image = _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    store = controller.get_data_store(UUID(image.data_store_id))
    image.transform = controller.data_to_world(
        scene.id, store, translation=(100.0, 0.0, 0.0)
    )

    low, high = _box_extent(controller, box)
    assert low[2] == pytest.approx(99.5)
    assert high[2] == pytest.approx(103.5)


def test_a_2d_view_draws_a_four_edge_frame_without_depth_test(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    controller.set_displayed_axes(scene.id, (1, 2))

    node = _gfx(controller, box).node
    positions = np.asarray(node.geometry.positions.data)
    assert positions.shape == (8, 3)
    np.testing.assert_allclose(positions[:, 2], 0.0)
    np.testing.assert_allclose(positions.min(axis=0)[:2], (-0.5, -0.5))
    np.testing.assert_allclose(positions.max(axis=0)[:2], (7.5, 5.5))
    assert not node.material.depth_test

    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert node.geometry.positions.data.shape == (24, 3)
    assert node.material.depth_test


def test_a_slice_move_does_not_rebuild(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    controller.set_displayed_axes(scene.id, (1, 2))
    geometry = _gfx(controller, box).node.geometry

    controller.update_slice_indices(scene.id, {0: 2.0})

    assert _gfx(controller, box).node.geometry is geometry


def test_a_store_extent_change_moves_the_box(controller):
    """Stores announce their own changes; no reslice_visual call is needed."""
    scene = _scene(controller)
    store = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    controller.add_points(store, scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    store.positions = np.array([[0.0, 0.0, 0.0], [9.0, 1.0, 1.0]], dtype=np.float32)

    assert _box_extent(controller, box)[1][2] == pytest.approx(9.0)


def test_an_in_place_edit_moves_the_box_once_announced(controller):
    scene = _scene(controller)
    store = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    controller.add_points(store, scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    store.positions[1, 0] = 9.0
    assert _box_extent(controller, box)[1][2] == pytest.approx(1.0)  # unseen

    store.notify_changed("extent")
    assert _box_extent(controller, box)[1][2] == pytest.approx(9.0)


def test_a_contents_change_does_not_rebuild_the_box(controller):
    scene = _scene(controller)
    store = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    controller.add_points(store, scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    geometry = _gfx(controller, box).node.geometry

    store.colors = np.ones((2, 4), dtype=np.float32)

    assert _gfx(controller, box).node.geometry is geometry


def test_reslice_visual_no_longer_refreshes_the_box(controller):
    """reslice_visual means "load data"; trail and LOD edits call it too."""
    scene = _scene(controller)
    store = PointsMemoryStore(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    )
    visual = controller.add_points(store, scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    calls: list = []
    original = controller._refresh_scene_overlays
    controller._refresh_scene_overlays = lambda *a, **k: calls.append(a)

    controller.reslice_visual(visual.id)

    controller._refresh_scene_overlays = original
    assert calls == []
    assert box.visible


def test_a_hidden_box_is_caught_up_when_shown(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    box.visible = False
    geometry = _gfx(controller, box).node.geometry

    controller.set_displayed_axes(scene.id, (1, 2))
    assert _gfx(controller, box).node.geometry is geometry  # skipped while hidden

    box.visible = True

    node = _gfx(controller, box).node
    assert node.visible
    assert node.geometry.positions.data.shape == (8, 3)


def test_an_unchanged_scene_does_not_rebuild(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    geometry = _gfx(controller, box).node.geometry

    controller._refresh_scene_overlays(scene.id)

    assert _gfx(controller, box).node.geometry is geometry


# ---------------------------------------------------------------------------
# Live fields and events
# ---------------------------------------------------------------------------


def test_appearance_fields_are_live_and_announced(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    events = _record(controller)

    box.appearance.color = (1.0, 0.0, 0.0, 1.0)
    box.appearance.thickness = 4.0
    box.visible = False

    node = _gfx(controller, box).node
    np.testing.assert_allclose(tuple(node.material.color), (1.0, 0.0, 0.0, 1.0))
    assert node.material.thickness == pytest.approx(4.0)
    assert not node.visible
    assert [(e.overlay_id, e.field_name) for e in events] == [
        (box.id, "appearance.color"),
        (box.id, "appearance.thickness"),
        (box.id, "visible"),
    ]


def test_a_replaced_appearance_is_applied_and_stays_connected(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    old = box.appearance
    events = _record(controller)

    box.appearance = SceneBoundingBoxAppearance(thickness=7.0, render_order=5)

    node = _gfx(controller, box).node
    assert node.material.thickness == pytest.approx(7.0)
    assert node.render_order == 5
    assert events[-1].field_name == "appearance"

    box.appearance.thickness = 2.0
    assert node.material.thickness == pytest.approx(2.0)
    old.thickness = 9.0  # the orphaned model no longer reaches the render layer
    assert node.material.thickness == pytest.approx(2.0)


def test_update_overlay_field_stamps_the_source_id(controller):
    scene = _scene(controller)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    events = _record(controller)
    source = uuid4()

    controller.update_overlay_field(
        box.id, "appearance.thickness", 3.0, source_id=source
    )

    assert box.appearance.thickness == 3.0
    assert events[-1].source_id == source
    assert events[-1].field_name == "appearance.thickness"


def test_the_incoming_bus_reaches_the_overlay(controller):
    scene = _scene(controller)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    controller.incoming_events.emit(
        OverlayUpdateEvent(
            source_id=uuid4(), overlay_id=box.id, field="visible", value=False
        )
    )

    assert box.visible is False


def test_set_overlay_visible_serves_both_categories(controller):
    scene = _scene(controller, dim="2d")
    controller.add_canvas(scene.id)
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    axes = controller.add_canvas_overlay(canvas_id, CenteredAxes2D(name="axes"))

    controller.set_overlay_visible(box.id, False)
    controller.set_overlay_visible(axes.id, False)

    assert box.visible is False
    assert axes.visible is False
    assert not _gfx(controller, axes)._line.visible
    with pytest.raises(KeyError):
        controller.set_overlay_visible(uuid4(), True)


def test_canvas_overlay_appearance_is_live(controller):
    scene = _scene(controller, dim="2d")
    controller.add_canvas(scene.id)
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    axes = controller.add_canvas_overlay(canvas_id, CenteredAxes2D(name="axes"))
    events = _record(controller)

    axes.appearance.line_thickness_px = 5.0
    axes.axis_a_label = "Y"

    gfx_axes = _gfx(controller, axes)
    assert gfx_axes._line.material.thickness == pytest.approx(5.0)
    assert [e.field_name for e in events] == [
        "appearance.line_thickness_px",
        "axis_a_label",
    ]


# ---------------------------------------------------------------------------
# Registration, removal and teardown
# ---------------------------------------------------------------------------


def test_registering_an_overlay_twice_is_rejected(controller):
    scene = _scene(controller)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    with pytest.raises(ValueError, match="already registered"):
        controller.add_scene_overlay(scene.id, box)


def test_remove_overlay_detaches_everything(controller):
    scene = _scene(controller)
    _image(controller, scene)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))
    node = _gfx(controller, box).node
    gfx_scene = controller._render_manager.get_scene(scene.id)
    assert node in gfx_scene.children

    controller.remove_overlay(box.id)

    assert scene.overlays == []
    assert node not in gfx_scene.children
    with pytest.raises(KeyError):
        controller.get_overlay(box.id)
    events = _record(controller)
    box.visible = False  # the bridge is gone
    assert events == []


def test_remove_canvas_drops_its_overlays(controller):
    scene = _scene(controller, dim="2d")
    controller.add_canvas(scene.id)
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    axes = controller.add_canvas_overlay(canvas_id, CenteredAxes2D(name="axes"))

    controller.remove_canvas(canvas_id)

    assert axes.id not in controller._overlays


def test_remove_scene_drops_its_overlays(controller):
    scene = _scene(controller)
    box = controller.add_scene_overlay(scene.id, SceneBoundingBox(name="box"))

    controller.remove_scene(scene.id)

    assert box.id not in controller._overlays


def test_add_canvas_announces_the_canvas(controller):
    scene = _scene(controller, dim="2d")
    events: list = []
    controller._outgoing_events.subscribe(
        CanvasAddedEvent, events.append, entity_id=scene.id
    )

    controller.add_canvas(scene.id)

    assert [e.canvas_id for e in events] == controller.get_canvas_ids(scene.id)


def test_scene_overlays_are_restored_with_their_scene():
    """``from_model`` wires overlays already in ``scene.overlays``."""
    source = CellierController(gui="offscreen")
    scene = _scene(source, dim="2d")
    _image(source, scene)
    source.add_scene_overlay(
        scene.id,
        SceneBoundingBox(
            name="box", appearance=SceneBoundingBoxAppearance(thickness=3.0)
        ),
    )

    restored = CellierController.from_model(source.to_model())

    (restored_scene,) = restored._model.scenes.values()
    (box,) = restored_scene.overlays
    assert box.appearance.thickness == 3.0
    node = restored._overlays[box.id].gfx.node
    assert node.visible
    assert node.geometry.positions.data.shape == (8, 3)
    box.visible = False
    assert not node.visible


def test_canvas_overlays_are_restored_with_their_canvas(controller):
    """``add_canvas_model`` wires overlays already in ``canvas.overlays``."""
    source = CellierController(gui="offscreen")
    source_scene = _scene(source, dim="2d")
    source.add_canvas(source_scene.id)
    (source_canvas_id,) = source.get_canvas_ids(source_scene.id)
    source.add_canvas_overlay(source_canvas_id, CenteredAxes2D(name="axes"))
    canvas_model = source.to_model().scenes[source_scene.id].canvases[source_canvas_id]

    scene = _scene(controller, dim="2d")
    controller.add_canvas_model(scene.id, canvas_model)

    (axes,) = canvas_model.overlays
    assert controller._overlays[axes.id].owner_id == canvas_model.id
    axes.visible = False
    assert not controller._overlays[axes.id].gfx._line.visible


def test_scene_overlays_roundtrip_through_json(controller):
    """The discriminated union restores the concrete overlay type."""
    from cellier.scene.scene import Scene

    scene = _scene(controller)
    controller.add_scene_overlay(
        scene.id,
        SceneBoundingBox(
            name="box",
            appearance=SceneBoundingBoxAppearance(color=(1.0, 0.0, 0.0, 1.0)),
        ),
    )

    restored = Scene.model_validate_json(scene.model_dump_json())

    (box,) = restored.overlays
    assert isinstance(box, SceneBoundingBox)
    assert box.id == scene.overlays[0].id
    assert box.appearance.color == (1.0, 0.0, 0.0, 1.0)
