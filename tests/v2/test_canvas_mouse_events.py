"""Tests for canvas mouse event context-field wiring and gesture synthesis."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMouseRelease2DEvent,
    CanvasPickInfo,
    ImagePickEvent,
    ImagePickInfo,
    LinesPickInfo,
    MeshPickInfo,
    PointsPickEvent,
    PointsPickInfo,
    _CanvasRawPointerEvent,
)
from cellier.render.render_manager import RenderManager, _ImageDisplayedDataCoord
from cellier.render.visuals._slicing import round_world_to_voxel
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import AffineTransform, Axis
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    LinesMemoryAppearance,
    MeshFlatAppearance,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance
from tests._v2 import data_system


def _raw_2d(
    canvas_id,
    scene_id,
    action,
    *,
    button=1,
    buttons=(1,),
    modifiers=("Control",),
    gesture_id=None,
) -> _CanvasRawPointerEvent:
    return _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action=action,
        camera_type="2d",
        position_2d=np.array([1.0, 2.0], dtype=np.float64),
        ray=None,
        hit_visual_id=None,
        button=button,
        modifiers=modifiers,
        buttons=buttons,
        gesture_id=gesture_id,
    )


def _make_2d_controller():
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="2d", coordinate_system=cs, name="main")
    return controller, scene.id


def test_context_fields_propagate_to_public_event():
    controller, scene_id = _make_2d_controller()
    canvas_id = uuid4()
    received: list = []
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=uuid4())

    gid = uuid4()
    controller._on_raw_pointer_event(
        _raw_2d(
            canvas_id,
            scene_id,
            "press",
            button=1,
            buttons=(1, 2),
            modifiers=("Control", "Shift"),
            gesture_id=gid,
        )
    )

    assert len(received) == 1
    event = received[0]
    assert isinstance(event, CanvasMousePress2DEvent)
    assert event.button == 1
    assert event.buttons == (1, 2)
    assert event.modifiers == ("Control", "Shift")
    assert event.gesture_id == gid
    assert isinstance(event.pick_info, CanvasPickInfo)


def test_gesture_id_threads_through_phases():
    controller, scene_id = _make_2d_controller()
    canvas_id = uuid4()
    received: list = []
    owner = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=owner)
    controller.on_mouse_move_2d(canvas_id, received.append, owner_id=owner)
    controller.on_mouse_release_2d(canvas_id, received.append, owner_id=owner)

    gid = uuid4()
    for action in ("press", "move", "release"):
        controller._on_raw_pointer_event(
            _raw_2d(canvas_id, scene_id, action, gesture_id=gid)
        )

    assert [type(e) for e in received] == [
        CanvasMousePress2DEvent,
        CanvasMouseMove2DEvent,
        CanvasMouseRelease2DEvent,
    ]
    assert {e.gesture_id for e in received} == {gid}


# ---------------------------------------------------------------------------
# render_manager gesture synthesis
# ---------------------------------------------------------------------------


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


def _fake_pygfx_event(event_type, *, button=1, buttons=(1,)):
    return SimpleNamespace(
        type=event_type,
        x=10.0,
        y=20.0,
        button=button,
        buttons=buttons,
        modifiers=("Control",),
        pick_info={},
    )


def _fake_canvas_view():
    camera = SimpleNamespace(
        width=100.0,
        height=100.0,
        local=SimpleNamespace(position=(0.0, 0.0, 0.0)),
    )
    canvas = SimpleNamespace(get_logical_size=lambda: (100, 100))
    return SimpleNamespace(_canvas=canvas, _camera=camera, _dim="2d")


def _render_manager_with_canvas():
    rm = RenderManager()
    bus = _RecordingBus()
    rm._event_bus = bus
    canvas_id = uuid4()
    scene_id = uuid4()
    rm._canvas_to_scene[canvas_id] = scene_id
    rm._canvases[canvas_id] = _fake_canvas_view()
    return rm, bus, canvas_id


def test_gesture_synthesis_press_move_release_share_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_move"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_up"), canvas_id)

    actions = [e.action for e in bus.events]
    assert actions == ["press", "move", "release"]
    gesture_ids = {e.gesture_id for e in bus.events}
    assert len(gesture_ids) == 1
    assert next(iter(gesture_ids)) is not None
    assert bus.events[0].buttons == (1,)


def test_fresh_press_has_new_gesture_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_up"), canvas_id)
    rm._on_canvas_pointer_event(_fake_pygfx_event("pointer_down"), canvas_id)

    assert bus.events[0].gesture_id != bus.events[2].gesture_id


def test_hover_move_without_press_has_none_gesture_id():
    rm, bus, canvas_id = _render_manager_with_canvas()

    # A move with no preceding press (e.g. after release cleared state).
    hover = _fake_pygfx_event("pointer_move", buttons=())
    rm._on_canvas_pointer_event(hover, canvas_id)

    assert bus.events[0].action == "move"
    assert bus.events[0].gesture_id is None


# ---------------------------------------------------------------------------
# element-level pick detail extraction
# ---------------------------------------------------------------------------


def _controller_with_points():
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32
    )
    visual = controller.add_points(
        data=PointsMemoryStore(positions=positions),
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(),
        name="points",
    )
    return controller, scene.id, visual.id


def _controller_with_lines():
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    # 3 edges -> 6 vertices.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    visual = controller.add_lines(
        data=LinesMemoryStore(positions=positions),
        scene_id=scene.id,
        appearance=LinesMemoryAppearance(),
        name="lines",
    )
    return controller, scene.id, visual.id


def test_extract_points_pick_details():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    details = rm._extract_pick_details(scene_id, node, {"vertex_index": 2})
    assert details == PointsPickInfo(point_index=2)


def test_extract_points_missing_index_returns_none():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    assert rm._extract_pick_details(scene_id, node, {}) is None


def test_extract_lines_vertex_to_edge_mapping():
    controller, scene_id, visual_id = _controller_with_lines()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    # vertex_index 2k and 2k+1 both map to edge k.
    for vertex_index, edge_index in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]:
        details = rm._extract_pick_details(
            scene_id, node, {"vertex_index": vertex_index}
        )
        assert details == LinesPickInfo(edge_index=edge_index)


def _controller_with_mesh():
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("z", "y", "x"), name="world")
    scene = controller.add_scene(dim="3d", coordinate_system=cs, name="main")
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    visual = controller.add_mesh(
        data=MeshMemoryStore(positions=pos, indices=idx, name="msh"),
        scene_id=scene.id,
        appearance=MeshFlatAppearance(),
        name="mesh",
    )
    return controller, scene.id, visual.id


def test_extract_mesh_pick_details():
    controller, scene_id, visual_id = _controller_with_mesh()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    details = rm._extract_pick_details(scene_id, node, {"face_index": 1})
    assert details == MeshPickInfo(face_index=1)


def test_extract_mesh_missing_index_returns_none():
    controller, scene_id, visual_id = _controller_with_mesh()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)

    assert rm._extract_pick_details(scene_id, node, {}) is None


def test_a_hit_emits_the_pick_event_after_the_mouse_event():
    """Mouse events say that something was hit; pick events say what (3.7)."""
    controller, scene_id = _make_2d_controller()
    received: list = []
    canvas_id = uuid4()
    visual_id = uuid4()
    owner = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=owner)
    controller.on_pick(canvas_id, PointsPickEvent, received.append, owner_id=owner)

    gesture_id = uuid4()
    raw = _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action="press",
        camera_type="2d",
        position_2d=np.array([0.0, 0.0], dtype=np.float64),
        ray=None,
        hit_visual_id=visual_id,
        button=1,
        modifiers=("Shift",),
        buttons=(1,),
        gesture_id=gesture_id,
        pick_details=PointsPickInfo(point_index=3),
    )
    controller._on_raw_pointer_event(raw)

    mouse, pick = received
    assert isinstance(mouse, CanvasMousePress2DEvent)
    assert mouse.pick_info == CanvasPickInfo(hit_visual_id=visual_id)
    assert isinstance(pick, PointsPickEvent)
    assert pick.pick_info == PointsPickInfo(point_index=3)
    assert pick.visual_id == visual_id
    assert (pick.action, pick.camera_type) == ("press", "2d")
    assert (pick.button, pick.buttons, pick.modifiers) == (1, (1,), ("Shift",))
    assert pick.gesture_id == gesture_id
    np.testing.assert_array_equal(pick.world_coordinate, mouse.world_coordinate)
    assert pick.ray is None


def test_a_miss_emits_no_pick_event():
    controller, scene_id = _make_2d_controller()
    received: list = []
    canvas_id = uuid4()
    owner = uuid4()
    controller.on_mouse_press_2d(canvas_id, received.append, owner_id=owner)
    controller.on_pick(canvas_id, PointsPickEvent, received.append, owner_id=owner)

    controller._on_raw_pointer_event(_raw_2d(canvas_id, scene_id, "press"))

    (mouse,) = received
    assert mouse.pick_info.hit_visual_id is None


# ---------------------------------------------------------------------------
# Phase 4: readback gate
# ---------------------------------------------------------------------------


def _register_fake_canvas(rm, scene_id):
    bus = _RecordingBus()
    rm._event_bus = bus
    canvas_id = uuid4()
    rm._canvas_to_scene[canvas_id] = scene_id
    rm._canvases[canvas_id] = _fake_canvas_view()
    return bus, canvas_id


def test_gate_off_by_default_details_none_but_hit_visual_set():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)
    bus, canvas_id = _register_fake_canvas(rm, scene_id)

    ev = _fake_pygfx_event("pointer_down")
    ev.pick_info = {"world_object": node, "vertex_index": 1}
    rm._on_canvas_pointer_event(ev, canvas_id)

    raw = bus.events[0]
    assert raw.hit_visual_id == visual_id
    assert raw.pick_details is None


def test_gate_on_extracts_details():
    controller, scene_id, visual_id = _controller_with_points()
    rm = controller._render_manager
    node = rm._scenes[scene_id].get_active_node(visual_id)
    bus, canvas_id = _register_fake_canvas(rm, scene_id)
    rm.set_pick_details_enabled(canvas_id, True)

    ev = _fake_pygfx_event("pointer_down")
    ev.pick_info = {"world_object": node, "vertex_index": 1}
    rm._on_canvas_pointer_event(ev, canvas_id)

    assert bus.events[0].pick_details == PointsPickInfo(point_index=1)


def test_on_pick_subscription_enables_gate_and_unsubscribe_disables():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()

    assert rm._pick_details_enabled.get(canvas_id, False) is False

    handle = controller.on_pick(
        canvas_id, PointsPickEvent, lambda e: None, owner_id=uuid4()
    )
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_pick(handle)
    assert rm._pick_details_enabled.get(canvas_id) is False


def test_mouse_subscriptions_do_not_enable_the_gate():
    """Detail extraction runs only for pick subscribers (design 3.7)."""
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()

    handle = controller.on_mouse_press_2d(canvas_id, lambda e: None, owner_id=uuid4())
    assert rm._pick_details_enabled.get(canvas_id, False) is False
    controller.unsubscribe_mouse(handle)


def test_on_pick_refuses_a_mouse_event_type():
    import pytest

    controller, _scene_id, _ = _controller_with_points()
    with pytest.raises(TypeError, match="pick event type"):
        controller.on_pick(
            uuid4(), CanvasMousePress2DEvent, lambda e: None, owner_id=uuid4()
        )


def test_gate_stays_enabled_until_last_subscriber_removed():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()
    owner = uuid4()

    h1 = controller.on_pick(canvas_id, PointsPickEvent, lambda e: None, owner_id=owner)
    h2 = controller.on_pick(canvas_id, ImagePickEvent, lambda e: None, owner_id=owner)
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_pick(h1)
    assert rm._pick_details_enabled.get(canvas_id) is True

    controller.unsubscribe_pick(h2)
    assert rm._pick_details_enabled.get(canvas_id) is False


def test_direct_bus_subscription_does_not_enable_gate():
    controller, _scene_id, _ = _controller_with_points()
    rm = controller._render_manager
    canvas_id = uuid4()

    # Mimic the paint controller: subscribe directly on the bus.
    controller._outgoing_events.subscribe(
        CanvasMouseMove2DEvent,
        lambda e: None,
        entity_id=canvas_id,
        owner_id=uuid4(),
    )
    assert rm._pick_details_enabled.get(canvas_id, False) is False


# ---------------------------------------------------------------------------
# Image/labels pick coordinate is promoted into true (z, y, x) data order
# ---------------------------------------------------------------------------


def test_image_pick_coord_2d_promoted_in_data_axis_order():
    """The displayed-axis values land on their true data axes, not transposed.

    The render layer decodes the displayed coordinate in pygfx ``(x, y)`` order
    (column, row).  For a 2-D view of a (z, y, x) volume with z sliced, the
    promoted ``data_coordinate`` must be ``(z_slice, y=row, x=col)`` -- i.e. the
    pygfx order is reversed onto the ascending displayed axes.  The hit is
    synthetic (no visual), so the promotion is called directly: a pick event
    needs a real store to read the value from.
    """
    controller = CellierController()
    coordinate = controller._promote_pick_coordinate(
        (10.5, 3.5),
        displayed_axes=(1, 2),
        slice_indices={0: 5},
        world_ndim=3,
        hit_visual_id=uuid4(),
        collapsed_data_indices=None,
    )
    # axis 0 (z) = slice index, axis 1 (y) = row, axis 2 (x) = column.
    assert coordinate == (5.0, 3.5, 10.5)


def test_image_pick_coord_3d_promoted_in_data_axis_order():
    """3-D pick: pygfx ``(x, y, z)`` is reversed onto displayed axes (z, y, x)."""
    controller = CellierController()
    coordinate = controller._promote_pick_coordinate(
        (20.5, 4.5, 1.5),
        displayed_axes=(0, 1, 2),
        slice_indices={},
        world_ndim=3,
        hit_visual_id=uuid4(),
        collapsed_data_indices=None,
    )
    # (z, y, x) -- z and x swap relative to the pygfx (x, y, z) order.
    assert coordinate == (1.5, 4.5, 20.5)


def _image_visual_with_transform(
    world_axes,
    *,
    data_axis_names,
    shape,
    axis_map_by_name,
    scale_by_name=None,
    translation_by_name=None,
    broadcast_names=(),
    dim="2d",
):
    """A controller holding one in-memory image under a stated data -> world map."""
    controller = CellierController()
    cs = world_coordinate_system(world_axes, name="world")
    scene = controller.add_scene(dim=dim, coordinate_system=cs, name="main")
    store = ImageMemoryStore(
        data=np.zeros(shape, dtype=np.float32),
        name="img",
        data_coordinate_systems=[data_system(data_axis_names, sampling="discrete")],
    )
    data_cs = store.data_coordinate_systems[0]
    world = scene.dims.world_coordinate_system
    transform = AffineTransform.from_axis_map(
        data_cs,
        world,
        axis_map={
            data_cs.axis_by_name(data_name).id: world.axis_by_name(world_name).id
            for data_name, world_name in axis_map_by_name.items()
        },
        scale={
            data_cs.axis_by_name(name).id: value
            for name, value in (scale_by_name or {}).items()
        },
        translation={
            data_cs.axis_by_name(name).id: value
            for name, value in (translation_by_name or {}).items()
        },
        broadcast_output_axes=[world.axis_by_name(name).id for name in broadcast_names],
    )
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        name="img",
        transform=transform,
        single=InMemoryImageSingleAppearance(color_map="gray"),
    )
    return controller, scene, visual


def _press_image_pick(controller, scene, visual, displayed_data_coord):
    """Drive one 2-D press whose pick decoded *displayed_data_coord*."""
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())
    controller._on_raw_pointer_event(
        _CanvasRawPointerEvent(
            canvas_id=canvas_id,
            scene_id=scene.id,
            action="press",
            camera_type="2d",
            position_2d=np.array(displayed_data_coord, dtype=np.float64),
            ray=None,
            hit_visual_id=visual.id,
            button=1,
            modifiers=(),
            buttons=(1,),
            gesture_id=None,
            pick_details=_ImageDisplayedDataCoord(
                displayed_data_coord=tuple(displayed_data_coord)
            ),
        )
    )
    assert len(received) == 1
    return received[0].pick_info


def test_the_slice_position_is_pulled_back_through_the_transform():
    """D3 made slice_indices a world position; a data coordinate must not hold one.

    The image sits at ``Z = 2 z + 10``, so the world plane ``Z = 30`` is data
    plane ``z = 10``.  Reporting ``30`` would index eleven planes past the end
    of a twelve-plane volume, or the wrong plane on any volume big enough to
    accept it.

    The component is ``10.5``, not ``10``: every component of this tuple uses
    the ``[i, i + 1)`` convention so that ``floor`` is the one rule a consumer
    needs, and the centre of plane 10 is ``10.5``.
    """
    controller, scene, visual = _image_visual_with_transform(
        spatial_axes("z", "y", "x"),
        data_axis_names=("z", "y", "x"),
        shape=(12, 16, 20),
        axis_map_by_name={"z": "z", "y": "y", "x": "x"},
        scale_by_name={"z": 2.0},
        translation_by_name={"z": 10.0},
    )
    scene.dims.selection.displayed_axes = (1, 2)
    scene.dims.selection.slice_indices = {0: 30.0}

    details = _press_image_pick(controller, scene, visual, (10.5, 3.5))
    assert isinstance(details, ImagePickInfo)
    assert tuple(details.data_coordinate) == (10.5, 3.5, 10.5)


def test_a_slider_past_the_voxel_midpoint_reports_the_next_plane():
    """A collapsed axis reports the plane drawn, not the raw pulled-back position.

    ``Z = 2 z`` puts the world plane ``Z = 7`` at data ``z = 3.5`` -- exactly
    between two planes.  The selection assembler rounds half-up and fetches
    plane 4, so that is the plane on screen and the one the pick must name.
    Reporting the raw ``3.5`` would floor to 3: the neighbouring plane, which
    was never drawn.  That is the shape of the bug this convention exists to
    prevent.
    """
    controller, scene, visual = _image_visual_with_transform(
        spatial_axes("z", "y", "x"),
        data_axis_names=("z", "y", "x"),
        shape=(12, 16, 20),
        axis_map_by_name={"z": "z", "y": "y", "x": "x"},
        scale_by_name={"z": 2.0},
    )
    scene.dims.selection.displayed_axes = (1, 2)
    scene.dims.selection.slice_indices = {0: 7.0}

    details = _press_image_pick(controller, scene, visual, (1.0, 2.0))
    assert tuple(details.data_coordinate) == (4.5, 2.0, 1.0)
    assert int(np.floor(details.data_coordinate[0])) == round_world_to_voxel(3.5, 12)


def test_a_broadcast_world_axis_contributes_no_data_coordinate():
    """A store of lower rank than the world reports its own rank, not the world's.

    The image is ``zyx`` in a ``czyx`` world and broadcasts over ``c``, so the
    ``c`` slider says nothing about which voxel was hit.  Keying the result by
    world axis instead would put the ``c`` position on the store's ``z`` axis.
    """
    controller, scene, visual = _image_visual_with_transform(
        (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x")),
        data_axis_names=("z", "y", "x"),
        shape=(12, 16, 20),
        axis_map_by_name={"z": "z", "y": "y", "x": "x"},
        scale_by_name={"z": 2.0},
        broadcast_names=("c",),
        dim="3d",
    )
    scene.dims.selection.displayed_axes = (2, 3)
    scene.dims.selection.slice_indices = {0: 2.0, 1: 8.0}

    details = _press_image_pick(controller, scene, visual, (5.5, 6.5))
    assert tuple(details.data_coordinate) == (4.5, 6.5, 5.5)


def test_an_unresolvable_hit_keeps_the_world_positions_it_was_given():
    """No visual, no transform, nothing to pull back through.

    A removed visual or a synthetic id leaves the promotion with only the dims
    state, and inventing a transform for it would be worse than saying what it
    was told.  Such a hit emits no pick event -- there is no store to read --
    so the promotion is called directly.
    """
    controller = CellierController()
    coordinate = controller._promote_pick_coordinate(
        (10.5, 3.5),
        displayed_axes=(1, 2),
        slice_indices={0: 30.0},
        world_ndim=3,
        hit_visual_id=uuid4(),
        collapsed_data_indices=None,
    )
    assert coordinate == (30.0, 3.5, 10.5)
