"""Typed pick events and in-memory pick values (unified image design 3.7).

Mouse events report *that* something was hit; a typed pick event, subscribed
with ``on_pick``, reports *what*.  It follows the mouse event for the same
pointer event and carries the same gesture context.  In-memory image and
labels events carry the value under the pointer and are emitted synchronously.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.convenience import OrthoViewer, Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    GraphEdgePickInfo,
    GraphNodePickInfo,
    GraphPickEvent,
    ImagePickEvent,
    LabelsPickEvent,
    LinesPickEvent,
    LinesPickInfo,
    MeshPickEvent,
    MeshPickInfo,
    PointsPickEvent,
    PointsPickInfo,
    ViewRay,
)
from cellier.events._events import _CanvasRawPointerEvent
from cellier.render.render_manager import (
    _ImageDisplayedDataCoord,
    _LabelsDisplayedDataCoord,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageChannelAppearance

_MOUSE_2D = {
    "press": CanvasMousePress2DEvent,
    "move": CanvasMouseMove2DEvent,
    "release": CanvasMouseRelease2DEvent,
}


def _raw(
    canvas_id,
    scene_id,
    visual_id,
    details,
    *,
    action="press",
    camera="2d",
    gesture_id=None,
    buttons=(1,),
    modifiers=(),
):
    return _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action=action,
        camera_type=camera,
        position_2d=np.array([1.0, 2.0]) if camera == "2d" else None,
        ray=(
            ViewRay(origin=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]))
            if camera == "3d"
            else None
        ),
        hit_visual_id=visual_id,
        button=1,
        modifiers=modifiers,
        buttons=buttons,
        gesture_id=gesture_id,
        pick_details=details,
    )


def _scene(world, dim):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=world, dim=dim)
    return controller, scene


# ---------------------------------------------------------------------------
# 6a: one event type per visual family
# ---------------------------------------------------------------------------

_FAMILIES = [
    (PointsPickInfo(point_index=2), PointsPickEvent),
    (LinesPickInfo(edge_index=1), LinesPickEvent),
    (MeshPickInfo(face_index=4), MeshPickEvent),
    (GraphNodePickInfo(node_id="a", node_row=0), GraphPickEvent),
    (
        GraphEdgePickInfo(edge_index=3, source_node_id="a", target_node_id="b"),
        GraphPickEvent,
    ),
]


@pytest.mark.parametrize("camera", ["2d", "3d"])
@pytest.mark.parametrize(("details", "event_type"), _FAMILIES)
def test_a_hit_emits_its_familys_event_with_the_gesture_context(
    details, event_type, camera
):
    controller, scene = _scene(spatial_axes("z", "y", "x"), camera)
    canvas_id = uuid4()
    visual_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, event_type, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual_id,
            details,
            action="move",
            camera=camera,
            buttons=(1, 2),
            modifiers=("Alt",),
        )
    )

    (event,) = received
    assert isinstance(event, event_type)
    assert event.pick_info == details
    assert (event.source_id, event.scene_id, event.visual_id) == (
        canvas_id,
        scene.id,
        visual_id,
    )
    assert (event.action, event.camera_type) == ("move", camera)
    assert (event.button, event.buttons, event.modifiers) == (1, (1, 2), ("Alt",))
    assert (event.world_coordinate is not None) == (camera == "2d")
    assert (event.ray is not None) == (camera == "3d")


def test_another_familys_subscriber_hears_nothing():
    controller, scene = _scene(spatial_axes("z", "y", "x"), "3d")
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(
        _raw(canvas_id, scene.id, uuid4(), PointsPickInfo(point_index=0), camera="3d")
    )

    assert received == []


def test_the_pick_event_is_scoped_to_its_canvas():
    controller, scene = _scene(spatial_axes("z", "y", "x"), "3d")
    received: list = []
    controller.on_pick(uuid4(), PointsPickEvent, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(
        _raw(uuid4(), scene.id, uuid4(), PointsPickInfo(point_index=0), camera="3d")
    )

    assert received == []


def test_a_gesture_shares_its_id_and_hover_has_none():
    """Each pick event carries the gesture id of the mouse event it follows."""
    controller, scene = _scene(spatial_axes("z", "y", "x"), "2d")
    canvas_id = uuid4()
    owner = uuid4()
    received: list = []
    for event_type in _MOUSE_2D.values():
        controller._outgoing_events.subscribe(
            event_type, received.append, entity_id=canvas_id, owner_id=owner
        )
    controller.on_pick(canvas_id, PointsPickEvent, received.append, owner_id=owner)

    first, second = uuid4(), uuid4()
    for action, gesture_id in [
        ("press", first),
        ("move", first),
        ("move", first),
        ("release", first),
        ("move", None),
        ("press", second),
    ]:
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                uuid4(),
                PointsPickInfo(point_index=0),
                action=action,
                gesture_id=gesture_id,
            )
        )

    # Synchronous: every pick event directly follows its mouse event.
    assert len(received) == 12
    for mouse, pick in zip(received[::2], received[1::2], strict=True):
        assert isinstance(mouse, _MOUSE_2D[pick.action])
        assert pick.gesture_id == mouse.gesture_id
    picks = received[1::2]
    assert [p.gesture_id for p in picks] == [first] * 4 + [None, second]


def test_unsubscribe_pick_stops_the_events_and_the_value_reads():
    controller, scene = _scene(spatial_axes("z", "y", "x"), "3d")
    canvas_id = uuid4()
    received: list = []
    handle = controller.on_pick(
        canvas_id, PointsPickEvent, received.append, owner_id=uuid4()
    )
    controller.unsubscribe_pick(handle)

    controller._on_raw_pointer_event(
        _raw(canvas_id, scene.id, uuid4(), PointsPickInfo(point_index=0), camera="3d")
    )

    assert received == []
    assert controller._pick_event_counts == {}
    assert controller._render_manager._pick_details_enabled[canvas_id] is False


# ---------------------------------------------------------------------------
# 6b: in-memory image values
# ---------------------------------------------------------------------------

_CYX = [("c", "channel"), *spatial_axes("y", "x")]
_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]


def _coded(shape):
    """Every voxel's value spells its own index, e.g. ``c*100 + y*10 + x``."""
    grids = np.indices(shape)
    weights = [10 ** (len(shape) - 1 - axis) for axis in range(len(shape))]
    return sum(w * g for w, g in zip(weights, grids, strict=True)).astype(np.float32)


def _image(world, dim, shape, **kwargs):
    controller, scene = _scene(world, dim)
    controller.add_canvas(scene.id)
    store = ImageMemoryStore(data=_coded(shape))
    visual = controller.add_image(store, scene.id, **kwargs)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())
    return controller, scene, visual, canvas_id, received


def test_single_mode_reports_the_drawn_channel_synchronously():
    controller, scene, visual, canvas_id, received = _image(
        _CYX, "2d", (3, 8, 8), channel_axis=0
    )
    details = _ImageDisplayedDataCoord(
        displayed_data_coord=(5.5, 3.5), collapsed_data_indices=((0, 2),)
    )

    controller._on_raw_pointer_event(_raw(canvas_id, scene.id, visual.id, details))

    (event,) = received  # no await: in-memory values are read in the handler
    assert event.pick_info.data_coordinate == (2.5, 3.5, 5.5)
    assert event.pick_info.channel_values == {2: 235.0}


def test_2d_composite_reports_every_drawn_channel_from_the_render_layer():
    """End to end from the pick buffer: only drawn (visible) channels report."""
    channels = {i: InMemoryImageChannelAppearance() for i in range(3)}
    channels[1].visible = False
    controller, scene, visual, canvas_id, received = _image(
        _CYX, "2d", (3, 8, 8), channel_axis=0, composite=True, channels=channels
    )
    render_manager = controller._render_manager
    gfx_visual = render_manager._scenes[scene.id].get_visual(visual.id)
    gfx_visual.build_slice_request_2d(
        np.zeros(3),
        100.0,
        10.0,
        None,
        None,
        scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )
    winner = gfx_visual.slots[gfx_visual._slot_for_key[2]].node_2d
    details = render_manager._extract_pick_details(
        scene.id, winner, {"index": (5, 3), "pixel_coord": (0.0, 0.0)}
    )
    assert details.channel_index == 2
    assert details.drawn_channels == (0, 2)

    controller._on_raw_pointer_event(_raw(canvas_id, scene.id, visual.id, details))

    (event,) = received
    assert event.pick_info.data_coordinate == (2.5, 3.5, 5.5)
    assert event.pick_info.channel_values == {0: 35.0, 2: 235.0}


def test_3d_composite_reports_only_the_winners_channel():
    controller, scene, visual, canvas_id, received = _image(
        _CZYX,
        "3d",
        (3, 4, 8, 8),
        channel_axis=0,
        composite=True,
        channels={i: InMemoryImageChannelAppearance() for i in range(3)},
    )
    details = _ImageDisplayedDataCoord(
        displayed_data_coord=(5.5, 3.5, 1.5),
        collapsed_data_indices=(),
        channel_index=1,
        drawn_channels=(0, 1, 2),
    )

    controller._on_raw_pointer_event(
        _raw(canvas_id, scene.id, visual.id, details, camera="3d")
    )

    (event,) = received
    assert event.pick_info.data_coordinate == (1.5, 1.5, 3.5, 5.5)
    assert event.pick_info.channel_values == {1: 1135.0}


def test_no_channel_axis_reports_channel_zero():
    controller, scene, visual, canvas_id, received = _image(
        spatial_axes("z", "y", "x"), "2d", (4, 8, 8)
    )
    details = _ImageDisplayedDataCoord(
        displayed_data_coord=(4.5, 6.5), collapsed_data_indices=((0, 1),)
    )

    controller._on_raw_pointer_event(_raw(canvas_id, scene.id, visual.id, details))

    (event,) = received
    # (4, 8, 8) codes as 100*z + 10*y + x.
    assert event.pick_info.channel_values == {0: 164.0}


def test_a_voxel_outside_the_data_reports_no_values():
    controller, scene, visual, canvas_id, received = _image(
        spatial_axes("z", "y", "x"), "2d", (4, 8, 8)
    )
    details = _ImageDisplayedDataCoord(
        displayed_data_coord=(9.5, 1.5), collapsed_data_indices=((0, 1),)
    )

    controller._on_raw_pointer_event(_raw(canvas_id, scene.id, visual.id, details))

    (event,) = received
    assert event.pick_info.channel_values == {}


def test_values_are_not_read_without_a_subscriber_for_that_type():
    controller, scene = _scene(spatial_axes("z", "y", "x"), "2d")
    store = ImageMemoryStore(data=np.zeros((4, 8, 8), dtype=np.float32))
    visual = controller.add_image(store, scene.id)
    canvas_id = uuid4()
    controller.on_pick(canvas_id, PointsPickEvent, lambda e: None, owner_id=uuid4())
    reads: list = []
    controller._read_pick_values = lambda *args: reads.append(args)

    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            _ImageDisplayedDataCoord(displayed_data_coord=(1.5, 1.5)),
        )
    )

    assert reads == []


# ---------------------------------------------------------------------------
# 6b: in-memory labels values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("camera", ["2d", "3d"])
def test_labels_report_the_label_id_synchronously(camera):
    controller, scene = _scene(spatial_axes("z", "y", "x"), camera)
    data = np.zeros((4, 8, 8), dtype=np.int32)
    data[1:3, 2:6, 2:6] = 9
    visual = controller.add_labels(LabelMemoryStore(data=data), scene.id)
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, LabelsPickEvent, received.append, owner_id=uuid4())

    if camera == "2d":
        displayed = (3.5, 4.5)  # pygfx (x, y) on displayed (y, x)
        collapsed = ((0, 2),)
    else:
        displayed = (3.5, 4.5, 1.5)  # pygfx (x, y, z) on displayed (z, y, x)
        collapsed = ()
    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            _LabelsDisplayedDataCoord(
                displayed_data_coord=displayed, collapsed_data_indices=collapsed
            ),
            camera=camera,
        )
    )

    (event,) = received
    assert event.pick_info.value == 9
    index = tuple(int(np.floor(v)) for v in event.pick_info.data_coordinate)
    assert data[index] == 9


# ---------------------------------------------------------------------------
# The viewers mirror on_pick
# ---------------------------------------------------------------------------


def test_viewer_on_pick_defaults_its_owner_to_the_canvas():
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d")
    canvas_id = uuid4()
    received: list = []
    viewer.on_pick(canvas_id, PointsPickEvent, received.append)

    viewer.controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            viewer.scene.id,
            uuid4(),
            PointsPickInfo(point_index=1),
            camera="3d",
        )
    )
    assert len(received) == 1

    # Owned by the canvas: removing the canvas's subscriptions removes it.
    viewer.controller._outgoing_events.unsubscribe_all(canvas_id)
    viewer.controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            viewer.scene.id,
            uuid4(),
            PointsPickInfo(point_index=1),
            camera="3d",
        )
    )
    assert len(received) == 1

    counts = viewer.controller._pick_event_counts
    before = dict(counts)
    other = viewer.on_pick(canvas_id, PointsPickEvent, received.append)
    viewer.unsubscribe_pick(other)
    assert counts == before


def test_ortho_viewer_on_pick_forwards_to_the_controller():
    calls: list = []
    stub = SimpleNamespace(
        _controller=SimpleNamespace(
            on_pick=lambda *args, **kwargs: calls.append((args, kwargs)),
            unsubscribe_pick=lambda handle: calls.append(handle),
        )
    )
    canvas_id = uuid4()

    OrthoViewer.on_pick(stub, canvas_id, MeshPickEvent, print)
    OrthoViewer.unsubscribe_pick(stub, "handle")

    assert calls == [
        (
            (canvas_id, MeshPickEvent, print),
            {"owner_id": canvas_id, "weak": False},
        ),
        "handle",
    ]


def test_a_mouse_only_subscriber_gets_no_pick_events():
    controller, scene = _scene(spatial_axes("z", "y", "x"), "3d")
    canvas_id = uuid4()
    received: list = []
    controller.on_mouse_press_3d(canvas_id, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(
        _raw(canvas_id, scene.id, uuid4(), PointsPickInfo(point_index=0), camera="3d")
    )

    (event,) = received
    assert isinstance(event, CanvasMousePress3DEvent)
    assert not any(isinstance(e, LinesPickEvent) for e in received)
