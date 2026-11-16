"""Removing controller subscriptions by handle, owner, GC and canvas removal.

``on_pick`` keeps per-canvas and per-type counters that gate pick-detail
extraction and the image and labels value reads; they must follow the
subscription however it ends.  The one-shot helpers return a handle that
cancels the pending callback.
"""

from __future__ import annotations

import gc
from uuid import uuid4

import numpy as np
import pytest

from cellier import CellierController
from cellier.convenience import Viewer
from cellier.data import ImageMemoryStore
from cellier.events import (
    CanvasConnectedEvent,
    FrameRenderedEvent,
    ImagePickEvent,
    PointsPickEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
)
from cellier.scene.dims import spatial_axes, world_coordinate_system


class _Listener:
    def __init__(self) -> None:
        self.received: list = []

    def on_event(self, event) -> None:
        self.received.append(event)


def _gate(controller: CellierController, canvas_id) -> bool:
    return controller._render_manager._pick_details_enabled.get(canvas_id, False)


def _points_pick_event(canvas_id) -> PointsPickEvent:
    fields = dict.fromkeys(PointsPickEvent._fields)
    fields["source_id"] = canvas_id
    return PointsPickEvent(**fields)


@pytest.fixture
def controller():
    made = CellierController()
    yield made
    made.close()


def test_controller_unsubscribe_removes_any_on_subscription(controller):
    scene_id = uuid4()
    received: list = []
    handle = controller.on_dims_changed(scene_id, received.append, owner_id=uuid4())

    controller.unsubscribe(handle)

    assert not handle.active
    controller.unsubscribe(handle)  # already removed: no error


def test_unsubscribe_owner_releases_the_pick_gate(controller):
    canvas_id = uuid4()
    widget_id = uuid4()
    controller.on_pick(canvas_id, PointsPickEvent, print, owner_id=widget_id)
    controller.on_pick(canvas_id, ImagePickEvent, print, owner_id=widget_id)
    assert _gate(controller, canvas_id)

    controller.unsubscribe_owner(widget_id)

    assert not _gate(controller, canvas_id)
    assert controller._pick_event_counts == {}
    assert controller._pick_subscriptions == {}


def test_unsubscribe_owner_keeps_other_owners_pick_subscriptions(controller):
    canvas_id = uuid4()
    widget_id = uuid4()
    controller.on_pick(canvas_id, PointsPickEvent, print, owner_id=widget_id)
    kept = controller.on_pick(canvas_id, PointsPickEvent, print, owner_id=uuid4())

    controller.unsubscribe_owner(widget_id)

    assert kept.active
    assert _gate(controller, canvas_id)
    assert controller._pick_event_counts == {(canvas_id, PointsPickEvent): 1}


def test_a_collected_weak_pick_callback_releases_the_gate(controller):
    canvas_id = uuid4()
    listener = _Listener()
    handle = controller.on_pick(
        canvas_id, PointsPickEvent, listener.on_event, owner_id=uuid4(), weak=True
    )
    del listener
    gc.collect()

    # The bus finds the dead callback at the next emission of its type.
    controller._outgoing_events.emit(_points_pick_event(canvas_id))

    assert not handle.active
    assert not _gate(controller, canvas_id)
    assert controller._pick_event_counts == {}


def test_canvas_connected_handle_cancels_the_callback(controller):
    canvas_id = uuid4()
    fired: list = []
    handle = controller.on_canvas_connected(canvas_id, lambda: fired.append(1))
    assert handle.active

    handle.unsubscribe()
    controller._outgoing_events.emit(
        CanvasConnectedEvent(source_id=canvas_id, canvas_id=canvas_id, gui="qt")
    )

    assert fired == []


def test_canvas_first_frame_handle_goes_inactive_when_it_fires(controller):
    canvas_id = uuid4()
    fired: list = []
    handle = controller.on_canvas_first_frame(canvas_id, lambda: fired.append(1))

    for _ in range(2):
        controller._outgoing_events.emit(
            FrameRenderedEvent(
                source_id=canvas_id, canvas_id=canvas_id, frame_time_ms=1.0
            )
        )

    assert fired == [1]
    assert not handle.active


def _arm_readiness(controller, scene_id, owner, fired):
    """Arm the scene-readiness tracker with one visual pending."""
    visual_id = uuid4()

    def trigger() -> None:
        controller._outgoing_events.emit(
            ResliceStartedEvent(
                source_id=scene_id,
                scene_id=scene_id,
                canvas_id=None,
                visual_ids=(visual_id,),
            )
        )

    handle = controller._notify_when_resliced(
        scene_id, lambda: fired.append(1), owner, trigger
    )
    completed = ResliceCompletedEvent(
        source_id=scene_id,
        scene_id=scene_id,
        canvas_id=None,
        visual_id=visual_id,
        brick_count=0,
    )
    return handle, completed


def _owned(controller, owner) -> int:
    return sum(
        sub.owner_id == owner
        for subs in controller._outgoing_events._subs.values()
        for sub in subs
    )


def test_scene_ready_handle_cancels_every_tracker_subscription(controller):
    # Started, completed and visual-removed.
    scene_id, owner = uuid4(), uuid4()
    fired: list = []
    handle, completed = _arm_readiness(controller, scene_id, owner, fired)
    assert handle.active
    assert _owned(controller, owner) == 3

    handle.unsubscribe()
    controller._outgoing_events.emit(completed)

    assert fired == []
    assert _owned(controller, owner) == 0


def test_scene_ready_handle_goes_inactive_when_it_fires(controller):
    scene_id, owner = uuid4(), uuid4()
    fired: list = []
    handle, completed = _arm_readiness(controller, scene_id, owner, fired)

    controller._outgoing_events.emit(completed)

    assert fired == [1]
    assert not handle.active
    assert _owned(controller, owner) == 0


def test_on_scene_ready_with_nothing_to_load_returns_an_inactive_handle(controller):
    scene = controller.add_scene(
        displayed_axes=("z", "y", "x"),
        coordinate_system=world_coordinate_system(
            spatial_axes("z", "y", "x"), name="world"
        ),
        name="empty",
    )
    fired: list = []

    handle = controller.on_scene_ready(scene.id, lambda: fired.append(1))

    assert fired == [1]
    assert not handle.active


# ---------------------------------------------------------------------------
# Canvas and scene removal (real offscreen canvases)
# ---------------------------------------------------------------------------


@pytest.fixture
def viewer(qtbot):
    made = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="offscreen"
    )
    made.controller.camera_reslice_enabled = False
    data = np.random.default_rng(0).random((4, 6, 8), dtype=np.float32)
    made.add_image(ImageMemoryStore(data=data))
    made.add_canvas()
    yield made
    made.controller.close()


def test_removing_a_canvas_removes_pick_subscriptions_of_any_owner(viewer):
    controller = viewer.controller
    canvas_id = viewer.canvases[0]
    by_canvas = viewer.on_pick(canvas_id, PointsPickEvent, print)
    by_widget = controller.on_pick(canvas_id, ImagePickEvent, print, owner_id=uuid4())

    controller.remove_canvas(canvas_id)

    assert not by_canvas.active
    assert not by_widget.active
    assert canvas_id not in controller._pick_subscriptions
    assert controller._pick_event_counts == {}


def test_removing_a_scene_removes_its_canvases_pick_subscriptions(viewer):
    controller = viewer.controller
    canvas_id = viewer.canvases[0]
    handle = controller.on_pick(canvas_id, PointsPickEvent, print, owner_id=uuid4())

    controller.remove_scene(viewer.scene.id)

    assert not handle.active
    assert controller._pick_subscriptions == {}
    assert controller._pick_event_counts == {}


def test_close_deactivates_handles_and_drops_pick_state(qtbot):
    made = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="offscreen"
    )
    made.add_canvas()
    canvas_id = made.canvases[0]
    handle = made.on_pick(canvas_id, PointsPickEvent, print)

    made.controller.close()

    assert not handle.active
    assert made.controller._pick_subscriptions == {}
    assert made.controller._pick_event_counts == {}
