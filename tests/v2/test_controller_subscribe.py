"""CellierController.subscribe: the general form of the on_* methods."""

from __future__ import annotations

from uuid import uuid4

import pytest

import cellier.events as events
from cellier import CellierController
from cellier.events import (
    PICK_EVENT_TYPES,
    AppearanceUpdateEvent,
    CanvasSizeChangedEvent,
    ImagePickEvent,
    RenderConfigChangedEvent,
    TransformChangedEvent,
)


@pytest.fixture
def controller():
    made = CellierController()
    yield made
    made.close()


def _transform_event(visual_id) -> TransformChangedEvent:
    return TransformChangedEvent(
        source_id=uuid4(), scene_id=uuid4(), visual_id=visual_id, transform=None
    )


def test_subscribe_filters_on_the_routing_key(controller):
    wanted, other = uuid4(), uuid4()
    received: list = []
    controller.subscribe(
        TransformChangedEvent, received.append, owner_id=uuid4(), entity_id=wanted
    )

    controller._outgoing_events.emit(_transform_event(other))
    controller._outgoing_events.emit(_transform_event(wanted))

    assert [event.visual_id for event in received] == [wanted]


def test_subscribe_without_entity_receives_every_event(controller):
    received: list = []
    handle = controller.subscribe(
        TransformChangedEvent, received.append, owner_id=uuid4()
    )
    controller._outgoing_events.emit(_transform_event(uuid4()))
    controller._outgoing_events.emit(_transform_event(uuid4()))
    handle.unsubscribe()
    controller._outgoing_events.emit(_transform_event(uuid4()))

    assert len(received) == 2


def test_canvas_size_events_route_by_canvas(controller):
    canvas_id = uuid4()
    received: list = []
    controller.subscribe(
        CanvasSizeChangedEvent, received.append, owner_id=uuid4(), entity_id=canvas_id
    )

    for target in (uuid4(), canvas_id):
        controller._outgoing_events.emit(
            CanvasSizeChangedEvent(
                source_id=target, canvas_id=target, width=10, height=20
            )
        )

    assert [event.canvas_id for event in received] == [canvas_id]


def test_subscribe_refuses_pick_events(controller):
    with pytest.raises(TypeError, match="on_pick"):
        controller.subscribe(ImagePickEvent, print, owner_id=uuid4())


def test_subscribe_refuses_update_events(controller):
    with pytest.raises(TypeError, match="incoming_events"):
        controller.subscribe(AppearanceUpdateEvent, print, owner_id=uuid4())


def test_an_event_without_a_routing_key_refuses_an_entity_id(controller):
    with pytest.raises(ValueError, match="no routing key"):
        controller.subscribe(
            RenderConfigChangedEvent, print, owner_id=uuid4(), entity_id=uuid4()
        )
    controller.subscribe(RenderConfigChangedEvent, print, owner_id=uuid4())


def test_every_exported_event_a_user_receives_can_be_subscribed(controller):
    """Each emitted event type is in the outgoing union ``subscribe`` checks."""
    names = [
        name
        for name in events.__all__
        if name.endswith("Event")
        and not name.endswith("UpdateEvent")
        and name != "PickEvent"
    ]
    for name in names:
        event_type = getattr(events, name)
        if event_type in PICK_EVENT_TYPES:
            continue
        controller.subscribe(event_type, print, owner_id=uuid4())


def test_get_subscribers_lists_a_subscription(controller):
    owner = uuid4()
    visual_id = uuid4()
    controller.subscribe(
        TransformChangedEvent, print, owner_id=owner, entity_id=visual_id
    )

    infos = controller.get_subscribers(TransformChangedEvent, entity_id=visual_id)

    assert [info.owner_id for info in infos if info.entity_id == visual_id] == [owner]
