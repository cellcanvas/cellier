"""SubscriptionHandle and the bus's single removal path.

A handle removes its own subscription, and every route that removes a
subscription runs its ``on_remove`` cleanup exactly once (``clear`` skips it).
"""

from __future__ import annotations

import gc
from uuid import uuid4

import pytest

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.events import DimsChangedEvent, EventBus, SubscriptionHandle
from cellier.events._bus import (
    fire_registered,
    group_handles,
    inactive_handle,
    register_callback,
)


def _dims_event(scene_id=None) -> DimsChangedEvent:
    return DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id or uuid4(),
        dims_state=DimsState(
            axis_labels=("z", "y", "x"),
            selection=AxisAlignedSelectionState(displayed_axes=(0, 1, 2)),
        ),
        displayed_axes_changed=False,
    )


class _Listener:
    def __init__(self) -> None:
        self.received: list = []

    def on_event(self, event) -> None:
        self.received.append(event)


def test_handle_unsubscribe_stops_the_callback():
    bus = EventBus()
    received: list = []
    handle = bus.subscribe(DimsChangedEvent, received.append)
    assert handle.active

    handle.unsubscribe()
    bus.emit(_dims_event())

    assert received == []
    assert not handle.active


def test_handle_unsubscribe_twice_is_a_no_op():
    bus = EventBus()
    removed: list = []
    handle = bus.subscribe(DimsChangedEvent, print, on_remove=lambda: removed.append(1))
    handle.unsubscribe()
    handle.unsubscribe()
    bus.unsubscribe(handle)

    assert removed == [1]


def test_a_callback_can_unsubscribe_itself():
    bus = EventBus()
    received: list = []

    def once(event) -> None:
        received.append(event)
        handle.unsubscribe()

    handle = bus.subscribe(DimsChangedEvent, once)
    first = _dims_event()
    bus.emit(first)
    bus.emit(_dims_event())

    assert received == [first]
    assert not handle.active


def test_a_subscription_removed_mid_emission_is_not_called():
    bus = EventBus()
    received: list = []
    bus.subscribe(DimsChangedEvent, lambda event: second.unsubscribe())
    second = bus.subscribe(DimsChangedEvent, received.append)

    bus.emit(_dims_event())

    assert received == []


@pytest.mark.parametrize("route", ["handle", "bus", "owner"])
def test_on_remove_runs_once_for_each_explicit_route(route):
    bus = EventBus()
    owner = uuid4()
    removed: list = []
    handle = bus.subscribe(
        DimsChangedEvent,
        print,
        owner_id=owner,
        on_remove=lambda: removed.append(route),
    )

    if route == "handle":
        handle.unsubscribe()
    elif route == "bus":
        bus.unsubscribe(handle)
    else:
        bus.unsubscribe_all(owner)
    # A second removal by another route does nothing more.
    bus.unsubscribe_all(owner)
    handle.unsubscribe()

    assert removed == [route]
    assert not handle.active


def test_on_remove_runs_when_a_dead_weak_callback_is_cleaned_up():
    bus = EventBus()
    removed: list = []
    listener = _Listener()
    handle = bus.subscribe(
        DimsChangedEvent,
        listener.on_event,
        weak=True,
        on_remove=lambda: removed.append(1),
    )
    del listener
    gc.collect()

    # Still registered until the next emission of its event type finds it.
    assert handle.active
    bus.emit(_dims_event())

    assert removed == [1]
    assert not handle.active


def test_clear_skips_on_remove_and_deactivates_handles():
    bus = EventBus()
    removed: list = []
    handle = bus.subscribe(DimsChangedEvent, print, on_remove=lambda: removed.append(1))

    bus.clear()
    handle.unsubscribe()

    assert removed == []
    assert not handle.active


def test_unsubscribe_all_finishes_when_a_cleanup_raises():
    bus = EventBus()
    owner = uuid4()
    removed: list = []

    def boom() -> None:
        raise RuntimeError("cleanup failed")

    first = bus.subscribe(DimsChangedEvent, print, owner_id=owner, on_remove=boom)
    second = bus.subscribe(
        DimsChangedEvent, print, owner_id=owner, on_remove=lambda: removed.append(1)
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        bus.unsubscribe_all(owner)

    assert not first.active
    assert not second.active
    assert removed == [1]


def test_a_handle_does_not_keep_its_bus_alive():
    bus = EventBus()
    handle = bus.subscribe(DimsChangedEvent, print)
    del bus
    gc.collect()

    assert not handle.active
    handle.unsubscribe()  # must not raise


def test_a_handle_that_was_never_registered_is_inactive():
    bus = EventBus()
    handle = SubscriptionHandle(bus)

    assert not handle.active
    handle.unsubscribe()  # must not raise
    assert "inactive" in repr(handle)


# ---------------------------------------------------------------------------
# Handles that are not on a bus: list-backed callbacks and groups
# ---------------------------------------------------------------------------


def test_a_registered_callback_handle_removes_its_callback():
    registry: list = []
    received: list = []
    handle = register_callback(registry, received.append)

    assert isinstance(handle, SubscriptionHandle)
    assert handle.active
    fire_registered(registry, 1)
    handle.unsubscribe()
    handle.unsubscribe()  # safe to repeat
    fire_registered(registry, 2)

    assert received == [1]
    assert not handle.active
    assert registry == []


def test_one_callable_registered_twice_is_two_subscriptions():
    """Bound methods of one object compare equal; removal must not mix them up."""
    registry: list = []
    listener = _Listener()
    first = register_callback(registry, listener.on_event)
    second = register_callback(registry, listener.on_event)

    first.unsubscribe()

    assert not first.active
    assert second.active
    fire_registered(registry, "x")
    assert listener.received == ["x"]


def test_a_one_shot_callback_is_inactive_inside_itself():
    registry: list = []
    seen: list[bool] = []
    handle = register_callback(registry, lambda: seen.append(handle.active))

    fire_registered(registry, once=True)
    fire_registered(registry, once=True)

    assert seen == [False]
    assert not handle.active


@pytest.mark.parametrize("once", [False, True])
def test_a_registered_callback_unsubscribed_mid_pass_is_not_called(once):
    registry: list = []
    received: list[str] = []
    register_callback(registry, lambda: (received.append("a"), later.unsubscribe()))
    later = register_callback(registry, lambda: received.append("b"))

    fire_registered(registry, once=once)

    assert received == ["a"]


def test_the_bus_ignores_a_handle_that_is_not_its_own():
    bus = EventBus()
    handle = register_callback([], print)

    bus.unsubscribe(handle)  # must not raise

    assert handle.active


def test_an_inactive_handle_reads_inactive():
    handle = inactive_handle()

    assert not handle.active
    handle.unsubscribe()  # must not raise
    assert "inactive" in repr(handle)


def test_a_group_is_active_while_any_member_is():
    bus = EventBus()
    registry: list = []
    on_bus = bus.subscribe(DimsChangedEvent, print)
    in_list = register_callback(registry, print)
    group = group_handles([on_bus, in_list])

    on_bus.unsubscribe()
    assert group.active

    group.unsubscribe()
    assert not group.active
    assert not in_list.active
    group.unsubscribe()  # safe to repeat
