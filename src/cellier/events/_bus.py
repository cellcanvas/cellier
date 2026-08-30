"""EventBus implementation for cellier v2."""

from __future__ import annotations

import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID, uuid4

from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    CameraChangedEvent,
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    ChannelAppearanceChangedEvent,
    DataStoreContentsChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    FrameRenderedEvent,
    PickWriteChangedEvent,
    ResliceCancelledEvent,
    ResliceCompletedEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    TrailChangedEvent,
    TransformChangedEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualVisibilityChangedEvent,
)

# Map every event type to its canonical entity-filter field name.
_ENTITY_FIELD: dict[type, str] = {
    DimsChangedEvent: "scene_id",
    CameraChangedEvent: "scene_id",
    AppearanceChangedEvent: "visual_id",
    ChannelAppearanceChangedEvent: "visual_id",
    PickWriteChangedEvent: "visual_id",
    AABBChangedEvent: "visual_id",
    VisualVisibilityChangedEvent: "visual_id",
    DataStoreMetadataChangedEvent: "data_store_id",
    DataStoreContentsChangedEvent: "data_store_id",
    ResliceStartedEvent: "scene_id",
    ResliceCompletedEvent: "visual_id",
    ResliceCancelledEvent: "visual_id",
    FrameRenderedEvent: "canvas_id",
    VisualAddedEvent: "scene_id",
    VisualRemovedEvent: "scene_id",
    TrailChangedEvent: "visual_id",
    TransformChangedEvent: "visual_id",
    SceneAddedEvent: "scene_id",
    SceneRemovedEvent: "scene_id",
    CanvasMousePress2DEvent: "source_id",
    CanvasMouseMove2DEvent: "source_id",
    CanvasMouseRelease2DEvent: "source_id",
    CanvasMousePress3DEvent: "source_id",
    CanvasMouseMove3DEvent: "source_id",
    CanvasMouseRelease3DEvent: "source_id",
}


@dataclass
class SubscriberInfo:
    """Human-readable summary of one EventBus subscription.

    Returned by :meth:`EventBus.get_subscribers`.  Intended for
    interactive debugging and test assertions — not for production
    control flow.

    Attributes
    ----------
    callback_qualname : str
        Dotted qualified name of the registered callback, e.g.
        ``"QtDimsControl._on_dims_changed"``.  Reads ``"(dead)"``
        when the callback was held by a weak reference that has since
        been garbage-collected.
    callback_instance : object
        The bound object when the callback is a bound method; ``None``
        for plain functions or dead weak references.  In a debugger,
        inspect this to examine the live subscriber.
    owner_id : UUID or None
        UUID used to group this subscription for bulk removal via
        :meth:`EventBus.unsubscribe_all`.  Typically the owning
        widget's ``self._id`` or the controller's ``self._id``.
    entity_id : UUID or None
        The entity this subscription is scoped to, or ``None`` when
        the subscription receives events for all entities of that
        type.  See :meth:`EventBus.get_subscribers` for a full
        explanation of what constitutes an entity.
    is_weak : bool
        ``True`` when the bus holds only a weak reference to the
        callback, meaning the subscriber may be garbage-collected
        without explicit teardown.
    is_alive : bool
        ``True`` when the callback target is still reachable.  Always
        ``True`` for strong references; may become ``False`` for weak
        ones after the subscriber is garbage-collected.
    """

    callback_qualname: str
    callback_instance: Any
    owner_id: UUID | None
    entity_id: UUID | None
    is_weak: bool
    is_alive: bool


class SubscriptionHandle:
    """Opaque handle returned by ``EventBus.subscribe()``.

    Pass to ``EventBus.unsubscribe()`` to remove a single subscription.
    """

    __slots__ = ("_id",)

    def __init__(self) -> None:
        self._id: UUID = uuid4()

    def __repr__(self) -> str:
        return f"SubscriptionHandle({self._id})"


class _Subscription:
    """Internal record for one subscription.

    For weak subscriptions, liveness is tracked via raw ``weakref.ref``
    objects so that dead entries can be detected and cleaned up during
    emission without requiring any coordination from the subscriber.
    """

    __slots__ = (
        "_strong_cb",
        "_weak_func",
        "_weak_obj",
        "entity_id",
        "handle",
        "is_weak",
        "owner_id",
    )

    def __init__(
        self,
        handle: SubscriptionHandle,
        callback: Callable,
        entity_id: UUID | None,
        owner_id: UUID | None,
        weak: bool,
    ) -> None:
        self.handle = handle
        self.entity_id = entity_id
        self.owner_id = owner_id
        self.is_weak = weak
        self._strong_cb: Callable | None = None
        self._weak_obj: weakref.ref | None = None
        self._weak_func: weakref.ref | None = None

        if weak:
            if hasattr(callback, "__self__"):
                # bound method
                self._weak_obj = weakref.ref(callback.__self__)
                self._weak_func = weakref.ref(callback.__func__)
            else:
                self._weak_func = weakref.ref(callback)
        else:
            self._strong_cb = callback

    def is_alive(self) -> bool:
        """Return True if the callback target is still reachable."""
        if not self.is_weak:
            return True
        if self._weak_obj is not None:
            return self._weak_obj() is not None and self._weak_func() is not None
        return self._weak_func() is not None  # type: ignore[union-attr]

    def call(self, event) -> None:
        """Invoke the callback with *event*. Assumes ``is_alive()`` is True."""
        if not self.is_weak:
            self._strong_cb(event)  # type: ignore[misc]
            return
        if self._weak_obj is not None:
            obj = self._weak_obj()
            func = self._weak_func()
            if obj is not None and func is not None:
                func(obj, event)
        else:
            fn = self._weak_func()  # type: ignore[union-attr]
            if fn is not None:
                fn(event)

    def __repr__(self) -> str:
        if self._strong_cb is not None:
            qualname = getattr(self._strong_cb, "__qualname__", repr(self._strong_cb))
        elif self._weak_obj is not None:
            # Bound method stored as (weak instance, weak unbound function) pair.
            live_func = self._weak_func() if self._weak_func is not None else None
            qualname = (
                getattr(live_func, "__qualname__", "(dead)")
                if live_func is not None
                else "(dead)"
            )
        elif self._weak_func is not None:
            live_func = self._weak_func()
            qualname = (
                getattr(live_func, "__qualname__", "(dead)")
                if live_func is not None
                else "(dead)"
            )
        else:
            qualname = "(unknown)"
        reference_kind = "weak" if self.is_weak else "strong"
        liveness = "alive" if self.is_alive() else "dead"
        return (
            f"_Subscription(callback={qualname!r} "
            f"owner_id={self.owner_id} "
            f"entity_id={self.entity_id} "
            f"{reference_kind}/{liveness})"
        )


def _build_subscriber_info(subscription: _Subscription) -> SubscriberInfo:
    """Extract human-readable metadata from one internal subscription record."""
    if subscription._strong_cb is not None:
        live_callback = subscription._strong_cb
        qualname = getattr(live_callback, "__qualname__", repr(live_callback))
        instance = getattr(live_callback, "__self__", None)
    elif subscription._weak_obj is not None:
        # Bound method stored as (weak instance, weak unbound function) pair.
        live_instance = subscription._weak_obj()
        live_func = (
            subscription._weak_func() if subscription._weak_func is not None else None
        )
        qualname = (
            getattr(live_func, "__qualname__", "(dead)")
            if live_func is not None
            else "(dead)"
        )
        instance = live_instance
    elif subscription._weak_func is not None:
        # Plain function stored as a weak reference.
        live_func = subscription._weak_func()
        qualname = (
            getattr(live_func, "__qualname__", "(dead)")
            if live_func is not None
            else "(dead)"
        )
        instance = None
    else:
        qualname = "(unknown)"
        instance = None

    return SubscriberInfo(
        callback_qualname=qualname,
        callback_instance=instance,
        owner_id=subscription.owner_id,
        entity_id=subscription.entity_id,
        is_weak=subscription.is_weak,
        is_alive=subscription.is_alive(),
    )


class EventBus:
    """Synchronous, typed event bus for cellier v2.

    Subscribers declare the event type they want, optionally scoped to a
    specific entity (scene, visual, canvas, data store).  All dispatch is
    synchronous.  Handlers that need async work should call
    ``asyncio.create_task()`` and return immediately.

    Strong references are held by default.  Pass ``weak=True`` for
    subscribers that may be garbage-collected outside the controller's
    explicit teardown path.  Lambdas cannot be weakly referenced and will
    raise ``ValueError`` when ``weak=True``.
    """

    def __init__(self) -> None:
        # event_type → list of _Subscription
        self._subs: dict[type, list[_Subscription]] = defaultdict(list)
        # handle._id → (event_type, _Subscription) for O(1) single removal
        self._handle_index: dict[UUID, tuple[type, _Subscription]] = {}

    def subscribe(
        self,
        event_type: type,
        callback: Callable,
        *,
        entity_id: UUID | None = None,
        owner_id: UUID | None = None,
        weak: bool = False,
    ) -> SubscriptionHandle:
        """Register *callback* to be called when *event_type* is emitted.

        Parameters
        ----------
        event_type:
            The event class to subscribe to.
        callback:
            Callable that accepts one positional argument: the event instance.
        entity_id:
            When set, only events whose canonical entity field matches this
            UUID will be dispatched to *callback*.  ``None`` matches all.
        owner_id:
            Logical owner for bulk removal via ``unsubscribe_all(owner_id)``.
        weak:
            If True, hold only a weak reference to *callback*.  Lambdas
            cannot be weakly referenced and will raise ``ValueError``.

        Returns
        -------
        SubscriptionHandle
            Pass to ``unsubscribe()`` to remove this subscription individually.

        Raises
        ------
        ValueError
            If *weak=True* and *callback* is a lambda.
        """
        if weak and getattr(callback, "__name__", None) == "<lambda>":
            raise ValueError(
                "Cannot create a weak subscription to a lambda. "
                "Use a named method or function instead."
            )

        handle = SubscriptionHandle()
        sub = _Subscription(
            handle=handle,
            callback=callback,
            entity_id=entity_id,
            owner_id=owner_id,
            weak=weak,
        )
        self._subs[event_type].append(sub)
        self._handle_index[handle._id] = (event_type, sub)
        return handle

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """Remove the subscription identified by *handle*."""
        entry = self._handle_index.pop(handle._id, None)
        if entry is None:
            return
        event_type, sub = entry
        subs = self._subs.get(event_type, [])
        try:
            subs.remove(sub)
        except ValueError:
            pass

    def unsubscribe_all(self, owner_id: UUID) -> None:
        """Remove all subscriptions whose ``owner_id`` matches *owner_id*."""
        for subs in self._subs.values():
            to_remove = [s for s in subs if s.owner_id == owner_id]
            for sub in to_remove:
                subs.remove(sub)
                self._handle_index.pop(sub.handle._id, None)

    def emit(self, event) -> None:
        """Dispatch *event* to all matching subscribers.

        Subscribers are called in registration order.  If a subscriber
        raises an exception, remaining subscribers still run; the first
        exception is re-raised after all handlers have been called.

        Dead weak references are cleaned up lazily during emission.
        """
        event_type = type(event)
        subs = self._subs.get(event_type)
        if not subs:
            return

        entity_field = _ENTITY_FIELD.get(event_type)
        event_entity_id: UUID | None = (
            getattr(event, entity_field, None) if entity_field else None
        )

        first_exc: BaseException | None = None
        dead: list[_Subscription] = []

        for sub in list(subs):
            # Collect dead weak subscriptions for cleanup
            if sub.is_weak and not sub.is_alive():
                dead.append(sub)
                continue

            # Entity filter
            if sub.entity_id is not None and sub.entity_id != event_entity_id:
                continue

            try:
                sub.call(event)
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc

        # Clean up confirmed-dead weak subscriptions
        for sub in dead:
            subs.remove(sub)
            self._handle_index.pop(sub.handle._id, None)

        if first_exc is not None:
            raise first_exc

    def get_subscribers(
        self,
        event_type: type,
        *,
        entity_id: UUID | None = None,
    ) -> list[SubscriberInfo]:
        """Return a summary of every subscription registered for *event_type*.

        Intended for interactive debugging and introspection.  Do not
        use in production control flow.

        Parameters
        ----------
        event_type :
            The event class to query, e.g. ``DimsChangedEvent``.
        entity_id :
            Scope the results to subscriptions that would fire when an
            event targets this specific entity UUID.  Pass ``None``
            (the default) to return every subscription for
            *event_type* regardless of scope.

            **What is an entity?**

            Each event type is associated with exactly one model-layer
            object whose UUID acts as the routing key — the *entity*.
            The full mapping is:

            - Scene (keyed by ``scene_id``) — ``DimsChangedEvent``,
              ``CameraChangedEvent``, ``ResliceStartedEvent``,
              ``VisualAddedEvent``, ``VisualRemovedEvent``,
              ``SceneAddedEvent``, ``SceneRemovedEvent``
            - Visual (keyed by ``visual_id``) —
              ``AppearanceChangedEvent``, ``AABBChangedEvent``,
              ``VisualVisibilityChangedEvent``,
              ``TransformChangedEvent``, ``ResliceCompletedEvent``,
              ``ResliceCancelledEvent``
            - Canvas (keyed by ``canvas_id``) — ``FrameRenderedEvent``
            - Data store (keyed by ``data_store_id``) —
              ``DataStoreMetadataChangedEvent``,
              ``DataStoreContentsChangedEvent``

            Subscribing with ``entity_id=some_scene_id`` means the
            callback only fires for events whose ``scene_id`` matches.
            Subscribing with ``entity_id=None`` means the callback
            fires for every emission of that type.

            When this method receives *entity_id*, the results include
            both subscriptions scoped to that UUID **and** unscoped
            subscriptions (``entity_id=None``), because both would be
            called by the bus for that entity's events.

        Returns
        -------
        list[SubscriberInfo]
            One entry per matching subscription, in registration order.
        """
        all_subscriptions = self._subs.get(event_type, [])
        results: list[SubscriberInfo] = []
        for subscription in all_subscriptions:
            if entity_id is not None:
                # Mirror the dispatch logic in emit(): unscoped subscriptions
                # (entity_id=None) match every entity and are always included.
                if (
                    subscription.entity_id is not None
                    and subscription.entity_id != entity_id
                ):
                    continue
            results.append(_build_subscriber_info(subscription))
        return results
