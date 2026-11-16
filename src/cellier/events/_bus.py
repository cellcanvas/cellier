"""EventBus implementation."""

from __future__ import annotations

import functools
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID, uuid4

from cellier.events._events import (
    AABBChangedEvent,
    AppearanceChangedEvent,
    BackgroundChangedEvent,
    BackstopCompleteEvent,
    CameraChangedEvent,
    CameraInteractionEvent,
    CanvasAddedEvent,
    CanvasConnectedEvent,
    CanvasMouseMove2DEvent,
    CanvasMouseMove3DEvent,
    CanvasMousePress2DEvent,
    CanvasMousePress3DEvent,
    CanvasMouseRelease2DEvent,
    CanvasMouseRelease3DEvent,
    CanvasSizeChangedEvent,
    ChannelAppearanceChangedEvent,
    ClippingPlanesChangedEvent,
    DataStoreContentsChangedEvent,
    DataStoreMetadataChangedEvent,
    DimsChangedEvent,
    DimsInteractionEvent,
    DrawStatusChangedEvent,
    FrameRenderedEvent,
    GraphPickEvent,
    ImageCompositeChangedEvent,
    ImagePickEvent,
    LabelsPickEvent,
    LinesPickEvent,
    LoadingConfigChangedEvent,
    LodConfigChangedEvent,
    MaskedImagePickEvent,
    MeshPickEvent,
    MeshSectionChangedEvent,
    OverlayChangedEvent,
    PickWriteChangedEvent,
    PlaneGizmoChangedEvent,
    PlaneGizmoMovedEvent,
    PlaneInteractionEvent,
    PointsPickEvent,
    RenderPlanesChangedEvent,
    ResliceCompletedEvent,
    ResliceProgressEvent,
    ResliceStartedEvent,
    SceneAddedEvent,
    SceneRemovedEvent,
    SingleAppearanceChangedEvent,
    SliderAxesChangedEvent,
    TrailChangedEvent,
    TransformChangedEvent,
    TransformInteractionEvent,
    VisualAddedEvent,
    VisualRemovedEvent,
    VisualRenderChangedEvent,
    VisualVisibilityChangedEvent,
)

# Map every event type to its canonical entity-filter field name.
_ENTITY_FIELD: dict[type, str] = {
    DimsChangedEvent: "scene_id",
    DimsInteractionEvent: "scene_id",
    SliderAxesChangedEvent: "scene_id",
    CameraChangedEvent: "scene_id",
    CameraInteractionEvent: "scene_id",
    AppearanceChangedEvent: "visual_id",
    ChannelAppearanceChangedEvent: "visual_id",
    SingleAppearanceChangedEvent: "visual_id",
    ImageCompositeChangedEvent: "visual_id",
    PickWriteChangedEvent: "visual_id",
    AABBChangedEvent: "visual_id",
    MeshSectionChangedEvent: "visual_id",
    VisualVisibilityChangedEvent: "visual_id",
    VisualRenderChangedEvent: "visual_id",
    DataStoreMetadataChangedEvent: "data_store_id",
    DataStoreContentsChangedEvent: "data_store_id",
    ResliceStartedEvent: "scene_id",
    ResliceCompletedEvent: "visual_id",
    ResliceProgressEvent: "visual_id",
    BackstopCompleteEvent: "visual_id",
    LoadingConfigChangedEvent: "visual_id",
    LodConfigChangedEvent: "visual_id",
    ClippingPlanesChangedEvent: "visual_id",
    RenderPlanesChangedEvent: "visual_id",
    PlaneInteractionEvent: "visual_id",
    TransformInteractionEvent: "visual_id",
    PlaneGizmoChangedEvent: "canvas_id",
    PlaneGizmoMovedEvent: "gizmo_id",
    FrameRenderedEvent: "canvas_id",
    CanvasSizeChangedEvent: "canvas_id",
    CanvasConnectedEvent: "canvas_id",
    VisualAddedEvent: "scene_id",
    VisualRemovedEvent: "scene_id",
    TrailChangedEvent: "visual_id",
    TransformChangedEvent: "visual_id",
    DrawStatusChangedEvent: "scene_id",
    SceneAddedEvent: "scene_id",
    CanvasAddedEvent: "scene_id",
    OverlayChangedEvent: "overlay_id",
    BackgroundChangedEvent: "scene_id",
    SceneRemovedEvent: "scene_id",
    CanvasMousePress2DEvent: "source_id",
    CanvasMouseMove2DEvent: "source_id",
    CanvasMouseRelease2DEvent: "source_id",
    CanvasMousePress3DEvent: "source_id",
    CanvasMouseMove3DEvent: "source_id",
    CanvasMouseRelease3DEvent: "source_id",
    GraphPickEvent: "source_id",
    ImagePickEvent: "source_id",
    LabelsPickEvent: "source_id",
    MaskedImagePickEvent: "source_id",
    LinesPickEvent: "source_id",
    MeshPickEvent: "source_id",
    PointsPickEvent: "source_id",
}


@dataclass
class SubscriberInfo:
    """Human-readable summary of one EventBus subscription.

    Returned by `EventBus.get_subscribers`.  Intended for
    interactive debugging and test assertions — not for production
    control flow.

    Attributes
    ----------
    callback_qualname : str
        Dotted qualified name of the registered callback, e.g.
        ``"MyPanel.on_dims_changed"``.  Reads ``"(dead)"``
        when the callback was held by a weak reference that has since
        been garbage-collected.
    callback_instance : object
        The bound object when the callback is a bound method; ``None``
        for plain functions or dead weak references.  In a debugger,
        inspect this to examine the live subscriber.
    owner_id : UUID or None
        UUID used to group this subscription for bulk removal via
        `EventBus.unsubscribe_all`.  Typically the id of the owning
        widget, or of the controller for its own subscriptions.
    entity_id : UUID or None
        The entity this subscription is scoped to, or ``None`` when
        the subscription receives events for all entities of that
        type.  See `EventBus.get_subscribers` for a full
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
    """One subscription, returned by every ``on_*`` method.

    Call ``unsubscribe()`` to stop the callback.  The handle needs nothing
    else: it finds its own event bus, so code that keeps a handle does not
    also have to keep the controller or viewer that made it.

    Examples
    --------
    >>> handle = viewer.on_pick(canvas_id, ImagePickEvent, show_value)
    >>> handle.active
    True
    >>> handle.unsubscribe()  # no more calls to show_value
    >>> handle.active
    False

    A callback that should run once can cancel itself:

    >>> def first_change(event):
    ...     print(event.dims_state)
    ...     handle.unsubscribe()
    >>> handle = controller.on_dims_changed(scene_id, first_change, owner_id=me)

    Notes
    -----
    The handle holds only a weak reference to its bus, so a handle kept after
    the controller is gone does not keep the controller alive; it then reads
    as inactive and ``unsubscribe()`` does nothing.

    The viewers' startup hooks (``on_ready``, ``on_scene_ready``,
    ``on_startup_progress``, ``on_startup_stalled``) do not go through the
    bus, and an ``OrthoViewer`` subscription can span all four panels.  Their
    handles behave the same way: ``unsubscribe()`` removes the whole
    subscription, and ``active`` says whether any of it is left.
    """

    __slots__ = ("_bus_ref", "_id")

    def __init__(self, bus: EventBus) -> None:
        self._id: UUID = uuid4()
        self._bus_ref: weakref.ref[EventBus] = weakref.ref(bus)

    @property
    def active(self) -> bool:
        """Whether the subscription is still registered.

        ``False`` once it was removed by any route: ``unsubscribe()``, its
        owner's removal, a one-shot callback firing, a weak callback being
        garbage-collected (seen at the next emission of its event), or the
        controller closing.
        """
        bus = self._bus_ref()
        return bus is not None and self._id in bus._handle_index

    def unsubscribe(self) -> None:
        """Remove the subscription.

        Safe to call more than once, from inside the callback itself, and
        after the subscription was removed some other way.
        """
        bus = self._bus_ref()
        if bus is not None:
            bus.unsubscribe(self)

    def __repr__(self) -> str:
        state = "active" if self.active else "inactive"
        return f"SubscriptionHandle({self._id}, {state})"


class _CallbackHandle(SubscriptionHandle):
    """A handle for a callback kept in a plain list instead of on a bus.

    *entry* is the object stored in *registry*; the subscription is active
    while that exact object is in the list.  Whoever fires the list removes a
    one-shot entry before calling it (see ``fire_registered``).  A
    *registry* of ``None`` gives a handle that was never registered: the
    callback already ran.

    Not constructed directly; use ``register_callback``.
    """

    __slots__ = ("_entry", "_registry")

    def __init__(self, registry: list | None, entry: Callable | None) -> None:
        # The base slots: ``_id`` keeps ``EventBus.unsubscribe`` a no-op for
        # this handle, as for any handle from another bus.
        self._id = uuid4()
        self._registry = registry
        self._entry = entry

    @property
    def active(self) -> bool:
        """Whether the callback is still registered."""
        if self._registry is None:
            return False
        return any(item is self._entry for item in self._registry)

    def unsubscribe(self) -> None:
        """Remove the callback.  Safe to call more than once."""
        if self._registry is not None:
            _remove_identical(self._registry, self._entry)


class _HandleGroup(SubscriptionHandle):
    """One handle over several subscriptions, such as one per panel.

    ``unsubscribe()`` removes every member; ``active`` is ``True`` while any
    member is.
    """

    __slots__ = ("_members",)

    def __init__(self, members: list[SubscriptionHandle]) -> None:
        self._id = uuid4()
        self._members = members

    @property
    def active(self) -> bool:
        """Whether any member subscription is still registered."""
        return any(member.active for member in self._members)

    def unsubscribe(self) -> None:
        """Remove every member subscription.  Safe to call more than once."""
        for member in self._members:
            member.unsubscribe()


def _remove_identical(registry: list, entry: object) -> bool:
    """Remove *entry* from *registry* by identity; report whether it was there.

    ``list.remove`` compares with ``==``, and two bound methods of one
    object compare equal, so it could remove another registration.
    """
    for index, item in enumerate(registry):
        if item is entry:
            del registry[index]
            return True
    return False


def register_callback(registry: list, callback: Callable) -> SubscriptionHandle:
    """Append *callback* to *registry* and return a handle that removes it.

    The callback is stored behind a fresh wrapper, so registering the same
    callable twice gives two independent subscriptions.
    """
    entry = functools.partial(callback)
    registry.append(entry)
    return _CallbackHandle(registry, entry)


def group_handles(members: list[SubscriptionHandle]) -> SubscriptionHandle:
    """One handle over *members*, such as one subscription per panel.

    The list is kept, not copied, so members appended later still count.
    """
    return _HandleGroup(members)


def inactive_handle() -> SubscriptionHandle:
    """A handle for a callback that already ran and was never registered."""
    return _CallbackHandle(None, None)


def fire_registered(registry: list, *args: Any, once: bool = False) -> None:
    """Call every callback in *registry* with *args*.

    Iterates a snapshot, so a callback may register or unsubscribe others.
    A callback unsubscribed by an earlier one in the same pass is skipped,
    as on the bus.  With *once*, each callback is removed before it runs,
    so its handle already reads inactive inside the callback.
    """
    for entry in list(registry):
        if once:
            if not _remove_identical(registry, entry):
                continue
        elif not any(item is entry for item in registry):
            continue
        entry(*args)


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
        "entity_field",
        "entity_id",
        "handle",
        "is_weak",
        "on_remove",
        "owner_id",
        "removed",
    )

    def __init__(
        self,
        handle: SubscriptionHandle,
        callback: Callable,
        entity_id: UUID | None,
        owner_id: UUID | None,
        weak: bool,
        entity_field: str | None = None,
        on_remove: Callable[[], None] | None = None,
    ) -> None:
        self.handle = handle
        self.entity_id = entity_id
        self.entity_field = entity_field
        self.owner_id = owner_id
        self.is_weak = weak
        self.on_remove = on_remove
        # Set once the bus has dropped this record, so an emission already
        # iterating a snapshot of the list skips it.
        self.removed = False
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
    """Synchronous, typed event bus.

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
        entity_field: str | None = None,
        on_remove: Callable[[], None] | None = None,
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
        entity_field:
            Name of the event field *entity_id* is compared with, for this
            subscription only.  ``None`` (the default) uses the event type's
            canonical field.  Use it to filter on another id the event
            carries, e.g. ``"visual_id"`` for ``VisualRemovedEvent``, which
            is routed by ``scene_id``.
        on_remove:
            Zero-argument cleanup run once when the subscription is removed,
            whichever route removes it: `unsubscribe` (also through
            `SubscriptionHandle.unsubscribe`), `unsubscribe_all`,
            or the clean-up of a dead weak callback during `emit`.
            `clear` does not run it.  Use it for bookkeeping that must
            stay in step with the subscription, such as a subscriber count.

        Returns
        -------
        SubscriptionHandle
            Call its `unsubscribe` (or pass it to
            `unsubscribe`) to remove this subscription on its own.

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

        handle = SubscriptionHandle(self)
        sub = _Subscription(
            handle=handle,
            callback=callback,
            entity_id=entity_id,
            owner_id=owner_id,
            weak=weak,
            entity_field=entity_field,
            on_remove=on_remove,
        )
        self._subs[event_type].append(sub)
        self._handle_index[handle._id] = (event_type, sub)
        return handle

    def _remove(self, event_type: type, sub: _Subscription) -> None:
        """Drop *sub* and run its ``on_remove`` cleanup.

        The one place a subscription leaves the bus (except `clear`), so
        the cleanup runs exactly once whichever route removed it.  An
        exception from the cleanup propagates after the record is gone.
        """
        if sub.removed:
            return
        sub.removed = True
        subs = self._subs.get(event_type)
        if subs is not None and sub in subs:
            subs.remove(sub)
        self._handle_index.pop(sub.handle._id, None)
        if sub.on_remove is not None:
            sub.on_remove()

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """Remove the subscription identified by *handle*.

        Does nothing when it was already removed.
        """
        entry = self._handle_index.get(handle._id)
        if entry is not None:
            self._remove(*entry)

    def clear(self) -> None:
        """Drop every subscription without running ``on_remove`` cleanups.

        For teardown only.  Subscriptions hold **strong** references to their
        callbacks by default, and those callbacks are typically bound methods
        of the controller, its render visuals and its widgets -- so a bus that
        outlives its owner keeps that whole graph reachable.  Unsubscribing by
        owner requires knowing every owner id; this is the catch-all for
        ``CellierController.close``.  The cleanups are skipped because the
        state they keep in step is being torn down with the bus.
        """
        for subs in self._subs.values():
            for sub in subs:
                sub.removed = True
        self._subs.clear()
        self._handle_index.clear()

    def unsubscribe_all(self, owner_id: UUID) -> None:
        """Remove all subscriptions whose ``owner_id`` matches *owner_id*.

        Every matching subscription is removed even when a cleanup raises; the
        first such exception is re-raised afterwards.
        """
        first_exc: BaseException | None = None
        for event_type, subs in list(self._subs.items()):
            for sub in [s for s in subs if s.owner_id == owner_id]:
                try:
                    self._remove(event_type, sub)
                except Exception as exc:
                    if first_exc is None:
                        first_exc = exc
        if first_exc is not None:
            raise first_exc

    def emit(self, event) -> None:
        """Dispatch *event* to all matching subscribers.

        Subscribers are called in registration order.  If a subscriber
        raises an exception, remaining subscribers still run; the first
        exception is re-raised after all handlers have been called.

        A subscription removed by an earlier subscriber during the same
        emission is not called.  Dead weak references are cleaned up lazily
        during emission, which runs their ``on_remove`` cleanups.
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
            if sub.removed:
                continue
            # Collect dead weak subscriptions for cleanup
            if sub.is_weak and not sub.is_alive():
                dead.append(sub)
                continue

            # Entity filter
            if sub.entity_id is not None:
                if sub.entity_field is None:
                    target_id = event_entity_id
                else:
                    target_id = getattr(event, sub.entity_field, None)
                if sub.entity_id != target_id:
                    continue

            try:
                sub.call(event)
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc

        # Clean up confirmed-dead weak subscriptions
        for sub in dead:
            try:
                self._remove(event_type, sub)
            except Exception as exc:
                if first_exc is None:
                    first_exc = exc

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
              ``ResliceProgressEvent``, ``BackstopCompleteEvent``,
              ``LoadingConfigChangedEvent``,
              ``LodConfigChangedEvent``,
              ``ClippingPlanesChangedEvent``,
              ``RenderPlanesChangedEvent``,
              ``PlaneInteractionEvent``,
              ``TransformInteractionEvent``
            - Canvas (keyed by ``canvas_id``) — ``FrameRenderedEvent``,
              ``CanvasConnectedEvent``, ``CanvasSizeChangedEvent``,
              ``PlaneGizmoChangedEvent``
            - Data store (keyed by ``data_store_id``) —
              ``DataStoreMetadataChangedEvent``,
              ``DataStoreContentsChangedEvent``
            - Plane gizmo (keyed by ``gizmo_id``) —
              ``PlaneGizmoMovedEvent``

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
