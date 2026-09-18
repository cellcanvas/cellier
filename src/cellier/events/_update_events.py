"""Incoming update event NamedTuples for the cellier v2 IncomingEventBus.

GUI widgets (and other callers) emit these onto ``CellierController.incoming_events``
to request model mutations.  The controller subscribes handler methods that
dispatch each event to the corresponding ``update_*`` method, threading
``source_id`` end-to-end so that echo filtering works on the outgoing bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID


class AppearanceUpdateEvent(NamedTuple):
    """Request to set one appearance field on a visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``AppearanceChangedEvent``
        so the caller can echo-filter on ``on_visual_changed``.
    visual_id :
        Target visual.
    field :
        Attribute name on the appearance model, e.g. ``"clim"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


class DimsUpdateEvent(NamedTuple):
    """Request to update slice positions and/or displayed axes for a scene.

    Set whichever fields you want to change; leave the others as ``None``.
    ``slice_indices`` is **merged** into the scene's positions, so it names
    only the axes that moved.  Every axis keeps a position whatever is
    displayed, so a 2D/3D switch sends ``displayed_axes`` alone.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``DimsChangedEvent``.
    scene_id :
        Target scene.
    slice_indices :
        Mapping of axis index -> world slice position for the axes that
        moved, or ``None`` to leave every position unchanged.
    displayed_axes :
        Tuple of axis indices to display, or ``None`` to leave the
        current displayed axes unchanged.
    """

    source_id: UUID
    scene_id: UUID
    slice_indices: dict[int, float] | None
    displayed_axes: tuple[int, ...] | None


class SliderOverrideUpdateEvent(NamedTuple):
    """Request to force a world axis's slider shown or hidden, or clear that.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``SliderAxesChangedEvent``
        when the effective slider axes change.
    scene_id :
        Target scene.
    axis :
        World axis index.
    value :
        ``True`` force-shows, ``False`` force-hides, ``None`` returns the
        axis to automatic.
    """

    source_id: UUID
    scene_id: UUID
    axis: int
    value: bool | None


class AABBUpdateEvent(NamedTuple):
    """Request to set one AABB parameter field on a visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``AABBChangedEvent``.
    visual_id :
        Target visual.
    field :
        Attribute name on the AABB model, e.g. ``"enabled"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


class OverlayUpdateEvent(NamedTuple):
    """Request to set one field on an overlay model.

    Serves both overlay categories: the controller finds the overlay by id,
    whichever canvas or scene holds it.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``OverlayChangedEvent``.
    overlay_id :
        Target overlay.
    field :
        Dotted path on the overlay model, e.g. ``"visible"`` or
        ``"appearance.color"``.
    value :
        New value for the field.
    """

    source_id: UUID
    overlay_id: UUID
    field: str
    value: Any


class ChannelAppearanceUpdateEvent(NamedTuple):
    """Request to set one appearance field on one channel of an image visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``ChannelAppearanceChangedEvent``
        so the caller can echo-filter on its own subscription.
    visual_id :
        Target visual.
    channel_index :
        Index into ``visual.channels`` selecting the ``ChannelAppearance``.
    field :
        Attribute name on the channel appearance model, e.g. ``"clim"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    channel_index: int
    field: str
    value: Any


class SingleAppearanceUpdateEvent(NamedTuple):
    """Request to set one field on an image visual's single-mode appearance.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``SingleAppearanceChangedEvent``.
    visual_id :
        Target image visual.
    field :
        Attribute name on ``visual.single``, e.g. ``"clim"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


class ImageCompositeUpdateEvent(NamedTuple):
    """Request to switch an image visual between single and composite mode.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``ImageCompositeChangedEvent``.
    visual_id :
        Target image visual.
    composite :
        ``True`` for composite mode, ``False`` for single mode.
    """

    source_id: UUID
    visual_id: UUID
    composite: bool


class BackgroundUpdateEvent(NamedTuple):
    """Request to set one field on a scene's background appearance.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``BackgroundChangedEvent``
        so the caller can echo-filter on its own subscription.
    scene_id :
        Target scene.
    field :
        Attribute name on the background model, e.g. ``"top_color"``.
    value :
        New value for the field.
    """

    source_id: UUID
    scene_id: UUID
    field: str
    value: Any


class RenderConfigUpdateEvent(NamedTuple):
    """Request to set one field on a section of the render configuration.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``RenderConfigChangedEvent``
        so the caller can echo-filter on its own subscription.
    section :
        Which configuration section to write: ``"outline"``, ``"ambient_occlusion"`` or
        ``"temporal"``.
    field :
        Dotted attribute path within the section, e.g. ``"power"`` or
        ``"selection.inward_thickness"``.  The path is dotted because
        ``OutlineConfig`` nests two layer models; the controller resolves it
        and decides whether the write is a uniform or a shader recompile, so
        a widget never has to know which.
    value :
        New value for the field.
    """

    source_id: UUID
    section: str
    field: str
    value: Any


class VisualRenderUpdateEvent(NamedTuple):
    """Request to set one screen-space render field on a visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``VisualRenderChangedEvent``
        so the caller can echo-filter on its own subscription.
    visual_id :
        Target visual.
    field :
        ``"outline.slot"``, ``"outline.placement"``,
        ``"ambient_occlusion"`` or ``"outline_selected_labels"``.
    value :
        New value for the field.
    """

    source_id: UUID
    visual_id: UUID
    field: str
    value: Any


class TrailUpdateEvent(NamedTuple):
    """Request to set or clear the trail window on one axis of a graph visual.

    Fields
    ------
    source_id :
        Caller's UUID.  Stamped on the outgoing ``TrailChangedEvent`` so the
        caller can echo-filter on its own subscription.
    visual_id :
        Target graph visual.
    axis :
        The data-axis index ``GraphVisual.trail`` keys the window by.
    config :
        The complete ``TrailConfig`` for *axis*, or ``None`` to remove the
        window on that axis.  A whole config rather than one field, so one
        event can both switch a window on and say what it looks like.  Pass a
        config no other visual holds: the controller may adopt the object.
    """

    source_id: UUID
    visual_id: UUID
    axis: int
    config: Any


CellierUpdateEventTypes = (
    AppearanceUpdateEvent
    | DimsUpdateEvent
    | SliderOverrideUpdateEvent
    | AABBUpdateEvent
    | OverlayUpdateEvent
    | ChannelAppearanceUpdateEvent
    | SingleAppearanceUpdateEvent
    | ImageCompositeUpdateEvent
    | BackgroundUpdateEvent
    | RenderConfigUpdateEvent
    | VisualRenderUpdateEvent
    | TrailUpdateEvent
)


class SubscriptionSpec(NamedTuple):
    """One inbound event subscription declared by a cellier widget.

    Pass a list of these to ``CellierController.connect_widget`` so that the
    controller wires the widget's *handler* to the outgoing ``EventBus``
    without the widget ever holding a controller reference.

    Fields
    ------
    event_type :
        The event class to subscribe to, e.g. ``AppearanceChangedEvent``.
    handler :
        Callable receiving the event.  Typically a bound method on the widget,
        e.g. ``my_widget._on_visual_changed``.
    entity_id :
        Optional UUID to scope the subscription to a specific visual, scene,
        or canvas.  ``None`` subscribes to all entities for that event type.
    weak :
        If ``True``, hold only a weak reference to *handler*.  Cannot be used
        with lambdas.  Default is ``False``.
    """

    event_type: type
    handler: Callable[..., None]
    entity_id: UUID | None = None
    weak: bool = False
