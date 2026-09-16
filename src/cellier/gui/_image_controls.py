"""Toolkit-neutral half of the unified image control (unified image design 3.10).

Both front ends draw the same control -- a shared section, a mode switch, a
single-mode page and a composite page -- and have to answer the same questions
about it: what it starts with, which bus event a change to each field rides
on, and how a model value is shown.  Those answers live here once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    ChannelAppearanceChangedEvent,
    ChannelAppearanceUpdateEvent,
    ImageCompositeChangedEvent,
    SingleAppearanceChangedEvent,
    SingleAppearanceUpdateEvent,
    VisualVisibilityChangedEvent,
)
from cellier.gui._colormap_util import colormap_to_str

if TYPE_CHECKING:
    from uuid import UUID

#: Shared-section fields, in display order.
SHARED_FIELDS: tuple[str, ...] = (
    "visible",
    "transparency_mode",
    "interpolation",
    "attenuation",
)

#: Fields each mode page shows, in display order.  A channel row adds
#: ``visible`` in front.
MODE_FIELDS: tuple[str, ...] = (
    "color_map",
    "clim",
    "opacity",
    "render_mode",
    "iso_threshold",
)

#: The shared transparency modes, with ``"auto"`` standing for ``None``.
TRANSPARENCY_CHOICES: tuple[str, ...] = (
    "auto",
    "blend",
    "add",
    "multiply",
    "weighted_blend",
    "weighted_solid",
)

INTERPOLATION_CHOICES: tuple[str, ...] = ("nearest", "linear")

#: The inbound events the control listens to, per driven visual.
INBOUND_EVENT_TYPES: tuple[type, ...] = (
    AppearanceChangedEvent,
    VisualVisibilityChangedEvent,
    SingleAppearanceChangedEvent,
    ChannelAppearanceChangedEvent,
    ImageCompositeChangedEvent,
)

FIELD_LABELS: dict[str, str] = {
    "visible": "Visible",
    "transparency_mode": "Blending",
    "interpolation": "Interpolation",
    "attenuation": "Attenuation",
    "color_map": "Colormap",
    "clim": "Contrast",
    "opacity": "Opacity",
    "render_mode": "Render mode",
    "iso_threshold": "Threshold",
}

DEFAULT_COLORMAP_NAMES: list[str] = [
    "gray",
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "cividis",
    "red",
    "green",
    "blue",
    "magenta",
    "cyan",
]


def to_widget_value(field: str, value: Any) -> Any:
    """Show a model value in a control's terms."""
    if field == "color_map":
        return colormap_to_str(value)
    if field == "clim":
        return [float(value[0]), float(value[1])]
    if field == "transparency_mode":
        return "auto" if value is None else str(value)
    if field in ("opacity", "iso_threshold", "attenuation"):
        return float(value)
    return value


def to_model_value(field: str, value: Any) -> Any:
    """Turn a control's value back into what the model field takes."""
    if field == "clim":
        return (float(value[0]), float(value[1]))
    if field == "transparency_mode":
        return None if value in (None, "auto") else str(value)
    return value


def image_update_event(
    source_id: UUID,
    visual_id: UUID,
    page: str,
    field: str,
    value: Any,
    channel: int | None = None,
):
    """The update event that carries one edit of the image control.

    Parameters
    ----------
    source_id, visual_id : UUID
        The control's id and the visual being written.
    page : {"shared", "single", "channel"}
        Which model the field is on.
    field : str
        The field name.
    value : Any
        The control's value; converted with :func:`to_model_value`.
    channel : int or None
        The channel index, for ``page="channel"``.

    Returns
    -------
    NamedTuple
        An ``AppearanceUpdateEvent``, ``SingleAppearanceUpdateEvent`` or
        ``ChannelAppearanceUpdateEvent``.
    """
    model_value = to_model_value(field, value)
    if page == "shared":
        return AppearanceUpdateEvent(source_id, visual_id, field, model_value)
    if page == "single":
        return SingleAppearanceUpdateEvent(source_id, visual_id, field, model_value)
    if page == "channel":
        return ChannelAppearanceUpdateEvent(
            source_id, visual_id, int(channel), field, model_value
        )
    raise ValueError(f"Unknown image control page {page!r}.")


def mode_values(appearance: Any) -> dict[str, Any]:
    """A single or channel appearance, in control terms."""
    values = {
        field: to_widget_value(field, getattr(appearance, field))
        for field in MODE_FIELDS
    }
    if hasattr(appearance, "visible"):
        values["visible"] = bool(appearance.visible)
    return values


def image_control_values(
    visual: Any,
    *,
    fields: list[str] | tuple[str, ...],
    colormap_names: list[str] | None = None,
    clim_range: tuple[float, float] | None = None,
    channel_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Everything either front end needs to build the image control.

    Parameters
    ----------
    visual : BaseImageVisual
        The representative visual.
    fields : sequence of str
        The appearance fields the config asked for.  The shared section's
        blending and interpolation rows and the mode switch are always shown.
    colormap_names : list[str] or None
        Colormap choices.  A curated default when ``None``.
    clim_range : tuple[float, float] or None
        Contrast slider bounds.  Inferred from the current limits when
        ``None``.
    channel_labels : dict[int, str] or None
        Per-channel names; ``"Channel {i}"`` when absent.

    Returns
    -------
    dict
        Plain data, JSON-compatible apart from integer channel keys and
        ``"colormaps"``, which holds the models' own ``color_map`` values.
        A colormap built inline (``Colormap([...], name="white_green")``) is
        not in cmap's catalogue, so its name cannot be resolved back into a
        colormap; a toolkit that needs the real thing reads it from there,
        while the page dicts keep the JSON-safe name.
    """
    from cellier.gui._appearance_fields import literal_choices

    appearance = visual.appearance
    multiscale = hasattr(appearance, "lod_bias")
    shared = {
        "visible": bool(appearance.visible),
        "transparency_mode": to_widget_value(
            "transparency_mode", appearance.transparency_mode
        ),
        "interpolation": appearance.interpolation,
    }
    if multiscale:
        shared["attenuation"] = float(appearance.attenuation)

    channels = {int(k): mode_values(v) for k, v in visual.channels.items()}
    if clim_range is None:
        limits = [visual.single.clim, *(c.clim for c in visual.channels.values())]
        clim_range = (
            min([0.0, *(float(c[0]) for c in limits)]),
            max([1.0, *(float(c[1]) for c in limits)]),
        )
    labels = dict(channel_labels or {})
    return {
        "storage": "multiscale" if multiscale else "memory",
        "colormaps": {
            "single": getattr(visual.single, "color_map", None),
            "channels": {
                int(k): getattr(v, "color_map", None)
                for k, v in visual.channels.items()
            },
        },
        "has_channel_axis": visual.channel_axis is not None,
        "composite": bool(visual.composite),
        "fields": [str(field) for field in fields],
        "shared": shared,
        "single": mode_values(visual.single),
        "channels": channels,
        "channel_labels": {k: labels.get(k, f"Channel {k}") for k in channels},
        "render_modes": list(literal_choices(visual.single, "render_mode")),
        "transparency_modes": list(TRANSPARENCY_CHOICES),
        "interpolations": list(INTERPOLATION_CHOICES),
        "colormap_names": list(colormap_names)
        if colormap_names is not None
        else list(DEFAULT_COLORMAP_NAMES),
        "clim_range": [float(clim_range[0]), float(clim_range[1])],
    }


def inbound_target(event: Any) -> tuple[str, int | None, str, Any] | None:
    """Where an inbound event lands on the control.

    Returns
    -------
    tuple or None
        ``(page, channel, field, widget value)``, with ``page`` one of
        ``"shared"``, ``"single"``, ``"channel"`` or ``"composite"``; ``None``
        for an event the control does not show.  A replaced ``single`` model
        (``field_name=None``) is reported as ``("single", None, None, model)``.
    """
    if isinstance(event, VisualVisibilityChangedEvent):
        return ("shared", None, "visible", bool(event.visible))
    if isinstance(event, ImageCompositeChangedEvent):
        return ("composite", None, "composite", bool(event.composite))
    if isinstance(event, SingleAppearanceChangedEvent):
        if event.field_name is None:
            return ("single", None, None, event.new_value)
        if event.field_name not in MODE_FIELDS:
            return None
        return (
            "single",
            None,
            event.field_name,
            to_widget_value(event.field_name, event.new_value),
        )
    if isinstance(event, ChannelAppearanceChangedEvent):
        if event.field_name not in (*MODE_FIELDS, "visible"):
            return None
        return (
            "channel",
            int(event.channel_index),
            event.field_name,
            to_widget_value(event.field_name, event.new_value),
        )
    if isinstance(event, AppearanceChangedEvent):
        if event.field_name not in SHARED_FIELDS:
            return None
        return (
            "shared",
            None,
            event.field_name,
            to_widget_value(event.field_name, event.new_value),
        )
    return None
