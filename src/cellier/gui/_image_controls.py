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
    from collections.abc import Iterable
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

#: Decimal places for the fraction-like fields (opacity, attenuation), on both
#: front ends.  Fixed, unlike the data-unit fields, which follow the config's
#: ``decimals``.
FRACTION_DECIMALS: int = 2

#: The shortest the contrast-limits and iso-threshold slider tracks may be, in
#: logical pixels, on both front ends.  Their number labels take most of a
#: narrow dock's width; the dock grows to keep this much track instead.
MIN_TRACK_WIDTH_PX: int = 120

#: Render modes that use the iso threshold.  The threshold row is shown only
#: for these, and only in 3D.
THRESHOLD_MODES: tuple[str, ...] = ("iso", "smooth_iso")

#: Render modes that use the attenuation coefficient.  The attenuation row is
#: shown only for these, and only in 3D.
ATTENUATION_MODES: tuple[str, ...] = ("attenuated_mip",)

#: Fields whose rows are shown only when the scene displays three dimensions:
#: a 2D slice ignores them.
THREE_D_FIELDS: tuple[str, ...] = ("render_mode", "iso_threshold", "attenuation")

#: The display dimensionalities a scene can have.
DISPLAY_DIMENSIONS: tuple[int, ...] = (2, 3)

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


def validate_n_displayed_dimensions(value: Any) -> int:
    """Return *value* if it is a display dimensionality (2 or 3).

    Raises
    ------
    TypeError
        If *value* is not an int (``bool`` included).
    ValueError
        If *value* is not 2 or 3.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"n_displayed_dimensions must be an int, got {type(value).__name__}."
        )
    if value not in DISPLAY_DIMENSIONS:
        raise ValueError(f"n_displayed_dimensions must be 2 or 3, got {value}.")
    return value


def row_visible(
    field: str, render_modes: Iterable[str], n_displayed_dimensions: int
) -> bool:
    """Whether a row of the image control is shown.

    Both front ends call this for every row that can hide, each time anything
    it depends on changes, rather than adjusting rows one trigger at a time.

    Parameters
    ----------
    field : str
        The row's field.
    render_modes : iterable of str
        The render modes the row serves: its page's mode, or every channel's
        for the composite page's shared attenuation row.
    n_displayed_dimensions : int
        How many dimensions the scene displays (2 or 3).

    Returns
    -------
    bool
        ``render_mode`` rows are shown in 3D; ``iso_threshold`` rows in 3D for
        a mode in :data:`THRESHOLD_MODES`; ``attenuation`` rows in 3D for a
        mode in :data:`ATTENUATION_MODES`.  Every other row is always shown.
    """
    if field not in THREE_D_FIELDS:
        return True
    if n_displayed_dimensions != 3:
        return False
    if field == "iso_threshold":
        return any(mode in THRESHOLD_MODES for mode in render_modes)
    if field == "attenuation":
        return any(mode in ATTENUATION_MODES for mode in render_modes)
    return True


def displayed_dimensions(dims_state: Any) -> int | None:
    """How many dimensions *dims_state* displays; ``None`` if it does not say.

    A selection with no ``displayed_axes`` (the plane-selection stub) gives
    ``None``, and the control keeps what it had.
    """
    axes = getattr(getattr(dims_state, "selection", None), "displayed_axes", None)
    return None if axes is None else len(axes)


def display_seed(
    controller: Any, visual_ids: Iterable[UUID]
) -> tuple[tuple[UUID, ...], int]:
    """The scenes an image control follows, and what they display now.

    ``DimsChangedEvent`` fires only on a change and ``connect_widget`` does
    not replay state, so a control is seeded with the current value at
    construction and then follows the events.

    Parameters
    ----------
    controller : CellierController or None
        The controller the control is wired to.  ``None`` gives no scenes
        and 3 (every row that can hide is shown).
    visual_ids : iterable of UUID
        The visuals the control drives.

    Returns
    -------
    tuple
        ``(scene_ids, n_displayed_dimensions)``: each driven visual's scene,
        in order and without repeats, and the first scene's display
        dimensionality.  A control assumes its visuals share one.
    """
    if controller is None:
        return (), 3
    scene_ids: list[UUID] = []
    for visual_id in visual_ids:
        try:
            scene_id = controller.get_visual_scene_id(visual_id)
        except KeyError:
            continue
        if scene_id not in scene_ids:
            scene_ids.append(scene_id)
    n = None
    if scene_ids:
        n = displayed_dimensions(controller.get_scene(scene_ids[0]).dims)
    return tuple(scene_ids), 3 if n is None else n


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
    decimals: int = 2,
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
    decimals : int
        Decimal places for values in data units (the contrast limits and the
        iso threshold).  Fractions such as opacity always show 2.

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
        "decimals": int(decimals),
        "threshold_modes": list(THRESHOLD_MODES),
        "attenuation_modes": list(ATTENUATION_MODES),
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
