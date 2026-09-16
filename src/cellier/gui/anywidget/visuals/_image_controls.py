"""The unified image control for anywidget (unified image design 3.10).

The twin of ``QtImageControls``.  State is synced as a few dict traits --
``shared``, ``single`` and ``channels`` -- plus ``composite``.  The front end
writes a whole dict back; the Python side diffs it against the previous value
and emits one update event per changed field.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import ImageCompositeUpdateEvent
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._image_controls import (
    INBOUND_EVENT_TYPES,
    image_update_event,
    inbound_target,
    mode_values,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class AnywidgetImageControls(VisualIdGroup, anywidget.AnyWidget):
    """Bidirectional image appearance control for the anywidget GUI.

    Parameters
    ----------
    visual_id :
        The visual, or an ``OrthoViewer``'s panel siblings, driven together.
    values :
        The seed from :func:`cellier.gui._image_controls.image_control_values`.
    title :
        The heading.  Defaults to :data:`DEFAULT_TITLE`.
    """

    _esm = _STATIC / "image_controls.js"
    _css = _STATIC / "image_controls.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "Image"

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    storage = traitlets.Unicode("memory").tag(sync=True)
    has_channel_axis = traitlets.Bool(False).tag(sync=True)
    composite = traitlets.Bool(False).tag(sync=True)
    fields = traitlets.List([]).tag(sync=True)
    shared = traitlets.Dict({}).tag(sync=True)
    single = traitlets.Dict({}).tag(sync=True)
    channels = traitlets.Dict({}).tag(sync=True)
    """Channel index (str) to that channel's values."""
    channel_labels = traitlets.Dict({}).tag(sync=True)
    render_modes = traitlets.List([]).tag(sync=True)
    transparency_modes = traitlets.List([]).tag(sync=True)
    interpolations = traitlets.List([]).tag(sync=True)
    colormap_names = traitlets.List([]).tag(sync=True)
    clim_range = traitlets.List([0.0, 1.0]).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        values: dict[str, Any],
        *,
        title: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            title=self.DEFAULT_TITLE if title is None else title,
            storage=values["storage"],
            has_channel_axis=bool(values["has_channel_axis"]),
            composite=bool(values["composite"]),
            fields=list(values["fields"]),
            shared=dict(values["shared"]),
            single=dict(values["single"]),
            channels={str(k): dict(v) for k, v in values["channels"].items()},
            channel_labels={str(k): v for k, v in values["channel_labels"].items()},
            render_modes=list(values["render_modes"]),
            transparency_modes=list(values["transparency_modes"]),
            interpolations=list(values["interpolations"]),
            colormap_names=list(values["colormap_names"]),
            clim_range=list(values["clim_range"]),
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_dict_change, names=["shared", "single", "channels"])
        self.observe(self._on_composite_change, names="composite")

    @property
    def widget(self) -> AnywidgetImageControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list:
        """One subscription per inbound event type per driven visual."""
        specs = []
        for event_type in INBOUND_EVENT_TYPES:
            specs.extend(self._group_specs(event_type, self._on_event))
        return specs

    # ── widget -> model ─────────────────────────────────────────────────

    def _emit(self, page: str, field: str, value: Any, channel: int | None = None):
        for visual_id in self._visual_ids:
            self.changed.emit(
                image_update_event(self._id, visual_id, page, field, value, channel)
            )

    def _on_dict_change(self, change) -> None:
        if self._applying:
            return
        old = change["old"] or {}
        new = change["new"] or {}
        if change["name"] == "channels":
            for key, fields in new.items():
                previous = old.get(key, {})
                for field, value in fields.items():
                    if previous.get(field) != value:
                        self._emit("channel", field, value, int(key))
            return
        for field, value in new.items():
            if old.get(field) != value:
                self._emit(change["name"], field, value)

    def _on_composite_change(self, change) -> None:
        if self._applying:
            return
        try:
            for visual_id in self._visual_ids:
                self.changed.emit(
                    ImageCompositeUpdateEvent(self._id, visual_id, bool(change["new"]))
                )
        except Exception:
            self._set_trait("composite", bool(change["old"]))
            raise

    # ── model -> widget ─────────────────────────────────────────────────

    def _set_trait(self, name: str, value: Any) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    def _on_event(self, event) -> None:
        if event.source_id == self._id:
            return
        target = inbound_target(event)
        if target is None:
            return
        page, channel, field, value = target
        if page == "composite":
            self._set_trait("composite", value)
        elif page == "single" and field is None:
            replaced = mode_values(value)
            replaced.pop("visible", None)
            self._set_trait("single", {**self.single, **replaced})
        elif page in ("shared", "single"):
            current = getattr(self, page)
            if field in current or page == "single":
                self._set_trait(page, {**current, field: value})
        elif page == "channel":
            key = str(channel)
            if key in self.channels:
                channels = {k: dict(v) for k, v in self.channels.items()}
                channels[key][field] = value
                self._set_trait("channels", channels)
