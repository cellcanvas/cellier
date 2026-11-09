"""AABB (axis-aligned bounding box) control wired to the event bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import AABBChangedEvent, AABBUpdateEvent, SubscriptionSpec
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"

_FIELDS = ("enabled", "line_width", "color")


class AnywidgetAABBWidget(VisualIdGroup, anywidget.AnyWidget):
    """Bidirectional AABB parameter controls wired to the event bus.

    Mirrors ``QtAABBWidget``: an *enabled* checkbox, a *line_width* number
    input, and a *color* swatch, sharing one UUID so a single subscription
    covers all three fields.

    Wire to the controller after construction::

        aabb = AnywidgetAABBWidget(visual_id, initial_enabled=True, ...)
        controller.connect_widget(aabb, subscription_specs=aabb.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``aabb`` params this widget controls. A
        sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings.
    initial_enabled :
        Starting value for the *enabled* checkbox. Default ``False``.
    initial_line_width :
        Starting value for the *line_width* input. Default ``2.0``.
    initial_color :
        Starting CSS color string for the color swatch. Default ``"#ffffff"``.
    """

    _esm = _STATIC / "aabb.js"
    _css = _STATIC / "aabb.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "Bounding box"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    """What this control calls itself, drawn by its own front end.

    A control names itself rather than being named by whatever lays it out,
    which is what lets the dock stack controls and stop.
    """

    enabled = traitlets.Bool(False).tag(sync=True)
    line_width = traitlets.Float(2.0).tag(sync=True)
    color = traitlets.Unicode("#ffffff").tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_enabled: bool = False,
        initial_line_width: float = 2.0,
        initial_color: str = "#ffffff",
        **kwargs,
    ) -> None:
        super().__init__(
            enabled=bool(initial_enabled),
            line_width=float(initial_line_width),
            color=str(initial_color),
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_trait_change, names=list(_FIELDS))

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetAABBWidget:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget.

        ``closed`` tells the controller to drop this widget's subscriptions;
        the rest actually releases the widget.  See
        ``cellier.gui.anywidget._teardown`` for why both steps are needed --
        ``ipywidgets`` holds every widget, and every widget's ``layout``, in a
        process-global table that only ``close()`` clears.
        """
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires."""
        return self._group_specs(AABBChangedEvent, self._on_aabb_changed)

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_aabb_changed(self, event: AABBChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name not in _FIELDS:
            return
        self._set_field(event.field_name, event.new_value)

    def _set_field(self, name: str, value) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return
        self._emit_group(AABBUpdateEvent, change["name"], change["new"])
