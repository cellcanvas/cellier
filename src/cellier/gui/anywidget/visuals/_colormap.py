"""Colormap control wired to the cellier v2 event bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._colormap_util import colormap_to_str
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"

_DEFAULT_COLORMAP_NAMES = [
    "grays",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "hot",
    "cool",
    "bwr",
]


class AnywidgetColormapCombo(VisualIdGroup, anywidget.AnyWidget):
    """Bidirectional colormap selector wired to the cellier v2 bus.

    Mirrors ``QtColormapCombo``: one UUID per widget, source-ID echo
    filtering, and a narrow subscription to just the ``color_map`` field.

    Wire to the controller after construction::

        control = AnywidgetColormapCombo(visual_id, initial_colormap="grays")
        controller.connect_widget(
            control, subscription_specs=control.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``color_map`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_colormap :
        Starting colormap -- typically ``visual_model.appearance.color_map``.
    colormap_names :
        Available colormap names for the dropdown.  Defaults to a curated list.
    """

    _esm = _STATIC / "colormap_control.js"
    _css = _STATIC / "colormap_control.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "Colormap"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this is
    what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the two
    together.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    """What this control calls itself, drawn by its own front end.

    A control names itself rather than being named by whatever lays it out
    (``plans/label_ownership_unification.md``), which is what lets the dock
    stack controls and stop.
    """

    color_map = traitlets.Unicode("grays").tag(sync=True)
    colormap_names = traitlets.List([]).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_colormap: str = "grays",
        colormap_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            color_map=colormap_to_str(initial_colormap),
            colormap_names=list(colormap_names)
            if colormap_names is not None
            else list(_DEFAULT_COLORMAP_NAMES),
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_trait_change, names="color_map")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetColormapCombo:
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
        return self._group_specs(AppearanceChangedEvent, self._on_appearance_changed)

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "color_map":
            return
        self._set_field("color_map", colormap_to_str(event.new_value))

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
        self._emit_group(AppearanceUpdateEvent, "color_map", change["new"])
