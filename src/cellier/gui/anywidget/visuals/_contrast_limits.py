"""Contrast-limits slider wired to the cellier v2 event bus (anywidget)."""

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
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class AnywidgetClimRangeSlider(VisualIdGroup, anywidget.AnyWidget):
    """Bidirectional contrast-limits slider wired to the cellier v2 bus.

    Mirrors ``QtClimRangeSlider``: one UUID per widget, source-ID echo
    filtering, and a narrow subscription to just the ``clim`` field.  Emits
    throttled updates while dragging, like the dims sliders.

    Wire to the controller after construction::

        slider = AnywidgetClimRangeSlider(
            visual_id, clim_range=(0, 255), initial_clim=(0, 200)
        )
        controller.connect_widget(
            slider, subscription_specs=slider.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``clim`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    clim_range :
        ``(min, max)`` for the slider range.
    initial_clim :
        Starting value -- typically ``visual_model.appearance.clim``.
    """

    _esm = _STATIC / "contrast_limits.js"
    _css = _STATIC / "contrast_limits.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "Contrast limits"
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

    clim = traitlets.List([0.0, 1.0]).tag(sync=True)
    clim_range = traitlets.List([0.0, 1.0]).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        clim_range: tuple[float, float] | list = (0.0, 1.0),
        initial_clim: tuple[float, float] | list = (0.0, 1.0),
        **kwargs,
    ) -> None:
        super().__init__(
            clim=[float(initial_clim[0]), float(initial_clim[1])],
            clim_range=[float(clim_range[0]), float(clim_range[1])],
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_trait_change, names="clim")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetClimRangeSlider:
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
        if event.field_name != "clim":
            return
        value = event.new_value
        self._set_field("clim", [float(value[0]), float(value[1])])

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
        self._emit_group(AppearanceUpdateEvent, "clim", tuple(change["new"]))
