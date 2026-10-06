"""LOD-bias slider wired to the cellier v2 event bus (anywidget)."""

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


class AnywidgetLodBiasSlider(VisualIdGroup, anywidget.AnyWidget):
    """Single-value LOD-bias slider wired to the cellier v2 bus.

    Mirrors ``QtLodBiasSlider``.  Because changing ``settled_lod_bias`` triggers a
    reslice, the JS emits only on settled ``change`` (not on every drag
    ``input``), so only one reslice fires per drag interaction.

    Wire to the controller after construction::

        slider = AnywidgetLodBiasSlider(visual_id, initial_lod_bias=1.0)
        controller.connect_widget(
            slider, subscription_specs=slider.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``settled_lod_bias`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_lod_bias :
        Starting value -- typically ``visual_model.appearance.settled_lod_bias``.
    """

    _esm = _STATIC / "lod_bias.js"
    _css = _STATIC / "lod_bias.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = "LOD bias"
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

    settled_lod_bias = traitlets.Float(1.0).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_lod_bias: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(settled_lod_bias=float(initial_lod_bias), **kwargs)
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._applying = False
        self.observe(self._on_trait_change, names="settled_lod_bias")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetLodBiasSlider:
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
        if event.field_name != "settled_lod_bias":
            return
        self._set_field("settled_lod_bias", event.new_value)

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
        self._emit_group(AppearanceUpdateEvent, "settled_lod_bias", change["new"])
