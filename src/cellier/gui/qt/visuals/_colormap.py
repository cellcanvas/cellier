"""Colormap combobox wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class QtColormapCombo(VisualIdGroup):
    """Bidirectional colormap selector wired to the cellier v2 bus.

    Wraps a ``superqt.QColormapComboBox`` and keeps it in sync with
    ``MultiscaleImageAppearance.color_map`` via ``AppearanceChangedEvent``.  Follows the
    v2 widget pattern: one UUID per widget, source-ID echo filtering, and
    signal blocking when applying model-driven updates.

    Wire to the controller after construction::

        combo = QtColormapCombo(visual_id, initial_colormap="grays")
        controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``color_map`` field this widget controls.
        A sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings (design section 8.1).
    initial_colormap :
        Starting colormap — typically ``visual_model.appearance.color_map``.
    title :
        The name shown beside the control.  Defaults to
        :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = "Colormap"
    """Name shown when no ``title=`` is given.

    The renderer passes the title from the shared control vocabulary; this
    is what a directly-constructed widget calls itself, and
    ``test_composite_default_titles_match_the_shared_vocabulary`` pins the
    two together.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_colormap,
        title: str | None = None,
        parent=None,
    ) -> None:
        from superqt import QColormapComboBox

        from cellier.gui.qt.visuals._chrome import labelled_row

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._init_visual_ids(visual_id)

        # ── Qt seam 1: widget creation and signal wiring ─────────────────────
        self._combo = QColormapComboBox(parent)
        self._combo.setCurrentColormap(initial_colormap)
        self._combo.currentColormapChanged.connect(self._on_combo_changed)

        self._row = labelled_row(
            self.DEFAULT_TITLE if title is None else title, self._combo, parent
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The labelled row to insert into a layout.

        The control names itself (``plans/label_ownership_unification.md``);
        reach for :attr:`control` to drive the input directly.
        """
        return self._row

    @property
    def control(self):
        """The bare input inside the row."""
        return self._combo

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def add_colormaps(self, colormaps: Sequence[Any]) -> None:
        """Add colormaps to the combo box.

        Parameters
        ----------
        colormaps : Sequence[Any]
            Colormaps to add. Each item can be anything accepted by
            ``cmap.Colormap`` — e.g. a name string, a ``cmap.Colormap``
            instance, or a color-stop sequence.
        """
        self._combo.addColormaps(colormaps)

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
        return self._group_specs(AppearanceChangedEvent, self._on_visual_changed)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_visual_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name != "color_map":
            return  # a different appearance field changed; nothing to do
        self._set_value(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_combo_changed(self, colormap) -> None:
        self._emit_group(AppearanceUpdateEvent, "color_map", colormap)

    # ── Qt seam 2: push value without re-firing currentColormapChanged ────────

    def _set_value(self, colormap) -> None:
        self._combo.blockSignals(True)
        self._combo.setCurrentColormap(colormap)
        self._combo.blockSignals(False)
