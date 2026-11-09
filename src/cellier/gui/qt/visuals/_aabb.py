"""AABB (axis-aligned bounding box) control widget wired to the event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.events import AABBChangedEvent, AABBUpdateEvent, SubscriptionSpec
from cellier.gui._appearance_fields import VisualIdGroup

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class QtAABBWidget(VisualIdGroup):
    """Bidirectional AABB parameter controls wired to the event bus.

    Wraps three Qt sub-controls — a ``QCheckBox`` for *enabled*, a
    ``QDoubleSpinBox`` for *line_width*, and a color swatch + button for
    *color* — and keeps them in sync with ``AABBParams`` via
    ``AABBChangedEvent``.  Follows the bus-widget pattern: one UUID per
    widget, source-ID echo filtering, and signal blocking when applying
    model-driven updates.

    Wire to the controller after construction::

        aabb = QtAABBWidget(visual_id, initial_enabled=True, ...)
        controller.connect_widget(aabb, subscription_specs=aabb.subscription_specs())

    Parameters
    ----------
    visual_id :
        UUID of the visual whose ``aabb`` params this widget controls. A
        sequence drives every listed visual in lock-step -- the
        ``OrthoViewer``'s four panel siblings.
    initial_enabled :
        Starting value for the *enabled* checkbox.
    initial_line_width :
        Starting value for the *line_width* spinbox.
    initial_color :
        Starting CSS color string for the color swatch (e.g. ``"#ff00ff"``).
    title :
        The name shown on the group frame.  Defaults to
        :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = "Bounding box"
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
        initial_enabled: bool = False,
        initial_line_width: float = 2.0,
        initial_color: str = "#ffffff",
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        from cellier.gui.qt.visuals._chrome import titled_group

        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._current_color = initial_color

        # ── Qt seam 1: container ─────────────────────────────────────────────
        self._container = QWidget(parent)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        # enabled checkbox
        self._enabled_check = QCheckBox("Show")
        self._enabled_check.setChecked(initial_enabled)
        self._enabled_check.toggled.connect(self._on_enabled_changed)
        layout.addWidget(self._enabled_check)

        # line width row
        lw_row = QWidget()
        lw_layout = QHBoxLayout(lw_row)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width (px):"))
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 20.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setDecimals(1)
        self._line_width_spin.setValue(initial_line_width)
        self._line_width_spin.valueChanged.connect(self._on_line_width_changed)
        lw_layout.addWidget(self._line_width_spin)
        layout.addWidget(lw_row)

        # color row: swatch + button
        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.addWidget(QLabel("Color:"))
        self._color_swatch = QWidget()
        self._color_swatch.setFixedSize(24, 24)
        self._color_swatch.setToolTip("Current AABB color")
        color_layout.addWidget(self._color_swatch)
        self._color_btn = QPushButton("Choose...")
        self._color_btn.clicked.connect(self._on_color_btn_clicked)
        color_layout.addWidget(self._color_btn)
        color_layout.addStretch()
        layout.addWidget(color_row)

        # apply initial swatch color (no signal to block)
        self._apply_swatch(initial_color)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._container, parent
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The titled group to insert into a layout.

        The rows inside only mean something together, so the group names
        them; reach for :attr:`control` for the bare rows.
        """
        return self._group

    @property
    def control(self):
        """The bare row container inside the group."""
        return self._container

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
        return self._group_specs(AABBChangedEvent, self._on_aabb_changed)

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_aabb_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.field_name == "enabled":
            self._set_enabled(event.new_value)
        elif event.field_name == "line_width":
            self._set_line_width(event.new_value)
        elif event.field_name == "color":
            self._set_color(event.new_value)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_enabled_changed(self, value: bool) -> None:
        self._emit_group(AABBUpdateEvent, "enabled", value)

    def _on_line_width_changed(self, value: float) -> None:
        self._emit_group(AABBUpdateEvent, "line_width", value)

    def _on_color_btn_clicked(self) -> None:
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        initial = QColor(self._current_color)
        color = QColorDialog.getColor(initial, self._container, "Choose AABB color")
        if not color.isValid():
            return  # user cancelled
        css = color.name()  # e.g. "#ff00ff"
        self._emit_group(AABBUpdateEvent, "color", css)
        # Update local cache and swatch immediately — echo filtering will
        # suppress the round-tripped AABBChangedEvent.
        self._current_color = css
        self._apply_swatch(css)

    # ── Qt seam 2: push values without re-firing change signals ─────────────

    def _set_enabled(self, value: bool) -> None:
        self._enabled_check.blockSignals(True)
        self._enabled_check.setChecked(value)
        self._enabled_check.blockSignals(False)

    def _set_line_width(self, value: float) -> None:
        self._line_width_spin.blockSignals(True)
        self._line_width_spin.setValue(value)
        self._line_width_spin.blockSignals(False)

    def _set_color(self, value: str) -> None:
        # Swatch has no change signal — no blocking needed.
        self._current_color = value
        self._apply_swatch(value)

    def _apply_swatch(self, css_color: str) -> None:
        """Update the color swatch background without touching any signals."""
        self._color_swatch.setStyleSheet(
            f"background-color: {css_color}; border: 1px solid #888;"
        )
