"""Trail-window controls for a graph visual, wired to the cellier v2 event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.events import SubscriptionSpec, TrailChangedEvent, TrailUpdateEvent
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._trail import (
    TRAIL_LABELS,
    TRAIL_TITLE,
    TRAIL_TOOLTIPS,
    edited_window,
    enabled_label,
)
from cellier.visuals._graph_memory import TrailConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

#: Upper bound on a window's extent.  World units are dataset-specific -- a
#: timelapse in seconds runs into the thousands -- so the bound only has to
#: stay out of the way.
_MAXIMUM_EXTENT = 1.0e9


class QtTrailControls(VisualIdGroup):
    """Trail-window controls for one or more data axes of a graph visual.

    Each offered axis gets a block of four controls: whether the axis has a
    window at all, how far it reaches before and after the slice position,
    and whether it fades.  Switching an axis off removes its window from
    ``GraphVisual.trail``; switching it back on restores the window it had.

    Wire to the controller after construction::

        trail = QtTrailControls(visual_id, [(0, "t")], visual.trail)
        controller.connect_widget(trail, subscription_specs=trail.subscription_specs())

    Parameters
    ----------
    visual_id :
        The graph visual whose trail this widget controls.  A sequence drives
        every listed visual in lock-step, as on an ``OrthoViewer``.
    axes :
        ``(data-axis index, label)`` for each axis to offer, in display order.
    trail :
        The visual's current ``{axis: TrailConfig}``.  Read, never held.
    title :
        The name shown on the group frame.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = TRAIL_TITLE
    """Name shown when no ``title=`` is given.

    ``test_composite_default_titles_match_the_shared_vocabulary`` pins it to
    the title the renderers pass in.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        axes: Sequence[tuple[int, str]],
        trail: Mapping[int, TrailConfig],
        *,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import labelled_row, titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._axes = tuple(int(axis) for axis, _label in axes)
        if not self._axes:
            raise ValueError("QtTrailControls needs at least one axis to offer.")

        # The window each axis last had.  Kept while the axis is switched off,
        # so switching it back on restores what it was.
        self._windows: dict[int, TrailConfig] = {}
        self._enabled: dict[int, bool] = {}
        self._controls: dict[int, dict[str, object]] = {}

        self._container = QWidget(parent)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        for axis, label in axes:
            axis = int(axis)
            window = trail.get(axis)
            self._windows[axis] = (window or TrailConfig()).model_copy()
            self._enabled[axis] = window is not None

            enabled = QCheckBox(enabled_label(label))
            enabled.setToolTip(TRAIL_TOOLTIPS["enabled"])
            before = self._extent_spin(TRAIL_TOOLTIPS["before"])
            after = self._extent_spin(TRAIL_TOOLTIPS["after"])
            fade = QCheckBox(TRAIL_LABELS["fade"])
            fade.setToolTip(TRAIL_TOOLTIPS["fade"])

            layout.addWidget(enabled)
            layout.addWidget(labelled_row(TRAIL_LABELS["before"], before))
            layout.addWidget(labelled_row(TRAIL_LABELS["after"], after))
            layout.addWidget(fade)

            self._controls[axis] = {
                "enabled": enabled,
                "before": before,
                "after": after,
                "fade": fade,
            }
            self._show(axis)

            enabled.toggled.connect(lambda on, a=axis: self._on_enabled(a, on))
            before.valueChanged.connect(
                lambda value, a=axis: self._on_field(a, "before", float(value))
            )
            after.valueChanged.connect(
                lambda value, a=axis: self._on_field(a, "after", float(value))
            )
            fade.toggled.connect(lambda on, a=axis: self._on_field(a, "fade", bool(on)))

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._container, parent
        )

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The titled group to insert into a layout."""
        return self._group

    @property
    def control(self):
        """The bare container inside the group."""
        return self._container

    @property
    def offered_axes(self) -> tuple[int, ...]:
        """The data axes this widget offers, in display order.

        Named to match ``AnywidgetTrailControls``, whose ``axes`` is a synced
        trait describing the same thing for the browser.
        """
        return self._axes

    def is_enabled(self, axis: int) -> bool:
        """Whether *axis* currently has a window."""
        return self._enabled[axis]

    def window(self, axis: int) -> TrailConfig:
        """The window *axis* has, or last had before it was switched off."""
        return self._windows[axis].model_copy()

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``TrailChangedEvent`` subscription per driven visual."""
        return self._group_specs(TrailChangedEvent, self._on_trail_changed)

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_trail_changed(self, event: TrailChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        for axis in self._axes:
            window = event.trail.get(axis)
            if window is not None:
                self._windows[axis] = window.model_copy()
            self._enabled[axis] = window is not None
            self._show(axis)

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_enabled(self, axis: int, on: bool) -> None:
        self._enabled[axis] = bool(on)
        self._set_inputs_enabled(axis, on)
        self._emit(axis, self._windows[axis] if on else None)

    def _on_field(self, axis: int, field: str, value) -> None:
        self._windows[axis] = edited_window(self._windows[axis], field, value)
        if self._enabled[axis]:
            self._emit(axis, self._windows[axis])

    def _emit(self, axis: int, window: TrailConfig | None) -> None:
        for visual_id in self._visual_ids:
            # A copy per visual: the controller wires handlers to the config
            # object it adopts, so visuals must never share one.
            self.changed.emit(
                TrailUpdateEvent(
                    source_id=self._id,
                    visual_id=visual_id,
                    axis=axis,
                    config=None if window is None else window.model_copy(),
                )
            )

    # ── Qt plumbing ──────────────────────────────────────────────────────────

    @staticmethod
    def _extent_spin(tooltip: str):
        from qtpy.QtWidgets import QAbstractSpinBox, QDoubleSpinBox

        spin = QDoubleSpinBox()
        spin.setRange(0.0, _MAXIMUM_EXTENT)
        spin.setDecimals(3)
        spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        # Every committed value refetches the graph, so typing "4800" must
        # not reslice at 4, 48 and 480 on the way.
        spin.setKeyboardTracking(False)
        spin.setToolTip(tooltip)
        return spin

    def _show(self, axis: int) -> None:
        """Push *axis*'s window and on/off state into its controls, silently."""
        controls = self._controls[axis]
        window = self._windows[axis]
        values = {
            "enabled": self._enabled[axis],
            "before": window.before,
            "after": window.after,
            "fade": window.fade,
        }
        for name, control in controls.items():
            control.blockSignals(True)
            try:
                if name in ("enabled", "fade"):
                    control.setChecked(bool(values[name]))
                else:
                    control.setValue(float(values[name]))
            finally:
                control.blockSignals(False)
        self._set_inputs_enabled(axis, self._enabled[axis])

    def _set_inputs_enabled(self, axis: int, on: bool) -> None:
        for name in ("before", "after", "fade"):
            self._controls[axis][name].setEnabled(bool(on))
