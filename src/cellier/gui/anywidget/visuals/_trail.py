"""Trail-window controls for a graph visual, wired to the cellier v2 bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import SubscriptionSpec, TrailChangedEvent, TrailUpdateEvent
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._trail import (
    TRAIL_FIELDS,
    TRAIL_LABELS,
    TRAIL_TITLE,
    TRAIL_TOOLTIPS,
    enabled_label,
)
from cellier.gui.anywidget._teardown import close_aux_widgets
from cellier.visuals._graph_memory import TrailConfig

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class AnywidgetTrailControls(VisualIdGroup, anywidget.AnyWidget):
    """Trail-window controls for one or more data axes of a graph visual.

    Mirrors ``QtTrailControls``: per offered axis, whether it has a window,
    how far the window reaches before and after the slice position, and
    whether it fades.

    The browser holds one ``windows`` dict keyed by the axis index as a
    string.  Python keeps the full ``TrailConfig`` per axis beside it, so the
    fields the control does not show survive every edit.

    Parameters
    ----------
    visual_id :
        The graph visual whose trail this widget controls.  A sequence drives
        every listed visual in lock-step, as on an ``OrthoViewer``.
    axes :
        ``(data-axis index, label)`` for each axis to offer, in display order.
    trail :
        The visual's current ``{axis: TrailConfig}``.  Read, never held.
    """

    _esm = _STATIC / "trail.js"
    _css = _STATIC / "trail.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = TRAIL_TITLE
    """Name shown when no ``title=`` is given.

    ``test_composite_default_titles_match_the_shared_vocabulary`` pins it to
    the title the renderers pass in.
    """

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: ``[{"axis": int, "label": str}, ...]``: which axes to draw, in order.
    axes = traitlets.List([]).tag(sync=True)
    #: ``{str(axis): {"enabled", "before", "after", "fade"}}``.
    windows = traitlets.Dict({}).tag(sync=True)
    #: The labels and tooltips the controls show, read from ``cellier.gui._trail``.
    text = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        axes: Sequence[tuple[int, str]],
        trail: Mapping[int, TrailConfig],
        **kwargs,
    ) -> None:
        offered = [(int(axis), str(label)) for axis, label in axes]
        if not offered:
            raise ValueError("AnywidgetTrailControls needs at least one axis to offer.")
        windows = {
            axis: (trail.get(axis) or TrailConfig()).model_copy()
            for axis, _label in offered
        }
        enabled = {axis: trail.get(axis) is not None for axis, _label in offered}
        super().__init__(
            axes=[
                {"axis": axis, "label": enabled_label(label)} for axis, label in offered
            ],
            windows=self._serialise(windows, enabled),
            text={"labels": dict(TRAIL_LABELS), "tooltips": dict(TRAIL_TOOLTIPS)},
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._windows = windows
        self._enabled = enabled
        self._applying = False
        self.observe(self._on_windows_change, names="windows")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetTrailControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def offered_axes(self) -> tuple[int, ...]:
        """The data axes this widget offers, in display order."""
        return tuple(entry["axis"] for entry in self.axes)

    def is_enabled(self, axis: int) -> bool:
        """Whether *axis* currently has a window."""
        return self._enabled[axis]

    def window(self, axis: int) -> TrailConfig:
        """The window *axis* has, or last had before it was switched off."""
        return self._windows[axis].model_copy()

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``TrailChangedEvent`` subscription per driven visual."""
        return self._group_specs(TrailChangedEvent, self._on_trail_changed)

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_trail_changed(self, event: TrailChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        for axis in self._windows:
            window = event.trail.get(axis)
            if window is not None:
                self._windows[axis] = window.model_copy()
            self._enabled[axis] = window is not None
        self._applying = True
        try:
            self.windows = self._serialise(self._windows, self._enabled)
        finally:
            self._applying = False

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_windows_change(self, change) -> None:
        if self._applying:
            return  # bus -> widget write; do not echo back
        before = change["old"] or {}
        for key, state in (change["new"] or {}).items():
            previous = before.get(key) or {}
            if previous == state:
                continue
            axis = int(key)
            if axis not in self._windows:
                continue
            self._windows[axis] = self._windows[axis].model_copy(
                update={
                    "before": float(state.get("before", 0.0)),
                    "after": float(state.get("after", 0.0)),
                    "fade": bool(state.get("fade", False)),
                }
            )
            on = bool(state.get("enabled", False))
            was_on = self._enabled[axis]
            self._enabled[axis] = on
            if on or was_on:
                self._emit(axis, self._windows[axis] if on else None)

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

    @staticmethod
    def _serialise(
        windows: Mapping[int, TrailConfig], enabled: Mapping[int, bool]
    ) -> dict:
        return {
            str(axis): {
                "enabled": bool(enabled[axis]),
                **{field: getattr(window, field) for field in TRAIL_FIELDS},
            }
            for axis, window in windows.items()
        }
