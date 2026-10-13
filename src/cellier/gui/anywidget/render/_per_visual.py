"""Notebook panels for one visual's screen-space render settings.

The anywidget twins of ``cellier.gui.qt.render._per_visual``, drawing from
the same ``VISUAL_RENDER_CONTROLS`` spec.  See the Qt module for why there
are three outline-ish widgets rather than one.

Each panel syncs a small, flat description of its controls plus one trait per
field, and a single ESM renders all of them -- the shape of a panel is data,
not four near-identical renderers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    PickWriteChangedEvent,
    RenderConfigChangedEvent,
    SubscriptionSpec,
    VisualRenderChangedEvent,
    VisualRenderUpdateEvent,
)
from cellier.gui._render_controls import (
    AMBIENT_OCCLUSION_CHOICES,
    DEPTH_RANGE_LABELS,
    DEPTH_RANGE_MODES,
    DEPTH_RANGE_STEP,
    OUTLINE_MODE_CHOICES,
    PLACEMENT_CHOICES,
    SLOT_IS_COLOUR_MODES,
    VISUAL_RENDER_CONTROLS,
    VISUAL_RENDER_TITLES,
    coerce_depth_range,
    visual_path,
    with_api_path,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"

#: Mirrors ``QtLabelsOutlineControls``: the GPU capacity is 256, but a panel
#: that tall is not a panel.
MAX_LABEL_ROWS = 16


def _choice_options(choices, auto_suffix: str = "") -> list[list]:
    """Serialise a choice tuple for the JS, completing the Auto label."""
    options = []
    for label, value in choices:
        text = f"{label} ({auto_suffix})" if label == "Auto" and auto_suffix else label
        # ``None`` survives the JSON round trip as null, which the ESM maps
        # back by index rather than by value -- so the index is the key.
        options.append([text, value])
    return options


class AnywidgetVisualRenderPanel(anywidget.AnyWidget):
    """Base for a notebook panel driving one visual's screen-space settings.

    Parameters
    ----------
    visual_ids :
        Every visual this panel writes to: one on a ``Viewer``, four on an
        ``OrthoViewer``.
    values :
        Current field values, read off the model by the caller.
    """

    section: ClassVar[str] = ""

    _esm = _STATIC / "visual_render_panel.js"
    _css = _STATIC / "render_panel.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    #: The controls to draw, as flat dicts.
    controls = traitlets.List([]).tag(sync=True)
    title = traitlets.Unicode("").tag(sync=True)
    #: The live outline palette, so a slot control can show real colours.
    palette = traitlets.List([]).tag(sync=True)
    #: ``{field: value}``.  One dict rather than a trait per field: these
    #: panels are small and fixed, and a dict keyed by the dotted field name
    #: avoids the name mangling the global panels need.
    values = traitlets.Dict({}).tag(sync=True)

    def __init__(
        self,
        visual_ids: Sequence[UUID],
        values: dict[str, Any],
        *,
        palette: Sequence[Sequence[float]] = (),
        **kwargs,
    ) -> None:
        super().__init__(
            controls=self._describe_controls(values),
            title=VISUAL_RENDER_TITLES[self.section],
            palette=[list(entry) for entry in palette],
            values=self._serialise(values),
            **kwargs,
        )
        self._id = uuid4()
        self._visual_ids = list(visual_ids)
        self._applying = False
        self.observe(self._on_values_change, names=["values"])

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def _describe_controls(self, values: dict[str, Any]) -> list[dict]:
        auto = str(values.get("default_placement", ""))
        described = []
        for control in VISUAL_RENDER_CONTROLS[self.section]:
            entry = {
                "field": control.field,
                "label": control.label,
                "kind": control.kind,
                "tooltip": with_api_path(control.tooltip, visual_path(control.field)),
            }
            if control.field == "outline.placement":
                entry["options"] = _choice_options(PLACEMENT_CHOICES, auto)
            elif control.field == "ambient_occlusion":
                entry["options"] = _choice_options(AMBIENT_OCCLUSION_CHOICES)
            elif control.field == "outline_mode":
                entry["options"] = _choice_options(OUTLINE_MODE_CHOICES)
            elif control.kind == "label_selection":
                entry["max_rows"] = MAX_LABEL_ROWS
            elif control.kind == "range":
                # The first box runs from the minimum to 1, the second from
                # the first box's value to the maximum.
                entry["labels"] = list(DEPTH_RANGE_LABELS)
                entry["minimum"] = control.minimum
                entry["maximum"] = control.maximum
                entry["first_maximum"] = 1.0
                entry["decimals"] = control.decimals
                entry["step"] = DEPTH_RANGE_STEP
            described.append(self._adjust_for_mode(entry, values))
        return [entry for entry in described if entry is not None]

    def _adjust_for_mode(self, entry: dict, values: dict[str, Any]) -> dict | None:
        """Reshape or drop a control for the current mode.  Base: no change."""
        return entry

    @staticmethod
    def _serialise(values: dict[str, Any]) -> dict:
        """Values the JS can hold: label keys become strings in JSON."""
        out = dict(values)
        selection = out.get("outline_selected_labels")
        if selection is not None:
            out["outline_selected_labels"] = {
                str(k): int(v) for k, v in selection.items()
            }
        depth_range = out.get("outline_depth_range")
        if depth_range is not None:
            out["outline_depth_range"] = [float(v) for v in depth_range]
        return out

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self) -> AnywidgetVisualRenderPanel:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [
            SubscriptionSpec(
                event_type=VisualRenderChangedEvent,
                handler=self._on_visual_render_changed,
                entity_id=visual_id,
            )
            for visual_id in self._visual_ids
        ]

    # ------------------------------------------------------------------
    # widget -> model
    # ------------------------------------------------------------------

    def _on_values_change(self, change) -> None:
        if self._applying:
            return  # bus -> widget write; do not echo back
        before = change.old or {}
        for field, value in (change.new or {}).items():
            if before.get(field) == value:
                continue
            if field == "outline_selected_labels":
                value = {int(k): int(v) for k, v in value.items()}
            elif field == "outline.slot":
                value = int(value)
            elif field == "outline_depth_range":
                value = self._depth_range_from_widget(value)
                if value is None:
                    continue
            for visual_id in self._visual_ids:
                self.changed.emit(
                    VisualRenderUpdateEvent(
                        source_id=self._id,
                        visual_id=visual_id,
                        field=field,
                        value=value,
                    )
                )

    def _depth_range_from_widget(self, value) -> tuple[float, float] | None:
        """The pair to send for a range the front end wrote, or ``None``.

        The script keeps the pair in order itself; this is the same rule
        applied again, so that nothing a front end writes reaches the model
        as a pair it would refuse.  A value that is not a pair of numbers
        is dropped.
        """
        try:
            start, solid = (float(part) for part in value)
        except (TypeError, ValueError):
            return None
        pair = coerce_depth_range(start, solid)
        if pair != (start, solid):
            # Show what was sent.
            self._applying = True
            try:
                self.values = {**self.values, "outline_depth_range": list(pair)}
            finally:
                self._applying = False
        return pair

    # ------------------------------------------------------------------
    # model -> widget
    # ------------------------------------------------------------------

    def _on_visual_render_changed(self, event: VisualRenderChangedEvent) -> None:
        if event.source_id == self._id:
            return
        self._apply(event.field_name, event.new_value)

    def _apply(self, field: str, value: Any) -> None:
        if field not in {c["field"] for c in self.controls}:
            return  # a field this panel does not display
        if field == "outline_selected_labels":
            value = {str(k): int(v) for k, v in (value or {}).items()}
        elif field == "outline_depth_range":
            value = [float(v) for v in value]
        self._applying = True
        try:
            self.values = {**self.values, field: value}
        finally:
            self._applying = False


class _PaletteFollowingPanel(AnywidgetVisualRenderPanel):
    """A panel whose swatches come from the live outline palette."""

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [
            *super().subscription_specs(),
            # Unfiltered: a render-config change carries no entity id.
            SubscriptionSpec(
                event_type=RenderConfigChangedEvent,
                handler=self._on_render_config_changed,
            ),
        ]

    def _on_render_config_changed(self, event: RenderConfigChangedEvent) -> None:
        if event.section != "outline":
            return
        palette = getattr(event.config, "palette", None)
        if palette is None:
            return
        self.palette = [list(entry) for entry in palette]


class AnywidgetVisualOutlineControls(_PaletteFollowingPanel):
    """Outline slot and placement for one non-labels visual."""

    section: ClassVar[str] = "visual_outline"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_outline"]


class AnywidgetLabelsOutlineControls(_PaletteFollowingPanel):
    """Outline mode, slot and per-label selection for a labels visual.

    The panel changes shape with its mode, because ``outline.slot`` means
    two different things.  In whole-volume, all-boundaries and
    depth-threshold mode it chooses the colour, so it gets the swatch row
    every other visual has; in per-label mode it only decides whether the
    volume participates, so it gets a checkbox and the per-label rows
    appear.  The depth range row is drawn in depth-threshold mode only.
    Re-describing the controls is enough -- the ESM rebuilds on
    ``change:controls``.
    """

    section: ClassVar[str] = "labels_outline"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["labels_outline"]

    def __init__(self, visual_ids, values, **kwargs) -> None:
        from cellier.visuals._label_memory import DEFAULT_OUTLINE_DEPTH_RANGE

        # A panel built with no range shows the model's default.
        values = {"outline_depth_range": DEFAULT_OUTLINE_DEPTH_RANGE, **values}
        super().__init__(visual_ids, values, **kwargs)

    def _adjust_for_mode(self, entry: dict, values: dict[str, Any]) -> dict | None:
        mode = values.get("outline_mode") or "per_label"
        if entry["field"] == "outline_depth_range" and mode not in DEPTH_RANGE_MODES:
            return None  # no depth bands in this mode
        if mode in SLOT_IS_COLOUR_MODES:
            if entry["field"] == "outline_selected_labels":
                return None  # no per-label colours in these modes
            return entry
        if entry["field"] == "outline.slot":
            # Participation only; a swatch here would choose a colour that
            # the per-label rows below immediately override.
            return {**entry, "kind": "bool", "label": "Outline this volume"}
        return entry

    def _apply(self, field: str, value: Any) -> None:
        if field == "outline_depth_range" and field not in {
            c["field"] for c in self.controls
        }:
            # The row is not drawn in this mode, but the value is what it
            # shows when the mode is next entered.
            self._applying = True
            try:
                self.values = {**self.values, field: [float(v) for v in value]}
            finally:
                self._applying = False
            return
        super()._apply(field, value)
        if field == "outline_mode":
            self.controls = self._describe_controls(self.values)

    def _on_values_change(self, change) -> None:
        super()._on_values_change(change)
        before = (change.old or {}).get("outline_mode")
        after = (change.new or {}).get("outline_mode")
        if after != before:
            self.controls = self._describe_controls(self.values)


class AnywidgetVisualOcclusionControls(AnywidgetVisualRenderPanel):
    """Whether one visual receives ambient occlusion."""

    section: ClassVar[str] = "visual_occlusion"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_occlusion"]


class AnywidgetVisualPickingControls(AnywidgetVisualRenderPanel):
    """Whether one visual writes to the pick buffer.

    ``pick_write`` has an outgoing event of its own, so this panel follows
    ``PickWriteChangedEvent`` while still writing through the same seam.
    """

    section: ClassVar[str] = "visual_picking"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_picking"]

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [
            SubscriptionSpec(
                event_type=PickWriteChangedEvent,
                handler=self._on_pick_write_changed,
                entity_id=visual_id,
            )
            for visual_id in self._visual_ids
        ]

    def _on_pick_write_changed(self, event: PickWriteChangedEvent) -> None:
        if event.source_id == self._id:
            return
        self._apply("pick_write", event.pick_write)
