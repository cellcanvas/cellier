"""Qt panels for one visual's screen-space render settings.

Three widgets, not one, because ``outline.slot`` means two different things:

* on a mesh, an image, points or lines it is both the on/off switch **and**
  the colour, since the selection layer draws the region in
  ``palette[slot - 1]``;
* on a labels visual it only decides whether the volume participates -- the
  colour comes from :attr:`outline_selected_labels`, per label value.

A single widget covering both would have to disable half of itself, which is
the shape of a widget that should be two.  Occlusion is a third because it is
independent of outlining, and because both features are opt-in separately.

All three drive the model through ``VisualRenderUpdateEvent`` and follow it
through ``VisualRenderChangedEvent``, keyed by ``visual_id``.  The outline
widgets additionally follow ``RenderConfigChangedEvent`` so their swatches
track the palette they are choosing from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

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
    DEPTH_RANGE_SOLID_MAXIMUM,
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

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from uuid import UUID

#: Rows a label-selection editor will draw before it stops offering "Add".
#: The GPU capacity is 256; a panel that tall is not a panel.
MAX_LABEL_ROWS = 16


class QtVisualRenderPanel:
    """Base for a Qt panel driving one visual's screen-space settings.

    Parameters
    ----------
    visual_ids :
        Every visual this panel writes to.  One on a ``Viewer``; the four
        panel siblings on an ``OrthoViewer``, driven in lock-step.  The
        panel subscribes to all of them and fans a user edit out to each.
    values :
        Current field values, read off the model by the caller.
    parent :
        Optional Qt parent widget.
    """

    #: Key into :data:`VISUAL_RENDER_CONTROLS`.
    section: ClassVar[str] = ""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_ids: Sequence[UUID],
        values: dict[str, Any],
        *,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        self._id = uuid4()
        self._visual_ids = list(visual_ids)
        self._values = dict(values)
        self._appliers: dict[str, Callable[[Any], None]] = {}

        self._content = QWidget(parent)
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._build()

        from cellier.gui.qt.visuals._chrome import titled_group

        self._widget = titled_group(
            VISUAL_RENDER_TITLES[self.section], self._content, parent
        )

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Draw this panel's controls.  Subclasses override."""
        raise NotImplementedError

    def _specs(self):
        return VISUAL_RENDER_CONTROLS[self.section]

    def _spec(self, field: str):
        return next(c for c in self._specs() if c.field == field)

    def _tooltip(self, control) -> str:
        """A control's tooltip, with the attribute it drives appended."""
        return with_api_path(control.tooltip, visual_path(control.field))

    # ------------------------------------------------------------------
    # WidgetView contract
    # ------------------------------------------------------------------

    @property
    def widget(self):
        """The titled group to embed in a layout."""
        return self._widget

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One inbound subscription per visual this panel drives."""
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

    def _emit(self, field: str, value: Any) -> None:
        self._values[field] = value
        for visual_id in self._visual_ids:
            self.changed.emit(
                VisualRenderUpdateEvent(
                    source_id=self._id,
                    visual_id=visual_id,
                    field=field,
                    value=value,
                )
            )

    # ------------------------------------------------------------------
    # model -> widget
    # ------------------------------------------------------------------

    def _on_visual_render_changed(self, event: VisualRenderChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        applier = self._appliers.get(event.field_name)
        if applier is None:
            return  # a field this panel does not display
        self._values[event.field_name] = event.new_value
        applier(event.new_value)

    # ------------------------------------------------------------------
    # Control builders
    # ------------------------------------------------------------------

    def _add_choice(self, field: str, choices, *, auto_suffix: str = ""):
        """A menu over a small closed set, e.g. the placement or occlusion."""
        from qtpy.QtWidgets import QComboBox

        from cellier.gui.qt.visuals._chrome import labelled_row

        spec = self._spec(field)
        control = QComboBox(self._content)
        values = []
        for label, value in choices:
            text = (
                f"{label} ({auto_suffix})" if label == "Auto" and auto_suffix else label
            )
            control.addItem(text)
            values.append(value)
        current = self._values.get(field)
        control.setCurrentIndex(values.index(current) if current in values else 0)
        control.setToolTip(self._tooltip(spec))
        control.currentIndexChanged.connect(
            lambda index: self._emit(field, values[index])
        )

        def _apply(value) -> None:
            control.blockSignals(True)
            control.setCurrentIndex(values.index(value) if value in values else 0)
            control.blockSignals(False)

        self._appliers[field] = _apply
        self._layout.addWidget(labelled_row(spec.label, control, self._content))
        return control

    def _add_bool(self, field: str, *, to_model=bool, from_model=bool):
        """A checkbox that names itself.

        *to_model* and *from_model* let a checkbox drive a field that is not
        a bool -- the labels outline toggle writes slot 0 or 1.
        """
        from qtpy.QtWidgets import QCheckBox

        spec = self._spec(field)
        control = QCheckBox(spec.label, self._content)
        control.setChecked(from_model(self._values.get(field)))
        control.setToolTip(self._tooltip(spec))
        control.toggled.connect(lambda on: self._emit(field, to_model(on)))

        def _apply(value) -> None:
            control.blockSignals(True)
            control.setChecked(from_model(value))
            control.blockSignals(False)

        self._appliers[field] = _apply
        self._layout.addWidget(control)
        return control


class _QtPaletteFollowingPanel(QtVisualRenderPanel):
    """Base for the two outline panels.

    They share three things the other per-visual panels do not need: a
    palette, a subscription that keeps it current, and the swatch row that
    offers it.  Both are needed by the labels panel too -- in whole-volume
    mode ``outline.slot`` chooses the colour exactly as it does elsewhere.

    Parameters
    ----------
    visual_ids :
        The visuals to drive.
    values :
        Current field values.
    palette :
        The current ``render_config.outline.palette``.  Followed live.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        visual_ids: Sequence[UUID],
        values: dict[str, Any],
        *,
        palette: Sequence[Sequence[float]] = (),
        parent=None,
    ) -> None:
        self._palette = [tuple(entry) for entry in palette]
        super().__init__(visual_ids, values, parent=parent)

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Also follow the palette, which is what the swatches offer."""
        return [
            *super().subscription_specs(),
            # Unfiltered: a render-config change has no entity id.  The
            # handler keeps only the outline palette.
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
        self._palette = [tuple(entry) for entry in palette]
        self._on_palette_changed()

    def _emit(self, field: str, value: Any) -> None:
        super()._emit(field, value)
        if field == "outline.slot":
            self._restyle_swatches()

    def _on_palette_changed(self) -> None:
        """Redraw whatever this panel draws from the palette."""
        self._redraw_swatches()

    # ------------------------------------------------------------------
    # The slot swatches
    # ------------------------------------------------------------------

    def _add_slot_swatches(self) -> None:
        from qtpy.QtWidgets import QButtonGroup, QHBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import labelled_row

        spec = self._spec("outline.slot")
        self._swatch_host = QWidget(self._content)
        self._swatch_layout = QHBoxLayout(self._swatch_host)
        self._swatch_layout.setContentsMargins(0, 0, 0, 0)
        self._swatch_layout.setSpacing(3)
        self._swatch_group = QButtonGroup(self._swatch_host)
        self._swatch_group.setExclusive(True)
        self._swatch_host.setToolTip(self._tooltip(spec))

        self._redraw_swatches()
        self._appliers["outline.slot"] = lambda _value: self._restyle_swatches()
        self._layout.addWidget(
            labelled_row(spec.label, self._swatch_host, self._content)
        )

    def _swatch_style(self, slot: int, rgba) -> str:
        """Stylesheet for one swatch, marked when it is the chosen slot.

        The chosen slot is marked by a heavy border rather than by Qt's own
        checked indicator, which an explicit background colour hides.
        """
        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        current = int(self._values.get("outline.slot", 0) or 0)
        border = "2px solid black" if current == slot else "1px solid #888"
        return f"background-color: {rgba_to_hex(as_rgba(rgba))}; border: {border};"

    def _restyle_swatches(self) -> None:
        """Move the chosen marking without rebuilding the row.

        Called from the user's own click, where deleting the button whose
        signal is being handled would be a crash rather than a refresh.
        Rebuilding is only needed when the *palette* changes, because only
        then does the number of swatches change.
        """
        if not getattr(self, "_swatch_buttons", None):
            return
        current = int(self._values.get("outline.slot", 0) or 0)
        for slot, button, rgba in self._swatch_buttons:
            if rgba is None:  # the Off button
                button.setChecked(current == 0)
                continue
            button.setChecked(current == slot)
            button.setStyleSheet(self._swatch_style(slot, rgba))

    def _redraw_swatches(self) -> None:
        """Rebuild the row: the palette can change length underneath it."""
        from qtpy.QtWidgets import QToolButton

        for button in list(self._swatch_group.buttons()):
            self._swatch_group.removeButton(button)
        while self._swatch_layout.count():
            item = self._swatch_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        current = int(self._values.get("outline.slot", 0) or 0)
        self._swatch_buttons: list = []

        off = QToolButton(self._swatch_host)
        off.setText("Off")
        off.setCheckable(True)
        off.setChecked(current == 0)
        off.setToolTip("Not outlined")
        off.clicked.connect(lambda: self._emit("outline.slot", 0))
        self._swatch_group.addButton(off)
        self._swatch_layout.addWidget(off)
        self._swatch_buttons.append((0, off, None))

        for index, entry in enumerate(self._palette):
            slot = index + 1
            button = QToolButton(self._swatch_host)
            button.setFixedSize(22, 22)
            button.setCheckable(True)
            button.setChecked(current == slot)
            button.setStyleSheet(self._swatch_style(slot, entry))
            button.setToolTip(f"Slot {slot}")
            button.clicked.connect(
                lambda _=False, s=slot: self._emit("outline.slot", s)
            )
            self._swatch_group.addButton(button)
            self._swatch_layout.addWidget(button)
            self._swatch_buttons.append((slot, button, entry))

        self._swatch_layout.addStretch()


class QtVisualOutlineControls(_QtPaletteFollowingPanel):
    """Outline slot and placement for one non-labels visual.

    The slot is a row of swatches rather than a number, because a slot
    number means nothing to someone thinking "outline this one in magenta".
    The swatches are the live palette, so the control can only offer slots
    that exist -- which removes the "slot N has no palette entry" failure by
    construction rather than by warning.

    Parameters
    ----------
    visual_ids :
        The visuals to drive.
    values :
        ``outline.slot`` and ``outline.placement``, plus
        ``default_placement`` -- what ``Auto`` resolves to for this visual
        type, computed by the caller.
    palette :
        The current ``render_config.outline.palette``.  Followed live.
    parent :
        Optional Qt parent widget.
    """

    section: ClassVar[str] = "visual_outline"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_outline"]

    def _build(self) -> None:
        self._add_slot_swatches()
        self._add_choice(
            "outline.placement",
            PLACEMENT_CHOICES,
            auto_suffix=str(self._values.get("default_placement", "")),
        )


class QtLabelsOutlineControls(_QtPaletteFollowingPanel):
    """Outline participation and the per-label selection for a labels visual.

    A labels visual is outlined *per label*, so its own slot only makes it
    eligible for the boundaries layer -- hence a checkbox rather than the
    swatch row.  The colour of each selected label comes from the rows
    below, which is where the palette slots are chosen.

    Parameters
    ----------
    visual_ids :
        The visuals to drive.
    values :
        ``outline.slot``, ``outline.placement``, ``outline_mode``,
        ``outline_selected_labels``, ``outline_depth_range`` and
        ``default_placement``.
    palette :
        The current outline palette, for the per-row slot swatches.
    parent :
        Optional Qt parent widget.
    """

    section: ClassVar[str] = "labels_outline"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["labels_outline"]

    def _build(self) -> None:
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        self._add_choice("outline_mode", OUTLINE_MODE_CHOICES)

        # The mode-dependent half lives in its own host so it can be rebuilt
        # without disturbing the controls around it.
        self._mode_host = QWidget(self._content)
        self._mode_layout = QVBoxLayout(self._mode_host)
        self._mode_layout.setContentsMargins(0, 0, 0, 0)
        self._mode_layout.setSpacing(2)
        self._layout.addWidget(self._mode_host)

        self._add_choice(
            "outline.placement",
            PLACEMENT_CHOICES,
            auto_suffix=str(self._values.get("default_placement", "")),
        )
        self._build_mode_section()

        # Changing the mode reshapes the panel, and it has to reshape on
        # *both* routes.  The applier below covers a change made elsewhere;
        # ``_emit`` covers the user's own click, which never reaches the
        # applier -- the echo of it comes back stamped with this widget's
        # source id and is discarded, correctly, as its own.
        mode_applier = self._appliers["outline_mode"]

        def _apply_mode(value) -> None:
            mode_applier(value)
            self._build_mode_section()

        self._appliers["outline_mode"] = _apply_mode

    def _emit(self, field: str, value: Any) -> None:
        super()._emit(field, value)
        if field == "outline_mode":
            # Safe to rebuild from inside the combo's own signal: the combo
            # lives in the outer layout, not the section being replaced.
            self._build_mode_section()

    def _on_visual_render_changed(self, event: VisualRenderChangedEvent) -> None:
        if (
            event.field_name == "outline_depth_range"
            and event.source_id != self._id
            and "outline_depth_range" not in self._appliers
        ):
            # The row is not drawn in this mode, but the value is what it
            # shows when the mode is next entered.
            self._values["outline_depth_range"] = tuple(event.new_value)
            return
        super()._on_visual_render_changed(event)

    def _mode(self) -> str:
        return self._values.get("outline_mode") or "per_label"

    def _build_mode_section(self) -> None:
        """Draw the controls whose meaning depends on the mode.

        In whole-volume, all-boundaries and depth-threshold mode
        ``outline.slot`` chooses the colour, so it gets the same swatch row
        every other visual has.  In per-label mode it only decides whether
        the volume participates, and the colour comes from the rows below --
        so it gets a checkbox, and the rows appear.  Depth-threshold mode
        also gets the row for its depth range.
        """
        while self._mode_layout.count():
            item = self._mode_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for field in (
            "outline.slot",
            "outline_selected_labels",
            "outline_depth_range",
        ):
            self._appliers.pop(field, None)

        # The builders append to ``self._layout``; point them at the host.
        outer, self._layout = self._layout, self._mode_layout
        try:
            if self._mode() in SLOT_IS_COLOUR_MODES:
                self._add_slot_swatches()
            else:
                self._add_bool(
                    "outline.slot",
                    to_model=lambda on: 1 if on else 0,
                    from_model=bool,
                )
                self._add_label_rows()
            if self._mode() in DEPTH_RANGE_MODES:
                self._add_depth_range()
        finally:
            self._layout = outer

    # ------------------------------------------------------------------
    # The depth range
    # ------------------------------------------------------------------

    def _add_depth_range(self) -> None:
        """Two spin boxes for the ``(start, solid)`` pair, sent as one value.

        The pair goes to the model whole, so the model never sees a start
        above its solid value on the way from one range to another.
        """
        from qtpy.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

        from cellier.gui.qt.visuals._chrome import labelled_row

        spec = self._spec("outline_depth_range")
        tooltip = self._tooltip(spec)
        host = QWidget(self._content)
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        boxes = []
        for text in DEPTH_RANGE_LABELS:
            label = QLabel(text, host)
            label.setToolTip(tooltip)
            box = QDoubleSpinBox(host)
            box.setDecimals(spec.decimals)
            box.setSingleStep(DEPTH_RANGE_STEP)
            # A value is sent when it is committed, not at every keystroke:
            # "0.0" on the way to "0.05" is not a range.
            box.setKeyboardTracking(False)
            box.setToolTip(tooltip)
            row.addWidget(label)
            row.addWidget(box, 1)
            boxes.append(box)
        start_box, solid_box = boxes
        start_box.setRange(spec.minimum, 1.0)
        solid_box.setRange(spec.minimum, DEPTH_RANGE_SOLID_MAXIMUM)

        def _show(pair) -> None:
            start, solid = pair
            for box, value in ((start_box, start), (solid_box, solid)):
                box.blockSignals(True)
                box.setValue(float(value))
                box.blockSignals(False)

        def _changed(_value) -> None:
            pair = coerce_depth_range(start_box.value(), solid_box.value())
            _show(pair)
            self._emit("outline_depth_range", pair)

        _show(self._depth_range())
        start_box.valueChanged.connect(_changed)
        solid_box.valueChanged.connect(_changed)

        self._depth_range_boxes = (start_box, solid_box)
        self._appliers["outline_depth_range"] = _show
        self._layout.addWidget(labelled_row(spec.label, host, self._content))

    def _depth_range(self) -> tuple[float, float]:
        from cellier.visuals._label_memory import DEFAULT_OUTLINE_DEPTH_RANGE

        start, solid = (
            self._values.get("outline_depth_range") or DEFAULT_OUTLINE_DEPTH_RANGE
        )
        return float(start), float(solid)

    def _on_palette_changed(self) -> None:
        """Whichever half is on screen draws from the palette."""
        if self._mode() in SLOT_IS_COLOUR_MODES:
            self._redraw_swatches()
        else:
            self._redraw_rows()

    # ------------------------------------------------------------------
    # The per-label rows
    # ------------------------------------------------------------------

    def _add_label_rows(self) -> None:
        from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import labelled_row

        spec = self._spec("outline_selected_labels")
        host = QWidget(self._content)
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        host.setToolTip(self._tooltip(spec))

        self._rows_host = QWidget(host)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        column.addWidget(self._rows_host)

        self._add_row_button = QPushButton("Add label", host)
        self._add_row_button.clicked.connect(self._on_add_row)
        column.addWidget(self._add_row_button)

        self._redraw_rows()
        self._appliers["outline_selected_labels"] = lambda _value: self._redraw_rows()
        self._layout.addWidget(labelled_row(spec.label, host, self._content))

    def _selection(self) -> dict[int, int]:
        return dict(self._values.get("outline_selected_labels") or {})

    def _on_add_row(self) -> None:
        """Append a label one past the highest already listed."""
        selection = self._selection()
        if len(selection) >= MAX_LABEL_ROWS:
            return
        next_label = max(selection, default=0) + 1
        selection[next_label] = 1
        self._emit("outline_selected_labels", selection)
        self._redraw_rows()

    def _redraw_rows(self) -> None:
        from qtpy.QtWidgets import (
            QHBoxLayout,
            QPushButton,
            QSpinBox,
            QToolButton,
            QWidget,
        )

        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        selection = self._selection()
        for label_value, slot in sorted(selection.items()):
            row = QWidget(self._rows_host)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(3)

            value_spin = QSpinBox(row)
            value_spin.setRange(0, 2**31 - 1)
            value_spin.setValue(int(label_value))
            value_spin.setToolTip("Label value")
            value_spin.valueChanged.connect(
                lambda new, old=label_value: self._on_label_renamed(old, new)
            )
            layout.addWidget(value_spin)

            swatch = QToolButton(row)
            swatch.setFixedSize(20, 20)
            colour = (
                self._palette[slot - 1]
                if 1 <= slot <= len(self._palette)
                else (0.0, 0.0, 0.0, 0.0)
            )
            swatch.setStyleSheet(
                f"background-color: {rgba_to_hex(as_rgba(colour))}; "
                "border: 1px solid #888;"
            )
            swatch.setToolTip(f"Slot {slot} -- click to cycle")
            swatch.clicked.connect(lambda _=False, lv=label_value: self._cycle_slot(lv))
            layout.addWidget(swatch)

            remove = QPushButton("x", row)
            remove.setFixedWidth(22)
            remove.setToolTip("Stop outlining this label")
            remove.clicked.connect(lambda _=False, lv=label_value: self._remove_row(lv))
            layout.addWidget(remove)
            layout.addStretch()

            self._rows_layout.addWidget(row)

        self._add_row_button.setEnabled(len(selection) < MAX_LABEL_ROWS)

    def _on_label_renamed(self, old: int, new: int) -> None:
        selection = self._selection()
        if new == old or new in selection:
            return
        selection[new] = selection.pop(old)
        self._emit("outline_selected_labels", selection)
        self._redraw_rows()

    def _cycle_slot(self, label_value: int) -> None:
        """Step a row to the next palette slot, wrapping at the end."""
        selection = self._selection()
        if not self._palette:
            return
        selection[label_value] = selection.get(label_value, 0) % len(self._palette) + 1
        self._emit("outline_selected_labels", selection)
        self._redraw_rows()

    def _remove_row(self, label_value: int) -> None:
        selection = self._selection()
        selection.pop(label_value, None)
        self._emit("outline_selected_labels", selection)
        self._redraw_rows()


class QtVisualOcclusionControls(QtVisualRenderPanel):
    """Whether one visual receives ambient occlusion.

    Identical for every visual type, and independent of outlining -- which
    is why it is its own opt-in widget rather than a row in the outline one.
    """

    section: ClassVar[str] = "visual_occlusion"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_occlusion"]

    def _build(self) -> None:
        self._add_choice("ambient_occlusion", AMBIENT_OCCLUSION_CHOICES)


class QtVisualPickingControls(QtVisualRenderPanel):
    """Whether one visual writes to the pick buffer.

    Shown whenever either render-settings widget is, because both features
    are derived from the pick buffer: choosing an outline slot turns this
    back on and warns, and without a control the warning names a field
    nothing on screen shows.

    ``pick_write`` has an outgoing event of its own, so this panel follows
    ``PickWriteChangedEvent`` rather than ``VisualRenderChangedEvent`` while
    still writing through the same seam.
    """

    section: ClassVar[str] = "visual_picking"
    DEFAULT_TITLE: ClassVar[str] = VISUAL_RENDER_TITLES["visual_picking"]

    def _build(self) -> None:
        self._add_bool("pick_write")

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
        self._values["pick_write"] = event.pick_write
        self._appliers["pick_write"](event.pick_write)
