"""Shared plumbing for the Qt render-settings panels.

The three panels -- outlines, ambient occlusion, temporal accumulation --
differ entirely in *which* controls they hold and not at all in how those
controls reach the model, so the bus contract lives here once.

Two things distinguish these panels from the appearance widgets in
``cellier.gui.qt.visuals``:

* **There is no entity id.**  Render configuration belongs to the
  ``RenderManager``, not to a scene, a visual or a canvas, so the panel
  subscribes to every ``RenderConfigChangedEvent`` and filters on
  ``section`` itself.
* **A panel is one widget, not one per field.**  These settings are only
  meaningful as a group -- a radius means nothing without the strength and
  the sample count beside it -- so each panel is a single composite
  ``WidgetView`` with one ``changed`` / ``closed`` pair, structurally like
  ``QtChannelList`` rather than like ``QtLodBiasSlider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    RenderConfigChangedEvent,
    RenderConfigUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._render_controls import (
    RENDER_CONTROLS,
    next_palette_color,
    render_config_path,
    with_api_path,
)
from cellier.render._config import MAX_OUTLINE_SLOT

if TYPE_CHECKING:
    from collections.abc import Callable


class QtRenderConfigPanel:
    """Base for a Qt panel driving one section of the render config.

    Subclasses set :attr:`section` and build their controls in ``_build``,
    registering each one with :meth:`_register` so the inbound handler can
    push a change back into it.

    Wire to the controller after construction::

        panel = QtAmbientOcclusionControls(controller.render_config.ambient_occlusion)
        controller.connect_widget(panel, subscription_specs=panel.subscription_specs())

    Parameters
    ----------
    parent :
        Optional Qt parent widget.
    """

    #: Which render-config section this panel drives.
    section: ClassVar[str] = ""

    #: Title of the group box drawn around the panel.
    title: ClassVar[str] = ""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(self, parent=None, *, slot_usage=None) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

        self._id = uuid4()
        # Reads ``{slot: visual count}``.  Derived state, like the effective
        # occlusion radius, so it is a callable rather than a value.
        self._slot_usage = slot_usage
        # field name -> callable that pushes a value into the control with
        # its signals blocked.  Populated by ``_register``.
        self._appliers: dict[str, Callable[[Any], None]] = {}
        # Read-only labels that restate derived state, refreshed on demand
        # rather than driven by a field.
        self._readouts: list[Callable[[], None]] = []

        self._container = QWidget(parent)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        self._container.setMinimumWidth(260)

        # The panel draws its own name, as the anywidget twin always has and as
        # every appearance control on both toolkits already does
        # (``plans/label_ownership_unification.md``).  The renderer used to wrap
        # it instead, which left a directly-constructed Qt panel anonymous where
        # its twin named itself -- and kept the two dock builders from being one
        # function (``plans/gui_backend_seam.md`` D6).
        from cellier.gui.qt.visuals._chrome import titled_group

        self._titled = titled_group(self.title, self._container, parent)
        self._titled.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

    # ------------------------------------------------------------------
    # Public interface (the WidgetView contract)
    # ------------------------------------------------------------------

    @property
    def widget(self):
        """The Qt widget to embed in a layout -- the panel inside its own title.

        The bare content is :attr:`content`, for a caller that wants to place
        the rows itself.
        """
        return self._titled

    @property
    def content(self):
        """The untitled container holding the panel's rows."""
        return self._container

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Subscribe to every render-config change; filter by section here.

        ``entity_id`` is deliberately ``None``: there is no UUID a render
        config change could be keyed by.
        """
        return [
            SubscriptionSpec(
                event_type=RenderConfigChangedEvent,
                handler=self._on_render_config_changed,
            )
        ]

    def refresh_readouts(self) -> None:
        """Re-read any derived values this panel displays.

        Derived state -- the effective occlusion radius, the accumulated
        frame count -- changes without any field changing, so nothing on
        the bus announces it.  A host that wants those labels live calls
        this on a timer; they are correct at construction either way.
        """
        for readout in self._readouts:
            readout()

    # ------------------------------------------------------------------
    # widget -> model (outbound)
    # ------------------------------------------------------------------

    def _emit(self, field: str, value: Any) -> None:
        self.changed.emit(
            RenderConfigUpdateEvent(
                source_id=self._id,
                section=self.section,
                field=field,
                value=value,
            )
        )

    def _make_emit(self, field: str) -> Callable[[Any], None]:
        def _on_change(value: Any) -> None:
            self._emit(field, value)

        return _on_change

    # ------------------------------------------------------------------
    # model -> widget (inbound)
    # ------------------------------------------------------------------

    def _on_render_config_changed(self, event: RenderConfigChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        if event.section != self.section:
            return  # another panel's section
        if event.field_name is None:
            # Whole-section replacement: refresh everything this panel shows.
            for field, applier in self._appliers.items():
                applier(_read_dotted(event.config, field))
            self.refresh_readouts()
            return
        applier = self._appliers.get(event.field_name)
        if applier is None:
            return  # a field this panel does not display
        applier(event.new_value)
        self.refresh_readouts()

    # ------------------------------------------------------------------
    # Control builders
    #
    # Each adds a named row, wires it to ``_emit``, and registers an
    # applier so an inbound change can push a value back in.
    # ------------------------------------------------------------------

    def _register(self, field: str, applier: Callable[[Any], None]) -> None:
        self._appliers[field] = applier

    def _build_from_spec(self, config: Any, *, skip: set[str] | None = None) -> None:
        """Draw every control the shared spec lists for this section.

        Labels, ranges and tooltips come from
        :data:`cellier.gui._render_controls.RENDER_CONTROLS` rather than
        from here, so the notebook panels say the same things without
        anyone keeping two lists in step.

        Parameters
        ----------
        config :
            The section config to read initial values from.
        skip :
            Fields this panel builds itself because they need behaviour a
            spec cannot express -- the occlusion radius and its auto mode.
        """
        skip = skip or set()
        pending: list = []
        current_group: str | None = None

        def _flush() -> None:
            if not pending:
                return
            controls = list(pending)
            pending.clear()
            if current_group is None:
                for control in controls:
                    self._add_spec_control(control, config)
                return
            self._add_group(
                current_group,
                lambda _p, controls=controls: [
                    self._add_spec_control(control, config) for control in controls
                ],
            )

        for control in RENDER_CONTROLS[self.section]:
            if control.field in skip:
                continue
            if control.group != current_group:
                _flush()
                current_group = control.group
            pending.append(control)
        _flush()

    def _tooltip(self, control) -> str:
        """A control's tooltip, with the attribute it drives appended."""
        return with_api_path(
            control.tooltip, render_config_path(self.section, control.field)
        )

    def _add_spec_control(self, control, config: Any):
        """Draw one spec control, seeded from *config*."""
        initial = _read_dotted(config, control.field)
        tooltip = self._tooltip(control)
        if control.kind == "bool":
            return self._add_checkbox(
                control.field, control.label, initial, tooltip=tooltip
            )
        if control.kind == "int":
            return self._add_int_spin(
                control.field,
                control.label,
                initial,
                int(control.minimum),
                int(control.maximum),
                tooltip=tooltip,
            )
        if control.kind == "float":
            return self._add_float_slider(
                control.field,
                control.label,
                initial,
                control.minimum,
                control.maximum,
                decimals=control.decimals,
                tooltip=tooltip,
            )
        if control.kind == "color":
            return self._add_color(
                control.field, control.label, initial, tooltip=tooltip
            )
        if control.kind == "palette":
            return self._add_palette(
                control.field, control.label, initial, tooltip=tooltip
            )
        raise ValueError(f"unknown render control kind: {control.kind!r}")

    def _add_row(self, label: str, control, parent=None) -> None:
        from cellier.gui.qt.visuals._chrome import labelled_row

        self._layout.addWidget(labelled_row(label, control, parent or self._container))

    def _add_checkbox(self, field: str, label: str, initial: bool, *, tooltip=""):
        """A checkbox that names itself, rather than being named by a row."""
        from qtpy.QtWidgets import QCheckBox

        control = QCheckBox(label, self._container)
        control.setChecked(bool(initial))
        if tooltip:
            control.setToolTip(tooltip)
        control.toggled.connect(self._make_emit(field))
        self._register(field, _checkbox_applier(control))
        self._layout.addWidget(control)
        return control

    def _add_int_spin(
        self, field: str, label: str, initial: int, low: int, high: int, *, tooltip=""
    ):
        from qtpy.QtWidgets import QSpinBox

        control = QSpinBox(self._container)
        control.setRange(low, high)
        control.setValue(int(initial))
        if tooltip:
            control.setToolTip(tooltip)
        control.valueChanged.connect(self._make_emit(field))
        self._register(field, _spin_applier(control))
        self._add_row(label, control)
        return control

    def _add_float_slider(
        self,
        field: str,
        label: str,
        initial: float,
        low: float,
        high: float,
        *,
        decimals: int = 2,
        tooltip: str = "",
    ):
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        control = QLabeledDoubleSlider(Qt.Orientation.Horizontal, self._container)
        control.setRange(low, high)
        control.setDecimals(decimals)
        control.setValue(float(initial))
        if tooltip:
            control.setToolTip(tooltip)
        control.valueChanged.connect(self._make_emit(field))
        self._register(field, _spin_applier(control))
        self._add_row(label, control)
        return control

    def _add_color(self, field: str, label: str, initial, *, tooltip: str = ""):
        """A clickable RGB swatch, with alpha on a row of its own.

        Two rows rather than one.  Alpha needs its own control because every
        colour here is float RGBA and both toolkits' colour inputs are
        RGB-only -- and it needs its own *row* because sharing one with the
        swatch left the slider a few pixels wide and its readout clipped.

        The swatch itself opens the picker; there is no separate button.  A
        colour swatch is the most obviously clickable thing on the row, so a
        "Choose..." button beside it is a second control for one action.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QToolButton
        from superqt import QLabeledDoubleSlider

        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        state = {"rgba": as_rgba(initial)}

        swatch = QToolButton(self._container)
        swatch.setFixedSize(36, 22)
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.setToolTip(tooltip or "Click to choose a colour")

        alpha = QLabeledDoubleSlider(Qt.Orientation.Horizontal, self._container)
        alpha.setRange(0.0, 1.0)
        alpha.setDecimals(2)
        alpha.setValue(state["rgba"][3])

        def _paint() -> None:
            swatch.setStyleSheet(
                f"background-color: {rgba_to_hex(state['rgba'])}; "
                "border: 1px solid #888;"
            )

        def _on_choose() -> None:
            from qtpy.QtGui import QColor
            from qtpy.QtWidgets import QColorDialog

            from cellier.gui._appearance_fields import hex_to_rgba

            chosen = QColorDialog.getColor(
                QColor(rgba_to_hex(state["rgba"])), self._container, "Choose color"
            )
            if not chosen.isValid():
                return  # user cancelled
            state["rgba"] = hex_to_rgba(chosen.name(), state["rgba"][3])
            _paint()
            self._emit(field, state["rgba"])

        def _on_alpha(value: float) -> None:
            red, green, blue, _a = state["rgba"]
            state["rgba"] = (red, green, blue, float(value))
            self._emit(field, state["rgba"])

        swatch.clicked.connect(_on_choose)
        alpha.valueChanged.connect(_on_alpha)

        def _apply(value) -> None:
            state["rgba"] = as_rgba(value)
            alpha.blockSignals(True)
            alpha.setValue(state["rgba"][3])
            alpha.blockSignals(False)
            _paint()

        _paint()
        self._register(field, _apply)
        self._add_row(label, swatch)
        self._add_row("Alpha", alpha)
        return swatch

    def _add_palette(self, field: str, label: str, initial, *, tooltip: str = ""):
        """The selection palette: a swatch per slot, with add and remove.

        Editing a colour recolours every visual in that slot at once, which
        is the point of a shared palette.

        **Length is editable, and it did not used to be.**  While the palette
        was only read by the shader its length was a configuration detail.
        Now that a per-visual control offers these entries as its choice set,
        the length is the number of groups a user can tell apart -- so a
        four-entry palette and six things to distinguish has to be solvable
        from here.

        Removal takes the **last** entry only.  Removing from the middle
        renumbers every slot above it, silently recolouring visuals that
        never changed; removing from the end can at worst orphan the highest
        slot, which is visible and reversible.  ``CellierController``'s
        palette route warns when that happens, naming the visuals.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QPushButton,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        state = {"entries": [as_rgba(entry) for entry in initial]}

        container = QWidget(self._container)
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)

        swatch_row = QWidget(container)
        swatches = QHBoxLayout(swatch_row)
        swatches.setContentsMargins(0, 0, 0, 0)
        swatches.setSpacing(3)
        column.addWidget(swatch_row)

        def _emit() -> None:
            self._emit(field, list(state["entries"]))

        def _choose(index: int) -> None:
            from qtpy.QtGui import QColor
            from qtpy.QtWidgets import QColorDialog

            from cellier.gui._appearance_fields import hex_to_rgba

            current = state["entries"][index]
            chosen = QColorDialog.getColor(
                QColor(rgba_to_hex(current)), container, f"Slot {index + 1} color"
            )
            if not chosen.isValid():
                return  # user cancelled
            entries = list(state["entries"])
            entries[index] = hex_to_rgba(chosen.name(), current[3])
            state["entries"] = entries
            _rebuild()
            _emit()

        def _rebuild() -> None:
            """Redraw the swatch row from ``state`` -- count can change."""
            while swatches.count():
                item = swatches.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            usage = self._slot_usage() if self._slot_usage is not None else {}
            for index, rgba in enumerate(state["entries"]):
                slot = index + 1
                cell = QWidget(swatch_row)
                cell_layout = QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(0)

                button = QToolButton(cell)
                button.setFixedSize(22, 22)
                button.setStyleSheet(
                    f"background-color: {rgba_to_hex(rgba)}; border: 1px solid #888;"
                )
                count = usage.get(slot, 0)
                # The slot number under each swatch is what connects this row
                # to ``visual.outline.slot`` and to the warnings, which both
                # speak in numbers while the control speaks in colour.
                caption = f"{slot}" if not count else f"{slot} x{count}"
                button.setToolTip(
                    f"Slot {slot}"
                    + (f" -- used by {count} visual(s)" if count else " -- unused")
                )
                button.clicked.connect(lambda _=False, i=index: _choose(i))
                cell_layout.addWidget(button)

                number = QLabel(caption, cell)
                number.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                number.setStyleSheet("font-size: 9px; opacity: 0.7;")
                cell_layout.addWidget(number)

                swatches.addWidget(cell)
            swatches.addStretch()

        def _add() -> None:
            if len(state["entries"]) >= MAX_OUTLINE_SLOT:
                return
            state["entries"] = [*state["entries"], next_palette_color(state["entries"])]
            _rebuild()
            _emit()

        def _remove() -> None:
            if len(state["entries"]) <= 1:
                return
            state["entries"] = state["entries"][:-1]
            _rebuild()
            _emit()

        buttons = QWidget(container)
        button_row = QHBoxLayout(buttons)
        button_row.setContentsMargins(0, 0, 0, 0)
        add_button = QPushButton("+", buttons)
        add_button.setFixedWidth(28)
        add_button.setToolTip(f"Add a slot (at most {MAX_OUTLINE_SLOT})")
        add_button.clicked.connect(_add)
        remove_button = QPushButton("-", buttons)
        remove_button.setFixedWidth(28)
        remove_button.setToolTip("Remove the last slot")
        remove_button.clicked.connect(_remove)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addStretch()
        column.addWidget(buttons)

        def _apply(value) -> None:
            state["entries"] = [as_rgba(entry) for entry in value]
            _rebuild()

        _rebuild()
        self._register(field, _apply)
        if tooltip:
            container.setToolTip(tooltip)
        if label:
            self._add_row(label, container)
        else:
            self._layout.addWidget(container)
        return container

    def _add_readout(self, label: str, read: Callable[[], str]):
        """A read-only label restating something derived, not a setting."""
        from qtpy.QtWidgets import QLabel

        control = QLabel("", self._container)

        def _refresh() -> None:
            control.setText(read())

        _refresh()
        self._readouts.append(_refresh)
        self._add_row(label, control)
        return control

    def _add_group(self, title: str, build: Callable[[Any], None]):
        """A titled sub-group, for rows that only mean something together."""
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import titled_group

        content = QWidget()
        inner = QVBoxLayout(content)
        inner.setContentsMargins(0, 0, 0, 0)

        outer_layout, self._layout = self._layout, inner
        try:
            build(content)
        finally:
            self._layout = outer_layout
        self._layout.addWidget(titled_group(title, content, self._container))


def _checkbox_applier(control) -> Callable[[Any], None]:
    def _apply(value: Any) -> None:
        control.blockSignals(True)
        control.setChecked(bool(value))
        control.blockSignals(False)

    return _apply


def _spin_applier(control) -> Callable[[Any], None]:
    def _apply(value: Any) -> None:
        control.blockSignals(True)
        control.setValue(value)
        control.blockSignals(False)

    return _apply


def _read_dotted(model: Any, field: str) -> Any:
    """Read a dotted attribute path off a config model."""
    target = model
    for name in field.split("."):
        target = getattr(target, name)
    return target
