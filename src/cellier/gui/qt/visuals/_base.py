"""Shared base for the single-field Qt appearance widgets.

Layer 1 of the three-layer design in ``plans/convenience_cleanup.md`` section
10.2: everything about talking to the cellier v2 bus, written once.  Layer 2
(one class per control type -- toggle, slider, spin, combo, colour) implements
the ``_build`` / ``_read`` / ``_apply`` seam; layer 3 binds a field name and a
label and is a few lines per field.

Layer 1 also builds the **label row**: a control names itself, here, rather
than being named by whatever lays it out
(``plans/label_ownership_unification.md``).  Before that, five of the six
layer-2 types rendered a bare control and were named only by the ``QGroupBox``
the appearance renderer wrapped them in -- so a widget used anywhere else was
anonymous, and the one type that *did* label itself (``QtToggle``) showed its
name twice.  The anywidget side always worked this way; this is Qt catching
up, which is why the row mirrors ``.cellier-app-row`` down to the 5.5em
right-aligned label column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from psygnal import Signal

from cellier.events import SubscriptionSpec
from cellier.gui._appearance_fields import (
    NO_MATCH,
    AppearanceFieldSpec,
    appearance_field_spec,
    normalize_visual_ids,
)
from cellier.gui.qt.visuals._chrome import labelled_row

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class _UnsetType:
    """Sentinel distinguishing "no initial value given" from ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


_UNSET = _UnsetType()


class QtAppearanceField:
    """Bidirectional control for **one** appearance field on one or more visuals.

    Subclasses bind ``_field`` and ``_label`` and implement the three-method
    seam.  Everything else -- the widget UUID, the ``changed`` / ``closed``
    signals, the subscriptions, the ``source_id`` echo filter, the re-entrancy
    guard, and building the outgoing ``AppearanceUpdateEvent`` -- lives here.

    ``visual_id`` accepts a single ``UUID`` or a sequence of them.  With a
    sequence the widget drives the whole group in lock-step: it returns one
    subscription per id and emits one update event per id, which is the
    pattern the per-channel list established for the ``OrthoViewer``'s four
    sibling visuals.

    Wire to the controller after construction::

        toggle = QtVisibleToggle(visual_id, initial_value=True)
        controller.connect_widget(
            toggle, subscription_specs=toggle.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        UUID of the visual whose field this widget controls, or a sequence of
        UUIDs to drive as one group.
    initial_value :
        Starting value -- typically read off the visual's appearance model.
        Defaults to the control type's ``_default_value``.
    parent :
        Optional Qt parent widget.
    """

    _field: ClassVar[str]
    _label: ClassVar[str]
    _default_value: ClassVar[Any] = None

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        parent=None,
    ) -> None:
        self._id = uuid4()
        self._visual_ids = normalize_visual_ids(visual_id)
        self._spec = self._make_spec()

        value = self._default_value if initial_value is _UNSET else initial_value
        self._control = self._build(value, parent)
        # ``_build`` is left alone: layer 2 keeps building exactly the control
        # it always did and knows nothing about being labelled.
        self._row = labelled_row(self._label, self._control, parent)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self):
        """The labelled row to insert into a layout.

        This is the whole control, name included -- what a caller wants in
        every case except reaching for the input itself, which is
        :attr:`control`.
        """
        return self._row

    @property
    def control(self):
        """The bare input inside the row -- the checkbox, slider or combo.

        For tests and for callers driving the input directly.  Layer 2's
        ``_read`` / ``_apply`` use ``self._control`` and never go through here.
        """
        return self._control

    @property
    def visual_ids(self) -> tuple[UUID, ...]:
        """The visual ids this widget drives, always as a tuple."""
        return self._visual_ids

    @property
    def field(self) -> str:
        """The appearance field name this widget writes."""
        return self._spec.name

    @property
    def value(self) -> Any:
        """The control's current value.

        A property, not a method, so it reads the same on both toolkits: the
        anywidget twin spells this as a synced trait, and code that had to ask
        which front end it was holding could not be written against either.

        Assigning to it behaves like a user edit -- the control moves *and* the
        change reaches the bus -- which is also what assigning to the anywidget
        trait does.  To push a value in without emitting (an inbound bus
        write), use ``_apply``.
        """
        return self._read()

    @value.setter
    def value(self, new_value: Any) -> None:
        self._apply(new_value)
        self._emit(self._read())

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return one inbound subscription per visual id (subscribe-to-all).

        Subscribing to every id rather than only the first means the widget
        stays correct when a sibling is written by something other than this
        widget -- see ``multichannel_widget_design.md`` section 11.3.
        """
        return [
            SubscriptionSpec(
                event_type=self._spec.inbound_event_type,
                handler=self._on_inbound_event,
                entity_id=visual_id,
            )
            for visual_id in self._visual_ids
        ]

    # ── model -> widget ──────────────────────────────────────────────────────

    def _on_inbound_event(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own change; ignore
        value = self._spec.inbound_value(event)
        if value is NO_MATCH:
            return  # a field this widget does not drive
        self._apply(value)

    # ── widget -> model ──────────────────────────────────────────────────────

    def _emit(self, value: Any) -> None:
        """Emit one update event per driven target, built by the spec."""
        for target_id in self._visual_ids:
            self.changed.emit(self._spec.outbound_event(self._id, target_id, value))

    # ── Subclass seam ────────────────────────────────────────────────────────

    @classmethod
    def _make_spec(cls) -> AppearanceFieldSpec:
        """Return the spec describing the field this class drives.

        An appearance field by default.  Overridden by the overlay field
        classes (``cellier.gui._overlay_fields.OverlayFieldMixin``), which
        reuse every layer-2 control unchanged and swap only the spec.
        """
        return appearance_field_spec(cls._field, cls._label)

    def _build(self, initial_value: Any, parent) -> Any:
        """Construct the Qt control, seeded with *initial_value*.

        Implementations must connect the control's change signal to
        ``self._emit``.
        """
        raise NotImplementedError

    def _read(self) -> Any:
        """Return the control's current value."""
        raise NotImplementedError

    def _apply(self, value: Any) -> None:
        """Push *value* into the control **without** re-firing its signals."""
        raise NotImplementedError


class QtToggle(QtAppearanceField):
    """Layer 2: a ``QCheckBox`` for a boolean appearance field."""

    _default_value: ClassVar[Any] = False

    def _build(self, initial_value: Any, parent):
        from qtpy.QtWidgets import QCheckBox

        # No text: the row's label names it, like every other control type.
        box = QCheckBox(parent)
        box.setChecked(bool(initial_value))
        box.toggled.connect(self._emit)
        return box

    def _read(self) -> bool:
        return bool(self._control.isChecked())

    def _apply(self, value: Any) -> None:
        self._control.blockSignals(True)
        self._control.setChecked(bool(value))
        self._control.blockSignals(False)


class QtBoundedSlider(QtAppearanceField):
    """Layer 2: a labelled slider for a float field the **model** bounds.

    The range comes from the field's ``ge``/``le`` metadata and there is
    deliberately no keyword to widen it: a wider slider would build a control
    that emits values pydantic then rejects (design section 6.5.1 proposal 3).
    Layer 3 binds ``_default_range`` by calling
    ``cellier.gui._appearance_fields.field_bounds`` on the model, so the two
    cannot drift.

    One documented exception: ``QtIsoThresholdSlider``.  ``iso_threshold``
    carries no ``ge``/``le`` -- a threshold is bounded relative to the data's
    intensity range, not absolutely -- so its ``(0.0, 1.0)`` is a convention
    for the normalised images cellier renders, stated rather than derived.
    """

    _default_value: ClassVar[Any] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
    _default_decimals: ClassVar[int] = 2

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        decimals: int | None = None,
        parent=None,
    ) -> None:
        # Resolved before super().__init__, which calls _build.
        self._decimals = self._default_decimals if decimals is None else decimals
        super().__init__(visual_id, initial_value=initial_value, parent=parent)

    def _build(self, initial_value: Any, parent):
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleSlider

        slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent)
        slider.setRange(*self._default_range)
        slider.setValue(float(initial_value))
        slider.setDecimals(self._decimals)
        slider.valueChanged.connect(self._emit)
        return slider

    def _read(self) -> float:
        return float(self._control.value())

    def _apply(self, value: Any) -> None:
        self._control.blockSignals(True)
        self._control.setValue(float(value))
        self._control.blockSignals(False)


class QtFloatSpin(QtAppearanceField):
    """Layer 2: a spin box for an **unbounded** float field.

    Only ``opacity`` carries ``ge``/``le``; every other float on the seven
    non-image appearance models is unconstrained, so the range here is a
    *widget* bound someone picked rather than a model constraint.  Layer 3
    binds a sensible default per field (design section 6.5.1 proposal 3) and
    every one is overridable at construction -- which is the escape hatch for
    the four ``*_space`` fields whose ``"world"`` setting makes screen-pixel
    bounds meaningless.
    """

    _default_value: ClassVar[Any] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 100.0)
    _default_step: ClassVar[float] = 0.1
    _default_decimals: ClassVar[int] = 2

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        value_range: tuple[float, float] | None = None,
        step: float | None = None,
        decimals: int | None = None,
        parent=None,
    ) -> None:
        self._range = self._default_range if value_range is None else value_range
        self._step = self._default_step if step is None else step
        self._decimals = self._default_decimals if decimals is None else decimals
        super().__init__(visual_id, initial_value=initial_value, parent=parent)

    def _build(self, initial_value: Any, parent):
        from qtpy.QtWidgets import QDoubleSpinBox

        spin = QDoubleSpinBox(parent)
        spin.setRange(*self._range)
        spin.setSingleStep(self._step)
        spin.setDecimals(self._decimals)
        spin.setValue(float(initial_value))
        spin.valueChanged.connect(self._emit)
        return spin

    def _read(self) -> float:
        return float(self._control.value())

    def _apply(self, value: Any) -> None:
        self._control.blockSignals(True)
        self._control.setValue(float(value))
        self._control.blockSignals(False)


class QtIntSpin(QtAppearanceField):
    """Layer 2: a spin box for an integer field.

    ``_shuffle`` adds a button writing a random value in range, for fields
    where the *number* is meaningless and only the resulting difference
    matters -- the labels colormap ``salt`` (design section 10.4).
    """

    _default_value: ClassVar[Any] = 0
    _default_range: ClassVar[tuple[int, int]] = (0, 2**31 - 1)
    _default_step: ClassVar[int] = 1
    _shuffle: ClassVar[bool] = False

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        value_range: tuple[int, int] | None = None,
        step: int | None = None,
        parent=None,
    ) -> None:
        self._range = self._default_range if value_range is None else value_range
        self._step = self._default_step if step is None else step
        super().__init__(visual_id, initial_value=initial_value, parent=parent)

    def _build(self, initial_value: Any, parent):
        from qtpy.QtWidgets import QHBoxLayout, QPushButton, QSpinBox, QWidget

        container = QWidget(parent)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        self._spin = QSpinBox(container)
        self._spin.setRange(*self._range)
        self._spin.setSingleStep(self._step)
        self._spin.setValue(int(initial_value))
        self._spin.valueChanged.connect(self._emit)
        row.addWidget(self._spin)

        if self._shuffle:
            button = QPushButton("Shuffle", container)
            button.clicked.connect(self._on_shuffle)
            row.addWidget(button)

        return container

    def _on_shuffle(self) -> None:
        import random

        self._spin.setValue(random.randint(*self._range))

    def _read(self) -> int:
        return int(self._spin.value())

    def _apply(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(int(value))
        self._spin.blockSignals(False)


class QtChoice(QtAppearanceField):
    """Layer 2: a combo box for a ``Literal`` field.

    Options come from the model's own annotation rather than a list restated
    here, so a mode added to a model appears automatically and the in-memory
    and multiscale variants of a field need no separate lists.  Pass
    ``choices=`` when constructing from a model instance
    (``literal_choices(appearance, field)``); ``_default_choices`` is the
    fallback for a widget built without one.
    """

    _default_value: ClassVar[Any] = ""
    _default_choices: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        choices: Sequence[str] | None = None,
        parent=None,
    ) -> None:
        self._choices = tuple(self._default_choices if choices is None else choices)
        super().__init__(visual_id, initial_value=initial_value, parent=parent)

    @property
    def choices(self) -> tuple[str, ...]:
        """The options offered, in model order."""
        return self._choices

    def _build(self, initial_value: Any, parent):
        from qtpy.QtWidgets import QComboBox

        combo = QComboBox(parent)
        combo.addItems(list(self._choices))
        index = combo.findText(str(initial_value))
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentTextChanged.connect(self._emit)
        return combo

    def _read(self) -> str:
        return str(self._control.currentText())

    def _apply(self, value: Any) -> None:
        index = self._control.findText(str(value))
        if index < 0:
            return  # a value this combo does not offer; leave the control alone
        self._control.blockSignals(True)
        self._control.setCurrentIndex(index)
        self._control.blockSignals(False)


class QtColorPicker(QtAppearanceField):
    """Layer 2: an RGB swatch plus a separate alpha slider.

    Every appearance ``color`` field is float RGBA, and both toolkits' colour
    inputs are RGB-only, so alpha needs its own control and the float-RGBA to
    hex conversion has to happen somewhere.  Written once here rather than per
    field class -- which is what section 10.3 warns is the cost this control
    type understates.
    """

    _default_value: ClassVar[Any] = (1.0, 1.0, 1.0, 1.0)

    def _build(self, initial_value: Any, parent):
        from qtpy.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
        from superqt import QLabeledDoubleSlider

        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        self._rgba = as_rgba(initial_value)

        container = QWidget(parent)
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)

        color_row = QWidget(container)
        row = QHBoxLayout(color_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._swatch = QWidget(color_row)
        self._swatch.setFixedSize(24, 24)
        row.addWidget(self._swatch)
        self._button = QPushButton("Choose...", color_row)
        self._button.clicked.connect(self._on_choose)
        row.addWidget(self._button)
        row.addStretch()
        column.addWidget(color_row)

        alpha_row = QWidget(container)
        alpha_layout = QHBoxLayout(alpha_row)
        alpha_layout.setContentsMargins(0, 0, 0, 0)
        alpha_layout.addWidget(QLabel("Alpha:", alpha_row))
        from qtpy.QtCore import Qt

        self._alpha = QLabeledDoubleSlider(Qt.Orientation.Horizontal, alpha_row)
        self._alpha.setRange(0.0, 1.0)
        self._alpha.setDecimals(2)
        self._alpha.setValue(self._rgba[3])
        self._alpha.valueChanged.connect(self._on_alpha_changed)
        alpha_layout.addWidget(self._alpha)
        column.addWidget(alpha_row)

        self._paint_swatch(rgba_to_hex(self._rgba))
        return container

    def _paint_swatch(self, hex_color: str) -> None:
        self._swatch.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #888;"
        )

    def _on_choose(self) -> None:
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        from cellier.gui._appearance_fields import hex_to_rgba, rgba_to_hex

        chosen = QColorDialog.getColor(
            QColor(rgba_to_hex(self._rgba)), self._control, "Choose color"
        )
        if not chosen.isValid():
            return  # user cancelled
        self._rgba = hex_to_rgba(chosen.name(), self._rgba[3])
        self._paint_swatch(chosen.name())
        self._emit(self._rgba)

    def _on_alpha_changed(self, value: float) -> None:
        r, g, b, _a = self._rgba
        self._rgba = (r, g, b, float(value))
        self._emit(self._rgba)

    def _read(self) -> tuple[float, float, float, float]:
        return self._rgba

    def _apply(self, value: Any) -> None:
        from cellier.gui._appearance_fields import as_rgba, rgba_to_hex

        self._rgba = as_rgba(value)
        self._alpha.blockSignals(True)
        self._alpha.setValue(self._rgba[3])
        self._alpha.blockSignals(False)
        self._paint_swatch(rgba_to_hex(self._rgba))
