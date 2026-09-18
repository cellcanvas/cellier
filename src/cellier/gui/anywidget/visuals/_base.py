"""Shared base for the single-field anywidget appearance widgets.

The anywidget half of ``cellier.gui.qt.visuals._base``; see that module for the
three-layer design.  Two things are specific to this side:

* the ``value`` trait is declared by **layer 2**, because its traitlets type is
  what makes a control type a control type;
* ``label`` is a synced trait so one ``.js`` asset can serve every field class
  of a control type -- the JS reads ``model.get("value")`` and
  ``model.get("label")`` and never a field-specific trait name.

Layer 2 also declares ``_esm`` / ``_css``.  ``AnyWidget.__init_subclass__``
only coerces those for the class that declares them in its own ``__dict__``,
so every layer-3 subclass shares the one ``FileContents`` -- pinned by
``test_esm_path_is_shared_by_subclasses_and_read_as_source``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import SubscriptionSpec
from cellier.gui._appearance_fields import (
    NO_MATCH,
    AppearanceFieldSpec,
    appearance_field_spec,
    normalize_visual_ids,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class _UnsetType:
    """Sentinel distinguishing "no initial value given" from ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


_UNSET = _UnsetType()


class AnywidgetAppearanceField(anywidget.AnyWidget):
    """Bidirectional control for **one** appearance field on one or more visuals.

    Mirrors ``QtAppearanceField``: same constructor keyword, same
    ``visual_id`` fan-out, same echo filtering, same outgoing event.

    Wire to the controller after construction::

        toggle = AnywidgetVisibleToggle(visual_id, initial_value=True)
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
    """

    _field: ClassVar[str]
    _label: ClassVar[str]
    _default_value: ClassVar[Any] = None

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    label = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        **kwargs,
    ) -> None:
        value = self._default_value if initial_value is _UNSET else initial_value
        super().__init__(value=self._coerce(value), label=self._label, **kwargs)

        self._id = uuid4()
        self._visual_ids = normalize_visual_ids(visual_id)
        self._spec = self._make_spec()
        self._applying = False
        self.observe(self._on_trait_change, names="value")

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def widget(self) -> AnywidgetAppearanceField:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def control(self) -> AnywidgetAppearanceField:
        """The bare input, under the name the Qt twin uses.

        On Qt the control sits inside a labelled row, so ``widget`` and
        ``control`` are different objects.  Here the widget *is* the control --
        the input lives in JavaScript -- so both return ``self``.  Providing
        the name anyway is what lets a caller reach for the input without
        asking which toolkit it has.
        """
        return self

    @property
    def visual_ids(self) -> tuple[UUID, ...]:
        """The visual ids this widget drives, always as a tuple."""
        return self._visual_ids

    @property
    def field(self) -> str:
        """The appearance field name this widget writes."""
        return self._spec.name

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
        """Return one inbound subscription per visual id (subscribe-to-all)."""
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

    def _apply(self, value: Any) -> None:
        """Push *value* into the trait without emitting an update event."""
        self._applying = True
        try:
            self.value = self._coerce(value)
        finally:
            self._applying = False

    # ── widget -> model ──────────────────────────────────────────────────────

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return
        self._emit(change["new"])

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

    @staticmethod
    def _coerce(value: Any) -> Any:
        """Cast *value* to the trait's type.

        The bus carries whatever the model holds, which is not always exactly
        the trait type (a numpy scalar, an int for a float trait).  Layer 2
        knows the type, so it does the cast.
        """
        return value


class AnywidgetToggle(AnywidgetAppearanceField):
    """Layer 2: a checkbox for a boolean appearance field."""

    _esm = _STATIC / "toggle.js"
    _css = _STATIC / "toggle.css"

    _default_value: ClassVar[Any] = False

    value = traitlets.Bool(False).tag(sync=True)

    @staticmethod
    def _coerce(value: Any) -> bool:
        return bool(value)


class AnywidgetBoundedSlider(AnywidgetAppearanceField):
    """Layer 2: a slider for a float field the **model** bounds.

    Mirrors ``QtBoundedSlider``.  ``min``/``max`` are synced traits so the one
    ``bounded_slider.js`` asset serves every field class of this type without
    knowing any field's name.

    Its one documented exception carries over too:
    ``AnywidgetIsoThresholdSlider``'s range is a convention rather than a model
    constraint.
    """

    _esm = _STATIC / "bounded_slider.js"
    _css = _STATIC / "bounded_slider.css"

    _default_value: ClassVar[Any] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 1.0)
    _default_step: ClassVar[float] = 0.01

    value = traitlets.Float(1.0).tag(sync=True)
    min = traitlets.Float(0.0).tag(sync=True)
    max = traitlets.Float(1.0).tag(sync=True)
    step = traitlets.Float(0.01).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        **kwargs,
    ) -> None:
        low, high = self._default_range
        kwargs.setdefault("min", low)
        kwargs.setdefault("max", high)
        kwargs.setdefault("step", self._default_step)
        super().__init__(visual_id, initial_value=initial_value, **kwargs)

    @staticmethod
    def _coerce(value: Any) -> float:
        return float(value)


class AnywidgetFloatSpin(AnywidgetAppearanceField):
    """Layer 2: a number input for an **unbounded** float field.

    Mirrors ``QtFloatSpin``, including that its range is a widget bound
    someone picked rather than a model constraint, and is overridable.
    """

    _esm = _STATIC / "float_spin.js"
    _css = _STATIC / "float_spin.css"

    _default_value: ClassVar[Any] = 1.0
    _default_range: ClassVar[tuple[float, float]] = (0.0, 100.0)
    _default_step: ClassVar[float] = 0.1

    value = traitlets.Float(1.0).tag(sync=True)
    min = traitlets.Float(0.0).tag(sync=True)
    max = traitlets.Float(100.0).tag(sync=True)
    step = traitlets.Float(0.1).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        value_range: tuple[float, float] | None = None,
        step: float | None = None,
        **kwargs,
    ) -> None:
        low, high = self._default_range if value_range is None else value_range
        kwargs.setdefault("min", float(low))
        kwargs.setdefault("max", float(high))
        kwargs.setdefault("step", self._default_step if step is None else float(step))
        super().__init__(visual_id, initial_value=initial_value, **kwargs)

    @staticmethod
    def _coerce(value: Any) -> float:
        return float(value)


class AnywidgetIntSpin(AnywidgetAppearanceField):
    """Layer 2: a number input for an integer field, optionally with shuffle.

    ``shuffle`` is a synced bool rather than a separate control type, so the
    one asset still serves every integer field: ``int_spin.js`` renders the
    button only when the trait is set.
    """

    _esm = _STATIC / "int_spin.js"
    _css = _STATIC / "int_spin.css"

    _default_value: ClassVar[Any] = 0
    _default_range: ClassVar[tuple[int, int]] = (0, 2**31 - 1)
    _default_step: ClassVar[int] = 1
    _shuffle: ClassVar[bool] = False

    value = traitlets.Int(0).tag(sync=True)
    min = traitlets.Int(0).tag(sync=True)
    max = traitlets.Int(2**31 - 1).tag(sync=True)
    step = traitlets.Int(1).tag(sync=True)
    shuffle = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        value_range: tuple[int, int] | None = None,
        step: int | None = None,
        **kwargs,
    ) -> None:
        low, high = self._default_range if value_range is None else value_range
        kwargs.setdefault("min", int(low))
        kwargs.setdefault("max", int(high))
        kwargs.setdefault("step", self._default_step if step is None else int(step))
        kwargs.setdefault("shuffle", self._shuffle)
        super().__init__(visual_id, initial_value=initial_value, **kwargs)

    @staticmethod
    def _coerce(value: Any) -> int:
        return int(value)


class AnywidgetChoice(AnywidgetAppearanceField):
    """Layer 2: a select for a ``Literal`` field.

    ``choices`` is a synced list, filled from the model's own annotation, so
    ``choice.js`` builds the options without knowing which field it serves.
    """

    _esm = _STATIC / "choice.js"
    _css = _STATIC / "choice.css"

    _default_value: ClassVar[Any] = ""
    _default_choices: ClassVar[tuple[str, ...]] = ()

    value = traitlets.Unicode("").tag(sync=True)
    choices = traitlets.List([]).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial_value: Any = _UNSET,
        choices: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault(
            "choices",
            list(self._default_choices if choices is None else choices),
        )
        super().__init__(visual_id, initial_value=initial_value, **kwargs)

    @staticmethod
    def _coerce(value: Any) -> str:
        return str(value)


class AnywidgetColorPicker(AnywidgetAppearanceField):
    """Layer 2: an RGB colour input plus a separate alpha slider.

    Mirrors ``QtColorPicker``.  The trait is the full float RGBA the model
    holds; the hex conversion the ``<input type=color>`` needs happens in the
    JS, from that one trait, so the Python side never speaks hex.
    """

    _esm = _STATIC / "color_picker.js"
    _css = _STATIC / "color_picker.css"

    _default_value: ClassVar[Any] = (1.0, 1.0, 1.0, 1.0)

    value = traitlets.List(traitlets.Float(), default_value=[1.0, 1.0, 1.0, 1.0]).tag(
        sync=True
    )

    @staticmethod
    def _coerce(value: Any) -> list:
        from cellier.gui._appearance_fields import as_rgba

        return list(as_rgba(value))
