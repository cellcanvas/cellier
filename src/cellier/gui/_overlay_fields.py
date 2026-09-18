"""Toolkit-neutral description of the overlay controls.

The overlay widgets reuse the appearance widgets' layer-2 controls -- toggle,
spin box, combo, colour picker -- unchanged, and differ only in how a change
travels: an overlay field is heard from on ``OverlayChangedEvent`` and written
with ``OverlayUpdateEvent``, keyed by the overlay's id rather than a visual's.
:class:`OverlayFieldSpec` says exactly that, and :class:`OverlayFieldMixin`
swaps it in, so a layer-3 overlay class is a mixin, a layer-2 base, a field
path and a label.

Field names are **dotted paths** on the overlay model: ``"visible"`` for the
overlay's own flag, ``"appearance.color"`` for an appearance field.  The
controller resolves the same path in ``update_overlay_field``.

The tables here say which controls each overlay type gets and which class
serves each one, for the layout dock (``OverlayControls``) and for anyone
assembling a panel by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from cellier.events import OverlayChangedEvent, OverlayUpdateEvent
from cellier.gui._appearance_fields import NO_MATCH

if TYPE_CHECKING:
    from uuid import UUID


@dataclass(frozen=True)
class OverlayFieldSpec:
    """How one overlay field is named, labelled, heard from and written.

    Parameters
    ----------
    name : str
        Dotted path on the overlay model, e.g. ``"appearance.color"``.  Also
        the ``field`` stamped on the outgoing ``OverlayUpdateEvent``.
    label : str
        Human-readable label for the control.
    """

    name: str
    label: str

    inbound_event_type: ClassVar[type] = OverlayChangedEvent

    def inbound_value(self, event: Any) -> Any:
        """Return the value *event* carries for this field, or ``NO_MATCH``.

        A wholesale ``appearance`` replacement carries the new model, so an
        appearance field reads its own value off it.
        """
        field_name = getattr(event, "field_name", None)
        if field_name == self.name:
            return event.new_value
        prefix, _, attribute = self.name.partition(".")
        if field_name == prefix and attribute:
            return getattr(event.new_value, attribute, NO_MATCH)
        return NO_MATCH

    def outbound_event(self, source_id: UUID, target_id: UUID, value: Any) -> Any:
        """Build the ``OverlayUpdateEvent`` asking the controller for *value*."""
        return OverlayUpdateEvent(
            source_id=source_id, overlay_id=target_id, field=self.name, value=value
        )


class OverlayFieldMixin:
    """Mixin: make a layer-2 appearance control drive an overlay field.

    Mix in **before** the layer-2 base so its ``_make_spec`` wins::

        class QtOverlayVisibleToggle(OverlayFieldMixin, QtToggle):
            _field = "visible"
            _label = "Visible"

    The constructor's first argument is then an overlay id (or a sequence of
    them) rather than a visual id.
    """

    _field: ClassVar[str]
    _label: ClassVar[str]
    _visual_ids: tuple[UUID, ...]

    @classmethod
    def _make_spec(cls) -> OverlayFieldSpec:
        """Return the overlay spec for this class's field."""
        return OverlayFieldSpec(name=cls._field, label=cls._label)

    @property
    def overlay_ids(self) -> tuple[UUID, ...]:
        """The overlay ids this widget drives, always as a tuple."""
        return self._visual_ids


OVERLAY_FIELD_WIDGETS: dict[str, tuple[str, str]] = {
    # Every overlay.
    "visible": ("OverlayVisibleToggle", "Visible"),
    # SceneBoundingBox.
    "appearance.color": ("OverlayColorPicker", "Color"),
    "appearance.thickness": ("OverlayThicknessSpin", "Thickness"),
    # CenteredAxes2D.
    "appearance.corner": ("OverlayCornerCombo", "Corner"),
    "appearance.length_px": ("OverlayLengthSpin", "Length"),
    "appearance.line_thickness_px": ("OverlayLineThicknessSpin", "Thickness"),
    "appearance.axis_a_color": ("OverlayAxisAColorPicker", "Axis A color"),
    "appearance.axis_b_color": ("OverlayAxisBColorPicker", "Axis B color"),
    "appearance.show_labels": ("OverlayShowLabelsToggle", "Labels"),
    "appearance.label_color": ("OverlayLabelColorPicker", "Label color"),
    "appearance.font_size_px": ("OverlayFontSizeSpin", "Font size"),
}
"""Field path -> (class stem, title) for every overlay control.

Both toolkits resolve a widget by prefixing the stem with ``Qt`` or
``Anywidget``.  Keyed by field path rather than by overlay type: two overlay
types spelling a field alike share its control.
"""

OVERLAY_CONTROLS: dict[str, tuple[str, ...]] = {
    "scene_bounding_box": (
        "visible",
        "appearance.color",
        "appearance.thickness",
    ),
    "centered_axes_2d": (
        "visible",
        "appearance.corner",
        "appearance.length_px",
        "appearance.line_thickness_px",
        "appearance.axis_a_color",
        "appearance.axis_b_color",
        "appearance.show_labels",
        "appearance.label_color",
        "appearance.font_size_px",
    ),
}
"""Overlay type (its ``overlay_type`` discriminator) -> its controls, in order."""


def overlay_field_value(overlay: Any, field: str) -> Any:
    """Read the dotted *field* path off *overlay*."""
    value = overlay
    for part in field.split("."):
        value = getattr(value, part)
    return value


def overlay_field_widget_class(field: str, toolkit: str) -> type:
    """Return the class serving overlay *field* on *toolkit* (``qt``/``anywidget``).

    Raises
    ------
    KeyError
        For a field with no overlay control.
    """
    stem, _title = OVERLAY_FIELD_WIDGETS[field]
    if toolkit == "qt":
        import cellier.gui.qt.overlays as module

        return getattr(module, f"Qt{stem}")
    import cellier.gui.anywidget.overlays as module

    return getattr(module, f"Anywidget{stem}")
