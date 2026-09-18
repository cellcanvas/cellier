"""Toolkit-neutral description of a single appearance field.

Both toolkits' appearance-field widgets (``cellier.gui.qt.visuals._base`` and
``cellier.gui.anywidget.visuals._base``) need the same three answers about the
field they drive: what it is called, what to label it, and which bus event
carries a change to it back from the model.  The first two are trivial; the
third is not uniform, which is the whole reason this module exists.

``Controller._make_appearance_handler`` routes ``visible`` to
``VisualVisibilityChangedEvent`` (payload ``visible``) and every other
appearance field to ``AppearanceChangedEvent`` (payload ``field_name`` /
``new_value``).  A widget base that assumed one event type would work for 23 of
the 24 fields and silently never update for ``visible``.  Keeping the exception
here means the per-field widget classes stay a few lines each and neither
toolkit has to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, get_args

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    VisualVisibilityChangedEvent,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


class _NoMatch:
    """Sentinel: an inbound event that this field should ignore."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NO_MATCH"


NO_MATCH = _NoMatch()


@dataclass(frozen=True)
class AppearanceFieldSpec:
    """How one appearance field is named, labelled, and heard from.

    Parameters
    ----------
    name : str
        Attribute name on the visual's appearance model, e.g. ``"wireframe"``.
        This is also the ``field`` stamped on the outgoing
        ``AppearanceUpdateEvent``.
    label : str
        Human-readable label for the control.
    """

    name: str
    label: str

    inbound_event_type: ClassVar[type] = AppearanceChangedEvent

    def inbound_value(self, event: Any) -> Any:
        """Return the new value carried by *event*, or ``NO_MATCH``.

        ``NO_MATCH`` means the event concerns a different field and the widget
        should ignore it.
        """
        if getattr(event, "field_name", None) != self.name:
            return NO_MATCH
        return event.new_value

    def outbound_event(self, source_id: UUID, target_id: UUID, value: Any) -> Any:
        """Build the update event that asks the controller for *value*.

        The spec builds it, not the widget base, so a control type can drive
        something other than a visual appearance -- an overlay field, say --
        by swapping the spec alone (``cellier.gui._overlay_fields``).

        Parameters
        ----------
        source_id : UUID
            The widget's id, stamped for echo filtering.
        target_id : UUID
            The visual the change is for.
        value : Any
            The new value.
        """
        return AppearanceUpdateEvent(
            source_id=source_id, visual_id=target_id, field=self.name, value=value
        )


@dataclass(frozen=True)
class VisibleFieldSpec(AppearanceFieldSpec):
    """``visible``, which travels on its own event with its own payload."""

    inbound_event_type: ClassVar[type] = VisualVisibilityChangedEvent

    def inbound_value(self, event: Any) -> Any:
        """Return ``event.visible``.

        The event type is field-specific, so an event of this type always
        concerns this field -- there is nothing to match on.
        """
        return event.visible


_SPECIAL_FIELD_SPECS: dict[str, type[AppearanceFieldSpec]] = {
    "visible": VisibleFieldSpec,
}


def appearance_field_spec(name: str, label: str) -> AppearanceFieldSpec:
    """Return the spec for the appearance field *name*.

    Fields whose changes travel on a non-standard bus event get a dedicated
    subclass; everything else gets the plain ``AppearanceChangedEvent`` spec.
    Widget classes call this rather than choosing a spec themselves, so a field
    that turns out to be special later needs no change in either toolkit.
    """
    return _SPECIAL_FIELD_SPECS.get(name, AppearanceFieldSpec)(name=name, label=label)


def normalize_visual_ids(visual_id: UUID | Sequence[UUID]) -> tuple[UUID, ...]:
    """Normalise a single visual id or a sequence of them to a tuple.

    An appearance widget drives one visual on a ``Viewer`` and the whole
    sibling group on an ``OrthoViewer``; accepting either shape keeps the
    single-visual call sites unchanged.
    """
    if isinstance(visual_id, (list, tuple)):
        ids = tuple(visual_id)
        if not ids:
            raise ValueError("visual_id sequence must not be empty")
        return ids
    return (visual_id,)


class VisualIdGroup:
    """Mixin: drive one visual or a whole sibling group in lock-step.

    A ``Viewer`` gives a control one visual; an ``OrthoViewer`` gives it the
    four panel visuals that share a data store, and the control must keep them
    equal.  the per-channel list established the shape (design section 8.1) and
    this generalises it to the appearance and AABB widgets:

    1. ``visual_id`` accepts a ``UUID`` or a sequence of them;
    2. one ``SubscriptionSpec`` per id -- subscribe-to-*all*, so the control
       stays correct when a sibling is written by something other than itself.
       Subscribing only to the first would make "panel 0 represents the group"
       a load-bearing invariant with no way to self-heal;
    3. one update event per id on every edit.

    A user edit therefore produces N echoes rather than one.  They are dropped
    by the ``source_id`` filter, and any that survive apply idempotently --
    pinned for appearance and for AABB by
    ``tests/convenience/test_cleanup_assumptions.py``.

    Mix in **before** the toolkit base so the plain-``object`` MRO stays first::

        class AnywidgetLodBiasSlider(VisualIdGroup, anywidget.AnyWidget): ...
    """

    _visual_ids: tuple[UUID, ...]

    def _init_visual_ids(self, visual_id: UUID | Sequence[UUID]) -> None:
        """Normalise and store the driven ids.  Call from ``__init__``."""
        self._visual_ids = normalize_visual_ids(visual_id)

    @property
    def visual_ids(self) -> tuple[UUID, ...]:
        """The visual ids this widget drives, always as a tuple."""
        return self._visual_ids

    @property
    def _visual_id(self) -> UUID:
        """The first driven id.

        Kept so single-visual call sites and tests written before the group
        form read the same attribute they always did.
        """
        return self._visual_ids[0]

    def _group_specs(self, event_type: type, handler) -> list:
        """One inbound subscription per driven id."""
        from cellier.events import SubscriptionSpec

        return [
            SubscriptionSpec(
                event_type=event_type, handler=handler, entity_id=visual_id
            )
            for visual_id in self._visual_ids
        ]

    def _emit_group(self, event_cls: type, field: str, value: Any) -> None:
        """Emit one update event per driven id on ``self.changed``."""
        for visual_id in self._visual_ids:
            self.changed.emit(
                event_cls(
                    source_id=self._id,
                    visual_id=visual_id,
                    field=field,
                    value=value,
                )
            )


# ── Reading a control's shape off the model ─────────────────────────────────
#
# Three things a control needs that the *model* already states: which values a
# ``Literal`` field admits, what bounds a ``ge``/``le`` field carries, and how
# an RGBA colour maps to the hex string every colour input speaks.  Deriving
# them rather than restating them is what keeps a control from drifting away
# from the field it drives -- adding a render mode to a model, or tightening a
# bound, needs no edit here (design section 6.5.1 proposals 3 and 4).

DEBUG_FIELD_VALUES: dict[str, frozenset[str]] = {
    "render_mode": frozenset({"gradient_debug"}),
}
"""Literal values that exist for developers and must not reach a control.

``gradient_debug`` renders ``abs(normalize(gradient))`` as RGB to diagnose
brick-boundary normal discontinuities -- a diagnostic, not a rendering choice,
and a user who picks it sees a broken-looking image (design section 6.5.1
proposal 4).  Denylisting rather than allowlisting keeps the combo in sync
with the model automatically: a new *real* mode appears with no change here.
"""


def literal_choices(appearance: Any, field: str) -> tuple[str, ...]:
    """Return the values *field* admits on *appearance*'s model, user-facing only.

    Reads the field's ``Literal`` annotation, so the in-memory and multiscale
    variants of a field need no separate lists -- each model answers for
    itself.  Debug-only values are dropped per :data:`DEBUG_FIELD_VALUES`.

    Returns an empty tuple when the field is not a ``Literal``.
    """
    model_fields = getattr(type(appearance), "model_fields", None)
    if not model_fields or field not in model_fields:
        return ()
    args = get_args(model_fields[field].annotation)
    denied = DEBUG_FIELD_VALUES.get(field, frozenset())
    return tuple(arg for arg in args if isinstance(arg, str) and arg not in denied)


def field_bounds(model_cls: type, field: str) -> tuple[float, float] | None:
    """Return *field*'s ``(ge, le)`` bounds on *model_cls*, or ``None``.

    A control built from these can only emit values the model already accepts,
    which is why ``BoundedSlider`` takes its range from here and deliberately
    offers no widening keyword: a wider range would build a control whose
    values pydantic then rejects.
    """
    model_fields = getattr(model_cls, "model_fields", None)
    if not model_fields or field not in model_fields:
        return None
    low = high = None
    for constraint in model_fields[field].metadata:
        low = getattr(constraint, "ge", None) if low is None else low
        high = getattr(constraint, "le", None) if high is None else high
    if low is None or high is None:
        return None
    return (float(low), float(high))


def rgba_to_hex(rgba: Sequence[float]) -> str:
    """Convert float RGBA in [0, 1] to a ``#rrggbb`` string.

    Alpha is **dropped**: colour inputs in both toolkits are RGB-only, so the
    fourth component travels on its own control (design section 10.3).
    """
    r, g, b = (max(0.0, min(1.0, float(c))) for c in tuple(rgba)[:3])
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def hex_to_rgba(
    hex_color: str, alpha: float = 1.0
) -> tuple[float, float, float, float]:
    """Convert ``#rrggbb`` plus a separate *alpha* to float RGBA in [0, 1]."""
    text = hex_color.lstrip("#")
    if len(text) == 3:  # "#abc" shorthand
        text = "".join(ch * 2 for ch in text)
    r, g, b = (int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, float(alpha))


def as_rgba(value: Any) -> tuple[float, float, float, float]:
    """Coerce a colour value to a 4-tuple of floats, filling alpha with 1.0.

    The bus carries whatever the model holds -- a tuple, a list, a numpy row --
    and a colour control needs exactly four floats.
    """
    values = [float(component) for component in value]
    while len(values) < 4:
        values.append(1.0)
    return (values[0], values[1], values[2], values[3])


APPEARANCE_FIELD_WIDGETS: dict[str, tuple[str, str]] = {
    # Universal -- both fields are on ``BaseAppearance``.
    "visible": ("VisibleToggle", "Visible"),
    "opacity": ("OpacitySlider", "Opacity"),
    # Labels.  The kind is ``labels_render_mode`` rather than ``render_mode``
    # because the image models spell that field ``mip``/``iso``/``minip``:
    # same name, different control, which is why a config class maps field to
    # kind rather than the kind being the field (design section 6.5.2
    # decision 6).
    "labels_render_mode": ("LabelsRenderModeCombo", "Render mode"),
    "salt": ("SaltSpin", "Salt"),
    "background_label": ("BackgroundLabelSpin", "Background label"),
    # Mesh, points and lines all spell the uniform colour ``color``.
    "color": ("UniformColorPicker", "Color"),
    # Mesh.
    "side": ("SideCombo", "Side"),
    "wireframe": ("WireframeToggle", "Wireframe"),
    "wireframe_thickness": ("WireframeThicknessSpin", "Wireframe thickness"),
    "shininess": ("ShininessSpin", "Shininess"),
    "flat_shading": ("FlatShadingToggle", "Flat shading"),
    # Points.
    "size": ("SizeSpin", "Size"),
    "size_space": ("SizeSpaceCombo", "Size space"),
    # Lines.
    "thickness": ("ThicknessSpin", "Thickness"),
    "thickness_space": ("ThicknessSpaceCombo", "Thickness space"),
    # Graph -- two mirrored groups.
    "node_visible": ("NodeVisibleToggle", "Nodes visible"),
    "node_color": ("NodeColorPicker", "Node color"),
    "node_size": ("NodeSizeSpin", "Node size"),
    "node_size_space": ("NodeSizeSpaceCombo", "Node size space"),
    "edge_visible": ("EdgeVisibleToggle", "Edges visible"),
    "edge_color": ("EdgeColorPicker", "Edge color"),
    "edge_thickness": ("EdgeThicknessSpin", "Edge thickness"),
    "edge_thickness_space": ("EdgeThicknessSpaceCombo", "Edge thickness space"),
}
"""Control kind -> (field-class stem, group title), for the single-field controls.

The one place a control kind is tied to the classes that serve it.  Both
renderers resolve a widget from this by prefixing the stem with ``Qt`` or
``Anywidget``, which is what lets 22 field classes reach the dock through one
generic builder instead of 22 dispatch entries per toolkit.

The title is here rather than derived from the field name because it is the
toolkit-neutral name for the control -- checked against each field class's
``_label``, which is what both front ends actually draw, by
``test_the_shared_widget_table_agrees_with_the_field_classes``.

Not listed: the multi-field image controls (``color_map``, ``clim``,
``render``, ``lod_bias``) and ``aabb``/``dataset_info``, which predate this
design and keep their bespoke builders.
"""


def field_widget_class(kind: str, toolkit: str) -> type:
    """Return the layer-3 class serving *kind* on *toolkit* (``qt``/``anywidget``).

    Raises ``KeyError`` for a kind with no single-field control, which is the
    renderer's cue to fall back to its bespoke builder table.
    """
    stem, _title = APPEARANCE_FIELD_WIDGETS[kind]
    if toolkit == "qt":
        import cellier.gui.qt.visuals as module

        return getattr(module, f"Qt{stem}")
    import cellier.gui.anywidget.visuals as module

    return getattr(module, f"Anywidget{stem}")
