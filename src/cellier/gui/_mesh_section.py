"""The "2D section" controls of a mesh, decided once for both toolkits.

A mesh in a 2D view is drawn as its cross-section
(:class:`~cellier.visuals.MeshSectionConfig`): an outline and a fill, cut at
the slice plane or through the scene's slab.  The Qt and anywidget controls
(``QtMeshSectionControls``, ``AnywidgetMeshSectionControls``) draw one row
per field; this module says which rows, and carries edits to the bus and
model changes back.

Nothing here imports a toolkit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from cellier.events import (
    DimsChangedEvent,
    MeshSectionChangedEvent,
    MeshSectionUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._image_controls import displayed_dimensions
from cellier.gui._loading import LoadingConfigField, error_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from uuid import UUID

MESH_SECTION_TITLE = "2D section"
"""The control's name: its ``DEFAULT_TITLE`` on both toolkits."""

#: The row label of each ``MeshSectionConfig`` field, in display order.
_MESH_SECTION_LABELS: dict[str, str] = {
    "outline": "Outline",
    "fill": "Fill",
    "outline_width": "Outline width",
    "mode": "Mode",
}

#: What each row says when the pointer rests on it.
_MESH_SECTION_TOOLTIPS: dict[str, str] = {
    "outline": "Draw the curve where the surface crosses the plane.",
    "fill": (
        "Fill the area the outline encloses. An open surface has no fill "
        "where its cut does not close."
    ),
    "outline_width": "Outline thickness, in screen pixels.",
    "mode": (
        "cut: draw the one plane at the slider's position; the scene's "
        "thickness on a spatial axis is ignored. "
        "slab: draw what lies inside the scene's slab, flattened."
    ),
}

#: The outline width's range and step, in screen pixels.
OUTLINE_WIDTH_RANGE: tuple[float, float, float] = (0.5, 20.0, 0.5)


def mesh_section_fields() -> list[LoadingConfigField]:
    """Describe every ``MeshSectionConfig`` field, read off the model.

    Choices come from the ``Literal`` annotations, so the control cannot
    drift from the config.

    Returns
    -------
    list[LoadingConfigField]
        In display order.  A ``"bool"`` is a checkbox, a ``"choice"`` a
        drop-down, and ``"fraction"`` a number.
    """
    from cellier.visuals import MeshSectionConfig

    fields = []
    for name, label in _MESH_SECTION_LABELS.items():
        annotation = MeshSectionConfig.model_fields[name].annotation
        if annotation is bool:
            field = LoadingConfigField(name, label, "bool")
        elif get_origin(annotation) is Literal:
            choices = tuple(str(choice) for choice in get_args(annotation))
            field = LoadingConfigField(name, label, "choice", choices)
        else:
            low, high, step = OUTLINE_WIDTH_RANGE
            field = LoadingConfigField(name, label, "fraction", (), low, high, step)
        fields.append(field._replace(tooltip=_MESH_SECTION_TOOLTIPS[name]))
    return fields


def section_display_seed(
    controller: Any, visual_ids: Iterable[UUID]
) -> dict[UUID, int]:
    """How many dimensions each scene of *visual_ids* displays now.

    ``DimsChangedEvent`` fires only on a change, so a section control is
    seeded with this at construction and then follows the events.

    Parameters
    ----------
    controller : CellierController or None
        The controller the control is wired to.  ``None`` gives no scenes.
    visual_ids : iterable of UUID
        The mesh visuals the control drives.

    Returns
    -------
    dict[UUID, int]
        Scene id to its number of displayed dimensions.  A scene whose dims
        do not say is left out.
    """
    displayed: dict[UUID, int] = {}
    if controller is None:
        return displayed
    for visual_id in visual_ids:
        try:
            scene_id = controller.get_visual_scene_id(visual_id)
        except KeyError:
            continue
        n = displayed_dimensions(controller.get_scene(scene_id).dims)
        if n is not None:
            displayed[scene_id] = n
    return displayed


class SectionVisibility:
    """Whether a mesh section control is shown: only while a scene is 2D.

    The section settings do nothing in a 3D view, so the control hides when
    none of its scenes displays two dimensions and comes back when one does
    (an ``OrthoViewer`` group spans 2D and 3D scenes, and stays shown).  A
    control that follows no scene is always shown.

    Parameters
    ----------
    displayed : Mapping[UUID, int]
        Scene id to its number of displayed dimensions, from
        :func:`section_display_seed`.
    apply : Callable[[bool], None]
        Shows or hides the widget.
    """

    def __init__(
        self, displayed: Mapping[UUID, int] | None, apply: Callable[[bool], None]
    ) -> None:
        self._displayed: dict[UUID, int] = dict(displayed or {})
        self._apply = apply

    @property
    def visible(self) -> bool:
        """Whether the control is shown now."""
        return not self._displayed or 2 in self._displayed.values()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``DimsChangedEvent`` subscription per followed scene."""
        return [
            SubscriptionSpec(DimsChangedEvent, self.on_dims_changed, entity_id=sid)
            for sid in self._displayed
        ]

    def on_dims_changed(self, event: DimsChangedEvent) -> None:
        """Follow a scene's switch between 2D and 3D display."""
        if not event.displayed_axes_changed:
            return
        n = displayed_dimensions(event.dims_state)
        if n is None or event.scene_id not in self._displayed:
            return
        self._displayed[event.scene_id] = n
        self._apply(self.visible)


def _from_control_value(field: str, value: Any) -> Any:
    if field in ("outline", "fill"):
        return bool(value)
    if field == "outline_width":
        return float(value)
    return value


class MeshSectionEditor:
    """The toolkit-neutral half of a mesh section control.

    Holds the config the model last reported, sends edits, and tells the
    widget what to show.  An edit is one ``MeshSectionUpdateEvent`` per
    visual.  If the controller refuses it, the widget is told to show the
    last config again, with the error.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The mesh visuals edited together (an ``OrthoViewer`` panel group).
    section : Mapping[str, Any]
        Their current config, as ``MeshSectionConfig.model_dump()``.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[MeshSectionUpdateEvent], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[dict[str, Any], str], None]
        Draws a config and an error message ("" for none).
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        section: Mapping[str, Any],
        source_id: UUID,
        emit: Callable[[MeshSectionUpdateEvent], None],
        show: Callable[[dict[str, Any], str], None],
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self.config: dict[str, Any] = dict(section)
        self._source_id = source_id
        self._emit = emit
        self._show = show

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``MeshSectionChangedEvent`` subscription per visual."""
        return [
            SubscriptionSpec(MeshSectionChangedEvent, self.on_changed, entity_id=vid)
            for vid in self._visual_ids
        ]

    def edit(self, field: str, control_value: Any) -> None:
        """Send an edit of *field*; on refusal, show the last config and why."""
        value = _from_control_value(field, control_value)
        if self.config.get(field) == value:
            return
        try:
            for visual_id in self._visual_ids:
                self._emit(
                    MeshSectionUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        field=field,
                        value=value,
                    )
                )
        except Exception as error:
            self._show(dict(self.config), error_message(error))

    def on_changed(self, event: MeshSectionChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        self.config[event.field_name] = event.new_value
        self._show(dict(self.config), "")
