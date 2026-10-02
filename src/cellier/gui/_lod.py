"""The "LOD" controls of a multiscale mesh, for both toolkits.

A multiscale mesh keeps two levels loaded, the finest and one coarse level,
and its :class:`~cellier.visuals.GeometryLodConfig` says which one a dims
scrub loads and draws and which one a moving camera draws.  The Qt and
anywidget controls (``QtLodConfigControls``, ``AnywidgetLodConfigControls``)
draw one row per setting; this module says which rows, and carries edits to
the bus and model changes back.

``coarse_level`` has no row: which coarse level is kept is fixed when the
visual is added.

Nothing here imports a toolkit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

from cellier.events import (
    LodConfigChangedEvent,
    LodConfigUpdateEvent,
    SubscriptionSpec,
)
from cellier.gui._loading import LoadingConfigField, error_message

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from uuid import UUID

LOD_CONFIG_TITLE = "LOD"
"""The control's name: its ``DEFAULT_TITLE`` on both toolkits."""

#: The row label of each ``GeometryLodConfig`` field shown, in display order.
_LOD_CONFIG_LABELS: dict[str, str] = {
    "dims_drag": "Dims drag loads",
    "dims_drag_draw": "Dims drag draws",
    "camera_motion": "Camera motion draws",
}


#: What each row says when the pointer rests on it.
_LOD_CONFIG_TOOLTIPS: dict[str, str] = {
    "dims_drag": (
        "What each step of a slider drag loads. coarse: the coarse level, "
        "and the finest once when the drag ends. full: both levels at "
        "every step."
    ),
    "dims_drag_draw": (
        "What is drawn, during a slider drag, of a mesh the drag does not "
        "change. Both levels stay loaded; nothing is read."
    ),
    "camera_motion": (
        "What a 3D view draws while its camera moves. Both levels stay "
        "loaded; nothing is read. A 2D view always draws the finest."
    ),
}


def lod_config_fields() -> list[LoadingConfigField]:
    """Describe every ``GeometryLodConfig`` field that has a row.

    Choices come from the ``Literal`` annotations, so the control cannot
    drift from the config.

    Returns
    -------
    list[LoadingConfigField]
        In display order; each is a ``"choice"``, drawn as a drop-down.
    """
    from cellier.visuals import GeometryLodConfig

    return [
        LoadingConfigField(
            name,
            label,
            "choice",
            tuple(
                str(choice)
                for choice in get_args(GeometryLodConfig.model_fields[name].annotation)
            ),
            tooltip=_LOD_CONFIG_TOOLTIPS[name],
        )
        for name, label in _LOD_CONFIG_LABELS.items()
    ]


class LodConfigEditor:
    """The toolkit-neutral half of a level-of-detail control.

    Holds the config the model last reported, sends edits, and tells the
    widget what to show.  An edit is one ``LodConfigUpdateEvent`` per
    visual.  If the controller refuses it, the widget is told to show the
    last config again, with the error.

    Parameters
    ----------
    visual_ids : Iterable[UUID]
        The multiscale meshes edited together (an ``OrthoViewer`` panel
        group).
    lod : Mapping[str, Any]
        Their current config, as ``GeometryLodConfig.model_dump()``.
    source_id : UUID
        The widget's id, stamped on each edit.
    emit : Callable[[LodConfigUpdateEvent], None]
        Sends an edit (the widget's ``changed.emit``).
    show : Callable[[dict[str, Any], str], None]
        Draws a config and an error message ("" for none).
    """

    def __init__(
        self,
        visual_ids: Iterable[UUID],
        lod: Mapping[str, Any],
        source_id: UUID,
        emit: Callable[[LodConfigUpdateEvent], None],
        show: Callable[[dict[str, Any], str], None],
    ) -> None:
        self._visual_ids = tuple(visual_ids)
        self.config: dict[str, Any] = dict(lod)
        self._source_id = source_id
        self._emit = emit
        self._show = show

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``LodConfigChangedEvent`` subscription per visual."""
        return [
            SubscriptionSpec(LodConfigChangedEvent, self.on_changed, entity_id=vid)
            for vid in self._visual_ids
        ]

    def edit(self, field: str, value: Any) -> None:
        """Send an edit of *field*; on refusal, show the last config and why."""
        if self.config.get(field) == value:
            return
        try:
            for visual_id in self._visual_ids:
                self._emit(
                    LodConfigUpdateEvent(
                        source_id=self._source_id,
                        visual_id=visual_id,
                        field=field,
                        value=value,
                    )
                )
        except Exception as error:
            self._show(dict(self.config), error_message(error))

    def on_changed(self, event: LodConfigChangedEvent) -> None:
        """The model changed (this widget's edit or anyone's): show it."""
        self.config = event.lod.model_dump()
        self._show(dict(self.config), "")
