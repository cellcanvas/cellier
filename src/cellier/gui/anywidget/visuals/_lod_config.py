"""Level-of-detail controls for multiscale meshes (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._lod import LOD_CONFIG_TITLE, LodConfigEditor, lod_config_fields
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

_STATIC = Path(__file__).parent / "static"


class AnywidgetLodConfigControls(VisualIdGroup, anywidget.AnyWidget):
    """The level-of-detail settings of a multiscale mesh.

    Mirrors ``QtLodConfigControls``.  The front end is the settings-rows
    module the loading settings use (``loading_config.js``): it draws one
    row per entry of ``fields`` from the values in ``config``, and reports
    an edit by setting ``edit``; a refused edit comes back as ``error``,
    with ``config`` unchanged.

    Parameters
    ----------
    visual_id :
        The visual, or a group of them, edited together.
    lod :
        The current config, as ``GeometryLodConfig.model_dump()``.
    """

    _esm = _STATIC / "loading_config.js"
    _css = _STATIC / "loading_config.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = LOD_CONFIG_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: One description per row: name, label, kind, choices, min, max, step.
    fields = traitlets.List([]).tag(sync=True)
    #: The settings as their controls hold them.
    config = traitlets.Dict({}).tag(sync=True)
    #: The reason the last edit was refused, or "".
    error = traitlets.Unicode("").tag(sync=True)
    #: Set by the front end: ``{"field", "value", "serial"}``.
    edit = traitlets.Dict({}).tag(sync=True)
    #: Read by the shared front end for a level row; these settings have none.
    coarsest_text = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        lod: Mapping[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            fields=[field._asdict() for field in lod_config_fields()], **kwargs
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._editor = LodConfigEditor(
            self._visual_ids, lod, self._id, self.changed.emit, self._show
        )
        self._show(dict(lod), "")
        self.observe(self._on_edit, names="edit")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetLodConfigControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``LodConfigChangedEvent`` subscription per visual."""
        return self._editor.subscription_specs()

    # -- widget -> model -------------------------------------------------------

    def _on_edit(self, change) -> None:
        edit = change["new"] or {}
        if "field" in edit:
            self._editor.edit(edit["field"], edit.get("value"))

    # -- model -> widget -------------------------------------------------------

    def _show(self, config: dict[str, Any], error: str) -> None:
        with self.hold_sync():
            # coarse_level has no row, and None does not belong in the rows.
            self.config = {
                name: value for name, value in config.items() if value is not None
            }
            self.error = error
