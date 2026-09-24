"""Loading-settings controls for multiscale visuals (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._loading import (
    COARSEST_LEVEL_TEXT,
    LOADING_CONFIG_TITLE,
    LoadingConfigEditor,
    loading_config_fields,
    to_control_value,
)
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec

_STATIC = Path(__file__).parent / "static"


class AnywidgetLoadingConfigControls(VisualIdGroup, anywidget.AnyWidget):
    """Every ``ProgressiveLoadingConfig`` field of a multiscale visual.

    Mirrors ``QtLoadingConfigControls``.  The front end draws one row per
    entry of ``fields`` from the values in ``config``, and reports an edit
    by setting ``edit``; a refused edit comes back as ``error``, with
    ``config`` unchanged.

    Parameters
    ----------
    visual_id :
        The visual, or a group of them, edited together.
    loading :
        The current config, as ``ProgressiveLoadingConfig.model_dump()``.
    n_levels :
        The visual's level count, the most "Backstop level" offers.
    """

    _esm = _STATIC / "loading_config.js"
    _css = _STATIC / "loading_config.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = LOADING_CONFIG_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    #: One description per row: name, label, kind, choices, min, max, step.
    fields = traitlets.List([]).tag(sync=True)
    #: The settings as their controls hold them (``backstop_level`` 0 is
    #: "coarsest").
    config = traitlets.Dict({}).tag(sync=True)
    #: The reason the last edit was refused, or "".
    error = traitlets.Unicode("").tag(sync=True)
    #: Set by the front end: ``{"field", "value", "serial"}``.
    edit = traitlets.Dict({}).tag(sync=True)
    coarsest_text = traitlets.Unicode(COARSEST_LEVEL_TEXT).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        loading: Mapping[str, Any],
        n_levels: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            fields=[field._asdict() for field in loading_config_fields(n_levels)],
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._editor = LoadingConfigEditor(
            self._visual_ids, loading, self._id, self.changed.emit, self._show
        )
        self._show(dict(loading), "")
        self.observe(self._on_edit, names="edit")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetLoadingConfigControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``LoadingConfigChangedEvent`` subscription per visual."""
        return self._editor.subscription_specs()

    # -- widget -> model -------------------------------------------------------

    def _on_edit(self, change) -> None:
        edit = change["new"] or {}
        if "field" in edit:
            self._editor.edit(edit["field"], edit.get("value"))

    # -- model -> widget -------------------------------------------------------

    def _show(self, config: dict[str, Any], error: str) -> None:
        with self.hold_sync():
            self.config = {
                name: to_control_value(name, value) for name, value in config.items()
            }
            self.error = error
