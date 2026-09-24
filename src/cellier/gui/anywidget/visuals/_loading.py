"""Loading indicator for multiscale visuals (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._loading import LOADING_TITLE, IndicatorState, LoadingIndicatorModel
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import LoadingProgress, SubscriptionSpec

_STATIC = Path(__file__).parent / "static"


class AnywidgetLoadingIndicator(VisualIdGroup, anywidget.AnyWidget):
    """How far a multiscale visual's data has loaded.

    Mirrors ``QtLoadingIndicator``: a bar of target chunks resident over
    needed, and a status line, under a heading titled ``title``.  Read-only:
    it listens to ``ResliceProgressEvent`` and emits nothing.

    Parameters
    ----------
    visual_id :
        The visual, or a group of them, shown summed.
    initial :
        Their progress now, so an indicator built mid-load starts right.
    """

    _esm = _STATIC / "loading.js"
    _css = _STATIC / "loading.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = LOADING_TITLE
    """Name shown when no ``title=`` is given."""

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    maximum = traitlets.Int(1).tag(sync=True)
    value = traitlets.Int(0).tag(sync=True)
    text = traitlets.Unicode("").tag(sync=True)
    busy = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial: Mapping[UUID, LoadingProgress | None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._model = LoadingIndicatorModel(self._visual_ids, initial or {}, self._show)
        self._show(self._model.state)

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self) -> AnywidgetLoadingIndicator:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def state(self) -> IndicatorState:
        """What the indicator shows now."""
        return self._model.state

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ResliceProgressEvent`` subscription per visual."""
        return self._model.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, state: IndicatorState) -> None:
        with self.hold_sync():
            self.maximum = state.maximum
            self.value = state.value
            self.text = state.text
            self.busy = state.busy
