"""Loading indicator for multiscale visuals (Qt)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._loading import LOADING_TITLE, IndicatorState, LoadingIndicatorModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import LoadingProgress, SubscriptionSpec


class QtLoadingIndicator(VisualIdGroup):
    """How far a multiscale visual's data has loaded.

    A bar of target chunks resident over needed, and a status line: the
    coarse overview (the backstop) first, then the detail, then
    ``"Loaded"``.  Both sit in a group titled :data:`DEFAULT_TITLE`, like the
    image and bounding-box controls.  Read-only: it listens to
    ``ResliceProgressEvent`` and emits nothing
    (``plans/progressive_loading_design_v3.md`` 5.13).

    Wire to the controller after construction::

        indicator = QtLoadingIndicator(
            visual_id, initial={visual_id: controller.loading_progress(visual_id)}
        )
        controller.connect_widget(
            indicator, subscription_specs=indicator.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), shown summed.
    initial :
        Their progress now, so an indicator built mid-load starts right.
    title :
        The group's title.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = LOADING_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        initial: Mapping[UUID, LoadingProgress | None] | None = None,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        box = QWidget(parent)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._bar = QProgressBar(box)
        self._bar.setTextVisible(False)
        self._bar.setMaximumHeight(8)
        self._text = QLabel(box)
        self._text.setWordWrap(True)
        layout.addWidget(self._bar)
        layout.addWidget(self._text)
        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, box, parent
        )

        self._model = LoadingIndicatorModel(self._visual_ids, initial or {}, self._show)
        self._show(self._model.state)

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self):
        """The titled group to insert into a layout."""
        return self._group

    @property
    def state(self) -> IndicatorState:
        """What the indicator shows now."""
        return self._model.state

    @property
    def text(self) -> str:
        """The status line as drawn."""
        return self._text.text()

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``ResliceProgressEvent`` subscription per visual."""
        return self._model.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, state: IndicatorState) -> None:
        self._bar.setRange(0, state.maximum)
        self._bar.setValue(state.value)
        self._text.setText(state.text)
