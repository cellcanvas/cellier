"""Loading-settings controls for multiscale visuals (Qt)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._loading import (
    COARSEST_LEVEL_TEXT,
    LOADING_CONFIG_TITLE,
    LoadingConfigEditor,
    loading_config_fields,
    to_control_value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec


class QtLoadingConfigControls(VisualIdGroup):
    """Every ``ProgressiveLoadingConfig`` field of a multiscale visual.

    One row per field: the backstop on or off, its level ("coarsest" for
    ``None``), its extent and slot cap, and what a dims slider tick loads.
    An edit is sent as ``LoadingConfigUpdateEvent`` and applied at once.
    An invalid combination (``dims_drag="backstop"`` with the backstop off)
    is refused by the controller: the rows show the settings again and the
    reason below them; nothing is corrected.

    Wire to the controller after construction::

        controls = QtLoadingConfigControls(visual_id, loading=config.model_dump())
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    loading :
        The current config, as ``ProgressiveLoadingConfig.model_dump()``.
    n_levels :
        The visual's level count, the most "Backstop level" offers.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = LOADING_CONFIG_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        loading: Mapping[str, Any],
        n_levels: int | None = None,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QLabel,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        content = QWidget(parent)
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        column.addLayout(form)

        self._inputs: dict[str, Any] = {}
        for field in loading_config_fields(n_levels):
            name = field.name
            if field.kind == "bool":
                widget = QCheckBox(content)
                widget.toggled.connect(lambda v, n=name: self._editor.edit(n, v))
            elif field.kind == "choice":
                widget = QComboBox(content)
                widget.addItems(list(field.choices))
                widget.currentTextChanged.connect(
                    lambda v, n=name: self._editor.edit(n, v)
                )
            elif field.kind == "level":
                widget = QSpinBox(content)
                widget.setRange(int(field.minimum), int(field.maximum))
                # 0 stands for None, the coarsest level.
                widget.setSpecialValueText(COARSEST_LEVEL_TEXT)
                widget.valueChanged.connect(lambda v, n=name: self._editor.edit(n, v))
            else:
                widget = QDoubleSpinBox(content)
                widget.setRange(field.minimum, field.maximum)
                widget.setSingleStep(field.step)
                widget.setDecimals(2)
                widget.valueChanged.connect(lambda v, n=name: self._editor.edit(n, v))
            self._inputs[name] = widget
            form.addRow(field.label, widget)

        self._error = QLabel(content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, content, parent
        )
        self._editor = LoadingConfigEditor(
            self._visual_ids, loading, self._id, self.changed.emit, self._show
        )
        self._show(dict(loading), "")

    # -- Public interface ------------------------------------------------------

    @property
    def widget(self):
        """The titled group to insert into a layout."""
        return self._group

    @property
    def config(self) -> dict[str, Any]:
        """The settings as last reported by the model."""
        return dict(self._editor.config)

    @property
    def error(self) -> str:
        """The reason the last edit was refused, or ""."""
        return self._error.text()

    def input(self, field: str):
        """The control for one field (for tests and scripting)."""
        return self._inputs[field]

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One ``LoadingConfigChangedEvent`` subscription per visual."""
        return self._editor.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, config: dict[str, Any], error: str) -> None:
        for name, widget in self._inputs.items():
            value = to_control_value(name, config.get(name))
            widget.blockSignals(True)
            try:
                if hasattr(widget, "setChecked"):
                    widget.setChecked(bool(value))
                elif hasattr(widget, "setCurrentText"):
                    widget.setCurrentText(str(value))
                else:
                    widget.setValue(value)
            finally:
                widget.blockSignals(False)
        self._error.setText(error)
        self._error.setVisible(bool(error))
