"""Level-of-detail controls for multiscale meshes (Qt)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._lod import LOD_CONFIG_TITLE, LodConfigEditor, lod_config_fields

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from cellier.events import SubscriptionSpec


class QtLodConfigControls(VisualIdGroup):
    """The level-of-detail settings of a multiscale mesh.

    One row per ``GeometryLodConfig`` setting: what a dims scrub loads, what
    it draws for a mesh it does not change, and what a moving camera draws
    in a 3D view.  Each is ``coarse`` or ``full``.  An edit is sent as
    ``LodConfigUpdateEvent`` and applies in the next frame; nothing is read.

    Wire to the controller after construction::

        controls = QtLodConfigControls(visual.id, lod=visual.lod.model_dump())
        controller.connect_widget(
            controls, subscription_specs=controls.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual, or a group of them (an ``OrthoViewer``'s panel
        siblings), edited together.
    lod :
        The current config, as ``GeometryLodConfig.model_dump()``.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = LOD_CONFIG_TITLE
    """Name shown when no ``title=`` is given."""

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        *,
        lod: Mapping[str, Any],
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

        from cellier.gui.qt.visuals._chrome import titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)

        content = QWidget(parent)
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        column.addLayout(form)

        self._inputs: dict[str, Any] = {}
        for field in lod_config_fields():
            name = field.name
            widget = QComboBox(content)
            widget.addItems(list(field.choices))
            widget.currentTextChanged.connect(lambda v, n=name: self._editor.edit(n, v))
            self._inputs[name] = widget
            form.addRow(field.label, widget)
            widget.setToolTip(field.tooltip)
            form.labelForField(widget).setToolTip(field.tooltip)

        self._error = QLabel(content)
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color: #d04040")
        self._error.setVisible(False)
        column.addWidget(self._error)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, content, parent
        )
        self._editor = LodConfigEditor(
            self._visual_ids, lod, self._id, self.changed.emit, self._show
        )
        self._show(dict(lod), "")

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
        """One ``LodConfigChangedEvent`` subscription per visual."""
        return self._editor.subscription_specs()

    # -- model -> widget -------------------------------------------------------

    def _show(self, config: dict[str, Any], error: str) -> None:
        for name, widget in self._inputs.items():
            if name not in config:
                continue
            widget.blockSignals(True)
            try:
                widget.setCurrentText(str(config[name]))
            finally:
                widget.blockSignals(False)
        self._error.setText(error)
        self._error.setVisible(bool(error))
