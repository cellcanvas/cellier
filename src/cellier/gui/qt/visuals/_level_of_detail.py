"""The "Level of detail" control wired to the cellier v2 event bus (Qt)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from psygnal import Signal

from cellier.events import (
    AppearanceChangedEvent,
    AppearanceUpdateEvent,
    DimsChangedEvent,
    SubscriptionSpec,
)
from cellier.gui._appearance_fields import VisualIdGroup
from cellier.gui._image_controls import (
    displayed_dimensions,
    validate_n_displayed_dimensions,
)
from cellier.gui._level_of_detail import (
    COARSEST_LABEL,
    LEVEL_OF_DETAIL_FIELDS,
    LEVEL_OF_DETAIL_TITLE,
    SETTLED_BIAS_LABEL,
    SETTLED_BIAS_RANGE,
    from_control_value,
    moving_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID


class QtLevelOfDetailControls(VisualIdGroup):
    """Settled bias slider and "coarsest while moving" checkbox.

    Drives ``settled_lod_bias``, ``coarsest_while_moving_3d`` and
    ``coarsest_while_moving_2d`` on a multiscale image or labels
    appearance.  The checkbox shows and edits the setting of the view the
    control is in: the 3D one while :attr:`n_displayed_dimensions` is 3,
    the 2D one otherwise.

    Changing ``settled_lod_bias`` reslices, so its ``AppearanceUpdateEvent``
    is emitted when the slider is released, not on every tick.

    Wire to the controller after construction::

        control = QtLevelOfDetailControls(visual_id, values)
        controller.connect_widget(
            control, subscription_specs=control.subscription_specs()
        )

    Parameters
    ----------
    visual_id :
        The visual the control drives.  A sequence drives every listed
        visual in lock-step (the ``OrthoViewer``'s panel siblings).
    values :
        ``cellier.gui._level_of_detail.level_of_detail_values`` of the
        visual's appearance.  Missing keys take the model's defaults.
    n_displayed_dimensions :
        What the view displays now, 2 or 3.
    scene_ids :
        Scenes whose ``DimsChangedEvent`` the control follows to keep
        :attr:`n_displayed_dimensions` current.
    lod_range :
        ``(min, max)`` of the slider.
    decimals :
        Decimal places shown beside the slider.
    title :
        The group's name.  Defaults to :data:`DEFAULT_TITLE`.
    parent :
        Optional Qt parent widget.
    """

    DEFAULT_TITLE = LEVEL_OF_DETAIL_TITLE

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        values: Mapping[str, Any] | None = None,
        *,
        n_displayed_dimensions: int = 3,
        scene_ids: Sequence[UUID] = (),
        lod_range: tuple[float, float] = SETTLED_BIAS_RANGE,
        decimals: int = 2,
        title: str | None = None,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget
        from superqt import QLabeledDoubleSlider

        from cellier.gui.qt.visuals._chrome import labelled_row, titled_group

        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._scene_ids = tuple(scene_ids)
        self._n_displayed_dimensions = validate_n_displayed_dimensions(
            n_displayed_dimensions
        )
        values = dict(values or {})
        self._moving = {
            "coarsest_while_moving_3d": bool(
                values.get("coarsest_while_moving_3d", True)
            ),
            "coarsest_while_moving_2d": bool(
                values.get("coarsest_while_moving_2d", False)
            ),
        }

        self._container = QWidget(parent)
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(*lod_range)
        self._slider.setDecimals(decimals)
        self._slider.setValue(float(values.get("settled_lod_bias", 1.0)))
        # On release only: a bias change reslices.
        self._slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(labelled_row(SETTLED_BIAS_LABEL, self._slider))

        self._moving_check = QCheckBox(COARSEST_LABEL)
        self._moving_check.setChecked(self._moving[self.moving_field])
        self._moving_check.toggled.connect(self._on_moving_toggled)
        layout.addWidget(self._moving_check)

        self._group = titled_group(
            self.DEFAULT_TITLE if title is None else title, self._container, parent
        )

    # -- public interface -----------------------------------------------------

    @property
    def widget(self):
        """The titled group to insert into a layout."""
        return self._group

    @property
    def control(self):
        """The bare row container inside the group."""
        return self._container

    @property
    def n_displayed_dimensions(self) -> int:
        """What the view displays, 2 or 3; picks the checkbox's field."""
        return self._n_displayed_dimensions

    @n_displayed_dimensions.setter
    def n_displayed_dimensions(self, value: int) -> None:
        self._n_displayed_dimensions = validate_n_displayed_dimensions(value)
        self._set_checked(self._moving[self.moving_field])

    @property
    def moving_field(self) -> str:
        """The appearance field the checkbox edits now."""
        return moving_field(self._n_displayed_dimensions)

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """One appearance subscription per visual, one dims one per scene."""
        specs = self._group_specs(AppearanceChangedEvent, self._on_appearance_changed)
        specs.extend(
            SubscriptionSpec(
                event_type=DimsChangedEvent,
                handler=self._on_dims_changed,
                entity_id=scene_id,
            )
            for scene_id in self._scene_ids
        )
        return specs

    # -- model -> widget ------------------------------------------------------

    def _on_appearance_changed(self, event) -> None:
        if event.source_id == self._id:
            return
        if event.field_name not in LEVEL_OF_DETAIL_FIELDS:
            return
        if event.field_name == "settled_lod_bias":
            self._slider.blockSignals(True)
            self._slider.setValue(float(event.new_value))
            self._slider.blockSignals(False)
            return
        self._moving[event.field_name] = bool(event.new_value)
        if event.field_name == self.moving_field:
            self._set_checked(bool(event.new_value))

    def _on_dims_changed(self, event) -> None:
        """Follow a scene's switch between 2D and 3D display."""
        if not event.displayed_axes_changed:
            return
        n = displayed_dimensions(event.dims_state)
        if n is not None and n != self._n_displayed_dimensions:
            self.n_displayed_dimensions = n

    def _set_checked(self, checked: bool) -> None:
        self._moving_check.blockSignals(True)
        self._moving_check.setChecked(checked)
        self._moving_check.blockSignals(False)

    # -- widget -> model ------------------------------------------------------

    def _on_slider_released(self) -> None:
        self._emit_group(
            AppearanceUpdateEvent,
            "settled_lod_bias",
            from_control_value("settled_lod_bias", self._slider.value()),
        )

    def _on_moving_toggled(self, checked: bool) -> None:
        field = self.moving_field
        self._moving[field] = bool(checked)
        self._emit_group(AppearanceUpdateEvent, field, bool(checked))
