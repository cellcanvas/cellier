"""The "Level of detail" control wired to the cellier v2 event bus (anywidget)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anywidget
import traitlets
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
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

_STATIC = Path(__file__).parent / "static"


class AnywidgetLevelOfDetailControls(VisualIdGroup, anywidget.AnyWidget):
    """Settled bias slider and "coarsest while moving" checkbox.

    Mirrors ``QtLevelOfDetailControls``.  The front end shows one checkbox,
    bound to ``coarsest_while_moving_3d`` while ``n_displayed_dimensions`` is
    3 and to ``coarsest_while_moving_2d`` otherwise.  The bias is sent on a
    settled ``change`` of the slider, not while it is dragged, because a
    bias change reslices.

    Parameters
    ----------
    visual_id :
        The visual the control drives, or a sequence driven in lock-step.
    values :
        ``cellier.gui._level_of_detail.level_of_detail_values`` of the
        visual's appearance.  Missing keys take the model's defaults.
    n_displayed_dimensions :
        What the view displays now, 2 or 3.
    scene_ids :
        Scenes whose ``DimsChangedEvent`` the control follows.
    """

    _esm = _STATIC / "level_of_detail.js"
    _css = _STATIC / "level_of_detail.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    DEFAULT_TITLE = LEVEL_OF_DETAIL_TITLE

    title = traitlets.Unicode(DEFAULT_TITLE).tag(sync=True)
    labels = traitlets.Dict().tag(sync=True)
    bias_range = traitlets.List(traitlets.Float()).tag(sync=True)

    settled_lod_bias = traitlets.Float(1.0).tag(sync=True)
    coarsest_while_moving_3d = traitlets.Bool(True).tag(sync=True)
    coarsest_while_moving_2d = traitlets.Bool(False).tag(sync=True)
    n_displayed_dimensions = traitlets.Int(3).tag(sync=True)

    @traitlets.validate("n_displayed_dimensions")
    def _validate_n_displayed_dimensions(self, proposal):
        return validate_n_displayed_dimensions(proposal["value"])

    def __init__(
        self,
        visual_id: UUID | Sequence[UUID],
        values: Mapping[str, Any] | None = None,
        *,
        n_displayed_dimensions: int = 3,
        scene_ids: Sequence[UUID] = (),
        **kwargs,
    ) -> None:
        values = dict(values or {})
        super().__init__(
            settled_lod_bias=float(values.get("settled_lod_bias", 1.0)),
            coarsest_while_moving_3d=bool(values.get("coarsest_while_moving_3d", True)),
            coarsest_while_moving_2d=bool(
                values.get("coarsest_while_moving_2d", False)
            ),
            n_displayed_dimensions=n_displayed_dimensions,
            labels={"bias": SETTLED_BIAS_LABEL, "moving": COARSEST_LABEL},
            bias_range=list(SETTLED_BIAS_RANGE),
            **kwargs,
        )
        self._id = uuid4()
        self._init_visual_ids(visual_id)
        self._scene_ids = tuple(scene_ids)
        self._applying = False
        self.observe(self._on_trait_change, names=list(LEVEL_OF_DETAIL_FIELDS))

    # -- public interface -----------------------------------------------------

    @property
    def widget(self) -> AnywidgetLevelOfDetailControls:
        """An ``AnyWidget`` is itself the embeddable element."""
        return self

    @property
    def moving_field(self) -> str:
        """The appearance field the checkbox edits now."""
        return moving_field(self.n_displayed_dimensions)

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget."""
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

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

    def _on_appearance_changed(self, event: AppearanceChangedEvent) -> None:
        if event.source_id == self._id:
            return  # echo of our own change
        if event.field_name not in LEVEL_OF_DETAIL_FIELDS:
            return
        self._set_field(
            event.field_name, from_control_value(event.field_name, event.new_value)
        )

    def _on_dims_changed(self, event) -> None:
        """Follow a scene's switch between 2D and 3D display."""
        if not event.displayed_axes_changed:
            return
        n = displayed_dimensions(event.dims_state)
        if n is not None:
            self.n_displayed_dimensions = n

    def _set_field(self, name: str, value) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # -- widget -> model ------------------------------------------------------

    def _on_trait_change(self, change) -> None:
        if self._applying:
            return
        self._emit_group(
            AppearanceUpdateEvent,
            change["name"],
            from_control_value(change["name"], change["new"]),
        )
