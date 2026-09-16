"""Dims anywidget -- per-axis slice sliders plus an optional 2D/3D toggle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import anywidget
import traitlets
from psygnal import Signal

from cellier.events import (
    DimsChangedEvent,
    DimsUpdateEvent,
    SliderAxesChangedEvent,
    SubscriptionSpec,
)
from cellier.gui._axis_values import (
    DiscreteAxisValues,
    coerce_axis_values,
    nearest_value_index,
)
from cellier.gui._constants import DIMS_SLIDER_THROTTLE_MS
from cellier.gui._dims import initial_slice_indices
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from cellier.gui._axis_values import AxisValues
    from cellier.scene.scene import Scene

_STATIC = Path(__file__).parent / "static"


class AnywidgetDimsPanel(anywidget.AnyWidget):
    """Renders per-axis slice sliders and an optional 2D/3D toggle button.

    Satisfies the :class:`cellier.gui._protocol.WidgetView` contract so
    ``CellierController.connect_widget`` wires it the same way as Qt widgets.
    The toggle button is included automatically when constructed with
    *axes_2d*/*axes_3d* (see :meth:`from_scene`), e.g. omitted for a scene
    with fewer than 3 axes.

    Construct via :meth:`from_scene`, then wire with::

        dims = AnywidgetDimsPanel.from_scene(scene, axis_values)
        controller.connect_widget(dims, subscription_specs=dims.subscription_specs())
    """

    _esm = _STATIC / "dims_panel.js"
    _css = _STATIC / "dims_panel.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    slice_indices = traitlets.Dict().tag(sync=True)
    axis_labels = traitlets.Dict().tag(sync=True)
    axis_values = traitlets.Dict().tag(sync=True)
    """Axis index (str) to that axis's serialised ``AxisValues``."""

    discrete_index = traitlets.Dict().tag(sync=True)
    """Axis index (str) to the slider position of each discrete axis.

    Derived here from ``slice_indices`` whenever it changes, so the rule for
    which listed value a between-values position shows lives in Python
    (:func:`~cellier.gui._axis_values.nearest_value_index`) rather than being
    copied into ``dims_panel.js``.
    """

    displayed_axes = traitlets.List().tag(sync=True)
    slider_axes = traitlets.List(allow_none=True, default_value=None).tag(sync=True)
    """World axes that get a slider when not displayed; ``None`` means all.

    Read from ``scene.slider_axes`` and kept current by
    ``SliderAxesChangedEvent``.
    """

    throttle_ms = traitlets.Int(DIMS_SLIDER_THROTTLE_MS).tag(sync=True)
    """How often a slider drag reaches the bus, in ms.

    Synced rather than hard-coded in ``dims_panel.js`` so this and the Qt
    front end coalesce drags at the same rate -- see
    :data:`cellier.gui._constants.DIMS_SLIDER_THROTTLE_MS`.
    """

    has_toggle = traitlets.Bool(False).tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    # Incremented by the JS click handler; observed on the Python side.
    _clicks = traitlets.Int(0).tag(sync=True)

    def __init__(
        self,
        *,
        scene_id: UUID,
        axis_values: Mapping[int, AxisValues],
        axis_labels: dict,
        slice_indices: dict,
        displayed_axes: list | tuple = (),
        slider_axes: list | tuple | None = None,
        axes_2d: tuple[int, ...] | None = None,
        axes_3d: tuple[int, ...] | None = None,
        **kwargs,
    ) -> None:
        has_toggle = axes_2d is not None and axes_3d is not None
        is_3d = len(displayed_axes) == 3
        coerced = coerce_axis_values(axis_values)
        slices = {str(k): float(v) for k, v in slice_indices.items()}
        super().__init__(
            slice_indices=slices,
            axis_labels={str(k): str(v) for k, v in axis_labels.items()},
            axis_values={
                str(k): spec.model_dump(mode="json") for k, spec in coerced.items()
            },
            discrete_index=_discrete_positions(coerced, slices),
            displayed_axes=[int(a) for a in displayed_axes],
            slider_axes=None if slider_axes is None else [int(a) for a in slider_axes],
            has_toggle=has_toggle,
            label=("Switch to 2D" if is_3d else "Switch to 3D") if has_toggle else "",
            **kwargs,
        )
        self._id = uuid4()
        self._scene_id = scene_id
        self._axis_values = coerced
        self._applying = False
        self._axes_2d = axes_2d
        self._axes_3d = axes_3d
        # The displayed axes the model last reported while a toggle's emit was
        # in flight, or ``None`` if it reported none; see ``_on_toggle_click``.
        self._model_displayed_during_toggle: tuple[int, ...] | None = None

        self.observe(self._on_slice_indices, names="slice_indices")
        self.observe(self._on_toggle_click, names="_clicks")

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        axis_values: Mapping[int, AxisValues],
    ) -> AnywidgetDimsPanel:
        """Build a dims panel from a live scene.

        Includes the 2D/3D toggle when the scene has 3 or more axes *and*
        declares both render modes: 3D displays the last three axis indices,
        2D the last two.  A scene that renders only one way -- each panel of
        an ``OrthoViewer`` -- gets no toggle, because there is nothing to
        switch to.
        """
        axis_values = coerce_axis_values(axis_values)
        axis_labels_list = scene.dims.axis_labels
        axis_labels = dict(enumerate(axis_labels_list))
        selection = scene.dims.selection

        axes_2d: tuple[int, ...] | None = None
        axes_3d: tuple[int, ...] | None = None
        # The toggle is offered only when the *scene* says it can render both
        # ways.  An OrthoViewer panel declares exactly one mode -- three slice
        # views and one volume -- so offering to switch it put the scene into a
        # mode it was never configured for: the reslice produced no geometry,
        # the scene had no bounds, and the panel went blank
        # (``plans/gui_backend_seam.md`` D18).  A plain ``Viewer`` declares
        # both, so its toggle is unaffected.
        modes = {str(mode) for mode in getattr(scene, "render_modes", ())}
        if len(axis_labels_list) >= 3 and {"2d", "3d"} <= modes:
            ndim = len(axis_labels_list)
            axes_3d = tuple(range(ndim - 3, ndim))
            axes_2d = tuple(range(ndim - 2, ndim))

        return cls(
            scene_id=scene.id,
            axis_values=axis_values,
            axis_labels=axis_labels,
            slice_indices=initial_slice_indices(selection, axis_values),
            displayed_axes=getattr(selection, "displayed_axes", ()),
            slider_axes=scene.slider_axes,
            axes_2d=axes_2d,
            axes_3d=axes_3d,
        )

    @property
    def widget(self) -> AnywidgetDimsPanel:
        return self

    def subscription_specs(self) -> list[SubscriptionSpec]:
        return [
            SubscriptionSpec(
                event_type=DimsChangedEvent,
                handler=self._on_dims_changed,
                entity_id=self._scene_id,
            ),
            SubscriptionSpec(
                event_type=SliderAxesChangedEvent,
                handler=self._on_slider_axes_changed,
                entity_id=self._scene_id,
            ),
        ]

    def close(self) -> None:
        """Unsubscribe from the bus and release the widget.

        ``closed`` tells the controller to drop this widget's subscriptions;
        the rest actually releases the widget.  See
        ``cellier.gui.anywidget._teardown`` for why both steps are needed --
        ``ipywidgets`` holds every widget, and every widget's ``layout``, in a
        process-global table that only ``close()`` clears.
        """
        self.closed.emit()
        close_aux_widgets(self)
        super().close()

    # ------------------------------------------------------------------
    # model -> widget
    # ------------------------------------------------------------------

    def _on_dims_changed(self, event: DimsChangedEvent) -> None:
        selection = event.dims_state.selection
        self._model_displayed_during_toggle = tuple(selection.displayed_axes)
        # Slider values are skipped on our own echo: a drag has moved on
        # since it sent them.  The positions ride on the event rather than on
        # ``dims_state``: the render layer takes the region instead (D5), but
        # a slider that something else moved still has to resync.
        if event.source_id != self._id:
            new_slices = dict(self.slice_indices)
            for axis, value in event.slice_indices.items():
                new_slices[str(axis)] = float(value)
            self._set_field("slice_indices", new_slices)
        # The displayed axes are applied even from our own echo: the event is
        # the model's state, and applying it twice is harmless.
        self._apply_displayed(tuple(selection.displayed_axes))

    def _on_slider_axes_changed(self, event: SliderAxesChangedEvent) -> None:
        self._set_field("slider_axes", [int(a) for a in event.slider_axes])

    def _apply_displayed(self, displayed_axes: tuple[int, ...]) -> None:
        """Hide *displayed_axes*' sliders and label the toggle for the other mode."""
        self._set_field("displayed_axes", [int(a) for a in displayed_axes])
        if self.has_toggle:
            is_3d = len(displayed_axes) == 3
            self.label = "Switch to 2D" if is_3d else "Switch to 3D"

    def _set_field(self, name: str, value) -> None:
        self._applying = True
        try:
            setattr(self, name, value)
        finally:
            self._applying = False

    # ------------------------------------------------------------------
    # widget -> model
    # ------------------------------------------------------------------

    def _on_slice_indices(self, change) -> None:
        # Refreshed for every change, from JS or from the bus, so a discrete
        # slider always shows the value nearest to the current position.
        self.discrete_index = _discrete_positions(self._axis_values, self.slice_indices)
        if self._applying:
            return
        old = change.get("old") or {}
        # Only the axes that moved: ``update_slice_indices`` merges.
        updates = {
            int(axis): float(value)
            for axis, value in (change.get("new") or {}).items()
            if axis not in old or float(old[axis]) != float(value)
        }
        if not updates:
            return
        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=updates,
                displayed_axes=None,
            )
        )

    def _on_toggle_click(self, change) -> None:
        if self._axes_2d is None or self._axes_3d is None:
            return
        previous = tuple(int(a) for a in self.displayed_axes)
        target_displayed = self._axes_2d if len(previous) == 3 else self._axes_3d

        # Only ``displayed_axes`` is sent: every axis keeps its slice position
        # in the model whatever is displayed (D36).  The panel moves itself
        # *before* the emit: the controller echoes the change stamped with our
        # own source_id, and a handler downstream of the change can fail
        # (``plans/gui_backend_seam.md`` D17).
        self._apply_displayed(tuple(target_displayed))
        self._model_displayed_during_toggle = None
        try:
            self.changed.emit(
                DimsUpdateEvent(
                    source_id=self._id,
                    scene_id=self._scene_id,
                    slice_indices=None,
                    displayed_axes=tuple(target_displayed),
                )
            )
        except Exception:
            # Refused before the model changed (a composited axis cannot be
            # displayed, design 3.4): go back.  Changed and then a later
            # handler raised: keep what the model reported.
            reported = self._model_displayed_during_toggle
            self._apply_displayed(previous if reported is None else reported)
            raise

    def _shown_value(self, axis: int, value: float) -> float:
        """The world value the slider for *axis* shows for position *value*."""
        spec = self._axis_values.get(axis)
        if isinstance(spec, DiscreteAxisValues):
            return spec.values[nearest_value_index(spec.values, value)]
        return value


def _discrete_positions(
    axis_values: Mapping[int, AxisValues], slice_indices: Mapping[str, float]
) -> dict[str, int]:
    """Slider position of every discrete axis that has a slice value."""
    return {
        str(axis): nearest_value_index(spec.values, float(slice_indices[str(axis)]))
        for axis, spec in axis_values.items()
        if isinstance(spec, DiscreteAxisValues) and str(axis) in slice_indices
    }
