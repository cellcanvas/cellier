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
    SubscriptionSpec,
)
from cellier.gui._constants import DIMS_SLIDER_THROTTLE_MS
from cellier.gui._dims import initial_slice_indices
from cellier.gui.anywidget._teardown import close_aux_widgets

if TYPE_CHECKING:
    from uuid import UUID

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

        dims = AnywidgetDimsPanel.from_scene(scene, axis_ranges)
        controller.connect_widget(dims, subscription_specs=dims.subscription_specs())
    """

    _esm = _STATIC / "dims_panel.js"
    _css = _STATIC / "dims_panel.css"

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    slice_indices = traitlets.Dict().tag(sync=True)
    axis_labels = traitlets.Dict().tag(sync=True)
    axis_ranges = traitlets.Dict().tag(sync=True)
    displayed_axes = traitlets.List().tag(sync=True)
    stacked_axes = traitlets.List().tag(sync=True)
    non_displayed = traitlets.List().tag(sync=True)

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
        axis_ranges: dict,
        axis_labels: dict,
        slice_indices: dict,
        displayed_axes: list | tuple = (),
        stacked_axes: list | tuple = (),
        non_displayed: list | tuple = (),
        axes_2d: tuple[int, ...] | None = None,
        axes_3d: tuple[int, ...] | None = None,
        **kwargs,
    ) -> None:
        has_toggle = axes_2d is not None and axes_3d is not None
        is_3d = len(displayed_axes) == 3
        super().__init__(
            slice_indices={str(k): int(v) for k, v in slice_indices.items()},
            axis_labels={str(k): str(v) for k, v in axis_labels.items()},
            axis_ranges={
                str(k): [float(lo), float(hi)] for k, (lo, hi) in axis_ranges.items()
            },
            displayed_axes=[int(a) for a in displayed_axes],
            stacked_axes=[int(a) for a in stacked_axes],
            non_displayed=[int(a) for a in non_displayed],
            has_toggle=has_toggle,
            label=("Switch to 2D" if is_3d else "Switch to 3D") if has_toggle else "",
            **kwargs,
        )
        self._id = uuid4()
        self._scene_id = scene_id
        self._applying = False
        self._axes_2d = axes_2d
        self._axes_3d = axes_3d

        self.observe(self._on_slice_indices, names="slice_indices")
        self.observe(self._on_toggle_click, names="_clicks")

    @classmethod
    def from_scene(
        cls,
        scene: Scene,
        axis_ranges: dict[int, tuple[float, float]],
        *,
        non_displayed: tuple[int, ...] = (),
    ) -> AnywidgetDimsPanel:
        """Build a dims panel from a live scene.

        Includes the 2D/3D toggle when the scene has 3 or more axes *and*
        declares both render modes: 3D displays the last three axis indices,
        2D the last two.  A scene that renders only one way -- each panel of
        an ``OrthoViewer`` -- gets no toggle, because there is nothing to
        switch to.
        """
        axis_labels_list = scene.dims.coordinate_system.axis_labels
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
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            slice_indices=initial_slice_indices(selection, axis_ranges),
            displayed_axes=getattr(selection, "displayed_axes", ()),
            stacked_axes=getattr(selection, "stacked_axes", ()),
            non_displayed=non_displayed,
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
            )
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
        if event.source_id == self._id:
            return
        selection = event.dims_state.selection
        new_slices = dict(self.slice_indices)
        for axis, value in selection.slice_indices.items():
            new_slices[str(axis)] = int(value)
        self._set_field("slice_indices", new_slices)
        self._set_field("displayed_axes", [int(a) for a in selection.displayed_axes])
        stacked = getattr(selection, "stacked_axes", ())
        self._set_field("stacked_axes", [int(a) for a in stacked])

        # Relabel the toggle purely from the event -- this is what lets it
        # stay correct even when displayed_axes changed via some other
        # caller, not just this widget's own button.
        if self.has_toggle:
            is_3d = len(selection.displayed_axes) == 3
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
        if self._applying:
            return
        self._emit_dims()

    def _emit_dims(self) -> None:
        hidden = (
            set(self.displayed_axes) | set(self.stacked_axes) | set(self.non_displayed)
        )
        updates = {
            int(axis): int(value)
            for axis, value in self.slice_indices.items()
            if int(axis) not in hidden
        }
        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=updates,
                displayed_axes=None,
            )
        )

    def _on_toggle_click(self, change) -> None:
        is_3d = len(self.displayed_axes) == 3
        target_displayed = self._axes_2d if is_3d else self._axes_3d
        target_set = set(target_displayed)

        # self.slice_indices already holds a live, correct value for every
        # axis (including hidden ones) -- no separate "saved position"
        # bookkeeping needed.
        new_slices = {
            int(axis): int(value)
            for axis, value in self.slice_indices.items()
            if int(axis) not in target_set and int(axis) not in set(self.stacked_axes)
        }
        # Applied *before* the emit, not after.  The controller echoes this
        # change back stamped with our own source_id, so _on_dims_changed's
        # filter ignores it -- the widget has to move itself either way.
        # Doing it first is what keeps the button honest when something
        # downstream of the emit fails: afterwards, one raising handler left
        # the scene in 2D while this panel still showed 3D, with no slider
        # for the axis it had just hidden (``plans/gui_backend_seam.md`` D17).
        # slice_indices already holds a value for every axis regardless of
        # display state (see ``initial_slice_indices``), so it needs no update.
        self._set_field("displayed_axes", [int(a) for a in target_displayed])
        self.label = "Switch to 2D" if not is_3d else "Switch to 3D"

        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=new_slices,
                displayed_axes=target_displayed,
            )
        )
