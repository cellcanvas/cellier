"""Widgets for scene components.

The QtDimsControl and QtCanvasWidget classes are modified from
the _QDimsSliders and _QArrayViewer classes from ndv, respectively.
https://github.com/pyapp-kit/ndv/blob/main/src/ndv/views/_qt/_array_view.py

NDV license:
BSD 3-Clause License

Copyright (c) 2023, Talley Lambert

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider

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
from cellier.gui._dims import initial_slice_indices as seed_slice_indices

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cellier.gui._axis_values import AxisValues

#: Styles the continuous sliders only.  A styled groove stops Qt drawing
#: native tick marks, and a discrete axis's integer slider relies on those
#: ticks to show where its values are, so the rules are scoped to the
#: ``QLabeledDoubleSlider`` a continuous axis uses.
SLIDER_STYLE = """
QLabeledDoubleSlider QSlider::groove:horizontal {
    border: 1px solid #bbb;
    background: white;
    height: 10px;
    border-radius: 4px;
}

QLabeledDoubleSlider QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #eee, stop:1 #ccc);
    border: 1px solid #777;
    width: 13px;
    margin-top: -7px;
    margin-bottom: -7px;
    border-radius: 4px;
}

QLabeledDoubleSlider QSlider::add-page:horizontal {
    background: #fff;
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QLabeledDoubleSlider QSlider::sub-page:horizontal {
    background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
        stop: 0 #66e, stop: 1 #bbf);
    background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
        stop: 0 #bbf, stop: 1 #55f);
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QLabeledDoubleSlider QSlider::handle:horizontal:hover {
background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #fff, stop:1 #ddd);
border: 1px solid #444;
border-radius: 4px;
}

QLabel { font-size: 12px; }
"""


class QtDimsControl:
    """Bidirectional dims slider panel + 2D/3D toggle wired to the cellier v2 bus.

    Composes a ``QWidget`` container (with a ``QFormLayout``) holding one
    slider per axis, plus (when *axes_2d*/*axes_3d* are given) a
    toggle button that switches the scene between its 2D and 3D axis sets.
    An axis shows a slider when it is one of the scene's ``slider_axes`` and
    is not displayed.

    Follows the v2 widget pattern:

    - One ``UUID`` (``self._id``) shared by the sliders and the toggle.
    - Subscribed to ``DimsChangedEvent`` via ``controller.connect_widget``.
    - Echo-filters its own changes using ``source_id``.
    - Suppresses re-entrant slider signals with ``blockSignals`` when applying
      model-driven updates.
    - The toggle never relabels itself optimistically on click -- it waits
      for the echoed ``DimsChangedEvent``, same as the sliders, so there is a
      single source of truth for "what is currently displayed."

    Wire to the controller after construction::

        control = QtDimsControl(scene_id, axis_values=..., axis_labels=...)
        controller.connect_widget(
            control, subscription_specs=control.subscription_specs()
        )

    Parameters
    ----------
    scene_id :
        UUID of the scene whose slice indices this widget controls.
    axis_values :
        Mapping of axis index to that axis's slider values, in world units.
        A ``ContinuousAxisValues`` axis gets a double-valued slider over
        ``[min, max]`` -- a slice position is a world position, not a voxel
        index (D3).  A ``DiscreteAxisValues`` axis gets an integer slider
        over the positions of its values, with a readout showing the value
        or its label; it emits the world value, never the position.
    axis_labels :
        Mapping of axis index to display label, e.g. ``{0: "z", 1: "y", 2: "x"}``.
    initial_slice_indices :
        Starting slider values; typically ``scene.dims.selection.slice_indices``.
    initial_displayed_axes :
        Axes to hide initially; typically ``scene.dims.selection.displayed_axes``.
    slider_axes :
        The world axes that get a slider when not displayed; typically
        ``scene.slider_axes``.  ``None`` (the default) gives every axis in
        *axis_values* one.  Kept current by ``SliderAxesChangedEvent``.
    axes_2d :
        Axis indices to display when toggling to 2D, or ``None`` to omit the
        toggle button entirely (e.g. a scene with fewer than 3 axes).
    axes_3d :
        Axis indices to display when toggling to 3D, or ``None`` to omit the
        toggle button entirely.
    parent :
        Optional Qt parent widget for the internal container.
    """

    changed: Signal = Signal(object)
    closed: Signal = Signal()

    def __init__(
        self,
        scene_id,
        axis_values: Mapping[int, AxisValues],
        axis_labels: dict[int, str],
        *,
        initial_slice_indices: dict[int, float] | None = None,
        initial_displayed_axes: tuple[int, ...] = (),
        slider_axes: tuple[int, ...] | None = None,
        debounce_ms: int | None = None,
        axes_2d: tuple[int, ...] | None = None,
        axes_3d: tuple[int, ...] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._scene_id = scene_id
        self._slider_axes: tuple[int, ...] | None = (
            None if slider_axes is None else tuple(slider_axes)
        )
        # Axes moved since the last submit; only these are sent, because
        # ``update_slice_indices`` merges.
        self._dirty_axes: set[int] = set()
        # The displayed axes the model last reported while a toggle's emit was
        # in flight, or ``None`` if it reported none.  Decides whether a
        # raising toggle rolls the control back.
        self._model_displayed_during_toggle: tuple[int, ...] | None = None
        self._axes_2d = axes_2d
        self._axes_3d = axes_3d

        # Single-shot QTimer for rate-limiting rapid slider moves.
        # Fires at most once per interval; a dirty flag ensures the final
        # position is always submitted even if it landed between ticks.
        self._rate_limit_timer = QTimer()
        self._rate_limit_timer.setSingleShot(True)
        self._rate_limit_timer.setInterval(
            DIMS_SLIDER_THROTTLE_MS if debounce_ms is None else debounce_ms
        )
        self._rate_limit_timer.timeout.connect(self._on_rate_limit_tick)
        self._slider_dirty = False

        # ── Qt seam 1: build container and sliders ───────────────────────────
        self._container = QWidget(parent)
        self._container.setStyleSheet(SLIDER_STYLE)

        layout = QFormLayout(self._container)
        layout.setSpacing(2)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setContentsMargins(0, 0, 0, 0)

        self._axis_values = coerce_axis_values(axis_values)
        # The slider itself, per axis.  A continuous axis's slider is also its
        # form row; a discrete axis's sits in a row widget beside its readout.
        self._sliders: dict[int, QLabeledDoubleSlider | QSlider] = {}
        self._rows: dict[int, QWidget] = {}
        self._readouts: dict[int, QLabel] = {}
        self._displayed_axes: tuple[int, ...] = initial_displayed_axes
        _initial = initial_slice_indices or {}

        for axis, spec in self._axis_values.items():
            if isinstance(spec, DiscreteAxisValues):
                row = self._build_discrete_row(axis, spec)
                self._set_value(axis, _initial.get(axis, spec.values[0]))
            else:
                sld = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
                # Display only; before setValue so the first position is not
                # rounded to the previous precision.
                sld.setDecimals(spec.decimals)
                sld.setRange(spec.min, spec.max)
                sld.setValue(_initial.get(axis, spec.min))
                # Capture `axis` by value in the default-argument closure.
                sld.valueChanged.connect(
                    lambda value, ax=axis: self._on_slider_changed(ax, value)
                )
                self._sliders[axis] = sld
                row = sld
            layout.addRow(axis_labels.get(axis, str(axis)), row)
            self._rows[axis] = row

        self._toggle_button: QPushButton | None = None
        if axes_2d is not None and axes_3d is not None:
            is_3d = len(initial_displayed_axes) == 3
            self._toggle_button = QPushButton(
                "Switch to 2D" if is_3d else "Switch to 3D"
            )
            self._toggle_button.clicked.connect(self._on_toggle_click)
            layout.addRow(self._toggle_button)

        self._update_visibility(initial_displayed_axes)

    # ── Public interface ─────────────────────────────────────────────────────

    @property
    def has_toggle(self) -> bool:
        """Whether this control offers a 2D/3D toggle.

        Named to match ``AnywidgetDimsPanel.has_toggle`` so a caller can ask
        either front end the same question.
        """
        return self._toggle_button is not None

    @property
    def widget(self) -> QWidget:
        """The Qt widget to insert into a layout.

        Qt seam 1: replace with the backend element for other toolkits.
        """
        return self._container

    @property
    def slider_axes(self) -> tuple[int, ...] | None:
        """The world axes that get a slider when not displayed.

        ``None`` means every axis this control was built with.
        """
        return self._slider_axes

    def current_index(self) -> dict[int, float]:
        """Return the current world value of every slider regardless of visibility.

        A discrete axis reports the world value at its slider position, not
        the position itself.
        """
        return {axis: self._world_value(axis) for axis in self._sliders}

    def close(self) -> None:
        """Emit ``closed`` to trigger bus unsubscription via the controller."""
        self.closed.emit()

    def subscription_specs(self) -> list[SubscriptionSpec]:
        """Return the inbound subscription this widget requires.

        Pass the result to ``CellierController.connect_widget``.
        """
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

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_dims_changed(self, event) -> None:
        sel = event.dims_state.selection
        self._model_displayed_during_toggle = tuple(sel.displayed_axes)

        # Slider values are skipped on our own echo: a drag has moved on
        # since it sent them.  The positions ride on the event rather than on
        # ``dims_state``: the render layer takes the region instead (D5), but
        # a slider that something else moved still has to resync.
        if event.source_id != self._id:
            for axis in self._sliders:
                value = event.slice_indices.get(axis)
                if value is not None:
                    self._set_value(axis, value)

        # The displayed axes are applied even from our own echo: the event is
        # the model's state, and applying it twice is harmless.
        self._apply_displayed(tuple(sel.displayed_axes))

    def _on_slider_axes_changed(self, event: SliderAxesChangedEvent) -> None:
        self._slider_axes = tuple(event.slider_axes)
        self._update_visibility(self._displayed_axes)

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, axis: int, value: float) -> None:
        self._slider_dirty = True
        self._dirty_axes.add(axis)
        if not self._rate_limit_timer.isActive():
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _on_rate_limit_tick(self) -> None:
        if self._slider_dirty:
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _submit_slider_values(self) -> None:
        """Submit the values of the sliders moved since the last submit."""
        self._slider_dirty = False
        updates = {axis: self._world_value(axis) for axis in sorted(self._dirty_axes)}
        self._dirty_axes.clear()
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

    def _on_toggle_click(self) -> None:
        if self._axes_2d is None or self._axes_3d is None:
            return
        previous = tuple(self._displayed_axes)
        target_displayed = self._axes_2d if len(previous) == 3 else self._axes_3d

        # Only ``displayed_axes`` is sent: every axis keeps its slice position
        # in the model whatever is displayed (D36), so there is nothing to
        # restore.  The control moves itself *before* the emit so it stays
        # honest when a handler downstream of the change fails
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

    def _apply_displayed(self, displayed_axes: tuple[int, ...]) -> None:
        """Hide *displayed_axes*' sliders and label the toggle for the other mode."""
        self._update_visibility(displayed_axes)
        if self._toggle_button is not None:
            is_3d = len(displayed_axes) == 3
            self._toggle_button.setText("Switch to 2D" if is_3d else "Switch to 3D")

    # ── Discrete axes ────────────────────────────────────────────────────────

    def _build_discrete_row(self, axis: int, spec: DiscreteAxisValues) -> QWidget:
        """Build an integer slider over *spec*'s positions plus a readout."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(0, len(spec.values) - 1)
        sld.setSingleStep(1)
        sld.setPageStep(1)
        # Only when the axis asks for them.  A tick per value is redrawn on
        # every value change -- once per mouse move while dragging -- and the
        # macOS style rebuilds each mark through AppKit, so a long axis costs
        # tens of milliseconds per move.  See TICK_WARNING_LIMIT.
        if spec.draw_ticks:
            sld.setTickPosition(QSlider.TickPosition.TicksBelow)
            sld.setTickInterval(1)
        else:
            sld.setTickPosition(QSlider.TickPosition.NoTicks)
        readout = QLabel()
        readout.setMinimumWidth(40)

        sld.valueChanged.connect(
            lambda _position, ax=axis: self._on_discrete_slider_changed(ax)
        )
        row_layout.addWidget(sld, stretch=1)
        row_layout.addWidget(readout)
        self._sliders[axis] = sld
        self._readouts[axis] = readout
        return row

    def _on_discrete_slider_changed(self, axis: int) -> None:
        self._update_readout(axis)
        self._on_slider_changed(axis, self._world_value(axis))

    def _update_readout(self, axis: int) -> None:
        spec = self._axis_values[axis]
        position = self._sliders[axis].value()
        if spec.labels is not None:
            text = spec.labels[position]
        else:
            text = f"{spec.values[position]:g}"
        self._readouts[axis].setText(text)

    def _world_value(self, axis: int) -> float:
        """The world position the slider for *axis* currently selects."""
        spec = self._axis_values[axis]
        if isinstance(spec, DiscreteAxisValues):
            return spec.values[self._sliders[axis].value()]
        return self._sliders[axis].value()

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────────

    def _set_value(self, axis: int, value: float) -> None:
        """Show world position *value* on *axis*'s slider.

        A discrete axis shows the nearest listed value.  Nothing is written
        back when *value* falls between two: the renderer resolves it with
        the same rule, so the slider and the view already agree.
        """
        spec = self._axis_values[axis]
        sld = self._sliders[axis]
        sld.blockSignals(True)
        if isinstance(spec, DiscreteAxisValues):
            sld.setValue(nearest_value_index(spec.values, value))
        else:
            sld.setValue(value)
        sld.blockSignals(False)
        if isinstance(spec, DiscreteAxisValues):
            self._update_readout(axis)

    # ── Visibility helper ────────────────────────────────────────────────────

    def _update_visibility(self, displayed_axes: tuple[int, ...]) -> None:
        self._displayed_axes = tuple(displayed_axes)
        layout: QFormLayout = self._container.layout()
        for axis, row in self._rows.items():
            visible = axis not in displayed_axes and (
                self._slider_axes is None or axis in self._slider_axes
            )
            layout.setRowVisible(row, visible)


class QtCanvasWidget:
    """Wraps a render canvas above a ``QtDimsControl`` panel.

    Composes the two elements in a ``QVBoxLayout`` so that the canvas expands
    to fill available space while the dims control sits below it at a fixed
    height.  The dims control includes the 2D/3D toggle button (when the
    scene has 3+ axes), so no separate toggle widget is needed.

    Prefer constructing via :meth:`from_scene_and_canvas` rather than calling
    ``__init__`` directly.

    Parameters
    ----------
    canvas_view :
        A ``CanvasView`` instance; its ``.widget`` property provides the render
        surface to embed.
    dims_control :
        An already-constructed ``QtDimsControl`` instance.
    parent :
        Optional Qt parent widget.
    """

    def __init__(
        self,
        canvas_view,
        dims_control: QtDimsControl,
        *,
        parent: QWidget | None = None,
    ) -> None:
        self._dims_control = dims_control

        self._container = QWidget(parent)
        self._container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        canvas_qt_widget = canvas_view.widget
        canvas_qt_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(canvas_qt_widget, stretch=1)
        layout.addWidget(dims_control.widget)

    @classmethod
    def from_scene_and_canvas(
        cls,
        scene,
        canvas_view,
        axis_values: Mapping[int, AxisValues],
        *,
        parent: QWidget | None = None,
    ) -> QtCanvasWidget:
        """Construct from live scene and canvas objects.

        Derives axis labels and the initial dims state from *scene* so that
        callers only need to supply *axis_values* (which requires data-store
        knowledge not available on the dims model itself).  The 2D/3D toggle
        is included automatically when the scene has 3 or more axes: 3D
        displays the last three axis labels, 2D the last two.

        Call ``controller.connect_widget`` on the returned widget's
        ``dims_control`` after construction to wire subscriptions.

        Parameters
        ----------
        scene :
            The live ``Scene`` object whose dims this panel controls.
        canvas_view :
            The ``CanvasView`` whose ``.widget`` is the render surface.
        axis_values :
            Mapping of axis index to that axis's slider values, e.g.
            ``{0: ContinuousAxisValues(min=0, max=99)}``.
        parent :
            Optional Qt parent widget.
        """
        axis_values = coerce_axis_values(axis_values)
        axis_labels_list = scene.dims.axis_labels
        axis_labels = dict(enumerate(axis_labels_list))

        selection = scene.dims.selection
        initial_slice_indices = seed_slice_indices(selection, axis_values)
        initial_displayed_axes = getattr(selection, "displayed_axes", ())

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

        dims_control = QtDimsControl(
            scene_id=scene.id,
            axis_values=axis_values,
            axis_labels=axis_labels,
            initial_slice_indices=initial_slice_indices,
            initial_displayed_axes=initial_displayed_axes,
            slider_axes=scene.slider_axes,
            axes_2d=axes_2d,
            axes_3d=axes_3d,
            parent=parent,
        )
        return cls(canvas_view=canvas_view, dims_control=dims_control, parent=parent)

    # ── Public interface ─────────────────────────────────────────────────────

    def compose(self, host) -> object:
        """Hand this widget to *host* as a center leaf.

        Qt composes canvas-over-dims internally, so unlike
        ``AnywidgetCanvasView.compose`` there is nothing to arrange here.
        """
        return host.leaf(self)

    @property
    def widget(self) -> QWidget:
        """The outer ``QWidget`` to insert into a layout."""
        return self._container

    @property
    def dims_control(self) -> QtDimsControl:
        """The ``QtDimsControl`` panel embedded below the canvas."""
        return self._dims_control

    def close(self) -> None:
        """Unsubscribe the dims control from the bus."""
        self._dims_control.close()
