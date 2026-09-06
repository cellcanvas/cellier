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

from uuid import uuid4

from psygnal import Signal
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFormLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledSlider

from cellier.events import DimsChangedEvent, DimsUpdateEvent, SubscriptionSpec
from cellier.gui._constants import DIMS_SLIDER_THROTTLE_MS
from cellier.gui._dims import initial_slice_indices as seed_slice_indices

SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: 1px solid #bbb;
    background: white;
    height: 10px;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #eee, stop:1 #ccc);
    border: 1px solid #777;
    width: 13px;
    margin-top: -7px;
    margin-bottom: -7px;
    border-radius: 4px;
}

QSlider::add-page:horizontal {
    background: #fff;
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
        stop: 0 #66e, stop: 1 #bbf);
    background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
        stop: 0 #bbf, stop: 1 #55f);
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}

QSlider::handle:horizontal:hover {
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
    ``QLabeledSlider`` per axis, plus (when *axes_2d*/*axes_3d* are given) a
    toggle button that switches the scene between its 2D and 3D axis sets.
    Sliders for displayed axes are hidden; only sliced (non-displayed) axes
    are shown.

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

        control = QtDimsControl(scene_id, axis_ranges=..., axis_labels=...)
        controller.connect_widget(
            control, subscription_specs=control.subscription_specs()
        )

    Parameters
    ----------
    scene_id :
        UUID of the scene whose slice indices this widget controls.
    axis_ranges :
        Mapping of axis index to ``(min, max)`` for each slider.
    axis_labels :
        Mapping of axis index to display label, e.g. ``{0: "z", 1: "y", 2: "x"}``.
    initial_slice_indices :
        Starting slider values; typically ``scene.dims.selection.slice_indices``.
    initial_displayed_axes :
        Axes to hide initially; typically ``scene.dims.selection.displayed_axes``.
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
        axis_ranges: dict[int, tuple[int, int]],
        axis_labels: dict[int, str],
        *,
        initial_slice_indices: dict[int, int] | None = None,
        initial_displayed_axes: tuple[int, ...] = (),
        initial_stacked_axes: tuple[int, ...] = (),
        non_displayed_sliders: set[int] | None = None,
        debounce_ms: int | None = None,
        axes_2d: tuple[int, ...] | None = None,
        axes_3d: tuple[int, ...] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # ── Cellier layer ────────────────────────────────────────────────────
        self._id = uuid4()
        self._scene_id = scene_id
        self._non_displayed_sliders: set[int] = non_displayed_sliders or set()
        self._stacked_axes: tuple[int, ...] = initial_stacked_axes
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

        self._sliders: dict[int, QLabeledSlider] = {}
        self._displayed_axes: tuple[int, ...] = initial_displayed_axes
        _initial = initial_slice_indices or {}

        for axis, (min_val, max_val) in axis_ranges.items():
            sld = QLabeledSlider(Qt.Orientation.Horizontal)
            sld.setRange(min_val, max_val)
            sld.setValue(_initial.get(axis, min_val))
            # Capture `axis` by value in the default-argument closure.
            sld.valueChanged.connect(
                lambda value, ax=axis: self._on_slider_changed(ax, value)
            )
            layout.addRow(axis_labels.get(axis, str(axis)), sld)
            self._sliders[axis] = sld

        self._toggle_button: QPushButton | None = None
        if axes_2d is not None and axes_3d is not None:
            is_3d = len(initial_displayed_axes) == 3
            self._toggle_button = QPushButton(
                "Switch to 2D" if is_3d else "Switch to 3D"
            )
            self._toggle_button.clicked.connect(self._on_toggle_click)
            layout.addRow(self._toggle_button)

        self._update_visibility(initial_displayed_axes, initial_stacked_axes)

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
    def non_displayed_sliders(self) -> set[int]:
        """Axes excluded from slider display regardless of dims state (e.g. channel axis)."""
        return self._non_displayed_sliders

    @non_displayed_sliders.setter
    def non_displayed_sliders(self, axes: set[int]) -> None:
        self._non_displayed_sliders = axes
        self._update_visibility(self._displayed_axes)

    def current_index(self) -> dict[int, int]:
        """Return the current value of every slider regardless of visibility."""
        return {axis: sld.value() for axis, sld in self._sliders.items()}

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
            )
        ]

    # ── Cellier layer: model → widget ────────────────────────────────────────

    def _on_dims_changed(self, event) -> None:
        if event.source_id == self._id:
            return  # echo from our own slider move or toggle click; ignore

        sel = event.dims_state.selection

        # Update slider values for sliced axes.
        for axis in self._sliders:
            value = sel.slice_indices.get(axis)
            if value is not None:
                self._set_value(axis, value)

        # Show/hide sliders based on which axes are now displayed or stacked.
        stacked = getattr(sel, "stacked_axes", ())
        self._update_visibility(sel.displayed_axes, stacked)

        # Relabel the toggle purely from the event -- this is what lets it
        # stay correct even when displayed_axes changed via some other
        # caller, not just this widget's own button.
        if self._toggle_button is not None:
            is_3d = len(sel.displayed_axes) == 3
            self._toggle_button.setText("Switch to 2D" if is_3d else "Switch to 3D")

    # ── Cellier layer: widget → model ────────────────────────────────────────

    def _on_slider_changed(self, axis: int, value: int) -> None:
        self._slider_dirty = True
        if not self._rate_limit_timer.isActive():
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _on_rate_limit_tick(self) -> None:
        if self._slider_dirty:
            self._submit_slider_values()
            self._rate_limit_timer.start()

    def _submit_slider_values(self) -> None:
        """Submit current slider values for all sliced (non-displayed) axes."""
        self._slider_dirty = False
        updates = {
            axis: sld.value()
            for axis, sld in self._sliders.items()
            if axis not in self._displayed_axes
            and axis not in self._non_displayed_sliders
            and axis not in self._stacked_axes
        }
        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=updates,
                displayed_axes=None,
            )
        )

    def _on_toggle_click(self) -> None:
        is_3d = len(self._displayed_axes) == 3
        target_displayed = self._axes_2d if is_3d else self._axes_3d
        target_set = set(target_displayed)

        # current_index() already holds a live, correct value for every
        # axis (including hidden ones, since Qt widgets retain their value
        # while hidden) -- no separate "saved position" bookkeeping needed.
        new_slices = {
            axis: value
            for axis, value in self.current_index().items()
            if axis not in target_set and axis not in self._stacked_axes
        }
        # Applied *before* the emit, not after.  The controller echoes this
        # change back stamped with our own source_id, so _on_dims_changed's
        # filter ignores it -- the widget has to move itself either way.
        # Doing it first is what keeps the control honest when something
        # downstream of the emit fails (``plans/gui_backend_seam.md`` D17).
        for axis, value in new_slices.items():
            self._set_value(axis, value)
        self._update_visibility(target_displayed, self._stacked_axes)
        self._displayed_axes = tuple(target_displayed)
        self._toggle_button.setText("Switch to 2D" if not is_3d else "Switch to 3D")

        self.changed.emit(
            DimsUpdateEvent(
                source_id=self._id,
                scene_id=self._scene_id,
                slice_indices=new_slices,
                displayed_axes=target_displayed,
            )
        )

    # ── Qt seam 2: push value without re-firing valueChanged ─────────────────

    def _set_value(self, axis: int, value: int) -> None:
        sld = self._sliders[axis]
        sld.blockSignals(True)
        sld.setValue(value)
        sld.blockSignals(False)

    # ── Visibility helper ────────────────────────────────────────────────────

    def _update_visibility(
        self,
        displayed_axes: tuple[int, ...],
        stacked_axes: tuple[int, ...] = (),
    ) -> None:
        self._displayed_axes = displayed_axes
        self._stacked_axes = stacked_axes
        layout: QFormLayout = self._container.layout()
        for axis, sld in self._sliders.items():
            visible = (
                axis not in displayed_axes
                and axis not in self._non_displayed_sliders
                and axis not in stacked_axes
            )
            layout.setRowVisible(sld, visible)


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
        axis_ranges: dict[int, tuple[int, int]],
        *,
        parent: QWidget | None = None,
    ) -> QtCanvasWidget:
        """Construct from live scene and canvas objects.

        Derives axis labels and the initial dims state from *scene* so that
        callers only need to supply *axis_ranges* (which requires data-store
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
        axis_ranges :
            Mapping of axis index to ``(min, max)``, e.g.
            ``{0: (0, 99), 1: (0, 511), 2: (0, 511)}``.
        parent :
            Optional Qt parent widget.
        """
        axis_labels_list = scene.dims.coordinate_system.axis_labels
        axis_labels = dict(enumerate(axis_labels_list))

        selection = scene.dims.selection
        initial_slice_indices = seed_slice_indices(selection, axis_ranges)
        initial_displayed_axes = getattr(selection, "displayed_axes", ())
        initial_stacked_axes = getattr(selection, "stacked_axes", ())

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
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=initial_slice_indices,
            initial_displayed_axes=initial_displayed_axes,
            initial_stacked_axes=initial_stacked_axes,
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
