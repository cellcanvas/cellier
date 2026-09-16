"""Tests for the anywidget (notebook) front-end.

The whole module is skipped when ``rendercanvas.anywidget`` (and the
``anywidget`` package) is not importable, so CI without the optional extra
still passes.  These tests never construct a ``QApplication``.
"""

from __future__ import annotations

from uuid import uuid4

import pygfx as gfx
import pytest

from cellier.convenience._backend import ANYWIDGET_BACKEND
from cellier.convenience._hosts import _AnywidgetDockPanel
from cellier.convenience.layout._shared import APPEARANCE_DOCK_GAP_PX
from cellier.convenience.layout._walk import build_appearance_widgets

rendercanvas_anywidget = pytest.importorskip("rendercanvas.anywidget")

from cellier.controller import CellierController  # noqa: E402
from cellier.convenience.gui import InMemoryImageControlsConfig  # noqa: E402
from cellier.render.canvas_view import CanvasView  # noqa: E402
from cellier.scene.dims import (  # noqa: E402
    AxisAlignedSelection,
    DimsManager,
    spatial_axes,
    world_coordinate_system,
)
from cellier.scene.scene import Scene  # noqa: E402


def _make_scene(
    displayed_axes: tuple[int, ...],
    render_modes: set[str] | None = None,
) -> Scene:
    """Return a minimal Scene with the given displayed_axes and render_modes."""
    n_axes = max(displayed_axes) + 1
    axis_labels = tuple(f"axis_{i}" for i in range(n_axes))
    cs = world_coordinate_system(spatial_axes(*axis_labels), name="world")
    slice_indices = dict.fromkeys(range(n_axes), 0)
    dims = DimsManager(
        world_coordinate_system=cs,
        selection=AxisAlignedSelection(
            displayed_axes=displayed_axes,
            slice_indices=slice_indices,
        ),
    )
    resolved_modes = render_modes if render_modes is not None else {"2d", "3d"}
    return Scene(name="test_scene", dims=dims, render_modes=resolved_modes)


# ---------------------------------------------------------------------------
# Task 3 - canvas backend seam
# ---------------------------------------------------------------------------


def test_canvas_view_anywidget_creates_anywidget_canvas():
    """CanvasView(gui='anywidget') builds a rendercanvas anywidget canvas."""
    scene = gfx.Scene()
    canvas_view = CanvasView(
        uuid4(), uuid4(), get_scene_fn=lambda _sid: scene, dim="2d", gui="anywidget"
    )
    assert isinstance(canvas_view.widget, rendercanvas_anywidget.RenderCanvas)


def test_canvas_view_unknown_gui_raises():
    """An unknown gui value is rejected."""
    scene = gfx.Scene()
    with pytest.raises(ValueError, match="Unknown gui"):
        CanvasView(
            uuid4(), uuid4(), get_scene_fn=lambda _sid: scene, dim="2d", gui="tk"
        )


# ---------------------------------------------------------------------------
# Task 4 - gui threaded through the controller
# ---------------------------------------------------------------------------


def test_controller_gui_anywidget_creates_anywidget_canvas():
    """A gui='anywidget' controller builds anywidget canvases via add_canvas."""
    controller = CellierController(gui="anywidget")
    scene = controller.add_scene_model(_make_scene(displayed_axes=(1, 2)))

    controller.add_canvas(scene_id=scene.id)

    canvas_id = controller.get_canvas_ids(scene.id)[-1]
    canvas_view = controller.get_canvas_view(canvas_id)
    assert isinstance(canvas_view.widget, rendercanvas_anywidget.RenderCanvas)


def test_controller_default_gui_is_qt():
    """The controller defaults to the Qt gui."""
    controller = CellierController()
    assert controller._gui == "qt"


# ---------------------------------------------------------------------------
# Task 5 - Viewer / OrthoViewer accept gui
# ---------------------------------------------------------------------------


def test_viewer_gui_property():
    """Viewer exposes the chosen gui and threads it to the controller."""
    from cellier.convenience import Viewer

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    assert viewer.gui == "anywidget"
    assert viewer.controller._gui == "anywidget"


def test_ortho_viewer_gui_property():
    """OrthoViewer exposes the chosen gui and threads it to the controller."""
    from cellier.convenience import OrthoViewer

    viewer = OrthoViewer(spatial_axes("z", "y", "x"), gui="anywidget")
    assert viewer.gui == "anywidget"
    assert viewer.controller._gui == "anywidget"


def test_viewer_default_gui_is_qt():
    """Viewer defaults to the Qt gui."""
    from cellier.convenience import Viewer

    assert Viewer(spatial_axes("z", "y", "x")).gui == "qt"


# ---------------------------------------------------------------------------
# Task 6 - AnywidgetDimsPanel (includes the 2D/3D toggle)
# ---------------------------------------------------------------------------


def _dims_changed_event(source_id, scene_id, *, displayed, slices):
    from cellier._state import AxisAlignedSelectionState, DimsState
    from cellier.events import DimsChangedEvent

    selection = AxisAlignedSelectionState(displayed_axes=displayed)
    state = DimsState(axis_labels=("z", "y", "x"), selection=selection)
    return DimsChangedEvent(
        source_id=source_id,
        scene_id=scene_id,
        dims_state=state,
        displayed_axes_changed=False,
        slice_indices=dict(slices),
    )


def _make_dummy_anywidget():
    """A trivial, argument-free anywidget for host-composition tests."""
    from cellier.gui.anywidget import AnywidgetDatasetInfo

    return AnywidgetDatasetInfo()


def _make_dims_panel(*, with_toggle=False):
    from cellier.gui._axis_values import ContinuousAxisValues
    from cellier.gui.anywidget import AnywidgetDimsPanel

    return AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_values={
            0: ContinuousAxisValues(min=0.0, max=99.0),
            1: ContinuousAxisValues(min=0.0, max=511.0),
            2: ContinuousAxisValues(min=0.0, max=511.0),
        },
        axis_labels={0: "z", 1: "y", 2: "x"},
        slice_indices={0: 0, 1: 0, 2: 0},
        displayed_axes=(1, 2),
        axes_2d=(1, 2) if with_toggle else None,
        axes_3d=(0, 1, 2) if with_toggle else None,
    )


def test_panel_on_dims_changed_updates_without_emitting():
    """A model-driven dims change updates the trait but does not echo."""
    panel = _make_dims_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    event = _dims_changed_event(
        source_id=uuid4(),
        scene_id=panel._scene_id,
        displayed=(1, 2),
        slices={0: 42},
    )
    panel._on_dims_changed(event)

    assert panel.slice_indices["0"] == 42
    assert emitted == []  # programmatic write is guarded, no re-emit


def test_panel_echo_filtered_by_source_id():
    """A DimsChangedEvent originating from the panel itself is ignored."""
    panel = _make_dims_panel()
    event = _dims_changed_event(
        source_id=panel._id,
        scene_id=panel._scene_id,
        displayed=(1, 2),
        slices={0: 7},
    )
    panel._on_dims_changed(event)
    assert "0" not in panel.slice_indices or panel.slice_indices["0"] == 0


def test_panel_user_change_emits_dims_update_event():
    """A user-driven trait change emits a DimsUpdateEvent for sliced axes only."""
    from cellier.events import DimsUpdateEvent

    panel = _make_dims_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    # Simulate the JS save_changes write (string keys, all axes present).
    panel.slice_indices = {"0": 12, "1": 0, "2": 0}

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, DimsUpdateEvent)
    assert event.source_id == panel._id
    assert event.scene_id == panel._scene_id
    # Only the sliced axis 0 is reported; displayed axes 1, 2 are excluded.
    assert event.slice_indices == {0: 12}


def test_panel_without_toggle_axes_has_no_toggle():
    """A panel built without axes_2d/axes_3d omits the toggle."""
    panel = _make_dims_panel(with_toggle=False)
    assert panel.has_toggle is False
    assert panel.label == ""


def test_dims_panel_toggle_emits_dims_update_event():
    """Clicking the toggle emits a DimsUpdateEvent, not a direct model mutation."""
    from cellier.events import DimsUpdateEvent

    panel = _make_dims_panel(with_toggle=True)
    emitted = []
    panel.changed.connect(emitted.append)

    # Simulate a JS click (increment the synced counter).
    panel._clicks += 1

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, DimsUpdateEvent)
    assert event.source_id == panel._id
    assert event.displayed_axes == (0, 1, 2)
    assert event.slice_indices is None
    # The controller stamps its echoed DimsChangedEvent with our own
    # source_id, so it would be swallowed by the echo filter -- the panel
    # must relabel itself directly from the click, not by waiting for it.
    assert panel.label == "Switch to 2D"
    assert list(panel.displayed_axes) == [0, 1, 2]


def test_dims_panel_toggle_sends_only_displayed_axes():
    """The toggle never carries positions: the model keeps every one (D36)."""
    panel = _make_dims_panel(with_toggle=True)
    emitted = []
    panel.changed.connect(emitted.append)

    panel.slice_indices = {"0": 42, "1": 0, "2": 0}
    assert emitted[-1].slice_indices == {0: 42.0}
    emitted.clear()

    panel._clicks += 1  # 2D -> 3D
    assert emitted[-1].displayed_axes == (0, 1, 2)
    assert emitted[-1].slice_indices is None

    panel._clicks += 1  # 3D -> 2D
    assert emitted[-1].displayed_axes == (1, 2)
    assert emitted[-1].slice_indices is None
    assert panel.slice_indices["0"] == 42


def test_dims_panel_toggle_relabels_from_every_dims_event():
    """The toggle label follows the model, including our own echo.

    An echo carries the model's displayed axes, which is exactly what the
    label should describe; only slider values are skipped on an echo.
    """
    panel = _make_dims_panel(with_toggle=True)
    assert panel.label == "Switch to 3D"

    event = _dims_changed_event(
        source_id=uuid4(),
        scene_id=panel._scene_id,
        displayed=(0, 1, 2),
        slices={},
    )
    panel._on_dims_changed(event)
    assert panel.label == "Switch to 2D"

    echo_event = _dims_changed_event(
        source_id=panel._id,
        scene_id=panel._scene_id,
        displayed=(1, 2),
        slices={0: 99},
    )
    panel._on_dims_changed(echo_event)
    assert panel.label == "Switch to 3D"
    assert panel.slice_indices["0"] == 0


# ---------------------------------------------------------------------------
# Task 7 - LayoutHost seam
# ---------------------------------------------------------------------------


def test_resolve_host_jupyter():
    """resolve_host('jupyter') returns a JupyterHost."""
    pytest.importorskip("IPython")
    from cellier.convenience._hosts import JupyterHost, resolve_host

    assert isinstance(resolve_host("jupyter"), JupyterHost)


def test_resolve_host_marimo():
    """resolve_host('marimo') returns a MarimoHost."""
    pytest.importorskip("marimo")
    from cellier.convenience._hosts import MarimoHost, resolve_host

    assert isinstance(resolve_host("marimo"), MarimoHost)


def test_resolve_host_unknown_raises():
    """An unknown explicit host that is not running raises."""
    from cellier.convenience._hosts import resolve_host

    with pytest.raises(RuntimeError, match="No anywidget host detected"):
        resolve_host("not-a-host")


def test_jupyter_host_stack_builds_anywidget_box():
    """JupyterHost.stack composes its leaves into an AnywidgetBox."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.gui.anywidget import AnywidgetBox

    host = JupyterHost()
    widget_a = _make_dummy_anywidget()
    widget_b = _make_dummy_anywidget()
    box = host.stack([host.leaf(widget_a), host.leaf(widget_b)], direction="v")
    assert isinstance(box, AnywidgetBox)
    assert box.direction == "v"
    assert list(box.children) == [widget_a, widget_b]


def test_jupyter_host_grid_builds_nested_anywidget_box():
    """JupyterHost.grid nests horizontal AnywidgetBoxes inside a vertical one."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.gui.anywidget import AnywidgetBox

    host = JupyterHost()
    a, b, c, d = (_make_dummy_anywidget() for _ in range(4))
    grid = host.grid([[a, b], [c, d]])
    assert isinstance(grid, AnywidgetBox)
    assert grid.direction == "v"
    assert all(
        isinstance(row, AnywidgetBox) and row.direction == "h" for row in grid.children
    )


# ---------------------------------------------------------------------------
# Task 8 - builders return anywidget leaves with compose(host)
# ---------------------------------------------------------------------------


def _image_viewer():
    """A small anywidget Viewer with one in-memory image and a data store."""
    import numpy as np

    from cellier.convenience import Viewer, axis_values_from_viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore

    blobs = np.zeros((8, 8, 8), dtype=np.float32)
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="anywidget")
    store = ImageMemoryStore(data=blobs, name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, single={"color_map": "viridis", "clim": (0.0, 1.0)})
    return viewer, axis_values_from_viewer(viewer)


def _image_viewer_with_controls():
    """A small anywidget Viewer with one in-memory image and an appearance panel."""
    import numpy as np

    from cellier.convenience import Viewer, axis_values_from_viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore

    blobs = np.zeros((8, 8, 8), dtype=np.float32)
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="2d", gui="anywidget")
    store = ImageMemoryStore(data=blobs, name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(
        store,
        appearance={"color_map": "viridis", "clim": (0.0, 1.0)},
        controls=InMemoryImageControlsConfig(appearance=["color_map", "clim"]),
    )
    return viewer, axis_values_from_viewer(viewer)


def test_build_canvas_widget_anywidget_returns_view():
    """build_canvas_widget returns an AnywidgetCanvasView; no controls by default."""
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    assert isinstance(view, AnywidgetCanvasView)
    assert isinstance(view.canvas, rendercanvas_anywidget.RenderCanvas)
    # The view is canvas + dims only; appearance controls live in a dock.
    assert not hasattr(view, "controls")
    assert callable(view.compose)

    composed = view.compose(JupyterHost())
    from cellier.gui.anywidget import AnywidgetBox

    assert isinstance(composed, AnywidgetBox)
    assert composed.direction == "v"


def test_build_canvas_widget_anywidget_with_controls():
    """A registered controls config does not change the canvas view.

    Panel building belongs to the renderer, from a dock spec.  The canvas
    view composes identically whether or not a config is registered, which is
    what makes the removed in-canvas path (section 7.2) purely dead weight.
    """
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer_with_controls()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    assert isinstance(view, AnywidgetCanvasView)
    assert not hasattr(view, "controls")

    composed = view.compose(JupyterHost())
    from cellier.gui.anywidget import AnywidgetBox

    assert isinstance(composed, AnywidgetBox)
    # One column: canvas above dims.
    assert composed.direction == "v"


def test_build_canvas_widget_gui_conflict_raises():
    """A gui that conflicts with viewer.gui is rejected."""
    from cellier.convenience.gui import build_canvas_widget

    viewer, ranges = _image_viewer()
    with pytest.raises(ValueError, match="conflicts with viewer"):
        build_canvas_widget(viewer, ranges, gui="qt")


def test_build_canvas_widget_defaults_gui_from_viewer():
    """Omitting gui defaults to viewer.gui."""
    from cellier.convenience.gui import AnywidgetCanvasView, build_canvas_widget

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges)
    assert isinstance(view, AnywidgetCanvasView)


def test_build_ortho_grid_anywidget_returns_canvases():
    """build_ortho_grid_widget(gui='anywidget') returns a composable grid."""
    import numpy as np

    from cellier.convenience import OrthoViewer, axis_values_from_ortho
    from cellier.convenience._hosts import JupyterHost
    from cellier.convenience.gui import OrthoAnywidgetCanvases, build_ortho_grid_widget
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.gui.anywidget import AnywidgetBox

    viewer = OrthoViewer(spatial_axes("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32), name="blobs")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, single={"color_map": "viridis", "clim": (0.0, 1.0)})
    ranges = axis_values_from_ortho(viewer)

    grid = build_ortho_grid_widget(viewer, ranges, gui="anywidget")
    assert isinstance(grid, OrthoAnywidgetCanvases)
    assert set(grid.canvases) == {"xy", "xz", "yz", "vol"}

    composed = grid.compose(JupyterHost())
    assert isinstance(composed, AnywidgetBox)


# ---------------------------------------------------------------------------
# Task 9 - display() launcher
# ---------------------------------------------------------------------------


class _Node:
    """A composable node that can hold an attached close() (unlike a tuple)."""

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload


class _FakeHost(_AnywidgetDockPanel):
    """An imperative LayoutHost (Jupyter-like): renders in present, returns None.

    Inherits the real ``dock_panel`` rather than restating it: a double that
    reimplements the thing under test drifts from it silently, which is how an
    earlier version of this class ended up spacing dock columns differently
    from every real host.
    """

    def __init__(self):
        from cellier.convenience._backend import ANYWIDGET_BACKEND

        self.presented = None
        # The walk reads widget classes off the host's backend.
        self.backend = ANYWIDGET_BACKEND

    def leaf(self, widget):
        return _Node("leaf", widget)

    def stack(
        self,
        items,
        *,
        direction="v",
        align=None,
        min_width=None,
        gap=None,
        title=None,
    ):
        return _Node("stack", (direction, list(items), align, min_width, gap))

    def grid(self, rows):
        return _Node("grid", [list(r) for r in rows])

    def present(self, root):
        self.presented = root
        return None  # imperative: rendered as a side effect


class _FakeReturnValueHost(_FakeHost):
    """A return-value LayoutHost (marimo-like): present returns the renderable."""

    def present(self, root):
        self.presented = root
        return root


def test_display_imports_from_convenience():
    """display, Layout, AppearanceControls are importable."""
    from cellier.convenience import AppearanceControls, Layout, display

    assert callable(display)
    assert Layout is not None
    assert AppearanceControls is not None


def test_display_center_only_presents_and_returns_inert_handle(monkeypatch):
    """Layout with no docks: presents just the center, returns an inert handle."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience._launch import DisplayHandle
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    handle = _launch.display(viewer, Layout(center=view), fit="none")

    # present() was called exactly once.
    assert fake.presented is not None
    # No docks: root is just the canvas+dims v-stack from compose().
    assert fake.presented.kind == "stack"
    _direction, leaves, _align, _min_width, _gap = fake.presented.payload
    assert _direction == "v"
    assert len(leaves) == 2  # canvas leaf + dims leaf

    # display() returns an inert handle (not the renderable) for Jupyter.
    assert isinstance(handle, DisplayHandle)
    assert handle._repr_mimebundle_() == {}

    # close() is idempotent.
    handle.close()
    handle.close()


def test_display_left_dock_stacks_controls_beside_center(monkeypatch):
    """AppearanceControls in left_dock produces a h-stack: [panel, center]."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import AppearanceControls, Layout

    viewer, ranges = _image_viewer_with_controls()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, Layout(center=view, left_dock=AppearanceControls()), fit="none"
    )

    # Middle h-stack: [panel_stack, center_composed]
    presented = fake.presented
    assert presented.kind == "stack"
    direction, leaves, _align, _min_width, _gap = presented.payload
    assert direction == "h"
    assert len(leaves) == 2
    panel_node, center_node = leaves
    # The dock is a live slot, leafed once, so it can follow the viewer; the
    # controls are the slot's children rather than a host stack.  Two
    # appearance fields (color_map, clim) plus the always-on AABB widget.
    from cellier.gui.anywidget import AnywidgetSlot

    assert panel_node.kind == "leaf"
    slot = panel_node.payload
    assert isinstance(slot, AnywidgetSlot)
    assert len(slot.children) == 2
    # An explicit gap groups the split sub-widgets, distinct from the host's
    # default macro-layout spacing.  Read from the shared constant rather than
    # restated: it is the same number the Qt dock column spaces by, which is
    # the point of it living there.
    assert slot.gap == APPEARANCE_DOCK_GAP_PX
    assert center_node.kind == "stack"  # canvas+dims v-stack


def test_display_return_value_host_yields_renderable(monkeypatch):
    """A return-value host (marimo-like) gets the renderable back from display."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import AppearanceControls, Layout

    viewer, ranges = _image_viewer_with_controls()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeReturnValueHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    result = _launch.display(
        viewer, Layout(center=view, left_dock=AppearanceControls()), fit="none"
    )

    # Return-value host: display() yields the renderable so marimo renders it.
    assert result is fake.presented
    # teardown is attached best-effort to the renderable.
    assert callable(result.close)
    result.close()


def test_run_dispatches_to_display_for_anywidget(monkeypatch):
    """run() with gui='anywidget' calls display() and returns a DisplayHandle."""
    from cellier.convenience import _hosts, _launch
    from cellier.convenience._launch import DisplayHandle
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    view = build_canvas_widget(viewer, ranges, gui="anywidget")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    result = _launch.run(viewer, Layout(center=view), fit="none")

    assert fake.presented is not None
    assert isinstance(result, DisplayHandle)


def test_run_importable_from_convenience():
    """run is importable from cellier.convenience."""
    from cellier.convenience import run

    assert callable(run)


def test_layout_single_preset_no_docks():
    """Layout.single with no dock args produces a center-only layout."""
    from cellier.convenience.layout import Layout

    viewer, ranges = _image_viewer()
    from cellier.convenience.gui import build_canvas_widget

    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    layout = Layout.single(view)
    assert layout.center is view
    assert layout.left_dock is None
    assert layout.right_dock is None
    assert layout.top_dock is None
    assert layout.bottom_dock is None


def test_layout_single_preset_with_docks():
    """Layout.single wires appearance to the requested dock."""
    from cellier.convenience.layout import AppearanceControls, Layout

    viewer, ranges = _image_viewer()
    from cellier.convenience.gui import build_canvas_widget

    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    layout = Layout.single(view, appearance="left")
    assert isinstance(layout.left_dock, AppearanceControls)
    assert layout.bottom_dock is None
    assert layout.right_dock is None
    assert layout.top_dock is None


# ---------------------------------------------------------------------------
# Phase 2 - appearance controls (split gui.anywidget.visuals sub-widgets)
# ---------------------------------------------------------------------------


def _make_lod_bias_slider(**kwargs):
    from cellier.gui.anywidget.visuals import AnywidgetLodBiasSlider

    defaults = {"initial_lod_bias": 1.0}
    defaults.update(kwargs)
    return AnywidgetLodBiasSlider(uuid4(), **defaults)


def _make_aabb_widget(**kwargs):
    from cellier.gui.anywidget.visuals import AnywidgetAABBWidget

    defaults = {
        "initial_enabled": False,
        "initial_line_width": 2.0,
        "initial_color": "#ffffff",
    }
    defaults.update(kwargs)
    return AnywidgetAABBWidget(uuid4(), **defaults)


def _appearance_changed_event(source_id, visual_id, field, value):
    from cellier.events import AppearanceChangedEvent

    return AppearanceChangedEvent(
        source_id=source_id,
        visual_id=visual_id,
        field_name=field,
        new_value=value,
        requires_reslice=False,
    )


def test_lod_bias_slider_user_change_emits_appearance_update():
    """A user-driven lod_bias change emits AppearanceUpdateEvent."""
    from cellier.events import AppearanceUpdateEvent

    slider = _make_lod_bias_slider()
    emitted = []
    slider.changed.connect(emitted.append)

    slider.lod_bias = 2.0

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AppearanceUpdateEvent)
    assert event.field == "lod_bias"
    assert event.value == 2.0


def _aabb_changed_event(source_id, visual_id, field, value):
    from cellier.events import AABBChangedEvent

    return AABBChangedEvent(
        source_id=source_id,
        visual_id=visual_id,
        field_name=field,
        new_value=value,
    )


def test_aabb_widget_on_aabb_changed_updates_trait_without_emitting():
    """A model-driven AABBChangedEvent updates the trait but does not echo."""
    widget = _make_aabb_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    event = _aabb_changed_event(
        source_id=uuid4(),
        visual_id=widget._visual_id,
        field="enabled",
        value=True,
    )
    widget._on_aabb_changed(event)

    assert widget.enabled is True
    assert emitted == []


def test_aabb_widget_user_enabled_change_emits_aabb_update():
    """A user-driven enabled change emits AABBUpdateEvent."""
    from cellier.events import AABBUpdateEvent

    widget = _make_aabb_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget.enabled = True

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AABBUpdateEvent)
    assert event.source_id == widget._id
    assert event.visual_id == widget._visual_id
    assert event.field == "enabled"
    assert event.value is True


def test_aabb_widget_user_color_change_emits_aabb_update():
    """A user-driven color change emits AABBUpdateEvent with field 'color'."""
    from cellier.events import AABBUpdateEvent

    widget = _make_aabb_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget.color = "#ff0000"

    assert len(emitted) == 1
    event = emitted[0]
    assert isinstance(event, AABBUpdateEvent)
    assert event.field == "color"
    assert event.value == "#ff0000"


def test_build_appearance_widgets_anywidget_from_visual():
    """Builds one widget per spec, plus the always-on AABB.

    Names come from the shared ``appearance_specs`` layer so the two front
    ends name the same control the same way (stage 1, section 7.3), and each
    widget carries its own -- ``label`` on a single-field control, ``title``
    on a multi-row one (``plans/label_ownership_unification.md``).
    """
    import numpy as np

    from cellier.convenience import Viewer
    from cellier.convenience.gui._controls_config import InMemoryImageControlsConfig
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetImageControls,
    )

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32), name="t")
    viewer.controller.add_data_store(store)
    viewer.add_image(store, single={"color_map": "viridis", "clim": (0.0, 1.0)})

    visual = viewer.scene.visuals[0]
    controls_config = InMemoryImageControlsConfig(appearance=["color_map", "clim"])

    widgets = build_appearance_widgets(
        visual, controls_config, viewer.controller, None, backend=ANYWIDGET_BACKEND
    )

    assert [w.title for w in widgets] == ["Image", "Bounding box"]

    built = widgets
    # anywidget dynamically subclasses each widget at construction time, so
    # compare via isinstance rather than exact type equality.
    assert isinstance(built[0], AnywidgetImageControls)
    assert built[0].single["clim"] == [0.0, 1.0]
    # AABB is always wired alongside whenever the visual has one, regardless
    # of whether "aabb" was requested in the appearance field list -- this
    # mirrors ControlPanel's previous (pre-split) behaviour.
    assert isinstance(built[1], AnywidgetAABBWidget)


def test_renderer_builds_appearance_widgets_for_configured_visual(monkeypatch):
    """The anywidget renderer creates the split appearance widgets in the left dock."""
    import numpy as np

    from cellier.convenience import Viewer, _hosts, _launch, axis_values_from_viewer
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout import AppearanceControls, Layout
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.events import AABBChangedEvent, AppearanceChangedEvent
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetImageControls,
    )

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="anywidget")
    store = ImageMemoryStore(data=np.zeros((4, 4, 4), dtype=np.float32), name="t")
    viewer.controller.add_data_store(store)
    viewer.add_image(
        store,
        single={"color_map": "viridis", "clim": (0.0, 1.0)},
        controls=InMemoryImageControlsConfig(appearance=["color_map", "clim"]),
    )

    ranges = axis_values_from_viewer(viewer)
    view = build_canvas_widget(viewer, ranges, gui="anywidget")
    # The canvas view is canvas + dims only: the in-canvas controls column was
    # unreachable through the public API and is gone (section 7.2).
    assert not hasattr(view, "controls")

    fake = _FakeHost()
    monkeypatch.setattr(_hosts, "resolve_host", lambda host=None: fake)

    _launch.display(
        viewer, Layout(center=view, left_dock=AppearanceControls()), fit="none"
    )

    # The renderer placed a v-stack of sub-widgets as the left leaf: the image
    # control and the AABB (always wired alongside when the visual has one).
    presented = fake.presented
    _direction, leaves, _align, _min_width, _gap = presented.payload
    panel_node = leaves[0]
    assert panel_node.kind == "leaf"
    slot = panel_node.payload
    assert slot.gap == APPEARANCE_DOCK_GAP_PX  # shared with the Qt dock column
    widgets = list(slot.children)
    assert any(isinstance(w, AnywidgetImageControls) for w in widgets)
    assert any(isinstance(w, AnywidgetAABBWidget) for w in widgets)

    event_types = set()
    for w in widgets:
        event_types.update(s.event_type for s in w.subscription_specs())
    assert AppearanceChangedEvent in event_types
    assert AABBChangedEvent in event_types
