"""The two layout renderers produce the same thing from the same spec.

``test_render_controls_parity.py`` pins that both front ends draw the same
*controls*; this pins that they compose them the same way.  Every assertion
here failed when it was written -- see ``plans/gui_backend_unification.md``
section 2.1 -- so each one names a real difference between what a Qt user and
a notebook user saw from identical code.

The two renderers cannot share a construction path (one builds ``QWidget``s,
the other serialises to traits), so nothing but a test can hold them
together.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.layout._walk import render_dock
from cellier.data._dataset_info import DatasetInfo, MatrixSection, RowSection

_MESH_POSITIONS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
)
_MESH_INDICES = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
_RANGES = {0: (0.0, 2.0), 1: (0.0, 2.0), 2: (0.0, 2.0)}


def _mesh_store():
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore

    return MeshMemoryStore(positions=_MESH_POSITIONS, indices=_MESH_INDICES)


def _viewer(gui: str, *, configure_controls: bool = False):
    """A one-mesh viewer, with or without a recorded ``controls=`` config."""
    from cellier.convenience import MeshControlsConfig, Viewer
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    viewer = Viewer(("z", "y", "x"), dim="3d", gui=gui)
    viewer.add_mesh(
        _mesh_store(),
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=True) if configure_controls else None,
    )
    return viewer


# ---------------------------------------------------------------------------
# D1: closing a window releases what the renderer built -- all of it
# ---------------------------------------------------------------------------


def test_closing_a_qt_window_closes_the_center_too(qtbot):
    """The dock is not the only thing the renderer built.

    ``render_qt`` threaded its ``closeables`` list into the docks and not into
    the center, so ``QtCanvasWidget.close()`` was never called and the dims
    control stayed subscribed to the bus for the life of the controller.  The
    anywidget renderer closes its center leaves, which is why only Qt leaked.

    Asserted on the dims control specifically rather than on a subscription
    count, because a count shrinking says only that *something* was released.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.convenience import AppearanceControls, Layout
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = _viewer("qt", configure_controls=True)
    canvas = build_canvas_widget(viewer, _RANGES)
    window = render_qt(Layout(center=canvas, left_dock=AppearanceControls()), viewer)

    dims_id = canvas.dims_control._id
    assert dims_id in _subscribed_owners(viewer.controller), "sanity: it subscribed"

    window.close()

    assert dims_id not in _subscribed_owners(viewer.controller)


def _subscribed_owners(controller) -> set:
    return {
        getattr(subscription, "owner_id", None)
        for subscriptions in controller._outgoing_events._subs.values()
        for subscription in subscriptions
    }


# ---------------------------------------------------------------------------
# D2: a Grid cell holds its column on both toolkits
# ---------------------------------------------------------------------------


class _FakeLeaf:
    """A center leaf: ``compose(host)`` plus the ``.widget`` a host unwraps."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._label = None

    def compose(self, host):
        return host.leaf(self)

    @property
    def widget(self):
        from qtpy.QtWidgets import QLabel

        if self._label is None:
            self._label = QLabel(self.name)
        return self._label


class _RecordingHost:
    """A ``LayoutHost`` that records the tree instead of building widgets."""

    @property
    def backend(self):
        from cellier.convenience._backend import ANYWIDGET_BACKEND

        return ANYWIDGET_BACKEND

    def dock_panel(self, widgets, *, title=None):
        return (title, list(widgets))

    def leaf(self, widget):
        # Records the leaf by name, so a spec's shape is readable in a plain
        # assertion instead of as a tree of widget objects.
        return getattr(widget, "name", widget)

    def stack(self, items, *, direction="v", **kwargs):
        return (f"stack-{direction}", list(items))

    def grid(self, rows):
        return ("grid", [list(row) for row in rows])

    def present(self, root):
        return root


def test_the_walk_keeps_an_empty_grid_cell_empty():
    """``None`` leaves a cell empty; it does not close the gap.

    One walk serves both toolkits now, so this pins the walk rather than
    comparing two of them -- and what it pins is that ``None`` reaches the
    host, instead of being filtered out before the host can place the hole.
    """
    from cellier.convenience.layout._spec import Grid
    from cellier.convenience.layout._walk import render_center

    grid = Grid(cells=[[_FakeLeaf("a"), None], [None, _FakeLeaf("b")]])
    kind, rows = render_center(grid, _RecordingHost(), [])

    assert kind == "grid"
    assert rows == [["a", None], [None, "b"]]


@pytest.mark.parametrize("host_name", ["qt", "jupyter"])
def test_every_host_puts_the_same_cell_in_the_same_column(qtbot, host_name):
    """The part that is still two implementations: ``LayoutHost.grid``.

    Sharing the walk removed the divergence this file was written to catch,
    which also removed the thing the old test compared.  The hosts are where
    two implementations still meet one contract, so the assertion moves down
    a level rather than being deleted as "covered".
    """
    pytest.importorskip("qtpy")
    from cellier.convenience.layout._spec import Grid
    from cellier.convenience.layout._walk import render_center

    if host_name == "qt":
        from qtpy.QtWidgets import QApplication

        from cellier.convenience._hosts import QtLayoutHost

        QApplication.instance() or QApplication([])
        host = QtLayoutHost()
        leaf_a, leaf_b = _FakeLeaf("a"), _FakeLeaf("b")
    else:
        pytest.importorskip("ipywidgets")
        from cellier.convenience._hosts import JupyterHost

        host = JupyterHost()
        leaf_a, leaf_b = _LabelLeaf("a"), _LabelLeaf("b")

    container = render_center(Grid(cells=[[leaf_a, None], [None, leaf_b]]), host, [])
    assert _cell_positions(container) == {"a": (0, 0), "b": (1, 1)}


def _cell_positions(container) -> dict:
    """``{name: (row, column)}`` for either host's grid container."""
    from qtpy.QtWidgets import QGridLayout

    layout = getattr(container, "layout", None)
    layout = layout() if callable(layout) else None
    if isinstance(layout, QGridLayout):
        return {
            layout.itemAt(i).widget().text(): layout.getItemPosition(i)[:2]
            for i in range(layout.count())
        }
    # JupyterHost: an outer AnywidgetBox of per-row AnywidgetBoxes, where an
    # empty cell is a childless box holding its column.
    positions = {}
    for row_index, row_box in enumerate(container.children):
        for column_index, cell in enumerate(row_box.children):
            name = getattr(cell, "description", "")
            if name:
                positions[name] = (row_index, column_index)
    return positions


class _LabelLeaf:
    """An anywidget center leaf that composes to a named ipywidget."""

    def __init__(self, name: str) -> None:
        self.name = name

    def compose(self, host):
        import ipywidgets

        return host.leaf(ipywidgets.Label(description=self.name))


# ---------------------------------------------------------------------------
# D3: a dock whose contents all resolve to nothing produces no dock
# ---------------------------------------------------------------------------


def test_a_dock_stack_that_builds_nothing_produces_no_dock(qtbot):
    """``AppearanceControls`` on an unconfigured viewer builds nothing.

    Bare, both renderers already agreed -- the spec resolves to ``None`` and
    the dock is skipped.  Wrapped in a stack they did not: the Qt branch
    returned its container unconditionally, so an empty ``QDockWidget``
    titled "Left" appeared beside the canvas.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from qtpy.QtWidgets import QDockWidget

    from cellier.convenience import AppearanceControls, Layout, VStack
    from cellier.convenience.gui import build_canvas_widget
    from cellier.convenience.layout._qt_renderer import render_qt

    spec = VStack(items=[AppearanceControls()])

    qt_viewer = _viewer("qt")  # no controls= -> nothing to build
    window = render_qt(
        Layout(center=build_canvas_widget(qt_viewer, _RANGES), left_dock=spec),
        qt_viewer,
    )
    assert window.findChildren(QDockWidget) == []

    anywidget_viewer = _viewer("anywidget")
    assert render_dock(spec, anywidget_viewer, _RecordingHost(), []) is None


# ---------------------------------------------------------------------------
# D4: dataset-info sections are drawn in the order the store declared them
# ---------------------------------------------------------------------------


def _sample_dataset_info() -> DatasetInfo:
    """The section shape ``dataset_info_from_path`` actually builds.

    A matrix between two labelled row sections is the case that separated the
    two toolkits, and it is what every multiscale OME-Zarr store produces.
    """
    return DatasetInfo(
        sections=[
            RowSection(None, [("Store", "zarr")]),
            RowSection("Axes", [("z", "um")]),
            MatrixSection(
                "World to data",
                np.eye(2),
                row_labels=["z", "1"],
                col_labels=["z", "1"],
            ),
            RowSection("Scale levels", [("0", "10x10")], collapsed=True),
        ]
    )


_EXPECTED_SECTION_ORDER = ["Store", "Axes", "World to data", "Scale levels"]


def test_both_front_ends_draw_the_sections_in_declaration_order(qtbot):
    """Qt hoisted matrices above labelled row sections.

    Unlabelled rows and every matrix went into a top-level form, while
    labelled row sections became nested collapsibles appended after it -- so
    a store's declared order survived only when it happened to match that
    split.  For the pyramid above it did not: the transform matrix jumped
    over "Axes", and every multiscale OME-Zarr panel read differently
    depending on the toolkit.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.qt import QtDatasetInfo

    info = _sample_dataset_info()
    qt_labels = QtDatasetInfo.from_info(info).section_labels()
    anywidget_labels = AnywidgetDatasetInfo.from_info(info).section_labels()

    assert qt_labels == anywidget_labels == _EXPECTED_SECTION_ORDER


# ---------------------------------------------------------------------------
# D5: the render dock names its scope on both toolkits
# ---------------------------------------------------------------------------


def test_both_render_docks_carry_the_dock_title(qtbot):
    """ "Outline" beside "Outlines" is not a distinction anyone should notice.

    The Qt dock wraps its panels in a group box titled "Renderer effects" so
    the global settings cannot be mistaken for the per-visual ones docked on
    the other side of the canvas.  The notebook dock stacked its panels bare,
    so the notebook had exactly the confusion the heading exists to prevent.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.convenience.layout._spec import RenderControls
    from cellier.gui._render_controls import RENDER_DOCK_TITLE

    spec = RenderControls(sections=("outline",))

    qt_dock = render_dock(spec, _viewer("qt"), QtLayoutHost(), [])
    assert qt_dock.title() == RENDER_DOCK_TITLE

    anywidget_dock = render_dock(spec, _viewer("anywidget"), _TitleRecordingHost(), [])
    assert anywidget_dock[0] == RENDER_DOCK_TITLE


class _TitleRecordingHost(_RecordingHost):
    """Kept as a name for the dock-title case; the base already records it."""


# ---------------------------------------------------------------------------
# D11: both front ends coalesce slider drags at the same rate
# ---------------------------------------------------------------------------


def test_both_dims_panels_throttle_at_the_same_interval(qtbot):
    """The interval is what a user feels, so it cannot be declared twice.

    Qt throttles with a ``QTimer`` and anywidget with a ``setTimeout`` loop in
    ``dims_panel.js`` -- two implementations, which is fine, over what used to
    be two independent 50s, one in Python and one in JavaScript where no test
    could reach it.  The JS now reads a synced trait instead.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("superqt")
    from cellier.gui._constants import DIMS_SLIDER_THROTTLE_MS
    from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel
    from cellier.gui.qt._scene import QtDimsControl

    qt_dims = QtDimsControl(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 4.0)},
        axis_labels={0: "z"},
        initial_slice_indices={0: 0},
    )
    qtbot.addWidget(qt_dims.widget)
    anywidget_dims = AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_ranges={0: (0.0, 4.0)},
        axis_labels={0: "z"},
        slice_indices={0: 0},
    )

    assert qt_dims._rate_limit_timer.interval() == DIMS_SLIDER_THROTTLE_MS
    assert anywidget_dims.throttle_ms == DIMS_SLIDER_THROTTLE_MS


def test_the_dims_javascript_reads_the_interval_rather_than_declaring_one():
    """A hard-coded constant in the ESM is unreachable from any Python test."""
    from cellier.gui.anywidget import _dims_panel

    source = (_dims_panel._STATIC / "dims_panel.js").read_text()
    assert 'model.get("throttle_ms")' in source
    assert "THROTTLE_MS = 50" not in source


# ---------------------------------------------------------------------------
# D15: the ortho panels are labelled on every toolkit
# ---------------------------------------------------------------------------


def _ortho_viewer(gui: str):
    import cmap

    from cellier.convenience import OrthoViewer
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    viewer = OrthoViewer(("z", "y", "x"), gui=gui)
    viewer.add_image(
        ImageMemoryStore(data=np.random.rand(8, 8, 8).astype(np.float32)),
        appearance=InMemoryImageAppearance(color_map=cmap.Colormap("gray")),
    )
    return viewer


def test_both_toolkits_label_the_ortho_panels(qtbot):
    """``XY`` / ``XZ`` / ``YZ`` / ``3D`` over the four panels, on both.

    Qt drew these from a private ``_PANEL_LAYOUT`` and anywidget drew none of
    them, so the same four-panel viewer was annotated in a script and bare in
    a notebook.  One grid class reads one table now; this pins that neither
    side can quietly stop drawing them.
    """
    pytest.importorskip("qtpy")
    pytest.importorskip("ipywidgets")
    from qtpy.QtWidgets import QApplication, QLabel

    from cellier.convenience._hosts import JupyterHost, QtLayoutHost
    from cellier.convenience.gui import PANEL_LAYOUT, build_ortho_grid_widget

    expected = sorted(header for _row, _col, _key, header in PANEL_LAYOUT)

    QApplication.instance() or QApplication([])
    qt_grid = build_ortho_grid_widget(
        _ortho_viewer("qt"), {0: (0.0, 8.0), 1: (0.0, 8.0), 2: (0.0, 8.0)}
    )
    composed = qt_grid.compose(QtLayoutHost())
    qt_headers = sorted(
        label.text()
        for label in composed.findChildren(QLabel)
        if label.text() in set(expected)
    )

    anywidget_grid = build_ortho_grid_widget(
        _ortho_viewer("anywidget"), {0: (0.0, 8.0), 1: (0.0, 8.0), 2: (0.0, 8.0)}
    )
    anywidget_headers = sorted(_box_titles(anywidget_grid.compose(JupyterHost())))

    assert qt_headers == anywidget_headers == expected


def _box_titles(box) -> list[str]:
    """Every non-empty ``title`` in an ``AnywidgetBox`` tree."""
    titles = []
    if getattr(box, "title", ""):
        titles.append(box.title)
    for child in getattr(box, "children", []):
        titles.extend(_box_titles(child))
    return titles


# ---------------------------------------------------------------------------
# D16 / D17 / D18: the 2D/3D toggle
# ---------------------------------------------------------------------------


def _toggle_viewer(gui: str):
    """A single-scene 3D viewer -- one that legitimately offers a toggle."""
    import cmap

    from cellier.convenience import Viewer
    from cellier.data.image._image_memory_store import ImageMemoryStore
    from cellier.visuals import InMemoryImageAppearance

    viewer = Viewer(("z", "y", "x"), dim="3d", gui=gui)
    viewer.add_image(
        ImageMemoryStore(data=np.random.rand(8, 8, 8).astype(np.float32)),
        appearance=InMemoryImageAppearance(color_map=cmap.Colormap("gray")),
    )
    return viewer


def _toggled_to_2d(gui: str):
    """Toggle a 3D viewer's panel to 2D.

    Returns ``(selection, displayed, label, centre)`` -- the scene's selection
    after the toggle, what the panel believes it is showing, the button text,
    and the index a centred slice on the hidden axis should have.
    """
    from cellier.convenience import axis_ranges_from_viewer
    from cellier.convenience.gui import build_canvas_widget

    viewer = _toggle_viewer(gui)
    axis_ranges = axis_ranges_from_viewer(viewer)
    low, high = axis_ranges[0]
    centre = round((low + high) / 2.0)

    # Held, not dropped: the canvas widget owns the Qt sliders, and letting it
    # be collected deletes them out from under the control.
    canvas = build_canvas_widget(viewer, axis_ranges)
    panel = canvas.dims_control
    if gui == "anywidget":
        viewer.controller.connect_widget(
            panel, subscription_specs=panel.subscription_specs()
        )
        panel._on_toggle_click(None)
        displayed, label = tuple(panel.displayed_axes), panel.label
    else:
        panel._on_toggle_click()
        displayed, label = tuple(panel._displayed_axes), panel._toggle_button.text()
    return viewer.scene.dims.selection, displayed, label, centre


@pytest.mark.parametrize("gui", ["qt", "anywidget"])
async def test_the_toggle_hands_the_slicer_a_centred_index(qtbot, gui):
    """Switching a 3D scene to 2D must name a slice for the axis it hides.

    A scene showing all three axes carries no ``slice_indices`` at all, so the
    toggle had nothing to carry forward.  Qt read its sliders and got their
    unset minimum (the volume's edge); anywidget read the model and got
    nothing, so the slicer raised ``KeyError``.  Both now seed every axis from
    :func:`cellier.gui._dims.initial_slice_indices`.
    """
    pytest.importorskip("qtpy")
    if gui == "anywidget":
        pytest.importorskip("ipywidgets")

    selection, _displayed, _label, centre = _toggled_to_2d(gui)

    assert tuple(selection.displayed_axes) == (1, 2)
    # The centre of the hidden axis, not its edge -- derived from the axis
    # range rather than restated, so the volume's size stays a detail.
    assert dict(selection.slice_indices) == {0: centre}


@pytest.mark.parametrize("gui", ["qt", "anywidget"])
async def test_the_toggle_leaves_the_panel_agreeing_with_the_scene(qtbot, gui):
    """The button and sliders must describe the mode the scene is actually in.

    Both toggles emitted first and updated themselves after, so a raising
    handler downstream left the scene in 2D while the panel still showed 3D --
    no slider for the axis it had just hidden, and a button offering to do
    what it had already done.
    """
    pytest.importorskip("qtpy")
    if gui == "anywidget":
        pytest.importorskip("ipywidgets")

    selection, displayed, label, _centre = _toggled_to_2d(gui)

    assert displayed == tuple(selection.displayed_axes)
    assert label == "Switch to 3D"


@pytest.mark.parametrize("gui", ["qt", "anywidget"])
def test_only_a_scene_that_renders_both_ways_offers_a_toggle(qtbot, gui):
    """An ortho panel renders one way, so it must not offer the other.

    Each ``OrthoViewer`` scene declares a single ``render_modes`` entry --
    three slice views and one volume.  The toggle was offered on axis count
    alone, so pressing it put the scene into a mode it was never configured
    for: the reslice produced no geometry, the scene had no bounds, and the
    panel went blank.
    """
    pytest.importorskip("qtpy")
    if gui == "anywidget":
        pytest.importorskip("ipywidgets")
    from cellier.convenience import axis_ranges_from_ortho, axis_ranges_from_viewer
    from cellier.convenience.gui import build_canvas_widget, build_ortho_grid_widget

    ortho = _ortho_viewer(gui)
    grid = build_ortho_grid_widget(ortho, axis_ranges_from_ortho(ortho))
    for key, view in grid.canvases.items():
        modes = {str(mode) for mode in ortho.scenes[key].render_modes}
        assert len(modes) == 1, f"sanity: {key} declares one render mode"
        assert not view.dims_control.has_toggle, key

    # ...while a viewer whose scene renders both ways keeps its toggle.
    viewer = _toggle_viewer(gui)
    canvas = build_canvas_widget(viewer, axis_ranges_from_viewer(viewer))
    assert {"2d", "3d"} <= {str(m) for m in viewer.scene.render_modes}
    assert canvas.dims_control.has_toggle
