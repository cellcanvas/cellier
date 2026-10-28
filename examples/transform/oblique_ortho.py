# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cellier[pyside]",
# ]
# ///

"""Demonstration of oblique slicing in an orthoviewer.

This is a rotated multiscale volume in an orthoviewer. It is
rotated such that the axis-aligned slicing in the world coordinate
system results in oblique slicing in the data's coordinate system.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np

from cellier.convenience import (
    AppearanceControls,
    ImageControl,
    Layout,
    OrthoViewer,
    axis_values_from_ortho,
)
from cellier.convenience.gui import build_ortho_grid_widget
from cellier.data import LinesMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals import (
    CenteredAxes2D,
    CenteredAxes2DAppearance,
    LinesMemoryAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    PlaneOutline,
)

# (z, y, x) shape of the finest level.
SHAPE = (96, 128, 128)
# Starting rotation of the ``(z, x)`` plane, degrees.
ANGLE = 25.0
# Half-thickness of the ``xy`` panel's projection slab, world units.
SLAB = 12.0
# Length of each world axis in the 3D view, world units.
AXIS_LENGTH = 128.0
# Colour of each world axis, in the 3D view and in the 2D indicators.
AXIS_COLORS = {"x": (1.0, 0.3, 0.3), "y": (0.3, 0.9, 0.3), "z": (0.3, 0.55, 1.0)}
# Outline colours of the 3D view's planes for the xy, xz and yz panels.
PLANE_COLORS = ((1.0, 0.8, 0.2, 1.0), (0.3, 0.9, 1.0, 1.0), (1.0, 0.4, 0.8, 1.0))
# The 2D panels.
PANELS_2D = ("xy", "xz", "yz")
# The canvas background color
BACKGROUND = (0.4, 0.4, 0.4, 1.0)


def _tubes(shape: tuple[int, int, int]) -> np.ndarray:
    """A few bright tubes along y, at different depths, on a dim background."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    volume = np.full(shape, 0.05, dtype=np.float32)
    for depth, column, radius in ((30, 40, 6), (48, 64, 8), (66, 90, 5), (48, 20, 4)):
        tube = (z - depth) ** 2 + (x - column) ** 2 <= radius**2
        volume[np.broadcast_to(tube, shape)] = 0.6 + 0.1 * (radius % 3)
    volume += 0.3 * (np.sin(y / 9.0) > 0.7)
    return volume


def _write_pyramid(root: Path, volume: np.ndarray) -> MultiscaleZarrDataStore:
    """Write three levels (strides 1, 2, 4) as zarr v3 and open them."""
    import tensorstore as ts

    levels = [volume, volume[::2, ::2, ::2], volume[::4, ::4, ::4]]
    for index, level in enumerate(levels):
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
                "metadata": {
                    "shape": list(level.shape),
                    "data_type": "float32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [32, 32, 32]},
                    },
                },
                "create": True,
                "delete_existing": True,
            }
        ).result()[...].write(np.ascontiguousarray(level)).result()
    return MultiscaleZarrDataStore(
        zarr_path=str(root),
        scale_names=["s0", "s1", "s2"],
        level_scales=[(1.0,) * 3, (2.0,) * 3, (4.0,) * 3],
        level_translations=[(0.0,) * 3, (0.0,) * 3, (0.0,) * 3],
        name="tubes",
    )


def rotation(store, world, degrees: float) -> AffineTransform:
    """The ``data -> world`` turn of ``(z, x)`` by *degrees*, about the centre."""
    theta = np.deg2rad(degrees)
    linear = np.eye(3)
    linear[0, 0] = linear[2, 2] = np.cos(theta)
    linear[0, 2], linear[2, 0] = -np.sin(theta), np.sin(theta)
    centre = (np.asarray(SHAPE, float) - 1.0) / 2.0
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = centre - linear @ centre
    return AffineTransform.from_matrix(matrix, store.data_coordinate_systems[0], world)


def _add_world_axes(viewer: OrthoViewer) -> None:
    """World x (red), y (green) and z (blue) from the origin, in the 3D view."""
    # Vertex pairs in (z, y, x) order: one segment per axis.
    positions = np.zeros((6, 3), dtype=np.float32)
    colors = np.zeros((6, 4), dtype=np.float32)
    for axis, name in enumerate(("z", "y", "x")):
        positions[2 * axis + 1, axis] = AXIS_LENGTH
        colors[2 * axis : 2 * axis + 2] = (*AXIS_COLORS[name], 1.0)
    viewer.controller.add_lines(
        LinesMemoryStore(positions=positions, colors=colors, name="world axes"),
        viewer.scenes["vol"].id,
        # After the volume, depth-tested: hidden where a plane or the
        # volume is in front.  A MIP writes no depth, so drawn after the axes
        # it would paint over them.
        LinesMemoryAppearance(color_mode="vertex", thickness=3.0, render_order=1),
        name="world_axes",
    )


def _add_axis_indicators(viewer: OrthoViewer) -> None:
    """An indicator of its two world axes in each 2D panel's corner.

    A canvas overlay belongs to a canvas, so this runs once the panels have
    theirs.  The indicator follows the panel's camera; its directions are
    pygfx's: a 2D panel draws its second displayed axis along x and its
    first along y.
    """
    controller = viewer.controller
    for key in PANELS_2D:
        scene = viewer.scenes[key]
        names = scene.dims.world_coordinate_system.axis_names()
        up, right = (names[axis] for axis in scene.dims.selection.displayed_axes)
        for canvas_id in controller.get_canvas_ids(scene.id):
            controller.add_canvas_overlay(
                canvas_id,
                CenteredAxes2D(
                    name=f"axes_{key}",
                    axis_a_direction=(0.0, 1.0, 0.0),
                    axis_a_label=up,
                    axis_b_direction=(1.0, 0.0, 0.0),
                    axis_b_label=right,
                    appearance=CenteredAxes2DAppearance(
                        axis_a_color=(*AXIS_COLORS[up], 1.0),
                        axis_b_color=(*AXIS_COLORS[right], 1.0),
                        corner="bottom_left",
                        corner_offset_px=(30.0, 30.0),
                    ),
                ),
            )


def build(gui: str = "offscreen") -> OrthoViewer:
    """The populated orthoviewer (the capture CLI's entry point)."""
    viewer = OrthoViewer(spatial_axes("z", "y", "x"), gui=gui)
    viewer.set_background(BackgroundAppearance(mode="uniform", color=BACKGROUND))
    store = _write_pyramid(Path(tempfile.mkdtemp()), _tubes(SHAPE))
    viewer.controller.add_data_store(store)
    scene = viewer.scenes["xy"]
    viewer.controller._ensure_data_coordinate_systems(scene.id, store)
    world = scene.dims.world_coordinate_system
    visuals = viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(coarsest_while_moving_2d=True),
        render_config=MultiscaleImageRenderConfig(gpu_budget_bytes=256 * 1024**2),
        # "plane" acts in the 3D panel alone: it draws the slice planes.
        single=MultiscaleImageSingleAppearance(
            color_map="magma", clim=(0.0, 1.0), render_mode="plane"
        ),
        transform=rotation(store, world, ANGLE),
        name="tubes",
        controls=[ImageControl(fields=["render_mode"])],
    )
    # The xy panel projects along z through its own slab.
    xy = visuals["xy"]
    xy.appearance.slab_projection = "mip"
    xy.appearance.projection_axes = ("z",)
    viewer.controller.update_thickness(scene.id, {0: SLAB})
    # The 3D view: the volume's box, turned with it, and the world axes.
    visuals["vol"].aabb.enabled = True
    _add_world_axes(viewer)
    # Whole voxels, not center_slices()'s voxel faces (47.5, 63.5): a plane
    # drawn with nearest interpolation on a voxel face picks the voxel on
    # either side pixel by pixel, and looks speckled.
    viewer.set_slice_positions({axis: float(n // 2) for axis, n in enumerate(SHAPE)})
    # The 3D view's planes follow the 2D panels' slices, outlined.
    viewer.plane_controller.mode = "slices"
    viewer.plane_controller.outlines = tuple(
        PlaneOutline(enabled=True, color=color) for color in PLANE_COLORS
    )
    if gui == "offscreen":
        # The capture CLI would give the panels canvases after build(); give
        # them here, so the axis indicators have canvases to go on.
        for scene in viewer.scenes.values():
            viewer.controller.add_canvas(scene_id=scene.id, canvas_size=(600, 600))
        _add_axis_indicators(viewer)
    viewer.tubes = visuals  # what main() turns
    viewer.tubes_store = store
    return viewer


def _rotation_slider(viewer: OrthoViewer):
    """A dock with a slider that turns every panel's copy, as one drag."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget, QLabel, QSlider, QVBoxLayout, QWidget

    controller = viewer.controller
    visuals = viewer.tubes
    store = viewer.tubes_store
    worlds = {key: viewer.scenes[key].dims.world_coordinate_system for key in visuals}
    source = uuid4()

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(-90, 90)
    slider.setValue(int(ANGLE))
    label = QLabel(f"(z, x) rotation: {ANGLE:.0f} degrees")

    def _begin() -> None:
        for visual in visuals.values():
            controller.begin_transform_interaction(visual.id, source_id=source)

    def _end() -> None:
        for visual in visuals.values():
            controller.end_transform_interaction(visual.id, source_id=source)

    def _turn(degrees: int) -> None:
        label.setText(f"(z, x) rotation: {degrees} degrees")
        for key, visual in visuals.items():
            visual.transform = rotation(store, worlds[key], float(degrees))

    slider.sliderPressed.connect(_begin)
    slider.sliderReleased.connect(_end)
    slider.valueChanged.connect(_turn)

    body = QWidget()
    layout = QVBoxLayout(body)
    layout.addWidget(label)
    layout.addWidget(slider)
    dock = QDockWidget("Rotation")
    dock.setWidget(body)
    return dock


def main() -> None:
    """Show the orthoviewer, the render-mode control and the rotation slider."""
    from qtpy.QtCore import Qt

    from cellier.convenience import launch
    from cellier.convenience.layout._qt_renderer import render_qt

    viewer = build(gui="qt")
    canvases = build_ortho_grid_widget(viewer, axis_values_from_ortho(viewer))
    _add_axis_indicators(viewer)
    window = render_qt(Layout(center=canvases, right_dock=AppearanceControls()), viewer)
    window.addDockWidget(
        Qt.DockWidgetArea.BottomDockWidgetArea, _rotation_slider(viewer)
    )
    launch(viewer, window, fit="ready")


if __name__ == "__main__":
    main()
