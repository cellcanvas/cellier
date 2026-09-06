"""3D brick-shader viewer for real OME-Zarr files with 2D/3D toggle.

Opens a Qt viewer for any OME-Zarr multiscale volume using the
Kiln-style brick cache shader.  Physical scale and level transforms are
read automatically from the OME metadata.

Usage
-----
Launch the viewer::

    uv run scripts/v2/debug_multiscale_volume_brick_ome.py \\
        --zarr-file-path /path/to/my.ome.zarr

Remote stores are also supported::

    uv run scripts/v2/debug_multiscale_volume_brick_ome.py \\
        --zarr-file-path s3://my-bucket/my.ome.zarr

Controls
--------
- **Toggle 2D / 3D** button — switch between a 2D Z-slice view and the
  full 3D volume renderer.
- **Reslice now** button — manually trigger a brick re-request.
- **Render mode** dropdown *(3D only)* — switch between ISO, MIP, and attenuated MIP.
  Colormap auto-switches to viridis for MIP and grays for ISO.
- **Attenuation** slider *(3D, attenuated_mip only)* — depth attenuation coefficient.
- **Pipeline mode** radios *(3D only)* — toggle between full LOD +
  frustum culling and a flat "load all bricks at the coarsest level" mode.
- **ISO threshold** spinner *(3D only)* — adjust the isosurface threshold.
- **Z slice** spinner *(2D only)* — select the displayed Z plane.
- **Contrast limits** range slider — adjust the lower and upper contrast limits (both views).
- **LOD bias** spinner — scale the screen-space LOD thresholds (both views).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np


class OmeBrickViewer:
    """Main window for the OME-Zarr brick validation viewer."""

    def __init__(
        self,
        controller,
        scene,
        visual_model,
        canvas_widget: QtCanvasWidget,
        zarr_uri: str,
        n_levels: int,
        z_depth: int,
        clim_range: tuple[float, float],
        slider_decimals: int = 2,
    ):
        from PySide6 import QtCore, QtWidgets

        from cellier.gui.qt import QtDatasetInfo
        from cellier.gui.qt.visuals import (
            QtAABBWidget,
            QtClimRangeSlider,
            QtColormapCombo,
            QtLodBiasSlider,
            QtVolumeRenderControls,
        )

        self._controller = controller
        self._scene = scene
        self._visual_model = visual_model
        self._canvas_widget = canvas_widget
        self._dims_control = canvas_widget.dims_control
        self._n_levels = n_levels
        self._z_max = z_depth - 1
        self._active_mode = "3d"

        self._clim_slider = QtClimRangeSlider(
            visual_model.id,
            clim_range=clim_range,
            initial_clim=visual_model.appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._clim_slider, subscription_specs=self._clim_slider.subscription_specs()
        )
        self._colormap_combo = QtColormapCombo(
            visual_model.id,
            initial_colormap=visual_model.appearance.color_map,
        )
        controller.connect_widget(
            self._colormap_combo,
            subscription_specs=self._colormap_combo.subscription_specs(),
        )
        self._aabb_widget = QtAABBWidget(
            visual_model.id,
            initial_enabled=visual_model.aabb.enabled,
            initial_line_width=visual_model.aabb.line_width,
            initial_color=visual_model.aabb.color,
        )
        controller.connect_widget(
            self._aabb_widget, subscription_specs=self._aabb_widget.subscription_specs()
        )
        self._render_controls = QtVolumeRenderControls(
            visual_model.id,
            dtype_max=clim_range[1],
            initial_render_mode=visual_model.appearance.render_mode,
            initial_threshold=visual_model.appearance.iso_threshold,
            initial_attenuation=visual_model.appearance.attenuation,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._render_controls,
            subscription_specs=self._render_controls.subscription_specs(),
        )
        self._metadata_widget = QtDatasetInfo.from_path(zarr_uri)

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("MultiscaleVolumeBrick — OME-Zarr viewer")
        self._window.resize(1200, 750)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        # ── Canvas + dims sliders ─────────────────────────────────────
        root_layout.addWidget(canvas_widget.widget, stretch=1)

        # ── Side panel ────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)

        # Track 3D-only widgets for show/hide on toggle.
        self._widget_3d: list = []

        # ── Dataset metadata ──────────────────────────────────────────
        panel_layout.addWidget(self._metadata_widget.widget)

        # ── Shared: toggle + reslice + mode label ─────────────────────
        self._toggle_btn = QtWidgets.QPushButton("Toggle 2D / 3D")
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        panel_layout.addWidget(self._toggle_btn)

        self._reslice_btn = QtWidgets.QPushButton("Reslice now")
        self._reslice_btn.clicked.connect(self._on_reslice_clicked)
        panel_layout.addWidget(self._reslice_btn)

        self._mode_label = QtWidgets.QLabel("Mode: 3D")
        panel_layout.addWidget(self._mode_label)

        # ── 3D-only: pipeline mode ────────────────────────────────────
        mode_group = QtWidgets.QGroupBox("Pipeline mode")
        mode_layout = QtWidgets.QVBoxLayout(mode_group)
        self._lod_radio = QtWidgets.QRadioButton("LOD + frustum culling")
        coarsest = n_levels - 1
        self._flat_radio = QtWidgets.QRadioButton(f"Level {coarsest} all bricks (flat)")
        self._lod_radio.setChecked(True)
        self._lod_radio.toggled.connect(self._on_pipeline_mode_toggle)
        mode_layout.addWidget(self._lod_radio)
        mode_layout.addWidget(self._flat_radio)
        panel_layout.addWidget(mode_group)
        self._widget_3d.append(mode_group)

        # ── 3D-only: render mode + ISO threshold ─────────────────────
        render_group = QtWidgets.QGroupBox("Render")
        render_layout = QtWidgets.QVBoxLayout(render_group)
        render_layout.addWidget(self._render_controls.widget)
        panel_layout.addWidget(render_group)
        self._widget_3d.append(render_group)

        # ── Shared: colormap ──────────────────────────────────────────
        cmap_group = QtWidgets.QGroupBox("Colormap")
        cmap_layout = QtWidgets.QVBoxLayout(cmap_group)
        cmap_layout.addWidget(self._colormap_combo.widget)
        panel_layout.addWidget(cmap_group)

        # ── Shared: contrast limits ───────────────────────────────────
        clim_group = QtWidgets.QGroupBox("Contrast limits")
        clim_layout = QtWidgets.QVBoxLayout(clim_group)
        clim_layout.addWidget(self._clim_slider.widget)
        panel_layout.addWidget(clim_group)

        # ── Shared: LOD bias ──────────────────────────────────────────
        self._lod_bias_slider = QtLodBiasSlider(
            visual_model.id,
            initial_lod_bias=visual_model.appearance.lod_bias,
        )
        controller.connect_widget(
            self._lod_bias_slider,
            subscription_specs=self._lod_bias_slider.subscription_specs(),
        )
        bias_group = QtWidgets.QGroupBox("LOD bias")
        bias_layout = QtWidgets.QVBoxLayout(bias_group)
        bias_layout.addWidget(self._lod_bias_slider.widget)
        panel_layout.addWidget(bias_group)

        # ── Shared: bounding box ──────────────────────────────────────
        aabb_group = QtWidgets.QGroupBox("Bounding box")
        aabb_layout = QtWidgets.QVBoxLayout(aabb_group)
        aabb_layout.addWidget(self._aabb_widget.widget)
        panel_layout.addWidget(aabb_group)

        self._status_label = QtWidgets.QLabel("Mode: LOD + frustum culling")
        self._status_label.setWordWrap(True)
        panel_layout.addWidget(self._status_label)
        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    # ── Toggle ────────────────────────────────────────────────────────

    def _on_toggle_clicked(self) -> None:
        from cellier.events import DimsUpdateEvent

        if self._active_mode == "3d":
            self._controller.cancel_pending_slices(self._scene.id)
            self._active_mode = "2d"
            self._mode_label.setText("Mode: 2D")
            # Set slice_indices before changing displayed_axes so the dims
            # state is consistent when the geometry-rebuild event fires.
            # Read the last Z value the slider stored (retained even while hidden).
            current_z = self._dims_control.current_index().get(0, self._z_max // 2)
            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices={0: current_z},
                    displayed_axes=(1, 2),
                )
            )
            for w in self._widget_3d:
                w.setVisible(False)
        else:
            self._controller.cancel_pending_slices(self._scene.id)
            self._active_mode = "3d"
            self._mode_label.setText("Mode: 3D")
            self._controller.incoming_events.emit(
                DimsUpdateEvent(
                    source_id=self._controller._id,
                    scene_id=self._scene.id,
                    slice_indices={},
                    displayed_axes=(0, 1, 2),
                )
            )
            for w in self._widget_3d:
                w.setVisible(True)

    # ── Shared callbacks ──────────────────────────────────────────────

    def _on_reslice_clicked(self) -> None:
        self._controller.reslice_scene(self._scene.id)

    # ── 3D-only callbacks ─────────────────────────────────────────────

    def _on_pipeline_mode_toggle(self, lod_active: bool) -> None:
        if lod_active:
            self._controller.update_appearance_field(
                self._visual_model.id, "force_level", None
            )
            self._controller.update_appearance_field(
                self._visual_model.id, "frustum_cull", True
            )
            self._status_label.setText("Mode: LOD + frustum culling")
        else:
            coarsest = self._n_levels - 1
            self._controller.update_appearance_field(
                self._visual_model.id, "force_level", coarsest
            )
            self._controller.update_appearance_field(
                self._visual_model.id, "frustum_cull", False
            )
            self._status_label.setText(f"Mode: Level {coarsest} all bricks (flat)")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def _dtype_clim_max(dtype: np.dtype) -> float:
    """Return a sensible initial clim upper bound for the given dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
    """Return slider label decimal places appropriate for the given dtype."""
    if np.issubdtype(dtype, np.integer):
        return 0
    return 2


# ---------------------------------------------------------------------------
# ViewerModel builder
# ---------------------------------------------------------------------------


def _build_viewer_model(
    data_store,
    cs,
    voxel_to_world,
    initial_clim_max: float,
    depth_range: tuple[float, float],
    z_depth: int,
):
    """Construct the ViewerModel for the brick viewer (no render layer).

    Parameters
    ----------
    data_store : OMEZarrImageDataStore
        The opened OME-Zarr data store.
    cs : CoordinateSystem
        The world coordinate system (also used for axis_labels display).
    voxel_to_world : AffineTransform
        Level-0 voxel-to-world transform derived from physical scale.
    initial_clim_max : float
        Upper contrast-limit bound for the initial clim.
    depth_range : tuple[float, float]
        (near, far) clipping planes for the perspective camera.
    z_depth : int
        Number of Z voxels at level 0; used to set the initial Z slice index.

    Returns
    -------
    viewer_model : ViewerModel
        Fully assembled model, ready for CellierController.from_model.
    visual_model : MultiscaleImageVisual
        The single image visual, for post-construction access.
    """
    from cellier.scene import Canvas
    from cellier.scene.cameras import OrbitCameraController, PerspectiveCamera
    from cellier.scene.dims import AxisAlignedSelection, DimsManager
    from cellier.scene.scene import Scene
    from cellier.viewer_model import DataManager, ViewerModel
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )

    visual_model = MultiscaleImageVisual(
        name="volume",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=MultiscaleImageAppearance(
            color_map="grays",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=None,
            frustum_cull=True,
            iso_threshold=0.2,
            render_mode="smooth_iso",
            attenuation=1.0,
        ),
        render_config=MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=4096 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        ),
        transform=voxel_to_world,
    )
    visual_model.aabb.color = "#ff00ff"
    visual_model.aabb.enabled = True

    # Single scene supporting both 2D and 3D; starts in 3D mode.
    # Pre-set the Z slice index for when the user toggles to 2D.
    scene = Scene(
        name="main",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1, 2),
                slice_indices={0: z_depth // 2},
            ),
        ),
        render_modes={"2d", "3d"},
        lighting="none",
        visuals=[visual_model],
        canvases={},
    )
    canvas = Canvas(
        cameras={
            "3d": PerspectiveCamera(
                fov=70.0,
                near_clipping_plane=depth_range[0],
                far_clipping_plane=depth_range[1],
                controller=OrbitCameraController(enabled=True),
            )
        }
    )
    scene.canvases[canvas.id] = canvas

    viewer_model = ViewerModel(
        data=DataManager(stores={data_store.id: data_store}),
        scenes={scene.id: scene},
    )

    return viewer_model, visual_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_uri: str):
    from PySide6 import QtWidgets

    from cellier.controller import CellierController
    from cellier.data import OMEZarrImageDataStore
    from cellier.gui.qt import QtCanvasWidget
    from cellier.render._config import (
        CameraConfig,
        RenderManagerConfig,
        SlicingConfig,
    )
    from cellier.scene.dims import CoordinateSystem
    from cellier.transform import AffineTransform

    # ── Open the OME-Zarr store ───────────────────────────────────────
    print(f"Opening OME-Zarr store: {zarr_uri}")
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    print(f"  {data_store.n_levels} levels found.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  Level {i}: shape={shape}")
    print(f"  Axes:  {data_store.axis_names}")
    print(f"  Units: {data_store.axis_units}")

    # ── Extract level-0 physical scale from OME metadata (ZYX order) ─
    import yaozarrs

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]
    level_0_scale_zyx = np.array(ms.datasets[0].scale_transform.scale, dtype=np.float64)

    print(f"\n  Level-0 physical scale (ZYX): {level_0_scale_zyx}")

    # ── Compute world extents for AABB and depth range ────────────────
    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_extents_zyx = vox_shape_zyx * level_0_scale_zyx
    max_extent = float(world_extents_zyx.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)

    print(f"  World extents (ZYX): {world_extents_zyx}")
    print(f"  Max extent:          {max_extent:.3f}")
    print(
        f"  Depth range:         near={depth_range[0]:.2f}  far={depth_range[1]:.0f}\n"
    )

    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    # ── Voxel-to-world transform ──────────────────────────────────────
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_zyx)
    )

    # ── Initial appearance ────────────────────────────────────────────
    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    _DEBUG_MODE = "none"

    z_depth = data_store.level_shapes[0][0]

    # ── Build ViewerModel (no render layer) ───────────────────────────
    viewer_model, visual_model = _build_viewer_model(
        data_store=data_store,
        cs=cs,
        voxel_to_world=voxel_to_world,
        initial_clim_max=initial_clim_max,
        depth_range=depth_range,
        z_depth=z_depth,
    )

    # ── Construct controller from model ───────────────────────────────
    controller = CellierController.from_model(
        viewer_model,
        render_config=RenderManagerConfig(
            slicing=SlicingConfig(batch_size=32, render_every=4),
            camera=CameraConfig(reslice_enabled=False),
        ),
        widget_parent=None,
    )

    # ── Retrieve scene by name ────────────────────────────────────────
    scene = controller.get_scene_by_name("main")

    # ── Set debug mode on the 3D material ────────────────────────────
    vis = controller._render_manager._scenes[scene.id].get_visual(visual_model.id)
    vis.material_3d.debug_mode = _DEBUG_MODE

    # ── Canvas view ───────────────────────────────────────────────────
    canvas_view = controller.get_canvas_view(controller.get_canvas_ids(scene.id)[0])

    # Axis ranges derived from the finest level's voxel shape (ZYX order).
    # These use voxel indices (not world coords) because slice_indices in this
    # script are in voxel space.
    level0_shape = data_store.level_shapes[0]
    axis_ranges = {i: (0, level0_shape[i] - 1) for i in range(len(level0_shape))}

    canvas_widget = QtCanvasWidget.from_scene_and_canvas(
        scene, canvas_view, axis_ranges=axis_ranges
    )
    controller.connect_widget(
        canvas_widget.dims_control,
        subscription_specs=canvas_widget.dims_control.subscription_specs(),
    )

    # ── Show window ───────────────────────────────────────────────────
    viewer = OmeBrickViewer(
        controller,
        scene=scene,
        visual_model=visual_model,
        canvas_widget=canvas_widget,
        zarr_uri=zarr_uri,
        n_levels=data_store.n_levels,
        z_depth=z_depth,
        clim_range=(0.0, initial_clim_max),
        slider_decimals=slider_decimals,
    )
    viewer.window.show()

    # Fit camera from metadata-derived bounding box, then kick off first reslice.
    controller.fit_camera(scene.id)
    controller.reslice_scene(scene.id)

    app = QtWidgets.QApplication.instance()
    app.aboutToQuit.connect(canvas_widget.close)
    app.aboutToQuit.connect(viewer._clim_slider.close)
    app.aboutToQuit.connect(viewer._colormap_combo.close)
    app.aboutToQuit.connect(viewer._aabb_widget.close)
    app.aboutToQuit.connect(viewer._render_controls.close)
    app.aboutToQuit.connect(viewer._lod_bias_slider.close)
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-file-path",
        required=True,
        metavar="PATH",
        help=(
            "Path or URI to the OME-Zarr store "
            "(local path, file://, s3://, gs://, https://)."
        ),
    )
    args = parser.parse_args()

    zarr_input = args.zarr_file_path

    # Resolve local paths to an absolute file:// URI.
    if "://" not in zarr_input:
        local = Path(zarr_input)
        if not local.exists():
            print(
                f"Error: OME-Zarr store not found at '{local}'",
                file=sys.stderr,
            )
            sys.exit(1)
        zarr_uri = f"file://{local.resolve()}"
    else:
        zarr_uri = zarr_input

    import PySide6.QtAsyncio as QtAsyncio
    from PySide6.QtWidgets import QApplication

    # enable_debug_logging(categories=("perf"), level=logging.WARN)

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_uri), handle_sigint=True)
