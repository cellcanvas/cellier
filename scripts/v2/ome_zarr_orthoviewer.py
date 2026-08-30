"""OME-Zarr orthoviewer: 4-panel viewer (XY, XZ, YZ, 3D).

Displays an OME-Zarr volume in four panels arranged in a 2x2 grid:
- XY plane (slices along Z)
- XZ plane (slices along Y)
- YZ plane (slices along X)
- 3D volume at the coarsest resolution level with AABB enabled

All four panels share the same data store.  The side panel provides
separate rendering controls for the 2D views and the 3D view.

Usage
-----
    # Create a synthetic anisotropic test dataset then open it:
    uv run scripts/v2/ome_zarr_orthoviewer.py --make-example

    # Run with ome-zarr v0.5 file
    uv run scripts/v2/ome_zarr_orthoviewer.py --zarr-file-path /path/to/data.ome.zarr
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic example dataset (ExpA-like anisotropic scale)
# ---------------------------------------------------------------------------

# ExpA: Z=5.0 µm, Y=X=6.55 µm; only Y/X are downsampled per level (Z fixed).
_EXAMPLE_SCALE_Z = 5.0
_EXAMPLE_SCALE_YX = 6.550032422660492
_EXAMPLE_SHAPE_ZYX = (200, 300, 300)  # voxels at level 0
_EXAMPLE_N_LEVELS = 4
_EXAMPLE_N_BLOBS = 12
_EXAMPLE_BLOB_RADIUS_UM = 120.0  # physical radius (appears round if transforms OK)
_EXAMPLE_CHUNK_ZYX = (32, 32, 32)
_EXAMPLE_PATH = Path(__file__).parent / "example_anisotropic_blobs.ome.zarr"


def _make_example_blob_volume(
    shape_zyx: tuple[int, int, int],
    spacing_zyx: tuple[float, float, float],
    n_blobs: int,
    radius_um: float,
    seed: int = 42,
) -> np.ndarray:
    """Return a uint8 volume with blobs that are spherical in physical space.

    Because Z spacing > YX spacing, blobs are stored as oblate ellipsoids
    in voxel space.  A viewer with correct physical transforms renders them
    as perfect spheres.
    """
    nz, ny, nx = shape_zyx
    sz, sy, sx = spacing_zyx
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape_zyx, dtype=np.uint8)

    # Voxel radii per axis so the physical sphere fits.
    rz = int(np.ceil(radius_um / sz))
    ry = int(np.ceil(radius_um / sy))
    rx = int(np.ceil(radius_um / sx))

    for _ in range(n_blobs):
        cz = rng.integers(rz, nz - rz)
        cy = rng.integers(ry, ny - ry)
        cx = rng.integers(rx, nx - rx)

        lz = np.arange(-rz, rz + 1)
        ly = np.arange(-ry, ry + 1)
        lx = np.arange(-rx, rx + 1)
        ZZ, YY, XX = np.meshgrid(lz, ly, lx, indexing="ij")
        # Physical-distance ellipsoid mask → sphere in world space.
        mask = (ZZ * sz) ** 2 + (YY * sy) ** 2 + (XX * sx) ** 2 <= radius_um**2

        z0, z1 = max(0, cz - rz), min(nz, cz + rz + 1)
        y0, y1 = max(0, cy - ry), min(ny, cy + ry + 1)
        x0, x1 = max(0, cx - rx), min(nx, cx + rx + 1)

        mz0 = max(0, -(cz - rz))
        my0 = max(0, -(cy - ry))
        mx0 = max(0, -(cx - rx))
        mz1 = mz0 + (z1 - z0)
        my1 = my0 + (y1 - y0)
        mx1 = mx0 + (x1 - x0)

        volume[z0:z1, y0:y1, x0:x1] |= mask[mz0:mz1, my0:my1, mx0:mx1].astype(np.uint8)

    return volume


def make_example_zarr(output_path: Path = _EXAMPLE_PATH) -> Path:
    """Create a synthetic anisotropic OME-Zarr with spherical blobs.

    Uses the same Z/YX scale ratio as ExpA (5.0 : 6.55 µm).  Only Y and X
    are downsampled per level; Z stays fixed — identical to the real dataset.

    Blobs are physical spheres: they appear round in all slice planes when
    the viewer applies the correct coordinate transforms.

    Parameters
    ----------
    output_path : Path
        Directory to write the OME-Zarr store.  Created if absent.

    Returns
    -------
    Path
        Resolved path to the written store.
    """
    import zarr

    output_path = Path(output_path)
    if output_path.exists():
        print(f"Example dataset already exists at {output_path}")
        return output_path.resolve()

    print(f"Creating example dataset at {output_path} ...")
    sz, syx = _EXAMPLE_SCALE_Z, _EXAMPLE_SCALE_YX
    nz, ny, nx = _EXAMPLE_SHAPE_ZYX

    # Generate level-0 blob volume.
    data_l0 = _make_example_blob_volume(
        _EXAMPLE_SHAPE_ZYX,
        (sz, syx, syx),
        _EXAMPLE_N_BLOBS,
        _EXAMPLE_BLOB_RADIUS_UM,
    )

    root = zarr.open_group(str(output_path), mode="w")
    datasets_meta = []

    for level in range(_EXAMPLE_N_LEVELS):
        factor = 2**level
        if level == 0:
            data = data_l0
        else:
            # Only downsample Y and X; Z is never downsampled (same as ExpA).
            data = data_l0[:, ::factor, ::factor]

        arr = root.create_array(
            f"s{level}",
            shape=data.shape,
            chunks=_EXAMPLE_CHUNK_ZYX,
            dtype=np.uint8,
        )
        arr[:] = data

        datasets_meta.append(
            {
                "path": f"s{level}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [sz, syx * factor, syx * factor]},
                ],
            }
        )
        print(
            f"  Level {level}: shape={data.shape}  "
            f"scale=(z={sz}, yx={syx * factor:.4f})"
        )

    # Write OME-NGFF v0.5 metadata (nested under "ome" key).
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets_meta,
                "name": "blobs",
            }
        ],
    }

    print(
        f"Done. Physical size: z={nz * sz:.0f} µm, y={ny * syx:.0f} µm, x={nx * syx:.0f} µm"
    )
    print(f"Blob radius: {_EXAMPLE_BLOB_RADIUS_UM} µm  (spherical in world space)")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Slider color styles for each 2D panel
# ---------------------------------------------------------------------------


def _make_slider_style(color_a: str, color_b: str) -> str:
    return f"""
QSlider::groove:horizontal {{
    border: 1px solid #bbb;
    background: white;
    height: 10px;
    border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);
    border: 1px solid #777;
    width: 13px;
    margin-top: -7px;
    margin-bottom: -7px;
    border-radius: 4px;
}}
QSlider::add-page:horizontal {{
    background: #fff;
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0.2, x2:1, y2:1,
        stop:0 {color_a}, stop:1 {color_b});
    border: 1px solid #777;
    height: 10px;
    border-radius: 4px;
}}
QSlider::handle:horizontal:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fff, stop:1 #ddd);
    border: 1px solid #444;
    border-radius: 4px;
}}
QLabel {{ font-size: 12px; }}
"""


# Blue for XY (slicing Z), green for XZ (slicing Y), orange for YZ (slicing X)
SLIDER_STYLE_XY = _make_slider_style("#bbf", "#55f")
SLIDER_STYLE_XZ = _make_slider_style("#bfb", "#3a3")
SLIDER_STYLE_YZ = _make_slider_style("#fdb", "#c60")


# ---------------------------------------------------------------------------
# Multi-visual control helpers
# ---------------------------------------------------------------------------


class _MultiVisualClimSlider:
    """Contrast-limits range slider that updates multiple visuals at once."""

    from psygnal import Signal

    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        visual_ids: list,
        *,
        clim_range: tuple[float, float],
        initial_clim: tuple[float, float],
        decimals: int = 2,
        parent=None,
    ) -> None:
        from qtpy.QtCore import Qt
        from superqt import QLabeledDoubleRangeSlider

        from cellier.events import AppearanceUpdateEvent

        self._id = uuid4()
        self._visual_ids = visual_ids
        self._AppearanceUpdateEvent = AppearanceUpdateEvent

        self._slider = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal, parent)
        self._slider.setRange(*clim_range)
        self._slider.setValue(initial_clim)
        self._slider.setDecimals(decimals)
        self._slider.valueChanged.connect(self._on_changed)

    def _on_changed(self, value: tuple[float, float]) -> None:
        for vid in self._visual_ids:
            self.changed.emit(
                self._AppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    field="clim",
                    value=value,
                )
            )

    @property
    def widget(self):
        return self._slider

    def close(self) -> None:
        self.closed.emit()


class _MultiVisualColormapCombo:
    """Colormap combo box that updates multiple visuals at once."""

    from psygnal import Signal

    changed = Signal(object)
    closed = Signal()

    def __init__(
        self,
        visual_ids: list,
        *,
        initial_colormap,
        parent=None,
    ) -> None:
        from superqt import QColormapComboBox

        from cellier.events import AppearanceUpdateEvent

        self._id = uuid4()
        self._visual_ids = visual_ids
        self._AppearanceUpdateEvent = AppearanceUpdateEvent

        self._combo = QColormapComboBox(parent)
        self._combo.setCurrentColormap(initial_colormap)
        self._combo.currentColormapChanged.connect(self._on_changed)

    def _on_changed(self, colormap) -> None:
        for vid in self._visual_ids:
            self.changed.emit(
                self._AppearanceUpdateEvent(
                    source_id=self._id,
                    visual_id=vid,
                    field="color_map",
                    value=colormap,
                )
            )

    @property
    def widget(self):
        return self._combo

    def close(self) -> None:
        self.closed.emit()


# ---------------------------------------------------------------------------
# Main viewer class
# ---------------------------------------------------------------------------


class OmeZarrOrthoViewer:
    """4-panel orthoviewer window: XY, XZ, YZ slices and a 3D volume."""

    def __init__(
        self,
        controller,
        scenes: dict,
        visuals: dict,
        canvas_widgets: dict,
        clim_range: tuple[float, float],
        slider_decimals: int = 2,
        # Plane overlay (optional — pass None to omit the group)
        plane_visual=None,
        plane_store=None,
        vol_visual_id=None,
        initial_plane_opacity: float = 0.4,
        # Orientation overlays (pass None to omit checkboxes)
        axes_2d_overlay_ids: list | None = None,
        orient_3d_visual_ids: list | None = None,
    ):
        from PySide6 import QtCore, QtWidgets

        from cellier.gui.qt.visuals import (
            QtClimRangeSlider,
            QtColormapComboBox,
            QtVolumeRenderControls,
        )

        self._controller = controller
        self._scenes = scenes
        self._visuals = visuals
        self._canvas_widgets = canvas_widgets

        xy_id = visuals["xy"].id
        xz_id = visuals["xz"].id
        yz_id = visuals["yz"].id
        vol_id = visuals["vol"].id

        # ── 2D controls (shared across all three 2D views) ────────────────────
        self._2d_clim = _MultiVisualClimSlider(
            [xy_id, xz_id, yz_id],
            clim_range=clim_range,
            initial_clim=visuals["xy"].appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(self._2d_clim)
        self._2d_colormap = _MultiVisualColormapCombo(
            [xy_id, xz_id, yz_id],
            initial_colormap=visuals["xy"].appearance.color_map,
        )
        controller.connect_widget(self._2d_colormap)

        # ── 3D controls ───────────────────────────────────────────────────────
        self._3d_clim = QtClimRangeSlider(
            vol_id,
            clim_range=clim_range,
            initial_clim=visuals["vol"].appearance.clim,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._3d_clim, subscription_specs=self._3d_clim.subscription_specs()
        )
        self._3d_colormap = QtColormapComboBox(
            vol_id,
            initial_colormap=visuals["vol"].appearance.color_map,
        )
        controller.connect_widget(
            self._3d_colormap, subscription_specs=self._3d_colormap.subscription_specs()
        )
        self._3d_render = QtVolumeRenderControls(
            vol_id,
            dtype_max=clim_range[1],
            initial_render_mode=visuals["vol"].appearance.render_mode,
            initial_threshold=visuals["vol"].appearance.iso_threshold,
            decimals=slider_decimals,
        )
        controller.connect_widget(
            self._3d_render, subscription_specs=self._3d_render.subscription_specs()
        )

        # ── Main window ───────────────────────────────────────────────────────
        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("OME-Zarr Orthoviewer")
        self._window.resize(1400, 900)

        central = QtWidgets.QWidget()
        self._window.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        # ── 2x2 canvas grid ───────────────────────────────────────────────────
        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        panels = [
            (0, 0, "xy", "XY  (slice Z)"),
            (0, 1, "xz", "XZ  (slice Y)"),
            (1, 0, "yz", "YZ  (slice X)"),
            (1, 1, "vol", "3D Volume"),
        ]
        for row, col, key, label in panels:
            cell = QtWidgets.QWidget()
            cell_layout = QtWidgets.QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)

            lbl = QtWidgets.QLabel(label)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px;")
            cell_layout.addWidget(lbl)
            cell_layout.addWidget(canvas_widgets[key].widget, stretch=1)
            grid.addWidget(cell, row, col)

        root_layout.addWidget(grid_widget, stretch=1)

        # ── Side panel ────────────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        root_layout.addWidget(panel)

        # 2D group
        group_2d = QtWidgets.QGroupBox("2D Rendering")
        layout_2d = QtWidgets.QVBoxLayout(group_2d)

        clim_2d_box = QtWidgets.QGroupBox("Contrast limits")
        QtWidgets.QVBoxLayout(clim_2d_box).addWidget(self._2d_clim.widget)
        layout_2d.addWidget(clim_2d_box)

        cmap_2d_box = QtWidgets.QGroupBox("Colormap")
        QtWidgets.QVBoxLayout(cmap_2d_box).addWidget(self._2d_colormap.widget)
        layout_2d.addWidget(cmap_2d_box)

        if axes_2d_overlay_ids:
            from PySide6.QtWidgets import QCheckBox

            axes_2d_cb = QCheckBox("Show orientation axes")
            axes_2d_cb.setChecked(True)

            def _on_axes_2d_toggled(checked: bool) -> None:
                for oid in axes_2d_overlay_ids:
                    controller.set_overlay_visible(oid, checked)

            axes_2d_cb.toggled.connect(_on_axes_2d_toggled)
            layout_2d.addWidget(axes_2d_cb)

        panel_layout.addWidget(group_2d)

        # 3D group
        group_3d = QtWidgets.QGroupBox("3D Rendering")
        layout_3d = QtWidgets.QVBoxLayout(group_3d)

        clim_3d_box = QtWidgets.QGroupBox("Contrast limits")
        QtWidgets.QVBoxLayout(clim_3d_box).addWidget(self._3d_clim.widget)
        layout_3d.addWidget(clim_3d_box)

        cmap_3d_box = QtWidgets.QGroupBox("Colormap")
        QtWidgets.QVBoxLayout(cmap_3d_box).addWidget(self._3d_colormap.widget)
        layout_3d.addWidget(cmap_3d_box)

        render_3d_box = QtWidgets.QGroupBox("Render mode")
        QtWidgets.QVBoxLayout(render_3d_box).addWidget(self._3d_render.widget)
        layout_3d.addWidget(render_3d_box)

        if orient_3d_visual_ids:
            from PySide6.QtWidgets import QCheckBox

            from cellier.events import (
                AppearanceUpdateEvent as _AppearanceUpdateEvent,
            )

            orient_3d_cb = QCheckBox("Show orientation axes")
            orient_3d_cb.setChecked(True)
            _orient_3d_bid = uuid4()

            def _on_orient_3d_toggled(checked: bool) -> None:
                for vid in orient_3d_visual_ids:
                    controller.incoming_events.emit(
                        _AppearanceUpdateEvent(
                            source_id=_orient_3d_bid,
                            visual_id=vid,
                            field="visible",
                            value=checked,
                        )
                    )

            orient_3d_cb.toggled.connect(_on_orient_3d_toggled)
            layout_3d.addWidget(orient_3d_cb)

        panel_layout.addWidget(group_3d)

        # Plane Overlay group (only shown when plane objects are provided)
        if plane_visual is not None or vol_visual_id is not None:
            from PySide6.QtCore import Qt
            from superqt import QLabeledDoubleSlider

            group_planes = QtWidgets.QGroupBox("Plane Overlay")
            layout_planes = QtWidgets.QVBoxLayout(group_planes)

            # Volume opacity slider
            vol_opacity_label = QtWidgets.QLabel("Volume opacity")
            vol_opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            vol_opacity_slider.setRange(0.0, 1.0)
            vol_opacity_slider.setValue(1.0)
            vol_opacity_slider.setDecimals(2)

            def _on_vol_opacity_changed(value: float) -> None:
                if vol_visual_id is not None:
                    controller.update_appearance_field(vol_visual_id, "opacity", value)

            vol_opacity_slider.valueChanged.connect(_on_vol_opacity_changed)
            layout_planes.addWidget(vol_opacity_label)
            layout_planes.addWidget(vol_opacity_slider)

            # Plane opacity slider
            plane_opacity_label = QtWidgets.QLabel("Plane opacity")
            plane_opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            plane_opacity_slider.setRange(0.0, 1.0)
            plane_opacity_slider.setValue(initial_plane_opacity)
            plane_opacity_slider.setDecimals(2)

            def _on_plane_opacity_changed(value: float) -> None:
                if plane_visual is not None:
                    controller.update_appearance_field(
                        plane_visual.id, "opacity", value
                    )
                if plane_store is not None:
                    plane_store.colors = _make_plane_colors(value)
                    controller.reslice_visual(plane_visual.id)

            plane_opacity_slider.valueChanged.connect(_on_plane_opacity_changed)
            layout_planes.addWidget(plane_opacity_label)
            layout_planes.addWidget(plane_opacity_slider)

            _on_plane_opacity_changed(initial_plane_opacity)

            panel_layout.addWidget(group_planes)

        panel_layout.addStretch()

    @property
    def window(self):
        return self._window

    def close_widgets(self) -> None:
        for cw in self._canvas_widgets.values():
            cw.close()
        self._2d_clim.close()
        self._2d_colormap.close()
        self._3d_clim.close()
        self._3d_colormap.close()
        self._3d_render.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Slice-plane mesh helpers (Step 16)
# ---------------------------------------------------------------------------

# Colors that visually match the slider gradients in SLIDER_STYLE_XY/XZ/YZ.
_PLANE_COLOR_XY = (0.33, 0.33, 1.00)  # blue
_PLANE_COLOR_XZ = (0.23, 0.67, 0.23)  # green
_PLANE_COLOR_YZ = (0.80, 0.40, 0.00)  # orange

# Fraction of the shorter canvas world-extent used as the 2D axis line length.
_AXIS_FRACTION: float = 0.15

# Fractions of the shortest world dimension used to size the 3D axis mesh overlays.
# Final world-unit sizes are computed at runtime from world_max_zyx.min().
_AXIS_3D_LENGTH_FRACTION: float = 0.12  # arm length of each prism
_AXIS_3D_CUBE_SIDE_FRACTION: float = 0.024  # side of the origin cube
_AXIS_3D_PRISM_CROSS_SECTION_FRACTION: float = 0.020  # prism cross-section width

# Neutral grey used for all six faces of the origin cube.
_AXIS_3D_CUBE_COLOUR: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
# Number of triangular faces per box (6 faces × 2 triangles).
_N_FACES_PER_BOX: int = 12


def _box_faces_geometry(
    centre_zyx: np.ndarray,
    half_extents_zyx: np.ndarray,
    vertex_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and indices for one axis-aligned box.

    Uses 24 vertices (4 duplicated per face) so each face can carry an
    independent per-face colour without bleeding across face boundaries.

    Parameters
    ----------
    centre_zyx : np.ndarray
        Shape (3,) box centre in data order (z, y, x).
    half_extents_zyx : np.ndarray
        Shape (3,) half-widths along each data axis.
    vertex_offset : int
        Added to every index value so that multiple boxes can be
        concatenated into a single geometry array without index collisions.

    Returns
    -------
    positions : np.ndarray
        (24, 3) float32 vertex positions.
    indices : np.ndarray
        (12, 3) int32 triangle indices, shifted by vertex_offset.
    """
    cz, cy, cx = float(centre_zyx[0]), float(centre_zyx[1]), float(centre_zyx[2])
    hz, hy, hx = (
        float(half_extents_zyx[0]),
        float(half_extents_zyx[1]),
        float(half_extents_zyx[2]),
    )
    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    positions = np.array(
        [
            # Face 0: −Z
            [z0, y0, x0],
            [z0, y1, x0],
            [z0, y1, x1],
            [z0, y0, x1],
            # Face 1: +Z
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x1],
            [z1, y1, x0],
            # Face 2: −Y
            [z0, y0, x0],
            [z0, y0, x1],
            [z1, y0, x1],
            [z1, y0, x0],
            # Face 3: +Y
            [z0, y1, x0],
            [z1, y1, x0],
            [z1, y1, x1],
            [z0, y1, x1],
            # Face 4: −X
            [z0, y0, x0],
            [z1, y0, x0],
            [z1, y1, x0],
            [z0, y1, x0],
            # Face 5: +X
            [z0, y0, x1],
            [z0, y1, x1],
            [z1, y1, x1],
            [z1, y0, x1],
        ],
        dtype=np.float32,
    )

    base_indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # Face 0
            [4, 5, 6],
            [4, 6, 7],  # Face 1
            [8, 9, 10],
            [8, 10, 11],  # Face 2
            [12, 13, 14],
            [12, 14, 15],  # Face 3
            [16, 17, 18],
            [16, 18, 19],  # Face 4
            [20, 21, 22],
            [20, 22, 23],  # Face 5
        ],
        dtype=np.int32,
    )
    return positions, base_indices + vertex_offset


def _make_axis_set_geometry(
    axis_a: int,
    axis_b: int,
    axis_length: float,
    cube_side: float,
    prism_cross_section: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build combined positions and indices for one axis set: cube + 2 prisms.

    All geometry is centred at the origin.  Apply a translation transform to
    the mesh visual to place the set at any world position without rebuilding
    the geometry.

    The two prisms extend in the positive direction along axis_a and axis_b
    respectively.  Each prism's near face is flush with the corresponding cube
    face so the two primitives touch without overlapping.

    Parameters
    ----------
    axis_a, axis_b : int
        Data-order axis indices (0 = Z, 1 = Y, 2 = X) for the two prisms.
    axis_length : float
        Arm length of each prism in world units.
    cube_side : float
        Side length of the origin cube in world units.
    prism_cross_section : float
        Width and height of each prism's square cross-section in world units.

    Returns
    -------
    positions : np.ndarray
        (72, 3) float32 — 24 vertices × 3 boxes (cube + 2 prisms).
    indices : np.ndarray
        (36, 3) int32 — 12 triangles × 3 boxes.
    """
    half_cube = cube_side / 2.0
    half_length = axis_length / 2.0
    half_cross = prism_cross_section / 2.0

    origin = np.zeros(3, dtype=np.float64)
    cube_half_extents = np.full(3, half_cube, dtype=np.float64)
    cube_positions, cube_indices = _box_faces_geometry(
        origin, cube_half_extents, vertex_offset=0
    )

    centre_a = np.zeros(3, dtype=np.float64)
    centre_a[axis_a] = half_cube + half_length
    half_extents_a = np.full(3, half_cross, dtype=np.float64)
    half_extents_a[axis_a] = half_length
    prism_a_positions, prism_a_indices = _box_faces_geometry(
        centre_a, half_extents_a, vertex_offset=24
    )

    centre_b = np.zeros(3, dtype=np.float64)
    centre_b[axis_b] = half_cube + half_length
    half_extents_b = np.full(3, half_cross, dtype=np.float64)
    half_extents_b[axis_b] = half_length
    prism_b_positions, prism_b_indices = _box_faces_geometry(
        centre_b, half_extents_b, vertex_offset=48
    )

    positions = np.concatenate(
        [cube_positions, prism_a_positions, prism_b_positions], axis=0
    )
    indices = np.concatenate([cube_indices, prism_a_indices, prism_b_indices], axis=0)
    return positions, indices


def _make_axis_set_face_colors(
    axis_a_color_rgb: tuple[float, float, float],
    axis_b_color_rgb: tuple[float, float, float],
) -> np.ndarray:
    """Return (36, 4) float32 per-face RGBA colors for one axis set.

    Layout: 12 cube faces (neutral grey), 12 prism-A faces (axis_a_color_rgb),
    12 prism-B faces (axis_b_color_rgb).  All faces are fully opaque.

    Parameters
    ----------
    axis_a_color_rgb : tuple[float, float, float]
        RGB color for the first-axis prism faces.
    axis_b_color_rgb : tuple[float, float, float]
        RGB color for the second-axis prism faces.

    Returns
    -------
    np.ndarray
        (36, 4) float32 RGBA per-face color array.
    """
    cube_color = np.array(_AXIS_3D_CUBE_COLOUR, dtype=np.float32)
    color_a = np.array([*axis_a_color_rgb, 1.0], dtype=np.float32)
    color_b = np.array([*axis_b_color_rgb, 1.0], dtype=np.float32)
    return np.concatenate(
        [
            np.tile(cube_color, (_N_FACES_PER_BOX, 1)),
            np.tile(color_a, (_N_FACES_PER_BOX, 1)),
            np.tile(color_b, (_N_FACES_PER_BOX, 1)),
        ],
        axis=0,
    )


def _make_axis_meshes(
    controller,
    vol_scene,
    initial_centre_zyx: np.ndarray,
    world_min_extent: float,
) -> tuple:
    """Create the three axis-set mesh visuals in the 3D volume scene.

    One mesh visual is created per 2D view (XY, XZ, YZ).  Each contains a
    cube at the origin plus two rectangular prisms — one per displayed axis —
    with their geometry baked relative to the origin.  A translation transform
    positions the mesh at the camera centre; only the transform is updated on
    subsequent camera changes, not the geometry.

    Sizes are derived from *world_min_extent* via the fraction constants
    ``_AXIS_3D_LENGTH_FRACTION``, ``_AXIS_3D_CUBE_SIDE_FRACTION``, and
    ``_AXIS_3D_PRISM_CROSS_SECTION_FRACTION`` so the indicators scale
    consistently across datasets with different physical extents.

    Parameters
    ----------
    controller : CellierController
        The active controller used to register each mesh with the scene.
    vol_scene : Scene
        The 3D volume scene model that receives the mesh visuals.
    initial_centre_zyx : np.ndarray
        Shape (3,) initial world position in data order (z, y, x), applied
        as a translation transform to all three meshes at creation time.
    world_min_extent : float
        Shortest world-space dimension (``world_max_zyx.min()``), used to
        derive axis_length, cube_side, and prism_cross_section.

    Returns
    -------
    tuple[MeshVisual, MeshVisual, MeshVisual]
        ``(xy_axis_visual, xz_axis_visual, yz_axis_visual)`` — the three
        mesh visuals, one per 2D view.
    """
    from cellier.data import MeshMemoryStore
    from cellier.transform import AffineTransform
    from cellier.visuals import MeshFlatAppearance

    # Axis RGB colors match the plane-mesh and slider color convention.
    color_z = _PLANE_COLOR_XY  # blue  — data axis 0
    color_y = _PLANE_COLOR_XZ  # green — data axis 1
    color_x = _PLANE_COLOR_YZ  # orange — data axis 2

    # Per-view specification: (view_name, axis_a_index, axis_b_index, color_a, color_b)
    # XY view displays Y (axis 1) and X (axis 2).
    # XZ view displays Z (axis 0) and X (axis 2).
    # YZ view displays Z (axis 0) and Y (axis 1).
    view_specifications = [
        ("xy_axis_set", 1, 2, color_y, color_x),
        ("xz_axis_set", 0, 2, color_z, color_x),
        ("yz_axis_set", 0, 1, color_z, color_y),
    ]

    axis_length = _AXIS_3D_LENGTH_FRACTION * world_min_extent
    cube_side = _AXIS_3D_CUBE_SIDE_FRACTION * world_min_extent
    prism_cross_section = _AXIS_3D_PRISM_CROSS_SECTION_FRACTION * world_min_extent

    initial_translation = tuple(float(v) for v in initial_centre_zyx)
    initial_transform = AffineTransform.from_translation(initial_translation)

    axis_visuals = []
    for view_name, axis_a, axis_b, color_a, color_b in view_specifications:
        positions, indices = _make_axis_set_geometry(
            axis_a, axis_b, axis_length, cube_side, prism_cross_section
        )
        face_colors = _make_axis_set_face_colors(color_a, color_b)
        store = MeshMemoryStore(
            positions=positions,
            indices=indices,
            colors=face_colors,
            colors_layout="face",
            name=view_name,
        )
        appearance = MeshFlatAppearance(
            color_mode="face",
            side="both",
            opacity=1.0,
            render_order=1,
            depth_test=True,
            depth_write=True,
            transparency_mode="blend",
        )
        visual = controller.add_mesh(
            data=store,
            scene_id=vol_scene.id,
            appearance=appearance,
            name=view_name,
            transform=initial_transform,
        )
        axis_visuals.append(visual)

    xy_axis_visual, xz_axis_visual, yz_axis_visual = axis_visuals
    return xy_axis_visual, xz_axis_visual, yz_axis_visual


def _make_plane_positions(
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
) -> np.ndarray:
    """Return (12, 3) float32 positions for the three slice planes.

    Vertices are in (z, y, x) axis order.  Planes are quads split into
    two triangles each:
      - XY plane (slices Z): vertices 0–3
      - XZ plane (slices Y): vertices 4–7
      - YZ plane (slices X): vertices 8–11

    Parameters
    ----------
    z_world : float
        World-space Z position for the XY plane.
    y_world : float
        World-space Y position for the XZ plane.
    x_world : float
        World-space X position for the YZ plane.
    world_max_zyx : np.ndarray
        Shape (3,) — world-space extents [wz, wy, wx].

    Returns
    -------
    np.ndarray
        (12, 3) float32 array of vertex positions.
    """
    wz, wy, wx = (
        float(world_max_zyx[0]),
        float(world_max_zyx[1]),
        float(world_max_zyx[2]),
    )
    z = float(z_world)
    y = float(y_world)
    x = float(x_world)

    return np.array(
        [
            # XY plane (slices Z) — vertices 0–3
            [z, 0, 0],
            [z, wy, 0],
            [z, wy, wx],
            [z, 0, wx],
            # XZ plane (slices Y) — vertices 4–7
            [0, y, 0],
            [wz, y, 0],
            [wz, y, wx],
            [0, y, wx],
            # YZ plane (slices X) — vertices 8–11
            [0, 0, x],
            [wz, 0, x],
            [wz, wy, x],
            [0, wy, x],
        ],
        dtype=np.float32,
    )


def _make_plane_colors(opacity: float) -> np.ndarray:
    """Return (6, 4) float32 per-face RGBA colors for the three planes.

    Face order: XY faces 0–1 (blue), XZ faces 2–3 (green), YZ faces 4–5
    (orange).  Alpha is set to *opacity* uniformly.

    Parameters
    ----------
    opacity : float
        Alpha value 0–1 applied to all six faces.

    Returns
    -------
    np.ndarray
        (6, 4) float32 RGBA array.
    """
    rgb_xy = _PLANE_COLOR_XY
    rgb_xz = _PLANE_COLOR_XZ
    rgb_yz = _PLANE_COLOR_YZ
    a = float(opacity)
    return np.array(
        [
            [*rgb_xy, a],  # XY face 0
            [*rgb_xy, a],  # XY face 1
            [*rgb_xz, a],  # XZ face 0
            [*rgb_xz, a],  # XZ face 1
            [*rgb_yz, a],  # YZ face 0
            [*rgb_yz, a],  # YZ face 1
        ],
        dtype=np.float32,
    )


def _make_plane_mesh(
    controller,
    vol_scene,
    z_world: float,
    y_world: float,
    x_world: float,
    world_max_zyx: np.ndarray,
    initial_opacity: float = 0.4,
):
    """Create the three slice-plane quads as a single mesh visual.

    Parameters
    ----------
    controller : CellierController
        The active controller.
    vol_scene : Scene
        The 3D volume scene to add the mesh to.
    z_world, y_world, x_world : float
        Initial world-space slice positions.
    world_max_zyx : np.ndarray
        Shape (3,) world-space AABB extents.
    initial_opacity : float
        Starting alpha for the plane faces.

    Returns
    -------
    tuple[MeshMemoryStore, MeshVisual]
        The live store and model objects.
    """
    from cellier.data import MeshMemoryStore
    from cellier.visuals import MeshFlatAppearance

    positions = _make_plane_positions(z_world, y_world, x_world, world_max_zyx)
    colors = _make_plane_colors(initial_opacity)

    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # XY plane
            [4, 5, 6],
            [4, 6, 7],  # XZ plane
            [8, 9, 10],
            [8, 10, 11],  # YZ plane
        ],
        dtype=np.int32,
    )

    store = MeshMemoryStore(
        positions=positions,
        indices=indices,
        colors=colors,
        colors_layout="face",
        name="slice_planes",
    )

    appearance = MeshFlatAppearance(
        color_mode="face",
        side="both",
        opacity=initial_opacity,
        wireframe=False,
    )

    visual = controller.add_mesh(
        data=store,
        scene_id=vol_scene.id,
        appearance=appearance,
        name="slice_planes",
    )

    return store, visual


class _PlaneUpdater:
    """Wires 2D dims changes to the slice-plane mesh positions.

    One instance is created for the whole viewer.  Separate named methods
    are registered as callbacks for each 2D scene via
    ``controller.on_dims_changed`` so that no lambdas are used.

    Parameters
    ----------
    controller : CellierController
        The active controller.
    plane_store : MeshMemoryStore
        The store holding the three slice-plane quads.
    plane_visual : MeshVisual
        The model-layer visual (used for its ``.id``).
    world_max_zyx : np.ndarray
        Shape (3,) world-space AABB extents [wz, wy, wx].
    """

    def __init__(
        self,
        controller,
        plane_store,
        plane_visual,
        world_max_zyx: np.ndarray,
    ) -> None:
        self._id = uuid4()
        self._controller = controller
        self._plane_store = plane_store
        self._plane_visual = plane_visual
        self._world_max_zyx = world_max_zyx

        # Cache the current slice positions so we can reconstruct the full
        # positions array when only one axis changes.
        positions = plane_store.positions
        self._z_world = float(positions[0, 0])  # vertex 0, z-component
        self._y_world = float(positions[4, 1])  # vertex 4, y-component
        self._x_world = float(positions[8, 2])  # vertex 8, x-component

    def _update(self) -> None:
        """Recompute positions and push to GPU via reslice_visual."""
        self._plane_store.positions = _make_plane_positions(
            self._z_world,
            self._y_world,
            self._x_world,
            self._world_max_zyx,
        )
        self._controller.reslice_visual(self._plane_visual.id)

    def on_xy_dims_changed(self, event) -> None:
        """Called when the XY scene dims change (Z slider moved)."""
        slice_indices = event.dims_state.selection.slice_indices
        if 0 in slice_indices:
            self._z_world = float(slice_indices[0])
            self._update()

    def on_xz_dims_changed(self, event) -> None:
        """Called when the XZ scene dims change (Y slider moved)."""
        slice_indices = event.dims_state.selection.slice_indices
        if 1 in slice_indices:
            self._y_world = float(slice_indices[1])
            self._update()

    def on_yz_dims_changed(self, event) -> None:
        """Called when the YZ scene dims change (X slider moved)."""
        slice_indices = event.dims_state.selection.slice_indices
        if 2 in slice_indices:
            self._x_world = float(slice_indices[2])
            self._update()


class _OrientationUpdater:
    """Keeps orientation overlay positions in sync with camera and dims state.

    One instance is created after the controller is built.  Named methods are
    registered with ``controller.on_camera_changed`` and
    ``controller.on_dims_changed`` for each 2D scene.

    Coordinate mapping (pygfx order -> data order)
    -----------------------------------------------
    The pygfx orthographic camera is fitted with ``view_dir=(0, 0, -1)`` and
    ``up=(0, 1, 0)``.  After fitting, ``camera_state.position`` in pygfx order
    maps to data order as follows:

    * XY scene (displayed Y, X): pygfx_x -> data X (axis 2),
                                  pygfx_y -> data Y (axis 1).
    * XZ scene (displayed Z, X): pygfx_x -> data X (axis 2),
                                  pygfx_y -> data Z (axis 0).
    * YZ scene (displayed Z, Y): pygfx_x -> data Y (axis 1),
                                  pygfx_y -> data Z (axis 0).

    Parameters
    ----------
    controller : CellierController
        The active controller.
    xy_axis_visual : MeshVisual
        Axis-set mesh visual for the XY view (displays Y and X prisms).
    xz_axis_visual : MeshVisual
        Axis-set mesh visual for the XZ view (displays Z and X prisms).
    yz_axis_visual : MeshVisual
        Axis-set mesh visual for the YZ view (displays Z and Y prisms).
    world_max_zyx : np.ndarray
        Shape (3,) world-space extents ``[wz, wy, wx]``.
    """

    def __init__(
        self,
        controller,
        xy_axis_visual,
        xz_axis_visual,
        yz_axis_visual,
        world_max_zyx: np.ndarray,
    ):
        self._id = uuid4()
        self._controller = controller
        self._xy_axis_visual_id = xy_axis_visual.id
        self._xz_axis_visual_id = xz_axis_visual.id
        self._yz_axis_visual_id = yz_axis_visual.id

        # Cached camera centres in data order (z, y, x); seeded at mid-volume.
        mid = world_max_zyx / 2.0
        self._xy_centre_zyx = mid.copy()
        self._xz_centre_zyx = mid.copy()
        self._yz_centre_zyx = mid.copy()

        # Cached slice positions used when dims update but camera has not.
        self._z_world = float(mid[0])
        self._y_world = float(mid[1])
        self._x_world = float(mid[2])

    def _update_3d(self) -> None:
        from cellier.transform import AffineTransform

        labels = ("xy", "xz", "yz")
        for label, visual_id, centre_zyx in zip(
            labels,
            (
                self._xy_axis_visual_id,
                self._xz_axis_visual_id,
                self._yz_axis_visual_id,
            ),
            (
                self._xy_centre_zyx,
                self._xz_centre_zyx,
                self._yz_centre_zyx,
            ),
        ):
            translation = tuple(float(v) for v in centre_zyx)
            self._controller.set_visual_transform(
                visual_id,
                AffineTransform.from_translation(translation),
                reslice=False,
            )

    def on_xy_camera_changed(self, event) -> None:
        camera_state = event.camera_state
        # pygfx_x -> data X (axis 2), pygfx_y -> data Y (axis 1)
        self._xy_centre_zyx = np.array(
            [self._z_world, camera_state.position[1], camera_state.position[0]],
            dtype=np.float64,
        )
        self._update_3d()

    def on_xz_camera_changed(self, event) -> None:
        camera_state = event.camera_state
        # pygfx_x -> data X (axis 2), pygfx_y -> data Z (axis 0)
        self._xz_centre_zyx = np.array(
            [camera_state.position[1], self._y_world, camera_state.position[0]],
            dtype=np.float64,
        )
        self._update_3d()

    def on_yz_camera_changed(self, event) -> None:
        camera_state = event.camera_state
        # pygfx_x -> data Y (axis 1), pygfx_y -> data Z (axis 0)
        self._yz_centre_zyx = np.array(
            [camera_state.position[1], camera_state.position[0], self._x_world],
            dtype=np.float64,
        )
        self._update_3d()

    def on_xy_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 0 in slice_indices:
            self._z_world = float(slice_indices[0])
            self._xy_centre_zyx[0] = self._z_world
        self._update_3d()

    def on_xz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 1 in slice_indices:
            self._y_world = float(slice_indices[1])
            self._xz_centre_zyx[1] = self._y_world
        self._update_3d()

    def on_yz_dims_changed(self, event) -> None:
        slice_indices = event.dims_state.selection.slice_indices
        if 2 in slice_indices:
            self._x_world = float(slice_indices[2])
            self._yz_centre_zyx[2] = self._x_world
        self._update_3d()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dtype_clim_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def _dtype_decimals(dtype: np.dtype) -> int:
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
    z_mid_world: int,
    y_mid_world: int,
    x_mid_world: int,
):
    """Construct the ViewerModel for the orthoviewer (no render layer).

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
        (near, far) clipping planes for the perspective cameras.
    z_mid_world, y_mid_world, x_mid_world : int
        Initial slice positions (world coordinates, rounded integers).

    Returns
    -------
    viewer_model : ViewerModel
        Fully assembled model, ready for CellierController.from_model.
    visuals : dict
        Mapping ``{"xy", "xz", "yz", "vol"}`` → MultiscaleImageVisual.
    visuals : dict
        Mapping ``{"xy", "xz", "yz", "vol"}`` → MultiscaleImageVisual.
    """
    from cellier.scene import Canvas
    from cellier.scene.cameras import (
        OrbitCameraController,
        OrthographicCamera,
        PanZoomCameraController,
        PerspectiveCamera,
    )
    from cellier.scene.dims import AxisAlignedSelection, DimsManager
    from cellier.scene.scene import Scene
    from cellier.viewer_model import DataManager, ViewerModel
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageVisual,
    )

    def _make_2d_canvas() -> Canvas:
        return Canvas(
            cameras={
                "2d": OrthographicCamera(
                    near_clipping_plane=depth_range[0],
                    far_clipping_plane=depth_range[1],
                    controller=PanZoomCameraController(enabled=True),
                )
            }
        )

    def _make_3d_canvas() -> Canvas:
        return Canvas(
            cameras={
                "3d": PerspectiveCamera(
                    fov=70.0,
                    near_clipping_plane=depth_range[0],
                    far_clipping_plane=depth_range[1],
                    controller=OrbitCameraController(enabled=True),
                )
            }
        )

    coarsest_level = data_store.n_levels - 1

    common_2d_appearance = MultiscaleImageAppearance(
        color_map="grays",
        clim=(0.0, initial_clim_max),
        lod_bias=1.0,
        force_level=None,
        frustum_cull=True,
        iso_threshold=0.2,
        render_mode="mip",
    )
    common_render_config = MultiscaleImageRenderConfig(
        block_size=32,
        gpu_budget_bytes=512 * 1024**2,
        gpu_budget_bytes_2d=64 * 1024**2,
    )

    def _make_2d_visual(name: str) -> MultiscaleImageVisual:
        return MultiscaleImageVisual(
            name=name,
            data_store_id=str(data_store.id),
            level_transforms=data_store.level_transforms,
            appearance=common_2d_appearance,
            render_config=common_render_config,
            transform=voxel_to_world,
        )

    # ── XY scene (slice Z) ────────────────────────────────────────────────
    xy_visual = _make_2d_visual("xy_volume")
    xy_scene = Scene(
        name="xy",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(1, 2),
                slice_indices={0: z_mid_world},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[xy_visual],
        canvases={},
    )
    xy_canvas = _make_2d_canvas()
    xy_scene.canvases[xy_canvas.id] = xy_canvas

    # ── XZ scene (slice Y) ────────────────────────────────────────────────
    xz_visual = _make_2d_visual("xz_volume")
    xz_scene = Scene(
        name="xz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 2),
                slice_indices={1: y_mid_world},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[xz_visual],
        canvases={},
    )
    xz_canvas = _make_2d_canvas()
    xz_scene.canvases[xz_canvas.id] = xz_canvas

    # ── YZ scene (slice X) ────────────────────────────────────────────────
    yz_visual = _make_2d_visual("yz_volume")
    yz_scene = Scene(
        name="yz",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1),
                slice_indices={2: x_mid_world},
            ),
        ),
        render_modes={"2d"},
        lighting="none",
        visuals=[yz_visual],
        canvases={},
    )
    yz_canvas = _make_2d_canvas()
    yz_scene.canvases[yz_canvas.id] = yz_canvas

    # ── Volume scene (3D) ─────────────────────────────────────────────────
    vol_visual = MultiscaleImageVisual(
        name="vol_volume",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=MultiscaleImageAppearance(
            color_map="grays",
            clim=(0.0, initial_clim_max),
            lod_bias=1.0,
            force_level=coarsest_level,
            frustum_cull=False,
            iso_threshold=0.2,
            render_mode="iso",
        ),
        render_config=MultiscaleImageRenderConfig(
            block_size=32,
            gpu_budget_bytes=2048 * 1024**2,
            gpu_budget_bytes_2d=64 * 1024**2,
        ),
        transform=voxel_to_world,
    )
    vol_visual.aabb.enabled = True
    vol_visual.aabb.color = "#ff00ff"

    vol_scene = Scene(
        name="vol",
        dims=DimsManager(
            coordinate_system=cs,
            selection=AxisAlignedSelection(
                displayed_axes=(0, 1, 2),
                slice_indices={},
            ),
        ),
        render_modes={"3d"},
        lighting="none",
        visuals=[vol_visual],
        canvases={},
    )
    vol_canvas = _make_3d_canvas()
    vol_scene.canvases[vol_canvas.id] = vol_canvas

    # ── Assemble ViewerModel ──────────────────────────────────────────────
    viewer_model = ViewerModel(
        data=DataManager(stores={data_store.id: data_store}),
        scenes={
            xy_scene.id: xy_scene,
            xz_scene.id: xz_scene,
            yz_scene.id: yz_scene,
            vol_scene.id: vol_scene,
        },
    )

    visuals = {
        "xy": xy_visual,
        "xz": xz_visual,
        "yz": yz_visual,
        "vol": vol_visual,
    }

    return viewer_model, visuals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def async_main(zarr_uri: str) -> None:
    from PySide6 import QtWidgets

    from cellier.controller import CellierController
    from cellier.data import OMEZarrImageDataStore
    from cellier.gui.qt import QtCanvasWidget, QtDimsControl
    from cellier.render._config import (
        RenderManagerConfig,
        SlicingConfig,
        TemporalAccumulationConfig,
    )
    from cellier.scene.dims import CoordinateSystem
    from cellier.transform import AffineTransform
    from cellier.visuals import (
        CenteredAxes2D,
        CenteredAxes2DAppearance,
    )

    # ── Open OME-Zarr store ───────────────────────────────────────────────
    print(f"Opening OME-Zarr store: {zarr_uri}")
    data_store = OMEZarrImageDataStore.from_path(zarr_uri)
    print(f"  {data_store.n_levels} levels found.")
    for i, shape in enumerate(data_store.level_shapes):
        print(f"  Level {i}: shape={shape}")
    print(f"  Axes:  {data_store.axis_names}")
    print(f"  Units: {data_store.axis_units}")

    # ── Physical scale from OME metadata ─────────────────────────────────
    import yaozarrs

    group = yaozarrs.open_group(data_store.zarr_path)
    ome_image = group.ome_metadata()
    ms = ome_image.multiscales[data_store.multiscale_index]
    level_0_scale_zyx = np.array(ms.datasets[0].scale_transform.scale, dtype=np.float64)
    print(f"\n  Level-0 physical scale (ZYX): {level_0_scale_zyx}")

    # ── World extents and depth range ─────────────────────────────────────
    vox_shape_zyx = np.array(data_store.level_shapes[0], dtype=np.float64)
    world_extents_zyx = vox_shape_zyx * level_0_scale_zyx
    max_extent = float(world_extents_zyx.max())
    depth_range = (max(1.0, max_extent * 0.0001), max_extent * 10.0)
    print(f"  World extents (ZYX): {world_extents_zyx}")
    print(f"  Max extent: {max_extent:.3f}")
    print(f"  Depth range: near={depth_range[0]:.2f}  far={depth_range[1]:.0f}\n")

    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    voxel_to_world = AffineTransform.from_scale_and_translation(
        scale=tuple(level_0_scale_zyx)
    )

    initial_clim_max = _dtype_clim_max(data_store.dtype)
    slider_decimals = _dtype_decimals(data_store.dtype)
    clim_range = (0.0, initial_clim_max)

    level0_shape = data_store.level_shapes[0]
    # axis_ranges must be in world coordinates because slice_indices are
    # world coordinates (the rendering pipeline maps them through the
    # voxel-to-world inverse transform to get voxel indices).
    # World extent for axis i = (N_i - 1) * physical_scale_i.
    # QLabeledSlider is integer-only, so we round to the nearest integer.
    # Slider steps of 1 world unit → 1/scale_i voxels per step.
    world_max_zyx = (np.array(level0_shape, dtype=np.float64) - 1) * level_0_scale_zyx
    axis_ranges = {
        i: (0, round(float(world_max_zyx[i]))) for i in range(len(level0_shape))
    }

    # Initial slice positions at mid-volume, in world coordinates (rounded to int).
    z_mid_world = round(float(world_max_zyx[0]) / 2.0)
    y_mid_world = round(float(world_max_zyx[1]) / 2.0)
    x_mid_world = round(float(world_max_zyx[2]) / 2.0)

    # ── Build ViewerModel (no render layer) ───────────────────────────────
    viewer_model, visuals = _build_viewer_model(
        data_store=data_store,
        cs=cs,
        voxel_to_world=voxel_to_world,
        initial_clim_max=initial_clim_max,
        depth_range=depth_range,
        z_mid_world=z_mid_world,
        y_mid_world=y_mid_world,
        x_mid_world=x_mid_world,
    )

    # ── Construct controller from model ───────────────────────────────────
    controller = CellierController.from_model(
        viewer_model,
        render_config=RenderManagerConfig(
            slicing=SlicingConfig(batch_size=32, render_every=4),
            temporal=TemporalAccumulationConfig(enabled=False),
        ),
        widget_parent=None,
    )

    # ── Retrieve scenes by name ───────────────────────────────────────────
    xy_scene = controller.get_scene_by_name("xy")
    xz_scene = controller.get_scene_by_name("xz")
    yz_scene = controller.get_scene_by_name("yz")
    vol_scene = controller.get_scene_by_name("vol")

    scenes = {"xy": xy_scene, "xz": xz_scene, "yz": yz_scene, "vol": vol_scene}

    # ── 3D axis-set mesh overlays (one per 2D view) ───────────────────────
    initial_centre_zyx = np.array(
        [z_mid_world, y_mid_world, x_mid_world], dtype=np.float64
    )
    xy_axis_visual, xz_axis_visual, yz_axis_visual = _make_axis_meshes(
        controller=controller,
        vol_scene=vol_scene,
        initial_centre_zyx=initial_centre_zyx,
        world_min_extent=float(world_max_zyx.min()),
    )

    # ── Slice plane mesh overlay ──────────────────────────────────────────
    _INITIAL_PLANE_OPACITY = 1.0

    plane_store, plane_visual = _make_plane_mesh(
        controller,
        vol_scene,
        z_mid_world,
        y_mid_world,
        x_mid_world,
        world_max_zyx,
        initial_opacity=_INITIAL_PLANE_OPACITY,
    )

    plane_updater = _PlaneUpdater(
        controller=controller,
        plane_store=plane_store,
        plane_visual=plane_visual,
        world_max_zyx=world_max_zyx,
    )

    controller.on_dims_changed(
        xy_scene.id, plane_updater.on_xy_dims_changed, owner_id=plane_updater._id
    )
    controller.on_dims_changed(
        xz_scene.id, plane_updater.on_xz_dims_changed, owner_id=plane_updater._id
    )
    controller.on_dims_changed(
        yz_scene.id, plane_updater.on_yz_dims_changed, owner_id=plane_updater._id
    )

    # ── Canvas views ──────────────────────────────────────────────────────
    def _canvas_view(scene_id):
        canvas_id = controller.get_canvas_ids(scene_id)[0]
        return controller.get_canvas_view(canvas_id)

    # ── Canvas widgets with per-panel slider colors ───────────────────────
    def _make_canvas_widget(scene, slider_style):
        canvas_view = _canvas_view(scene.id)
        axis_labels = dict(enumerate(scene.dims.coordinate_system.axis_labels))
        selection = scene.dims.selection
        dims_control = QtDimsControl(
            scene_id=scene.id,
            axis_ranges=axis_ranges,
            axis_labels=axis_labels,
            initial_slice_indices=dict(getattr(selection, "slice_indices", {})),
            initial_displayed_axes=getattr(selection, "displayed_axes", ()),
        )
        controller.connect_widget(
            dims_control, subscription_specs=dims_control.subscription_specs()
        )
        dims_control.widget.setStyleSheet(slider_style)
        return QtCanvasWidget(canvas_view=canvas_view, dims_control=dims_control)

    _vol_cw = QtCanvasWidget.from_scene_and_canvas(
        vol_scene, _canvas_view(vol_scene.id), axis_ranges=axis_ranges
    )
    controller.connect_widget(
        _vol_cw.dims_control,
        subscription_specs=_vol_cw.dims_control.subscription_specs(),
    )
    canvas_widgets = {
        "xy": _make_canvas_widget(xy_scene, SLIDER_STYLE_XY),
        "xz": _make_canvas_widget(xz_scene, SLIDER_STYLE_XZ),
        "yz": _make_canvas_widget(yz_scene, SLIDER_STYLE_YZ),
        "vol": _vol_cw,
    }

    # ── Screen-space 2D axis overlays ─────────────────────────────────────
    # axis_a_direction=(0,1,0) is world +Y, which maps to screen-up for the
    # standard PanZoom camera (view_dir=(0,0,-1), up=(0,1,0)).
    # axis_b_direction=(1,0,0) is world +X, which maps to screen-right.
    # Labels differ per panel because different data axes are displayed.

    # XY panel: screen-up = data Y, screen-right = data X
    xy_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(xy_scene.id)[0],
        CenteredAxes2D(
            name="xy_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Y",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="X",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XZ, 1.0),
                axis_b_color=(*_PLANE_COLOR_YZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )

    # XZ panel: screen-up = data Z, screen-right = data X
    xz_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(xz_scene.id)[0],
        CenteredAxes2D(
            name="xz_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Z",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="X",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XY, 1.0),
                axis_b_color=(*_PLANE_COLOR_YZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )

    # YZ panel: screen-up = data Z, screen-right = data Y
    yz_axes_overlay = controller.add_canvas_overlay_model(
        controller.get_canvas_ids(yz_scene.id)[0],
        CenteredAxes2D(
            name="yz_axes",
            axis_a_direction=(0.0, 1.0, 0.0),
            axis_a_label="Z",
            axis_b_direction=(1.0, 0.0, 0.0),
            axis_b_label="Y",
            appearance=CenteredAxes2DAppearance(
                axis_a_color=(*_PLANE_COLOR_XY, 1.0),
                axis_b_color=(*_PLANE_COLOR_XZ, 1.0),
                label_color=(1.0, 0.0, 1.0, 1.0),
            ),
        ),
    )

    # ── Build viewer window ───────────────────────────────────────────────
    viewer = OmeZarrOrthoViewer(
        controller,
        scenes=scenes,
        visuals=visuals,
        canvas_widgets=canvas_widgets,
        clim_range=clim_range,
        slider_decimals=slider_decimals,
        plane_visual=plane_visual,
        plane_store=plane_store,
        vol_visual_id=visuals["vol"].id,
        initial_plane_opacity=_INITIAL_PLANE_OPACITY,
        axes_2d_overlay_ids=[
            xy_axes_overlay.id,
            xz_axes_overlay.id,
            yz_axes_overlay.id,
        ],
        orient_3d_visual_ids=[
            xy_axis_visual.id,
            xz_axis_visual.id,
            yz_axis_visual.id,
        ],
    )
    viewer.window.show()

    # ── 3D orientation overlay wiring ─────────────────────────────────────
    orient_updater = _OrientationUpdater(
        controller=controller,
        xy_axis_visual=xy_axis_visual,
        xz_axis_visual=xz_axis_visual,
        yz_axis_visual=yz_axis_visual,
        world_max_zyx=world_max_zyx,
    )

    controller.on_camera_changed(
        xy_scene.id, orient_updater.on_xy_camera_changed, owner_id=orient_updater._id
    )
    controller.on_camera_changed(
        xz_scene.id, orient_updater.on_xz_camera_changed, owner_id=orient_updater._id
    )
    controller.on_camera_changed(
        yz_scene.id, orient_updater.on_yz_camera_changed, owner_id=orient_updater._id
    )

    controller.on_dims_changed(
        xy_scene.id, orient_updater.on_xy_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        xz_scene.id, orient_updater.on_xz_dims_changed, owner_id=orient_updater._id
    )
    controller.on_dims_changed(
        yz_scene.id, orient_updater.on_yz_dims_changed, owner_id=orient_updater._id
    )

    # ── Fit cameras and trigger initial reslice ───────────────────────────
    for scene in scenes.values():
        controller.fit_camera(scene.id)
        controller.reslice_scene(scene.id)

    # Seed 3D orientation with post-fit camera state.
    def _seed_camera_event(scene_id):
        from cellier.events._events import CameraChangedEvent

        canvas_id = controller.get_canvas_ids(scene_id)[0]
        return CameraChangedEvent(
            source_id=canvas_id,
            scene_id=scene_id,
            camera_state=controller.get_camera_state(canvas_id),
        )

    orient_updater.on_xy_camera_changed(_seed_camera_event(xy_scene.id))
    orient_updater.on_xz_camera_changed(_seed_camera_event(xz_scene.id))
    orient_updater.on_yz_camera_changed(_seed_camera_event(yz_scene.id))

    app = QtWidgets.QApplication.instance()
    app.aboutToQuit.connect(viewer.close_widgets)
    close_event = asyncio.Event()
    app.aboutToQuit.connect(close_event.set)
    await close_event.wait()


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-file-path",
        metavar="PATH",
        default=None,
        help=(
            "Path or URI to the OME-Zarr store "
            "(local path, file://, s3://, gs://, https://)."
        ),
    )
    parser.add_argument(
        "--make-example",
        action="store_true",
        help=(
            "Create a synthetic anisotropic OME-Zarr (ExpA-like scale) with "
            "spherical blobs and open it in the viewer."
        ),
    )
    parser.add_argument(
        "--debug-2d-reslice",
        action="store_true",
        help="Enable DEBUG logging for the 2D reslice pipeline (tile selection & data regions).",
    )
    args = parser.parse_args()

    if not args.make_example and args.zarr_file_path is None:
        parser.error("one of --zarr-file-path or --make-example is required")

    if args.make_example:
        zarr_path = make_example_zarr()
        zarr_uri = f"file://{zarr_path}"
    else:
        zarr_input = args.zarr_file_path
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

    from cellier.logging import enable_debug_logging

    enable_debug_logging(categories=("perf",), level=logging.WARN)

    if args.debug_2d_reslice:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("cellier.2d_reslice_debug").setLevel(logging.DEBUG)

    app = QApplication([sys.argv[0]])
    QtAsyncio.run(async_main(zarr_uri), handle_sigint=True)
