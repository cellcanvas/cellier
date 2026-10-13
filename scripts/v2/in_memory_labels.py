#!/usr/bin/env python
"""Demo: 4D in-memory label volume with 2D/3D toggle and salt control.

World shape is (300, 300, 300) µm in Z/Y/X with an anisotropic voxel size:
  Z: 300/48 ≈ 6.25 µm/voxel  (coarser — simulates axial anisotropy)
  Y: 300/48 ≈ 6.25 µm/voxel
  X: 300/48 ≈ 6.25 µm/voxel

... wait — to make the scale *anisotropic*, we use different voxel counts:
  data shape  : (T=4, Z=30, Y=60, X=60)
  voxel scale : (Z=10 µm, Y=5 µm, X=5 µm)
  world extent: (Z=300 µm, Y=300 µm, X=300 µm)

Run with:
    uv run python scripts/v2/in_memory_labels.py
"""

from __future__ import annotations

import sys

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.controller import CellierController
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import InMemoryLabelsAppearance

# ── Data parameters ─────────────────────────────────────────────────────────
N_T = 4
N_Z = 150  # coarser Z (10 µm/voxel → 300 µm world)
N_Y = 300  # finer  Y  ( 5 µm/voxel → 300 µm world)
N_X = 300  # finer  X  ( 5 µm/voxel → 300 µm world)

# Anisotropic voxel size in µm (Z, Y, X)
VOXEL_SIZE_ZYX = (10.0, 5.0, 5.0)


def _make_4d_labels() -> np.ndarray:
    """Return (T, Z, Y, X) int32 label array; each T frame independently labeled."""
    from skimage.data import binary_blobs
    from skimage.measure import label as skimage_label

    frames = [
        skimage_label(
            binary_blobs(
                length=N_Z,
                n_dim=3,
                volume_fraction=0.04,
                rng=t * 17,
            )
        ).astype(np.int32)
        for t in range(N_T)
    ]
    # Resize each Z-frame to (N_Z, N_Y, N_X) — skimage generates cubes.
    # Re-label after zoom to preserve integer integrity.
    import scipy.ndimage as ndi

    resized = []
    for frame in frames:
        # Zoom to the target non-cubic shape.
        zoom_factors = (1.0, N_Y / N_Z, N_X / N_Z)
        zoomed = ndi.zoom(frame.astype(np.float32), zoom_factors, order=0)
        resized.append(zoomed.astype(np.int32))
    return np.stack(resized)  # (N_T, N_Z, N_Y, N_X)


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create label data ────────────────────────────────────────────────
    print("Generating label data...")
    labels = _make_4d_labels()
    print(f"  shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  unique labels: {np.unique(labels).size}")
    store = LabelMemoryStore(data=labels, name="demo_labels")

    # ── 2. Data-to-world scale ──────────────────────────────────────────────
    # Data axes: (T, Z, Y, X).  World axes in the scene: (t, z, y, x).
    # The transform maps data voxel index → world position in µm.
    # T axis has a scale of 1 (unitless frame index).
    scale_tzyx = np.array([1.0, *VOXEL_SIZE_ZYX])

    # ── 3. Set up the controller ────────────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("t", "z", "y", "x"))

    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="labels_scene",
        render_modes={"2d", "3d"},
    )

    # The transform names the two systems it maps between, so it is built
    # once both exist: the store's voxel space and the scene's world.
    data_to_world = controller.data_to_world(scene.id, store, scale=scale_tzyx)

    appearance = InMemoryLabelsAppearance(
        colormap_mode="random",
        render_mode="iso_categorical",
    )
    visual = controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="demo_labels",
        transform=data_to_world,
    )

    # ── 4. Build UI ─────────────────────────────────────────────────────────
    root = QWidget()
    outer_layout = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer_layout.addWidget(canvas_widget)

    # Controls
    controls = QWidget()
    ctrl_layout = QHBoxLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)

    toggle_btn = QPushButton("Switch to 2D")
    ctrl_layout.addWidget(toggle_btn)

    ctrl_layout.addWidget(QLabel("t:"))
    t_spin = QSpinBox()
    t_spin.setRange(0, N_T - 1)
    t_spin.setValue(0)
    ctrl_layout.addWidget(t_spin)

    ctrl_layout.addWidget(QLabel("z (2D only):"))
    z_spin = QSpinBox()
    z_spin.setRange(0, N_Z - 1)
    z_spin.setValue(N_Z // 2)
    z_spin.setEnabled(False)
    ctrl_layout.addWidget(z_spin)

    ctrl_layout.addWidget(QLabel("salt:"))
    salt_spin = QSpinBox()
    salt_spin.setRange(0, 9999)
    salt_spin.setValue(0)
    ctrl_layout.addWidget(salt_spin)

    ctrl_layout.addStretch()
    outer_layout.addWidget(controls)

    # World-size info label
    world_z = N_Z * VOXEL_SIZE_ZYX[0]
    world_y = N_Y * VOXEL_SIZE_ZYX[1]
    world_x = N_X * VOXEL_SIZE_ZYX[2]
    info = QLabel(
        f"Data: (T={N_T}, Z={N_Z}, Y={N_Y}, X={N_X})  |  "
        f"Voxel: ({VOXEL_SIZE_ZYX[0]:.1f}, {VOXEL_SIZE_ZYX[1]:.1f}, "
        f"{VOXEL_SIZE_ZYX[2]:.1f}) µm  |  "
        f"World: ({world_z:.0f}×{world_y:.0f}×{world_x:.0f}) µm"
    )
    outer_layout.addWidget(info)

    root.resize(960, 720)
    root.show()

    # ── 5. Toggle logic ─────────────────────────────────────────────────────
    _mode = ["3d"]

    def _apply_dims() -> None:
        if _mode[0] == "3d":
            # T axis: world position = t_index * scale_t (=1.0 since scale_t=1).
            scene.dims.selection.slice_indices = {
                0: int(t_spin.value() * scale_tzyx[0])
            }
            scene.dims.selection.displayed_axes = (1, 2, 3)
        else:
            # World-space positions for slice axes (T and Z).
            scene.dims.selection.slice_indices = {
                0: int(t_spin.value() * scale_tzyx[0]),
                1: int(z_spin.value() * scale_tzyx[1]),
            }
            scene.dims.selection.displayed_axes = (2, 3)
        controller.reslice_scene(scene.id)

    def _toggle() -> None:
        if _mode[0] == "3d":
            _mode[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            z_spin.setEnabled(True)
        else:
            _mode[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            z_spin.setEnabled(False)
        _apply_dims()

    toggle_btn.clicked.connect(_toggle)
    t_spin.valueChanged.connect(lambda _: _apply_dims())
    z_spin.valueChanged.connect(lambda _: _apply_dims())

    # Salt changes update the appearance uniform directly (no reslice).
    salt_spin.valueChanged.connect(lambda v: setattr(visual.appearance, "salt", v))

    # ── 6. Initial reslice + camera fit ─────────────────────────────────────
    _camera_fitted: set[str] = set()

    def _on_data_ready(event):
        mode = _mode[0]
        if mode not in _camera_fitted:
            _camera_fitted.add(mode)
            controller.look_at_visual(visual.id, canvas_id)

    controller.on_reslice_completed(visual.id, _on_data_ready, owner_id=controller._id)

    QTimer.singleShot(0, controller.reslice_all)

    QtAsyncio.run()


if __name__ == "__main__":
    main()
