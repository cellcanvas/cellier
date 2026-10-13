#!/usr/bin/env python
"""Demo: in-memory 3D image with toggleable 2D/3D view in a single scene.

Run with:
    uv run python scripts/v2/image_memory_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
import PySide6.QtAsyncio as QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.data import binary_blobs

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._image_memory import InMemoryImageAppearance


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    volume = binary_blobs(length=128, volume_fraction=0.03, n_dim=3).astype(np.float32)
    store = ImageMemoryStore(data=volume, name="demo_volume")

    # ── 2. Set up the controller ────────────────────────────────────────
    controller = CellierController()

    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    # Single scene that supports both 2D and 3D rendering.
    # Start in 3D; displayed_axes=(0,1,2).
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="volume",
        render_modes={"2d", "3d"},
    )

    appearance = InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0))
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="volume_view",
    )
    visual.aabb.color = "#ff00ff"

    # ── 3. Build UI ─────────────────────────────────────────────────────
    root = QWidget()
    outer_layout = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer_layout.addWidget(canvas_widget)

    # Controls row
    controls = QWidget()
    ctrl_layout = QHBoxLayout(controls)
    ctrl_layout.setContentsMargins(0, 0, 0, 0)

    z_depth = store.shape[0]
    ctrl_layout.addWidget(QLabel("Z slice:"))
    z_spin = QSpinBox()
    z_spin.setRange(0, z_depth - 1)
    z_spin.setValue(z_depth // 2)
    ctrl_layout.addWidget(z_spin)
    ctrl_layout.addStretch()

    aabb_check = QCheckBox("Show bounding box")
    aabb_check.setChecked(False)
    ctrl_layout.addWidget(aabb_check)

    toggle_btn = QPushButton("Switch to 2D")
    ctrl_layout.addWidget(toggle_btn)
    outer_layout.addWidget(controls)

    root.resize(900, 700)
    root.show()

    # ── 4. Toggle logic ─────────────────────────────────────────────────
    _current_dim = ["3d"]

    def _toggle() -> None:
        if _current_dim[0] == "3d":
            _current_dim[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            scene.dims.selection.slice_indices = {0: z_spin.value()}
            scene.dims.selection.displayed_axes = (1, 2)
        else:
            _current_dim[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            scene.dims.selection.displayed_axes = (0, 1, 2)

        controller.reslice_scene(scene.id)

    toggle_btn.clicked.connect(_toggle)

    def _on_z_changed(z: int) -> None:
        scene.dims.selection.slice_indices = {0: z}
        if _current_dim[0] == "2d":
            controller.reslice_scene(scene.id)

    z_spin.valueChanged.connect(_on_z_changed)

    def _on_aabb_toggled(checked: bool) -> None:
        visual.aabb.enabled = checked

    aabb_check.toggled.connect(_on_aabb_toggled)

    # ── 5. Initial reslice + camera fit ────────────────────────────────
    _camera_fitted: set[str] = set()

    def _on_data_ready(event):
        mode = _current_dim[0]
        if mode not in _camera_fitted:
            _camera_fitted.add(mode)
            controller.look_at_visual(visual.id, canvas_id)

    controller.on_reslice_completed(visual.id, _on_data_ready, owner_id=controller._id)

    QTimer.singleShot(0, controller.reslice_all)

    QtAsyncio.run()


if __name__ == "__main__":
    main()
