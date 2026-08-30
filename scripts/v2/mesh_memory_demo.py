#!/usr/bin/env python
"""Demo: in-memory mesh with toggleable 2D/3D view.

Creates a sphere isosurface via marching cubes.  In 2D mode the slab
filter shows only faces within ±0.5 of the current Z slice.

Run with:
    uv run python scripts/v2/mesh_memory_demo.py
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
from skimage.measure import marching_cubes

from cellier.controller import CellierController
from cellier.data import MeshMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import MeshPhongAppearance

RNG = np.random.default_rng(0)


def _make_sphere_mesh():
    """Sphere of radius 40 in a 100^3 volume via marching cubes."""
    size = 100
    c = size // 2
    z, y, x = np.mgrid[:size, :size, :size]
    vol = (x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2
    verts, faces, _, _ = marching_cubes(vol, level=40**2)
    # marching_cubes returns verts in (z, y, x) row order — correct for cellier.
    return verts.astype(np.float32), faces.astype(np.int32)


def _random_vertex_colors(n: int) -> np.ndarray:
    colors = RNG.random((n, 4)).astype(np.float32)
    colors[:, 3] = 1.0
    return colors


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Build mesh ────────────────────────────────────────────────
    verts, faces = _make_sphere_mesh()
    store = MeshMemoryStore(
        positions=verts,
        indices=faces,
        # Declared up front so the Randomize button can assign colours later;
        # the layout is never inferred from the array length.
        colors=_random_vertex_colors(verts.shape[0]),
        colors_layout="vertex",
        name="sphere",
    )
    z_min = int(verts[:, 0].min())
    z_max = int(verts[:, 0].max())
    z_mid = (z_min + z_max) // 2

    # ── 2. Controller and scene ──────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))
    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="sphere_scene",
        render_modes={"2d", "3d"},
        lighting="default",  # required for MeshPhongAppearance
    )

    appearance = MeshPhongAppearance(
        color=(0.4, 0.7, 1.0, 1.0),
        shininess=60.0,
        side="front",
    )
    visual = controller.add_mesh(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="sphere_mesh",
    )

    # ── 3. Build UI ──────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Cellier v2 — Mesh Memory Demo")
    outer = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer.addWidget(canvas_widget)

    controls = QWidget()
    ctrl = QHBoxLayout(controls)
    ctrl.setContentsMargins(4, 4, 4, 4)

    z_label = QLabel("Z slice:")
    z_spin = QSpinBox()
    z_spin.setRange(z_min, z_max)
    z_spin.setValue(z_mid)
    z_label.setVisible(False)
    z_spin.setVisible(False)
    ctrl.addWidget(z_label)
    ctrl.addWidget(z_spin)
    ctrl.addStretch()

    color_btn = QPushButton("Randomize Colors")
    toggle_btn = QPushButton("Switch to 2D")
    ctrl.addWidget(color_btn)
    ctrl.addWidget(toggle_btn)
    outer.addWidget(controls)

    root.resize(900, 700)
    root.show()

    # ── 4. Interaction ───────────────────────────────────────────────
    _dim = ["3d"]

    def _toggle() -> None:
        if _dim[0] == "3d":
            _dim[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            z_label.setVisible(True)
            z_spin.setVisible(True)
            scene.dims.selection.slice_indices = {0: z_spin.value()}
            scene.dims.selection.displayed_axes = (1, 2)
        else:
            _dim[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            z_label.setVisible(False)
            z_spin.setVisible(False)
            scene.dims.selection.displayed_axes = (0, 1, 2)
        controller.reslice_scene(scene.id)

    toggle_btn.clicked.connect(_toggle)

    def _on_z(z: int) -> None:
        scene.dims.selection.slice_indices = {0: z}
        if _dim[0] == "2d":
            controller.reslice_scene(scene.id)

    z_spin.valueChanged.connect(_on_z)

    def _randomize() -> None:
        """Assign new random per-vertex colors and trigger a reslice.

        Updates store.colors directly (psygnal EventedModel emits the
        field-change signal).  The store's colors_layout must already be
        set -- it is declared at construction below -- because the layout
        decides which rows a slice gathers and is never inferred.

        The appearance's color_mode is what decides whether the material
        uses them, and it is honoured verbatim (D20), so it has to be set
        too.
        """
        store.colors = _random_vertex_colors(store.n_vertices)
        visual.appearance.color_mode = "vertex"
        controller.reslice_scene(scene.id)

    color_btn.clicked.connect(_randomize)

    # ── 5. Camera fit on first data delivery ─────────────────────────
    _fitted: set[str] = set()

    def _on_data_ready(event):
        mode = _dim[0]
        if mode not in _fitted:
            _fitted.add(mode)
            if mode == "3d":
                controller.look_at_visual(
                    visual.id, canvas_id, view_direction=(-1, -1, -1), up=(0, 0, 1)
                )
            else:
                controller.look_at_visual(
                    visual.id, canvas_id, view_direction=(0, 0, -1), up=(0, 1, 0)
                )

    controller.on_reslice_completed(visual.id, _on_data_ready, owner_id=controller._id)

    # ── 6. Initial reslice ───────────────────────────────────────────
    QTimer.singleShot(0, controller.reslice_all)
    QtAsyncio.run()


if __name__ == "__main__":
    main()
