"""Demo: PointsMemoryVisual with 3D display, 2D z-slicing, and color randomisation.

Run with:
    uv run --script scripts/v2/points_memory_demo.py
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtAsyncio
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
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._points_memory import PointsMarkerAppearance

VOLUME_SIZE = 64
N_POINTS = 2000
RNG = np.random.default_rng(42)


def _make_positions() -> np.ndarray:
    """Return (N_POINTS, 3) float32 positions scattered in a VOLUME_SIZE cube.

    Columns are in (z, y, x) data order.
    """
    return RNG.uniform(0, VOLUME_SIZE, size=(N_POINTS, 3)).astype(np.float32)


def _random_colors(n: int) -> np.ndarray:
    """Return (n, 4) float32 RGBA with alpha=1."""
    colors = RNG.random(size=(n, 4)).astype(np.float32)
    colors[:, 3] = 1.0
    return colors


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    positions = _make_positions()
    initial_colors = _random_colors(N_POINTS)

    store = PointsMemoryStore(
        positions=positions,
        colors=initial_colors,
        name="demo_points",
    )

    # ── 2. Controller and scene ─────────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="points",
        render_modes={"2d", "3d"},
    )

    # A slice with no thickness is a plane: only what lies exactly on it is
    # drawn.  Give the sliced z axis a slab, so the 2D view shows the points
    # within half a unit of the slice.
    controller.update_thickness(scene.id, {0: 0.5})

    appearance = PointsMarkerAppearance(
        color=(0.2, 0.8, 1.0, 1.0),
        size=6.0,
        size_space="screen",
        color_mode="vertex",
    )
    visual = controller.add_points(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="point_cloud",
    )

    # ── 3. Build UI ─────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Cellier v2 — Points Memory Demo")
    outer = QVBoxLayout(root)
    controller.set_widget_parent(root)

    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    outer.addWidget(canvas_widget)

    # Controls row
    controls = QWidget()
    ctrl_layout = QHBoxLayout(controls)
    ctrl_layout.setContentsMargins(4, 4, 4, 4)

    z_label = QLabel("Z slice:")
    z_spin = QSpinBox()
    z_spin.setRange(0, VOLUME_SIZE - 1)
    z_spin.setValue(VOLUME_SIZE // 2)
    z_label.setVisible(False)
    z_spin.setVisible(False)
    ctrl_layout.addWidget(z_label)
    ctrl_layout.addWidget(z_spin)
    ctrl_layout.addStretch()

    color_btn = QPushButton("Randomize Colors")
    ctrl_layout.addWidget(color_btn)

    toggle_btn = QPushButton("Switch to 2D")
    ctrl_layout.addWidget(toggle_btn)

    outer.addWidget(controls)
    root.resize(900, 700)
    root.show()

    # ── 4. Interaction logic ────────────────────────────────────────────
    _current_dim = ["3d"]

    def _toggle() -> None:
        if _current_dim[0] == "3d":
            _current_dim[0] = "2d"
            toggle_btn.setText("Switch to 3D")
            z_label.setVisible(True)
            z_spin.setVisible(True)
            scene.dims.selection.slice_indices = {0: z_spin.value()}
            scene.dims.selection.displayed_axes = (1, 2)
        else:
            _current_dim[0] = "3d"
            toggle_btn.setText("Switch to 2D")
            z_label.setVisible(False)
            z_spin.setVisible(False)
            scene.dims.selection.displayed_axes = (0, 1, 2)
        controller.reslice_scene(scene.id)

    toggle_btn.clicked.connect(_toggle)

    def _on_z_changed(z: int) -> None:
        scene.dims.selection.slice_indices = {0: z}
        if _current_dim[0] == "2d":
            controller.reslice_scene(scene.id)

    z_spin.valueChanged.connect(_on_z_changed)

    def _randomize_colors() -> None:
        """Assign new random colors and trigger a reslice.

        Must update the store field (not mutate in place) so psygnal
        EventedModel detects the change.
        """
        store.colors = _random_colors(store.n_points)
        controller.reslice_scene(scene.id)

    color_btn.clicked.connect(_randomize_colors)

    # ── 5. Camera fit on first data delivery ────────────────────────────
    _camera_fitted: set[str] = set()

    def _on_data_ready(event):
        mode = _current_dim[0]
        if mode not in _camera_fitted:
            _camera_fitted.add(mode)
            if mode == "3d":
                controller.look_at_visual(
                    visual.id, canvas_id, view_direction=(-1, -1, -1), up=(0, 0, 1)
                )
            else:
                controller.look_at_visual(
                    visual.id, canvas_id, view_direction=(0, 0, -1), up=(0, 1, 0)
                )

    controller.on_reslice_completed(visual.id, _on_data_ready, owner_id=controller._id)

    # ── 6. Initial reslice ──────────────────────────────────────────────
    QTimer.singleShot(0, controller.reslice_all)

    QtAsyncio.run()


if __name__ == "__main__":
    main()
