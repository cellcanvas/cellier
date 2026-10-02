"""Demo: LinesMemoryVisual with 3D display, 2D z-slicing, and color randomisation.

Run with:
    uv run --script scripts/v2/lines_memory_demo.py
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
from cellier.data import LinesMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import LinesMemoryAppearance

VOLUME_SIZE = 64
N_SEGMENTS = 500
RNG = np.random.default_rng(42)


def _make_positions() -> np.ndarray:
    """Return (N_SEGMENTS * 2, 3) float32 positions for N_SEGMENTS random segments.

    Each consecutive pair of rows is one segment.  Columns are in (z, y, x)
    data order.  Segments are kept short (≤ 8 units) so that many survive
    the 2D slab filter and the display is still interesting.
    """
    starts = RNG.uniform(0, VOLUME_SIZE, size=(N_SEGMENTS, 3)).astype(np.float32)
    offsets = RNG.uniform(-4, 4, size=(N_SEGMENTS, 3)).astype(np.float32)
    ends = np.clip(starts + offsets, 0, VOLUME_SIZE).astype(np.float32)
    # Interleave start/end rows: [s0, e0, s1, e1, ...]
    positions = np.empty((N_SEGMENTS * 2, 3), dtype=np.float32)
    positions[0::2] = starts
    positions[1::2] = ends
    return positions


def _random_colors(n_vertices: int) -> np.ndarray:
    """Return (n_vertices, 4) float32 RGBA with alpha=1."""
    colors = RNG.random(size=(n_vertices, 4)).astype(np.float32)
    colors[:, 3] = 1.0
    return colors


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    positions = _make_positions()
    n_vertices = positions.shape[0]
    initial_colors = _random_colors(n_vertices)

    store = LinesMemoryStore(
        positions=positions,
        colors=initial_colors,
        name="demo_lines",
    )

    # ── 2. Controller and scene ─────────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("z", "y", "x"))

    scene = controller.add_scene(
        dim="3d",
        coordinate_system=cs,
        name="lines",
        render_modes={"2d", "3d"},
    )

    # A slice with no thickness is a plane: only what lies exactly on it is
    # drawn.  Give the sliced z axis a slab, so the 2D view shows the segments
    # within half a unit of the slice.
    controller.update_thickness(scene.id, {0: 0.5})

    appearance = LinesMemoryAppearance(
        color=(0.2, 0.9, 0.4, 1.0),
        thickness=2.0,
        thickness_space="screen",
        color_mode="vertex",
    )
    visual = controller.add_lines(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        name="line_segments",
    )

    # ── 3. Build UI ─────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Cellier v2 — Lines Memory Demo")
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
        store.colors = _random_colors(store.positions.shape[0])
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
