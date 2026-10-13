"""Demo: SyncPaintController on a 256x256 in-memory int32 label array.

Click and drag on the image to paint integer label IDs.  Different label
values produce different colors via the random colormap.  The toolbar
exposes brush value, brush radius, undo, commit, and abort.

Run with:
    uv run python scripts/v2/in_memory_paint_labels.py
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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellier.controller import CellierController
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import InMemoryLabelsAppearance, LabelMemoryVisual
from cellier.visuals._points_memory import PointsMarkerAppearance


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Data ─────────────────────────────────────────────────────────
    label_data = np.zeros((256, 256), dtype=np.int32)
    label_store = LabelMemoryStore(data=label_data, name="labels")

    # Magenta dots at known (y, x) data coordinates for coordinate verification.
    ANCHOR_COORDS = [
        (50, 50),
        (50, 200),
        (200, 50),
        (200, 200),
    ]
    point_positions = np.array(ANCHOR_COORDS, dtype=np.float32)
    points_store = PointsMemoryStore(positions=point_positions, name="anchors")

    # ── 2. Controller and 2D scene ──────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    appearance = InMemoryLabelsAppearance(colormap_mode="random")
    label_visual_model = LabelMemoryVisual(
        name="labels",
        data_store_id=str(label_store.id),
        appearance=appearance,
    )
    label_visual = controller.add_visual(
        scene.id, label_visual_model, data_store=label_store
    )

    points_appearance = PointsMarkerAppearance(
        color=(1.0, 0.0, 1.0, 1.0),
        size=20.0,
        size_space="screen",
        opacity=0.5,
        depth_test=False,
        render_order=100,
    )
    controller.add_points(
        data=points_store,
        scene_id=scene.id,
        appearance=points_appearance,
        name="anchors",
    )

    # ── 3. Qt window ────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("In-memory label paint demo")
    layout = QVBoxLayout(root)

    toolbar = QWidget(root)
    toolbar_layout = QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(4, 4, 4, 4)

    brush_value_label = QLabel("label id")
    brush_value_spin = QSpinBox()
    brush_value_spin.setRange(0, 255)
    brush_value_spin.setValue(1)

    brush_radius_label = QLabel("radius (vox)")
    brush_radius_spin = QSpinBox()
    brush_radius_spin.setRange(1, 30)
    brush_radius_spin.setValue(12)

    undo_btn = QPushButton("Undo")
    commit_btn = QPushButton("Commit")
    abort_btn = QPushButton("Abort")

    for w in (
        brush_value_label,
        brush_value_spin,
        brush_radius_label,
        brush_radius_spin,
        undo_btn,
        commit_btn,
        abort_btn,
    ):
        toolbar_layout.addWidget(w)
    toolbar_layout.addStretch(1)
    layout.addWidget(toolbar)

    controller.set_widget_parent(root)
    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    layout.addWidget(canvas_widget)
    root.resize(700, 750)

    # ── 4. Camera fit on first delivery ──────────────────────────────────
    _fitted = False

    def _on_first_data(event):
        nonlocal _fitted
        if not _fitted:
            _fitted = True
            controller.look_at_visual(label_visual.id, canvas_id)

    controller.on_reslice_completed(
        label_visual.id, _on_first_data, owner_id=controller._id
    )
    QTimer.singleShot(0, controller.reslice_all)

    # ── 5. Paint controller ──────────────────────────────────────────────
    paint_ctrl = controller.add_paint_controller(
        visual_id=label_visual.id,
        canvas_id=canvas_id,
        brush_value=float(brush_value_spin.value()),
        brush_radius_voxels=float(brush_radius_spin.value()),
    )

    def _on_brush_value_changed(value: int) -> None:
        paint_ctrl.brush_value = float(value)

    def _on_brush_radius_changed(value: int) -> None:
        paint_ctrl.brush_radius_voxels = float(value)

    def _on_undo_clicked() -> None:
        paint_ctrl.undo()

    def _on_commit_clicked() -> None:
        paint_ctrl.commit()
        QMessageBox.information(root, "Commit", "Stroke history cleared.")
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _on_abort_clicked() -> None:
        paint_ctrl.abort()
        QMessageBox.information(root, "Abort", "All strokes reverted.")
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    brush_value_spin.valueChanged.connect(_on_brush_value_changed)
    brush_radius_spin.valueChanged.connect(_on_brush_radius_changed)
    undo_btn.clicked.connect(_on_undo_clicked)
    commit_btn.clicked.connect(_on_commit_clicked)
    abort_btn.clicked.connect(_on_abort_clicked)

    root.show()
    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    main()
