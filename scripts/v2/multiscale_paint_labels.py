"""Demo: MultiscalePaintController on a 2-D OME-Zarr label store.

This demo paints in 2D into a multiscale labels image.

What happens when you paint
---------------------------
- ``_apply_brush`` stages the new integer label IDs into a tensorstore
  transaction (RAM only, owned by the WriteBuffer).
- The paint controller marks the dirty level-0 bricks and writes the
  values directly into the visual's GPU paint cache + paint LUT via
  ``controller._patch_painted_tiles_2d`` (internal — called by ``MultiscalePaintController``).
- The label shader composites the paint cache over the base sample so
  painted voxels show the correct random color on the very next frame —
  no slicer round-trip required.

Autosave (every 30 seconds, when dirty)
----------------------------------------
- The WriteBuffer flushes the current transaction to disk.
- The pyramid is rebuilt bottom-up for dirty bricks using stride-2
  decimation — all writes share the same transaction so the whole
  pyramid is atomic.
- A fresh WriteBuffer replaces the old one.  Undo history is preserved.
- The GPU paint textures are cleared and the base cache is evicted +
  resliced from disk.

Commit
-------
Same as autosave, but also clears undo history and tears down the
session.  After commit the buttons are disabled.

Abort
------
Discards the transaction without touching disk.

Run with::

    uv run scripts/v2/multiscale_paint_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr
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
from cellier.data import OMEZarrLabelDataStore
from cellier.scene.dims import CoordinateSystem
from cellier.visuals import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
    MultiscaleLabelVisual,
)

_DEMO_BLOCK_SIZE = 32
_DEMO_CHUNK = (64, 64)

_LEVEL_SHAPES = [(256, 256), (128, 128), (64, 64)]
_LEVEL_SCALES = [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]


def _ensure_demo_ome_zarr(target_dir: Path) -> Path:
    """Return the demo OME-Zarr label store at *target_dir*, creating it if missing."""
    if target_dir.exists():
        print(f"  Loading existing demo OME-Zarr from {target_dir}")
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(target_dir), mode="w")

    datasets = []
    for i, (shape, scale) in enumerate(zip(_LEVEL_SHAPES, _LEVEL_SCALES)):
        path = f"s{i}"
        arr = root.create_array(path, shape=shape, chunks=_DEMO_CHUNK, dtype=np.int32)
        arr[:] = np.zeros(shape, dtype=np.int32)
        datasets.append(
            {
                "path": path,
                "coordinateTransformations": [{"type": "scale", "scale": scale}],
            }
        )

    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": [
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": datasets,
                "name": "demo_labels",
            }
        ],
    }
    print(f"  Created demo int32 OME-Zarr (3 levels) at {target_dir}")
    return target_dir


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Demo data on disk ────────────────────────────────────────────
    zarr_path = Path(__file__).resolve().parent / "multiscale_paint_labels.ome.zarr"
    _ensure_demo_ome_zarr(zarr_path)

    zarr_uri = f"file://{zarr_path.resolve()}"
    data_store = OMEZarrLabelDataStore.from_path(zarr_uri, name="labels")
    print(
        f"  Opened label store: levels={data_store.n_levels} dtype={data_store.dtype}"
    )
    for i, shape in enumerate(data_store.level_shapes):
        print(f"    s{i}: {shape}")

    # ── 2. Controller and 2-D scene ─────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    appearance = MultiscaleLabelsAppearance(colormap_mode="random")
    render_config = MultiscaleLabelRenderConfig(
        block_size=_DEMO_BLOCK_SIZE,
        gpu_budget_bytes_2d=64 * 1024**2,
        paint_max_tiles=512,
    )
    label_visual_model = MultiscaleLabelVisual(
        name="labels",
        data_store_id=str(data_store.id),
        level_transforms=data_store.level_transforms,
        appearance=appearance,
        render_config=render_config,
    )
    label_visual = controller.add_visual(
        scene.id, label_visual_model, data_store=data_store
    )

    # ── 3. Qt window ─────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Multiscale label paint demo (OME-Zarr, pyramid rebuild)")
    layout = QVBoxLayout(root)

    toolbar = QWidget(root)
    toolbar_layout = QHBoxLayout(toolbar)
    toolbar_layout.setContentsMargins(4, 4, 4, 4)

    brush_value_label = QLabel("label ID")
    brush_value_spin = QSpinBox()
    brush_value_spin.setRange(1, 255)
    brush_value_spin.setValue(1)

    brush_radius_label = QLabel("radius (vox)")
    brush_radius_spin = QSpinBox()
    brush_radius_spin.setRange(1, 30)
    brush_radius_spin.setValue(8)

    undo_btn = QPushButton("Undo")
    commit_btn = QPushButton("Commit")
    abort_btn = QPushButton("Abort")

    autosave_title = QLabel("Autosave:")
    autosave_label = QLabel("never")

    for w in (
        brush_value_label,
        brush_value_spin,
        brush_radius_label,
        brush_radius_spin,
        undo_btn,
        commit_btn,
        abort_btn,
        autosave_title,
        autosave_label,
    ):
        toolbar_layout.addWidget(w)
    toolbar_layout.addStretch(1)
    layout.addWidget(toolbar)

    controller.set_widget_parent(root)
    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    layout.addWidget(canvas_widget)
    root.resize(700, 750)

    # ── 4. Camera fit on first delivery ─────────────────────────────────
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
        brush_value=int(brush_value_spin.value()),
        brush_radius_voxels=float(brush_radius_spin.value()),
        autosave_interval_s=30.0,
    )
    print(f"  Paint controller: {paint_ctrl!r}")

    def _on_brush_value_changed(value: int) -> None:
        paint_ctrl.brush_value = int(value)

    def _on_brush_radius_changed(value: int) -> None:
        paint_ctrl.brush_radius_voxels = float(value)

    def _on_undo_clicked() -> None:
        paint_ctrl.undo()

    def _on_commit_clicked() -> None:
        paint_ctrl.commit()
        status_timer.stop()
        QMessageBox.information(
            root,
            "Commit",
            f"Paint and pyramid flushed to disk.\n"
            f"All {data_store.n_levels} LOD levels updated.\n\n"
            f"Zoom out to verify painted labels appear at coarser zoom levels.\n\n"
            f"Path: {zarr_path}",
        )
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _on_abort_clicked() -> None:
        paint_ctrl.abort()
        status_timer.stop()
        QMessageBox.information(
            root, "Abort", "All staged paint discarded; on-disk zarr unchanged."
        )
        commit_btn.setEnabled(False)
        abort_btn.setEnabled(False)
        undo_btn.setEnabled(False)

    def _print_status() -> None:
        last_save = (
            paint_ctrl.last_autosave_time.strftime("%H:%M:%S")
            if paint_ctrl.last_autosave_time
            else "never"
        )
        if paint_ctrl.last_autosave_time:
            autosave_label.setText(f"{last_save} ({paint_ctrl.autosave_count} saves)")

    status_timer = QTimer()
    status_timer.setInterval(5000)
    status_timer.timeout.connect(_print_status)
    status_timer.start()

    brush_value_spin.valueChanged.connect(_on_brush_value_changed)
    brush_radius_spin.valueChanged.connect(_on_brush_radius_changed)
    undo_btn.clicked.connect(_on_undo_clicked)
    commit_btn.clicked.connect(_on_commit_clicked)
    abort_btn.clicked.connect(_on_abort_clicked)

    root.show()
    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    main()
