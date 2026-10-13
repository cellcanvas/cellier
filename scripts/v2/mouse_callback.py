"""Demo: canvas mouse events (press, move, release) with pick_info.

This script creates a 2D scene with a grayscale image and a small point cloud,
then prints mouse event details — event type, world coordinate, and which visual
(if any) was hit.

Run with:
    uv run python scripts/v2/mouse_callback.py

Notes
-----
- ``on_mouse_move`` fires on *all* pointer motion, not only while a button is
  held.  Callers that only want drag events must track press/release state
  themselves.
- ``pick_info.hit_visual_id`` is ``None`` for background hits.
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6 import QtAsyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from skimage.data import camera as camera_image

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events._events import (
    CanvasMouseMove2DEvent,
    CanvasMousePress2DEvent,
    CanvasMouseRelease2DEvent,
)
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._image_memory import InMemoryImageAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance

RNG = np.random.default_rng(0)


def main() -> None:
    app = QApplication(sys.argv)

    # ── 1. Create data ──────────────────────────────────────────────────
    image_data = camera_image().astype(np.float32) / 255.0  # shape (512, 512)
    image_store = ImageMemoryStore(data=image_data, name="camera_image")

    h, w = image_data.shape
    # 2D positions (y, x) to match the 2D coordinate system.
    point_positions = RNG.uniform(low=[[0, 0]], high=[[h, w]], size=(15, 2)).astype(
        np.float32
    )
    points_store = PointsMemoryStore(
        positions=point_positions,
        name="random_points",
    )

    # ── 2. Controller and 2D scene ──────────────────────────────────────
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))

    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="2d_scene",
        render_modes={"2d"},
    )

    appearance = InMemoryImageAppearance(color_map="grays", clim=(0.0, 1.0))
    image_visual = controller.add_image(
        data=image_store,
        scene_id=scene.id,
        appearance=appearance,
        name="image",
    )

    points_appearance = PointsMarkerAppearance(
        color=(1.0, 0.3, 0.3, 1.0),
        size=8.0,
        size_space="screen",
    )
    points_visual = controller.add_points(
        data=points_store,
        scene_id=scene.id,
        appearance=points_appearance,
        name="points",
    )

    # ── 3. Qt window ────────────────────────────────────────────────────
    root = QWidget()
    root.setWindowTitle("Mouse callback demo")
    layout = QVBoxLayout(root)

    status_label = QLabel("Move or click on the canvas.")
    layout.addWidget(status_label)

    controller.set_widget_parent(root)
    canvas_widget = controller.add_canvas(scene_id=scene.id)
    canvas_id = next(iter(scene.canvases))
    layout.addWidget(canvas_widget)

    root.resize(600, 550)

    # ── 4. Camera fit on first delivery ──────────────────────────────────────
    _fitted = False

    def _on_first_data(event):
        nonlocal _fitted
        if not _fitted:
            _fitted = True
            controller.look_at_visual(image_visual.id, canvas_id)

    controller.on_reslice_completed(
        image_visual.id, _on_first_data, owner_id=controller._id
    )
    QTimer.singleShot(0, controller.reslice_all)

    # ── 5. Mouse callbacks ───────────────────────────────────────────────
    owner_id = controller._id

    def _hit_name(hit_visual_id) -> str:
        if hit_visual_id is None:
            return "background"
        if hit_visual_id == image_visual.id:
            return "image"
        if hit_visual_id == points_visual.id:
            return "points"
        return f"unknown ({hit_visual_id})"

    def on_press(event: CanvasMousePress2DEvent) -> None:
        name = _hit_name(event.pick_info.hit_visual_id)
        msg = f"PRESS  world={np.round(event.world_coordinate, 2)}  hit={name}"
        print(msg)
        status_label.setText(msg)

    def on_move(event: CanvasMouseMove2DEvent) -> None:
        name = _hit_name(event.pick_info.hit_visual_id)
        msg = f"MOVE   world={np.round(event.world_coordinate, 2)}  hit={name}"
        status_label.setText(msg)

    def on_release(event: CanvasMouseRelease2DEvent) -> None:
        name = _hit_name(event.pick_info.hit_visual_id)
        msg = f"RELEASE world={np.round(event.world_coordinate, 2)}  hit={name}"
        print(msg)
        status_label.setText(msg)

    controller.on_mouse_press_2d(canvas_id, on_press, owner_id=owner_id)
    controller.on_mouse_move_2d(canvas_id, on_move, owner_id=owner_id)
    controller.on_mouse_release_2d(canvas_id, on_release, owner_id=owner_id)

    root.show()
    QtAsyncio.run(handle_sigint=True)


if __name__ == "__main__":
    main()
