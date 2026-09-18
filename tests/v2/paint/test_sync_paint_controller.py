"""Tests for :class:`cellier.paint.SyncPaintController`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.paint._history import PaintStrokeCommand
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals._label_memory import InMemoryLabelsAppearance

if TYPE_CHECKING:
    from cellier.paint import SyncPaintController


@pytest.fixture
def paint_setup(qtbot):
    """Build a 2D scene/canvas/labels visual and a SyncPaintController.

    The tests are ``async`` so pytest-asyncio (mode=auto) installs an
    event loop -- ``_write_values`` announces a store change, the controller
    reslices the visual, and reslicing calls ``asyncio.ensure_future``, which
    needs a running loop.
    """
    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("y", "x"), name="world")
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="paint_scene",
        render_modes={"2d"},
    )

    data = np.zeros((32, 32), dtype=np.int32)
    store = LabelMemoryStore(data=data, name="paint_store")
    visual = controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(),
        name="paint_labels",
    )

    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    paint_ctrl = controller.add_paint_controller(
        visual_id=visual.id,
        canvas_id=canvas_id,
        brush_value=1,
        brush_radius_voxels=2.0,
    )
    return controller, paint_ctrl, store, canvas_id


def _push_stroke(
    paint_ctrl: SyncPaintController,
    indices: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
) -> None:
    """Append a stroke directly to history (mimics what mouse handlers do)."""
    paint_ctrl._history.push(
        PaintStrokeCommand(
            visual_id=paint_ctrl._visual_id,
            data_store_id=paint_ctrl._data_store_id,
            voxel_indices=indices,
            old_values=old,
            new_values=new,
        )
    )


async def test_brush_paints_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[5, 5], [5, 6], [6, 5]], dtype=np.int64)
    values = np.array([1, 1, 1], dtype=np.int32)

    paint_ctrl._write_values(indices, values)

    assert store.data[5, 5] == 1
    assert store.data[5, 6] == 1
    assert store.data[6, 5] == 1


async def test_undo_reverts_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[10, 10], [10, 11]], dtype=np.int64)
    old = paint_ctrl._read_old_values(indices)
    new = np.array([1, 1], dtype=np.int32)

    paint_ctrl._write_values(indices, new)
    _push_stroke(paint_ctrl, indices, old, new)

    paint_ctrl.undo()

    assert store.data[10, 10] == 0
    assert store.data[10, 11] == 0


async def test_redo_replays_voxels(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    indices = np.array([[20, 20]], dtype=np.int64)
    old = paint_ctrl._read_old_values(indices)
    new = np.array([1], dtype=np.int32)
    paint_ctrl._write_values(indices, new)
    _push_stroke(paint_ctrl, indices, old, new)

    paint_ctrl.undo()
    assert store.data[20, 20] == 0

    paint_ctrl.redo()

    assert store.data[20, 20] == 1


async def test_abort_reverts_all_strokes(paint_setup):
    _controller, paint_ctrl, store, _canvas_id = paint_setup
    for r, c in [(3, 3), (4, 4)]:
        idx = np.array([[r, c]], dtype=np.int64)
        old = paint_ctrl._read_old_values(idx)
        new = np.array([1], dtype=np.int32)
        paint_ctrl._write_values(idx, new)
        _push_stroke(paint_ctrl, idx, old, new)
    assert store.data[3, 3] == 1
    assert store.data[4, 4] == 1

    paint_ctrl.abort()

    assert store.data[3, 3] == 0
    assert store.data[4, 4] == 0
    assert paint_ctrl._history.can_undo is False


async def test_commit_clears_history(paint_setup):
    _controller, paint_ctrl, _store, _canvas_id = paint_setup
    idx = np.array([[1, 1]], dtype=np.int64)
    _push_stroke(
        paint_ctrl,
        idx,
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
    )
    assert paint_ctrl._history.can_undo is True

    paint_ctrl.commit()

    assert paint_ctrl._history.can_undo is False


async def test_camera_controller_disabled_on_init(paint_setup):
    controller, _paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False


async def test_camera_controller_restored_on_commit(paint_setup):
    controller, paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False

    paint_ctrl.commit()

    assert canvas_view._controller.enabled is True


async def test_camera_controller_restored_on_abort(paint_setup):
    controller, paint_ctrl, _store, canvas_id = paint_setup
    canvas_view = controller._render_manager._canvases[canvas_id]
    assert canvas_view._controller.enabled is False

    paint_ctrl.abort()

    assert canvas_view._controller.enabled is True


async def test_add_paint_controller_unsupported_visual_raises(qtbot):
    """A visual type with no paint controller raises TypeError."""
    from cellier.data.points._points_memory_store import PointsMemoryStore
    from cellier.visuals._points_memory import PointsMarkerAppearance

    controller = CellierController()
    cs = world_coordinate_system(spatial_axes("y", "x"), name="world")
    scene = controller.add_scene(
        dim="2d",
        coordinate_system=cs,
        name="bad_scene",
        render_modes={"2d"},
    )
    points = PointsMemoryStore(
        positions=np.array([[1.0, 1.0]], dtype=np.float32),
        name="pts",
    )
    visual = controller.add_points(
        data=points,
        scene_id=scene.id,
        appearance=PointsMarkerAppearance(
            color=(1.0, 0.0, 0.0, 1.0), size=4.0, size_space="screen"
        ),
        name="points_visual",
    )
    controller.add_canvas(scene_id=scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]

    with pytest.raises(TypeError, match="No PaintController"):
        controller.add_paint_controller(
            visual_id=visual.id,
            canvas_id=canvas_id,
        )
