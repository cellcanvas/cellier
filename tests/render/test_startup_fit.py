"""The camera fit uses the box of the stores, and the 3D far plane grows to it.

The fit puts the camera on the sphere of the scene's displayed data box
(stores and transforms, not the pygfx scene graph), and a 3D camera's far
plane grows to ``FAR_MARGIN * d + o`` so the whole box stays in view.  Near is
never changed, and a range only ever grows.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.capture import _ensure_canvases, _load_data
from cellier.data import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.render.visuals._image import _MultiscaleImageSlot
from cellier.scene._camera_fit import fit_distance, fit_sphere, fitted_far
from cellier.scene.dims import spatial_axes
from cellier.visuals._image_memory import (
    InMemoryImageAppearance,
    InMemoryImageSingleAppearance,
)
from tests.render.test_pick_large_volume import _add as add_long_volume

DEFAULT_RANGE = (1.0, 8000.0)


@pytest.fixture
def viewer():
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="offscreen"
    )
    yield viewer
    viewer.controller.close()


@pytest.fixture
def planned(monkeypatch):
    """Every 3D brick plan made, as ``(n, 4)`` arrays, last column the x brick."""
    plans: list[np.ndarray] = []
    original = _MultiscaleImageSlot._plan_bricks

    def record(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        plans.append(np.asarray(out).copy())
        return out

    monkeypatch.setattr(_MultiscaleImageSlot, "_plan_bricks", record)
    return plans


def _view(viewer):
    controller = viewer.controller
    return controller._render_manager._canvases[
        controller.get_canvas_ids(viewer.scene.id)[0]
    ]


def _camera_model(viewer):
    canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
    return viewer.scene.canvases[canvas_id].cameras["3d"]


def _load(viewer) -> None:
    """Give the scene a canvas, fit and load, as a launch does."""
    _ensure_canvases(viewer, (300, 300))
    asyncio.run(_load_data(viewer))


def _add_image(viewer, shape=(8, 8, 8)):
    return viewer.add_image(
        ImageMemoryStore(data=np.zeros(shape, dtype=np.float32)),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(clim=(0.0, 1.0)),
    )


def _points(*rows) -> PointsMemoryStore:
    return PointsMemoryStore(positions=np.array(rows, dtype=np.float32).reshape(-1, 3))


def _fitted_position(view, box) -> np.ndarray:
    """Where a fit of *view*'s 3D camera to *box* puts it, pygfx ``(x, y, z)``."""
    centre, radius = fit_sphere(box)
    direction = np.asarray(view.fit_orientation()[0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return np.asarray(centre) - direction * fit_distance(radius, view.camera.fov)


def _box(low, high):
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


# -- the first plan --------------------------------------------------------------


@pytest.mark.parametrize(("length", "n_bricks"), [(12000, 188), (16000, 250)])
def test_the_first_plan_covers_a_long_volume(
    viewer, planned, tmp_path, length, n_bricks
):
    """The fixed 8000 far plane used to leave the far end of the volume unplanned.

    A 12000-voxel bar's fitted camera is about 12050 from its far end, so with
    far at 8000 only bricks 107..187 of 188 were planned.
    """
    data = np.zeros((24, 32, length), dtype=np.float32)
    add_long_volume(viewer, "image_multiscale", data, tmp_path)
    _load(viewer)

    first = planned[0]
    assert np.array_equal(np.unique(first[:, -1]), np.arange(n_bricks))
    near, far = _view(viewer)._depth_range_3d
    assert near == DEFAULT_RANGE[0]
    assert far > DEFAULT_RANGE[1]
    model = _camera_model(viewer)
    assert (model.near_clipping_plane, model.far_clipping_plane) == pytest.approx(
        (near, far)
    )


def test_a_small_scene_keeps_the_default_range(viewer):
    _add_image(viewer)
    _load(viewer)
    assert _view(viewer)._depth_range_3d == DEFAULT_RANGE
    model = _camera_model(viewer)
    assert (model.near_clipping_plane, model.far_clipping_plane) == DEFAULT_RANGE


def test_the_fit_is_to_the_sphere_of_the_store_box(viewer):
    _add_image(viewer, shape=(4, 6, 8))
    _load(viewer)
    view = _view(viewer)
    box = _box((-0.5, -0.5, -0.5), (3.5, 5.5, 7.5))
    np.testing.assert_allclose(
        view.camera.world.position, _fitted_position(view, box), atol=1e-4
    )


# -- the box changes after launch ------------------------------------------------


def test_adding_a_long_volume_grows_far_and_keeps_the_camera(viewer, planned, tmp_path):
    """The notebook case: launch with a small image, then add a large volume."""
    _add_image(viewer)
    _load(viewer)
    view = _view(viewer)
    position = np.array(view.camera.world.position)
    assert view._depth_range_3d == DEFAULT_RANGE

    data = np.zeros((24, 32, 12000), dtype=np.float32)
    add_long_volume(viewer, "image_multiscale", data, tmp_path)

    near, far = view._depth_range_3d
    assert near == DEFAULT_RANGE[0]
    box = viewer.controller._displayed_scene_box(viewer.scene.id)
    assert far == pytest.approx(fitted_far(box, view.camera.fov, view._fit_view_dir_3d))
    np.testing.assert_array_equal(view.camera.world.position, position)
    assert _camera_model(viewer).far_clipping_plane == pytest.approx(far)

    # A re-fit then plans the whole volume.
    planned.clear()
    asyncio.run(_load_data(viewer))
    assert any(
        np.array_equal(np.unique(plan[:, -1]), np.arange(188)) for plan in planned
    )


def test_an_empty_scene_is_fitted_when_its_data_arrives(viewer):
    store = _points()
    viewer.add_points(store)
    _load(viewer)
    controller = viewer.controller
    view = _view(viewer)
    (canvas_id,) = controller.get_canvas_ids(viewer.scene.id)
    assert canvas_id in controller._canvases_awaiting_fit
    unfitted = np.array(view.camera.world.position)

    store.positions = np.array([[0, 0, 0], [20, 10, 40]], dtype=np.float32)

    assert canvas_id not in controller._canvases_awaiting_fit
    position = np.array(view.camera.world.position)
    assert not np.allclose(position, unfitted)
    np.testing.assert_allclose(
        position, _fitted_position(view, _box((0, 0, 0), (20, 10, 40))), atol=1e-4
    )


def test_a_camera_placed_before_the_data_arrives_is_kept(viewer):
    store = _points()
    viewer.add_points(store)
    _load(viewer)
    controller = viewer.controller
    view = _view(viewer)
    (canvas_id,) = controller.get_canvas_ids(viewer.scene.id)
    state = view.capture_camera_state()
    controller.set_camera_state(canvas_id, state._replace(position=(100.0, 50.0, 25.0)))
    assert canvas_id not in controller._canvases_awaiting_fit
    placed = np.array(view.camera.world.position)

    store.positions = np.array([[0, 0, 0], [20, 10, 40]], dtype=np.float32)

    np.testing.assert_array_equal(view.camera.world.position, placed)


# -- what the fit measures ---------------------------------------------------------


def test_only_visible_visuals_fits_the_shown_ones_and_keeps_far_for_all(viewer):
    _add_image(viewer)
    far_away = viewer.add_points(_points([0, 0, 20000], [0, 0, 20001]))
    _load(viewer)
    view = _view(viewer)
    everything = viewer.controller._displayed_scene_box(viewer.scene.id)
    viewer.controller.set_visual_visible(far_away.id, False)

    viewer.fit_camera(only_visible_visuals=True)

    image_box = _box((-0.5, -0.5, -0.5), (7.5, 7.5, 7.5))
    np.testing.assert_allclose(
        view.camera.world.position, _fitted_position(view, image_box), atol=1e-4
    )
    assert view._depth_range_3d[1] >= fitted_far(
        everything, view.camera.fov, view._fit_view_dir_3d
    )


def test_hidden_visuals_count_by_default(viewer):
    _add_image(viewer)
    far_away = viewer.add_points(_points([0, 0, 200], [0, 0, 201]))
    viewer.controller.set_visual_visible(far_away.id, False)
    _load(viewer)
    view = _view(viewer)
    box = _box((-0.5, -0.5, -0.5), (7.5, 7.5, 201))
    np.testing.assert_allclose(
        view.camera.world.position, _fitted_position(view, box), atol=1e-4
    )


def test_only_visible_visuals_with_nothing_shown_does_nothing(viewer):
    image = _add_image(viewer)
    _load(viewer)
    view = _view(viewer)
    position = np.array(view.camera.world.position)
    viewer.controller.set_visual_visible(image.id, False)

    viewer.fit_camera(only_visible_visuals=True)

    np.testing.assert_array_equal(view.camera.world.position, position)
    assert not viewer.controller._canvases_awaiting_fit


def test_look_at_visual_fits_that_visual_not_the_scene(viewer):
    near = viewer.add_points(_points([0, 0, 0], [10, 10, 10]))
    viewer.add_points(_points([0, 0, 20000], [10, 10, 20010]))
    _load(viewer)
    controller = viewer.controller
    view = _view(viewer)
    (canvas_id,) = controller.get_canvas_ids(viewer.scene.id)
    scene_box = controller._displayed_scene_box(viewer.scene.id)

    controller.look_at_visual(near.id, canvas_id)

    np.testing.assert_allclose(
        view.camera.world.position,
        _fitted_position(view, _box((0, 0, 0), (10, 10, 10))),
        atol=1e-4,
    )
    assert view._depth_range_3d[1] >= fitted_far(
        scene_box, view.camera.fov, view._fit_view_dir_3d
    )


# -- the model --------------------------------------------------------------------


def test_the_grown_range_round_trips_through_a_file(viewer, tmp_path):
    viewer.add_points(_points([0, 0, 0], [10, 10, 20000]))
    _load(viewer)
    near, far = _view(viewer)._depth_range_3d
    assert far > DEFAULT_RANGE[1]
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    restored = Viewer.from_file(path, gui="offscreen")
    try:
        assert _view(restored)._depth_range_3d == pytest.approx((near, far))
        model = _camera_model(restored)
        assert (model.near_clipping_plane, model.far_clipping_plane) == pytest.approx(
            (near, far)
        )
    finally:
        restored.controller.close()
