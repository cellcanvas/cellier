"""Camera models drive the live cameras, and files restore camera poses.

Assigning a camera model's fields moves its pygfx camera (the camera shown
as a jump, the other one placed for later); the controller's own writes of
the live camera into the model do not loop back.  A canvas built from a
model with a pose starts there, and startup's default fit keeps it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.data import ImageMemoryStore
from cellier.scene.cameras import (
    OrbitCameraController,
    OrthographicCamera,
    PanZoomCameraController,
    PerspectiveCamera,
)
from cellier.scene.canvas import Canvas
from cellier.scene.dims import spatial_axes

AXES = ("z", "y", "x")


@pytest.fixture
def make_viewer(qtbot):
    made = []

    def _make(displayed_axes=AXES, **kwargs):
        viewer = Viewer(
            spatial_axes(*AXES),
            displayed_axes=displayed_axes,
            gui="offscreen",
            **kwargs,
        )
        viewer.controller.camera_reslice_enabled = False
        data = np.random.default_rng(0).random((8, 12, 16), dtype=np.float32)
        viewer.add_image(ImageMemoryStore(data=data))
        viewer.add_canvas()
        viewer.fit_camera()
        made.append(viewer)
        return viewer

    yield _make
    for viewer in made:
        viewer.controller.close()


def _canvas(viewer):
    canvas_id = viewer.canvases[0]
    model = viewer.scene.canvases[canvas_id]
    view = viewer.controller.get_canvas_view(canvas_id)
    return canvas_id, model, view


def test_assigning_the_shown_cameras_position_moves_it(make_viewer):
    viewer = make_viewer()
    _, model, _ = _canvas(viewer)
    camera = model.cameras["3d"]
    target = np.asarray(camera.position) + np.array([1.0, 2.0, 3.0], np.float32)

    camera.position = target

    np.testing.assert_allclose(viewer.get_camera_state().position, target, rtol=1e-6)
    np.testing.assert_allclose(camera.position, target, rtol=1e-6)


def test_assigning_fov_reaches_the_camera(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    camera = model.cameras["3d"]

    camera.fov = 30.0

    state = viewer.get_camera_state()
    assert state.fov == pytest.approx(30.0)
    assert view._camera_3d.fov == pytest.approx(30.0)


def test_assigning_up_direction_turns_the_camera(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    camera = model.cameras["3d"]
    forward_before = np.asarray(view._camera_3d.world.forward, np.float64)
    position_before = np.asarray(camera.position, np.float64)
    # Some up not along the view direction.
    up = np.cross(forward_before, [1.0, 0.0, 0.0])
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0, 1, 0])

    camera.up_direction = up

    np.testing.assert_allclose(view._camera_3d.world.forward, forward_before, atol=1e-5)
    np.testing.assert_allclose(view._camera_3d.world.reference_up, up, atol=1e-6)
    np.testing.assert_allclose(camera.position, position_before, atol=1e-4)
    # The model holds the camera's up, which the turn brought onto *up*.
    np.testing.assert_allclose(camera.up_direction, up, atol=1e-5)


def test_the_cameras_own_moves_are_mirrored_not_applied_back(make_viewer):
    viewer = make_viewer()
    _, model, _ = _canvas(viewer)
    camera = model.cameras["3d"]
    seen = []
    camera.events.position.connect(seen.append)
    state = viewer.get_camera_state()
    moved = state._replace(position=tuple(np.asarray(state.position) + 4.0))

    viewer.set_camera_state(moved)

    np.testing.assert_allclose(camera.position, moved.position, rtol=1e-6)
    assert len(seen) == 1
    # The live camera wrote its frustum too.
    assert np.any(camera.frustum != 0)


def test_assigning_the_other_cameras_pose_places_it_for_the_switch(make_viewer):
    viewer = make_viewer()
    canvas_id, model, view = _canvas(viewer)
    camera_2d = model.cameras["2d"]
    position = np.array([3.0, 4.0, 0.0], np.float32)

    camera_2d.position = position
    camera_2d.width = 3.0

    # Not shown yet: the 3D camera did not move, the 2D one did.
    assert view._dim == "3d"
    np.testing.assert_allclose(view._camera_2d.world.position, position)
    # Through the controller: ``Viewer.set_displayed_dimensions`` refits.
    viewer.controller.set_displayed_axes(viewer.scene.id, (1, 2))
    assert view._dim == "2d"
    state = viewer.get_camera_state()
    np.testing.assert_allclose(state.position, position)
    assert state.extent[0] == pytest.approx(3.0)
    assert viewer.controller._canvas_camera_placed(canvas_id)


def test_derived_frustum_cannot_be_assigned(make_viewer):
    viewer = make_viewer()
    _, model, _ = _canvas(viewer)
    with pytest.raises(AttributeError, match="frustum"):
        model.cameras["3d"].frustum = np.zeros((2, 4, 3), np.float32)


def test_controller_enabled_follows_the_model(make_viewer):
    viewer = make_viewer()
    canvas_id, model, view = _canvas(viewer)

    model.cameras["3d"].controller.enabled = False
    assert view._controller_3d.enabled is False
    model.cameras["3d"].controller.enabled = True
    assert view._controller_3d.enabled is True

    # The other dim's controller waits for the switch, and stays off.
    model.cameras["2d"].controller.enabled = False
    assert view._controller_2d.enabled is False
    viewer.set_displayed_dimensions(("y", "x"))
    assert view._controller_2d.enabled is False
    assert view._controller_3d.enabled is False

    viewer.controller.set_camera_controller_enabled(canvas_id, True)
    assert all(c.controller.enabled for c in model.cameras.values())
    assert view._controller_2d.enabled is True


def test_a_replaced_controller_model_is_applied(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)

    model.cameras["3d"].controller = OrbitCameraController(enabled=False)

    assert view._controller_3d.enabled is False


def test_a_removed_canvas_no_longer_follows_its_camera_models(make_viewer):
    viewer = make_viewer()
    canvas_id, model, _ = _canvas(viewer)
    camera = model.cameras["3d"]

    viewer.controller.remove_canvas(canvas_id)

    camera.position = np.array([9.0, 9.0, 9.0], np.float32)  # no error
    assert canvas_id not in viewer.controller._canvas_psygnal_handlers


def test_a_saved_viewer_comes_back_with_its_camera_pose(make_viewer, tmp_path):
    viewer = make_viewer()
    state = viewer.get_camera_state()
    moved = state._replace(
        position=tuple(np.asarray(state.position) + np.array([2.0, -1.0, 5.0])),
        extent=(state.extent[0] / 1.75, state.extent[1] / 1.75),
    )
    viewer.set_camera_state(moved)
    saved = viewer.get_camera_state()
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path, gui="offscreen")
    try:
        assert loaded.gui == "offscreen"
        restored = loaded.get_camera_state()
        np.testing.assert_allclose(restored.position, saved.position, rtol=1e-6)
        np.testing.assert_allclose(restored.rotation, saved.rotation, atol=1e-6)
        np.testing.assert_allclose(restored.extent, saved.extent, rtol=1e-6)
        canvas_id = loaded.canvases[0]
        # Startup's default fit keeps a placed camera.
        assert loaded.controller._canvas_camera_placed(canvas_id)
        # The 2D camera was never shown: it has no pose and is fitted later.
        view = loaded.controller.get_canvas_view(canvas_id)
        assert "2d" not in view._fitted
    finally:
        loaded.controller.close()


def test_a_saved_2d_viewer_comes_back_with_its_extent(make_viewer, tmp_path):
    viewer = make_viewer(displayed_axes=("y", "x"))
    state = viewer.get_camera_state()
    moved = state._replace(
        position=(state.position[0] + 3.0, state.position[1] - 2.0, state.position[2]),
        extent=(state.extent[0] * 2.0, state.extent[1] * 2.0),
    )
    viewer.set_camera_state(moved)
    saved = viewer.get_camera_state()
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path, gui="offscreen")
    try:
        restored = loaded.get_camera_state()
        np.testing.assert_allclose(restored.position, saved.position, rtol=1e-6)
        np.testing.assert_allclose(restored.extent, saved.extent, rtol=1e-6)
    finally:
        loaded.controller.close()


def test_a_new_canvas_with_default_cameras_is_not_placed(make_viewer):
    viewer = make_viewer()
    canvas_id = viewer.canvases[0]
    assert not viewer.controller._canvas_camera_placed(canvas_id)


def test_a_canvas_model_with_a_disabled_controller_starts_disabled(make_viewer):
    viewer = make_viewer()
    canvas = Canvas(
        cameras={
            "3d": PerspectiveCamera(controller=OrbitCameraController(enabled=False)),
            "2d": OrthographicCamera(controller=PanZoomCameraController()),
        }
    )
    viewer.controller.add_canvas_model(viewer.scene.id, canvas)
    view = viewer.controller.get_canvas_view(canvas.id)
    assert view._controller_3d.enabled is False


def test_a_saved_orthoviewer_comes_back_with_its_panel_poses(qtbot, tmp_path):
    ortho = OrthoViewer(spatial_axes(*AXES), gui="offscreen")
    ortho.controller.camera_reslice_enabled = False
    data = np.random.default_rng(0).random((8, 12, 16), dtype=np.float32)
    ortho.add_image(ImageMemoryStore(data=data))
    loaded = None
    try:
        # The layout builder makes the panel canvases; build the 3D one here.
        ortho.controller.add_canvas(scene_id=ortho.scenes["vol"].id)
        ortho.fit_camera("vol")
        state = ortho.get_camera_state("vol")
        ortho.set_camera_state(
            "vol", state._replace(position=tuple(np.asarray(state.position) + 3.0))
        )
        saved = ortho.get_camera_state("vol")
        path = tmp_path / "ortho.json"
        ortho.to_file(path)

        loaded = OrthoViewer.from_file(path, gui="offscreen")
        assert loaded.gui == "offscreen"
        restored = loaded.get_camera_state("vol")
        np.testing.assert_allclose(restored.position, saved.position, rtol=1e-6)
        vol_canvas = loaded.controller.get_canvas_ids(loaded.scenes["vol"].id)[0]
        assert loaded.controller._canvas_camera_placed(vol_canvas)
    finally:
        ortho.controller.close()
        if loaded is not None:
            loaded.controller.close()
