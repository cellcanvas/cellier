"""A restored 3D camera state brings back the point the camera orbits about.

pygfx's orbit controller pivots ``orbit_distance(fov, width, height)`` ahead of
the camera, and a wheel zoom changes the camera's width and height.  Before
``CameraState.extent`` carried the 3D width and height, restoring a view saved
before a zoom brought the camera back with the pivot somewhere else, and the
next click swung the camera away.  The width and height are the size of the
view at the orbit point, at every field of view; the distance is derived.  See
``plans/orbit_controller_state.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data import ImageMemoryStore
from cellier.events import (
    CameraChangedEvent,
    OrbitControllerState,
    PanZoomControllerState,
)
from cellier.scene import orbit_distance
from cellier.scene.dims import spatial_axes
from cellier.viewer_model import ViewerModel

AXES = ("z", "y", "x")


@pytest.fixture
def make_viewer(qtbot):
    made = []

    def _make(displayed_axes=AXES):
        viewer = Viewer(
            spatial_axes(*AXES), displayed_axes=displayed_axes, gui="offscreen"
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


def _zoom(view, delta: float) -> None:
    """Zoom the shown camera with its pygfx controller, then draw a frame.

    The canvas's controllers do not update the camera themselves
    (``auto_update=False``): the state they return is applied, as
    ``_draw_frame`` applies a ticked one.  The frame is what reports the move
    (event and model write-back), as for a wheel zoom.
    """
    controller = view._controller_3d if view.dim == "3d" else view._controller_2d
    view.camera.set_state(controller.zoom(delta, (0, 0, 640, 480), animate=False))
    view._draw_frame()


def _size(view) -> tuple[float, float]:
    camera = view._camera_3d
    return float(camera.width), float(camera.height)


@pytest.mark.parametrize("fov", [70.0, 30.0, 0.7, 179.5])
def test_orbit_distance_matches_the_pygfx_controller(make_viewer, fov):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    viewer.set_camera_state(viewer.get_camera_state()._replace(fov=fov))
    camera = view._camera_3d

    expected = np.linalg.norm(np.asarray(view._controller_3d.target))
    got = orbit_distance(float(camera.fov), *_size(view))

    assert got == pytest.approx(expected, rel=1e-9)
    assert view.orbit_distance() == got


def test_orbit_distance_is_none_at_zero_fov():
    assert orbit_distance(0.0, 10.0, 20.0) is None
    assert orbit_distance(0.4, 10.0, 20.0) is None
    assert orbit_distance(90.0, 10.0, 10.0) == pytest.approx(5.0)


def test_a_restored_state_brings_the_orbit_point_back(make_viewer):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    _zoom(view, 1.5)
    saved = viewer.get_camera_state()
    pivot = np.asarray(view.orbit_point())

    # The extent is the size at the pivot: it gives the distance to it.
    assert saved.extent == pytest.approx(_size(view))
    ahead = np.linalg.norm(pivot - np.asarray(saved.position))
    assert orbit_distance(saved.fov, *saved.extent) == pytest.approx(ahead, rel=1e-6)

    # A zoom about the centre keeps the pivot and moves the camera back.
    _zoom(view, -3.0)
    assert view.orbit_distance() > 2.0 * ahead

    # Without the extent (a state captured before 3D extents were recorded)
    # the camera comes back to the saved position but pivots about a point
    # far behind the old one.
    viewer.set_camera_state(saved._replace(extent=(0.0, 0.0)))
    assert not np.allclose(view.orbit_point(), pivot, rtol=1e-3)

    _zoom(view, -1.0)
    viewer.set_camera_state(saved)

    np.testing.assert_allclose(view.orbit_point(), pivot, rtol=1e-6, atol=1e-6)
    assert _size(view) == pytest.approx(saved.extent)


def test_a_state_without_an_extent_keeps_the_current_size(make_viewer):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    _zoom(view, 1.5)
    saved = viewer.get_camera_state()
    _zoom(view, -3.0)
    size = _size(view)

    viewer.set_camera_state(saved._replace(extent=(0.0, 0.0), controller_state=None))

    assert _size(view) == size
    np.testing.assert_allclose(view._camera_3d.world.position, saved.position)


def test_captured_states_carry_the_controller_kind(make_viewer):
    state_3d = make_viewer().get_camera_state()
    assert state_3d.controller_state == OrbitControllerState()

    viewer_2d = make_viewer(displayed_axes=("y", "x"))
    _, _, view = _canvas(viewer_2d)
    state_2d = viewer_2d.get_camera_state()
    assert state_2d.controller_state == PanZoomControllerState()
    # It applies, and so does a 2D state without one.
    viewer_2d.set_camera_state(state_2d)
    viewer_2d.set_camera_state(state_2d._replace(controller_state=None))
    assert view.capture_camera_state() == state_2d


def test_a_controller_state_of_the_other_kind_raises(make_viewer):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    before = view.capture_camera_state()
    with pytest.raises(ValueError, match="'pan_zoom' controller state"):
        viewer.set_camera_state(
            before._replace(
                position=(1.0, 2.0, 3.0), controller_state=PanZoomControllerState()
            )
        )
    # Checked before anything moved.
    assert view.capture_camera_state() == before

    viewer_2d = make_viewer(displayed_axes=("y", "x"))
    state_2d = viewer_2d.get_camera_state()
    with pytest.raises(ValueError, match="'orbit' controller state"):
        viewer_2d.set_camera_state(
            state_2d._replace(controller_state=OrbitControllerState())
        )


def test_a_zoom_writes_the_size_into_the_model(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    camera = model.cameras["3d"]
    # The fit was reported, and wrote the size.
    assert (camera.width, camera.height) == pytest.approx(_size(view))
    before = camera.orbit_distance

    _zoom(view, 1.5)

    assert (camera.width, camera.height) == pytest.approx(_size(view), rel=1e-12)
    assert camera.orbit_distance == pytest.approx(view.orbit_distance(), rel=1e-12)
    assert camera.orbit_distance < before


def test_assigning_the_size_moves_the_pivot(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    events = []
    viewer.controller.on_camera_changed(
        viewer.scene.id, events.append, owner_id=viewer.controller._id
    )
    camera = model.cameras["3d"]
    position = np.asarray(view._camera_3d.world.position).copy()
    forward = np.asarray(view._camera_3d.world.forward)
    width, height = _size(view)

    camera.width = 2.5 * width

    assert _size(view) == pytest.approx((2.5 * width, height))
    target = orbit_distance(camera.fov, 2.5 * width, height)
    np.testing.assert_allclose(
        view.orbit_point(), position + target * forward, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(view._camera_3d.world.position, position)
    # A change of the size alone is a camera change, and is reported.
    assert len(events) == 1
    assert isinstance(events[0], CameraChangedEvent)
    assert events[0].camera_state.extent == pytest.approx((2.5 * width, height))

    # ``None`` says nothing: the model gets the live size back.
    camera.height = None
    assert camera.height == pytest.approx(height)
    assert len(events) == 1

    # Toggling ``enabled`` leaves the size alone.
    camera.controller.enabled = False
    assert _size(view) == pytest.approx((2.5 * width, height))
    assert len(events) == 1


def test_assigning_the_3d_size_while_2d_is_shown(make_viewer):
    viewer = make_viewer(displayed_axes=("y", "x"))
    _, model, view = _canvas(viewer)
    assert view.dim == "2d"
    camera = model.cameras["3d"]

    camera.width = 42.0
    camera.height = 21.0

    # Placed for when the canvas switches to 3D; the model keeps it.
    assert _size(view) == (42.0, 21.0)
    assert (camera.width, camera.height) == pytest.approx((42.0, 21.0))
    assert view.orbit_point() is None


def test_a_saved_viewer_comes_back_with_its_orbit_point(make_viewer, tmp_path):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    _zoom(view, 1.5)
    saved = viewer.get_camera_state()
    pivot = np.asarray(view.orbit_point())
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    model_from_file = ViewerModel.from_file(path)
    (canvas,) = next(iter(model_from_file.scenes.values())).canvases.values()
    assert (canvas.cameras["3d"].width, canvas.cameras["3d"].height) == (
        pytest.approx(saved.extent, rel=1e-6)
    )

    loaded = Viewer.from_file(path, gui="offscreen")
    try:
        _, _, loaded_view = _canvas(loaded)
        restored = loaded.get_camera_state()
        np.testing.assert_allclose(restored.extent, saved.extent, rtol=1e-6)
        np.testing.assert_allclose(
            loaded_view.orbit_point(), pivot, rtol=1e-5, atol=1e-4
        )
    finally:
        loaded.controller.close()


def test_a_file_without_the_3d_size_still_loads(make_viewer, tmp_path):
    viewer = make_viewer()
    path = tmp_path / "viewer.json"
    viewer.to_file(path)
    text = json.loads(path.read_text())
    for scene in text["scenes"].values():
        for canvas in scene["canvases"].values():
            del canvas["cameras"]["3d"]["width"], canvas["cameras"]["3d"]["height"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(text))

    loaded = Viewer.from_file(old, gui="offscreen")
    try:
        # The camera keeps a size of its own; the model learns it.
        _, loaded_model, loaded_view = _canvas(loaded)
        camera = loaded_model.cameras["3d"]
        assert camera.width is None
        assert loaded_view.orbit_distance() is not None
        loaded.fit_camera()
        assert (camera.width, camera.height) == pytest.approx(_size(loaded_view))
    finally:
        loaded.controller.close()


# ---------------------------------------------------------------------------
# Field of view 0, and changes of the field of view
# ---------------------------------------------------------------------------


def test_at_zero_fov_the_extent_is_the_view_size(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    viewer.set_camera_state(viewer.get_camera_state()._replace(fov=0.0))

    state = viewer.get_camera_state()
    assert state.extent == pytest.approx(_size(view))
    assert view.orbit_distance() is None
    assert model.cameras["3d"].orbit_distance is None
    # The pivot is the camera itself.
    np.testing.assert_allclose(view.orbit_point(), state.position)

    # An orthographic zoom halves the size, and the model follows.
    _zoom(view, 1.0)
    assert _size(view) == pytest.approx((state.extent[0] / 2, state.extent[1] / 2))
    assert (model.cameras["3d"].width, model.cameras["3d"].height) == (
        pytest.approx(_size(view))
    )

    # Assigning resizes the view.
    model.cameras["3d"].width = 30.0
    assert _size(view)[0] == 30.0


def test_a_state_from_zero_fov_restores_a_perspective_pivot(make_viewer):
    viewer = make_viewer()
    _, _, view = _canvas(viewer)
    _zoom(view, 1.0)
    perspective = viewer.get_camera_state()
    pivot = np.asarray(view.orbit_point())
    viewer.set_camera_state(perspective._replace(fov=0.0))
    _zoom(view, -2.0)

    viewer.set_camera_state(perspective)

    np.testing.assert_allclose(view.orbit_point(), pivot, rtol=1e-6, atol=1e-6)


def test_a_fov_change_keeps_the_pivot_and_the_size(make_viewer):
    viewer = make_viewer()
    _, model, view = _canvas(viewer)
    pivot = np.asarray(view.orbit_point())
    size = _size(view)

    # What alt+wheel does.
    view._camera_3d.set_state(view._controller_3d.update_fov(-30.0, animate=False))
    view._draw_frame()

    np.testing.assert_allclose(view.orbit_point(), pivot, rtol=1e-6, atol=1e-6)
    state = viewer.get_camera_state()
    assert state.extent == pytest.approx(size)
    assert model.cameras["3d"].fov == pytest.approx(state.fov)
    assert model.cameras["3d"].orbit_distance == pytest.approx(view.orbit_distance())


def test_a_saved_3d_camera_at_zero_fov_waiting_behind_2d_comes_back(
    make_viewer, tmp_path
):
    viewer = make_viewer(displayed_axes=("y", "x"))
    _, model, _ = _canvas(viewer)
    camera_3d = model.cameras["3d"]
    # Placed for when the canvas switches to 3D.
    camera_3d.fov = 0.0
    camera_3d.position = np.array([5.0, 6.0, 40.0], np.float32)
    camera_3d.width = 33.0
    camera_3d.height = 22.0
    path = tmp_path / "viewer.json"
    viewer.to_file(path)

    loaded = Viewer.from_file(path, gui="offscreen")
    try:
        loaded_view = loaded.controller.get_canvas_view(loaded.canvases[0])
        assert loaded_view.dim == "2d"
        # The fov, then the size, reach the 3D camera kept in reserve.
        assert float(loaded_view._camera_3d.fov) == 0.0
        assert _size(loaded_view) == pytest.approx((33.0, 22.0))
    finally:
        loaded.controller.close()


def test_no_mouse_button_sets_the_pygfx_zoom(make_viewer):
    import pygfx as gfx

    # pygfx binds its fourth mouse button to "quickzoom", which sets
    # ``camera.zoom``: a change the camera state cannot see.
    assert "quickzoom" in {
        action[0] for action in gfx.OrbitController().controls.values()
    }
    viewer = make_viewer()
    _canvas_id, _model, view = _canvas(viewer)
    for controller in (view._controller_3d, view._controller_2d):
        actions = {action[0] for action in controller.controls.values()}
        assert "quickzoom" not in actions
        assert "zoom" in actions  # the wheel zoom (width and height) stays
