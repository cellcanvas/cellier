"""A 3D canvas whose camera has a field of view of 0 (orthographic).

pygfx puts such a camera at the centre of what it is fitted to, so the near
plane has to be behind it; and its controller zooms by changing the camera's
width and height and nothing else.  The canvas has to allow for both.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)

DEPTH_RANGE = (1.0, 8000.0)


def _setup(controller, store):
    """A fitted 3D canvas on a 16^3 image; returns (scene, canvas id, view)."""
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        render_config=MultiscaleImageRenderConfig(block_size=8),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
    )
    controller.add_canvas(scene_id=scene.id, depth_range_3d=DEPTH_RANGE)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    controller.fit_camera(scene.id)
    return scene, canvas_id, view


def _go_orthographic(controller, scene, view) -> None:
    view.camera.fov = 0.0
    controller.fit_camera(scene.id)


def _near_side_point(view) -> np.ndarray:
    """A world point between the camera's side of the scene and its centre."""
    camera = view.camera
    forward = np.asarray(camera.world.forward, dtype=np.float64)
    radius = 0.5 * float(camera.width)
    # Fitted at a field of view of 0, the camera sits at the scene's centre.
    return np.asarray(camera.world.position, dtype=np.float64) - 0.5 * radius * forward


def _clip_depth(view, point: np.ndarray) -> float:
    """The clip-space depth of *point*; inside the clip volume is 0 to 1."""
    clip = np.asarray(view.camera.camera_matrix) @ np.append(point, 1.0)
    return float(clip[2] / clip[3])


def _inside(corners: np.ndarray, point: np.ndarray) -> bool:
    """Whether *point* is between the near and far faces of a frustum."""
    near_centre = corners[0].mean(axis=0)
    far_centre = corners[1].mean(axis=0)
    axis = far_centre - near_centre
    t = float(np.dot(point - near_centre, axis) / np.dot(axis, axis))
    return 0.0 <= t <= 1.0


async def test_near_side_of_the_scene_is_inside_the_clip_volume_at_zero_fov(
    controller, multiscale_image_store
):
    scene, canvas_id, view = _setup(controller, multiscale_image_store)
    _go_orthographic(controller, scene, view)
    point = _near_side_point(view)

    # What the planners cull bricks against.
    selection = controller._selections_for_scene(scene.id)[canvas_id]
    request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
    assert _inside(request.frustum_corners, point)

    # What the renderer clips against.
    view._draw_frame()
    assert 0.0 <= _clip_depth(view, point) <= 1.0
    near, far = view.camera.depth_range
    assert (near, far) == (-DEPTH_RANGE[1], DEPTH_RANGE[1])


async def test_configured_depth_range_returns_above_zero_fov(
    controller, multiscale_image_store
):
    scene, _canvas_id, view = _setup(controller, multiscale_image_store)
    perspective_fov = float(view.camera.fov)
    _go_orthographic(controller, scene, view)
    view._draw_frame()

    view.camera.fov = perspective_fov
    view._draw_frame()

    assert tuple(view.camera.depth_range) == DEPTH_RANGE


async def test_depth_range_survives_a_round_trip_through_the_camera_state(
    controller, multiscale_image_store
):
    scene, canvas_id, view = _setup(controller, multiscale_image_store)
    perspective_fov = float(view.camera.fov)
    _go_orthographic(controller, scene, view)

    # The state, and so the camera model, holds the configured range.
    state = view.capture_camera_state()
    assert state.depth_range == DEPTH_RANGE

    controller.set_camera_state(canvas_id, state)
    assert tuple(view.camera.depth_range) == (-DEPTH_RANGE[1], DEPTH_RANGE[1])

    controller.set_camera_state(canvas_id, state._replace(fov=perspective_fov))
    assert tuple(view.camera.depth_range) == DEPTH_RANGE


async def test_set_depth_range_is_widened_at_zero_fov(
    controller, multiscale_image_store
):
    scene, canvas_id, view = _setup(controller, multiscale_image_store)
    _go_orthographic(controller, scene, view)

    controller.set_camera_depth_range(canvas_id, (2.0, 500.0))

    assert tuple(view.camera.depth_range) == (-500.0, 500.0)
    assert view.capture_camera_state().depth_range == (2.0, 500.0)


async def test_orthographic_zoom_is_a_camera_change(
    monkeypatch, controller, multiscale_image_store
):
    scene, _canvas_id, view = _setup(controller, multiscale_image_store)
    _go_orthographic(controller, scene, view)
    view._draw_frame()  # settle: the fit is a camera change of its own

    resets: list[int] = []
    original = view._accum_pass.reset
    monkeypatch.setattr(
        view._accum_pass, "reset", lambda: (resets.append(1), original())[1]
    )
    events = []
    controller.on_camera_changed(scene.id, events.append, owner_id=controller._id)

    view._draw_frame()
    assert not resets, "a still camera keeps its history"
    assert not events

    # What pygfx's orbit controller does to zoom at a field of view of 0.
    camera = view.camera
    extent = (0.5 * float(camera.width), 0.5 * float(camera.height))
    camera.width, camera.height = extent
    view._draw_frame()

    assert resets, "an orthographic zoom must discard the accumulation history"
    assert len(events) == 1, "an orthographic zoom must be reported"
    assert events[0].camera_state.extent == pytest.approx(extent)


async def test_camera_state_restores_the_orthographic_extent(
    controller, multiscale_image_store
):
    scene, canvas_id, view = _setup(controller, multiscale_image_store)
    _go_orthographic(controller, scene, view)
    camera = view.camera
    state = view.capture_camera_state()
    assert state.extent == (float(camera.width), float(camera.height))

    camera.width, camera.height = 3.0, 3.0
    controller.set_camera_state(canvas_id, state)
    assert (float(camera.width), float(camera.height)) == state.extent

    # A state saved before the extent was recorded leaves the camera's alone.
    controller.set_camera_state(canvas_id, state._replace(extent=(0.0, 0.0)))
    assert (float(camera.width), float(camera.height)) == state.extent


async def test_perspective_camera_state_has_no_extent(
    controller, multiscale_image_store
):
    _scene, _canvas_id, view = _setup(controller, multiscale_image_store)

    state = view.capture_camera_state()

    assert state.fov > 0
    assert state.extent == (0.0, 0.0)
    assert state.depth_range == DEPTH_RANGE
    assert tuple(view.camera.depth_range) == DEPTH_RANGE
