"""A get/set round trip of a 2D canvas's camera state changes nothing.

The 2D camera's depth range is automatic unless one is set: pygfx works it
out from what the camera is fitted to.  The camera state has to say so, and
not report a range of its own that then gets applied.
"""

from __future__ import annotations

import warnings

import numpy as np

from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


def _setup(controller, store):
    """A fitted 2D canvas on a gradient image; returns (scene, canvas id, view)."""
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id, canvas_size=(128, 128))
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller.get_canvas_view(canvas_id)
    controller.fit_camera(scene.id)
    return scene, canvas_id, view


async def test_automatic_depth_range_survives_a_round_trip(
    controller, gradient_image, reslice, offscreen_renderer
):
    scene, canvas_id, view = _setup(controller, gradient_image)
    await reslice(controller, scene.id)
    assert view.camera.depth_range is None
    before = controller.screenshot(canvas_id, size=(64, 64))
    assert len(np.unique(before.reshape(-1, 4), axis=0)) > 1

    state = controller.get_camera_state(canvas_id)
    assert state.depth_range is None

    # A changed extent, so the state differs and is applied.
    width, height = state.extent
    zoomed = state._replace(extent=(0.5 * width, 0.5 * height))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        controller.set_camera_state(canvas_id, zoomed)
        assert view.camera.depth_range is None
        assert controller.get_camera_state(canvas_id) == zoomed
        after = controller.screenshot(canvas_id, size=(64, 64))

    assert len(np.unique(after.reshape(-1, 4), axis=0)) > 1


async def test_automatic_depth_range_leaves_the_camera_model_alone(
    controller, gradient_image
):
    scene, canvas_id, _view = _setup(controller, gradient_image)
    camera_model = scene.canvases[canvas_id].cameras["2d"]
    configured = (camera_model.near_clipping_plane, camera_model.far_clipping_plane)

    state = controller.get_camera_state(canvas_id)
    controller._update_camera_model(scene.id, canvas_id, state)

    assert (
        camera_model.near_clipping_plane,
        camera_model.far_clipping_plane,
    ) == configured


async def test_set_depth_range_round_trips_on_a_2d_camera(controller, gradient_image):
    _scene, canvas_id, view = _setup(controller, gradient_image)

    controller.set_camera_depth_range(canvas_id, (-250.0, 250.0))
    state = controller.get_camera_state(canvas_id)
    assert state.depth_range == (-250.0, 250.0)

    controller.set_camera_state(canvas_id, state._replace(zoom=2.0 * state.zoom))
    assert tuple(view.camera.depth_range) == (-250.0, 250.0)
