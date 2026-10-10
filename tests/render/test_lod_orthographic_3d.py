"""An orthographic 3D camera picks one level from the screen pixel size.

At a field of view of 0 the 3D planners had no distance thresholds and fell
back to multiples of the level-0 volume diagonal, so any camera within one
diagonal planned the finest level however far the view was zoomed out.  A
pixel of an orthographic view covers the same world size everywhere, so the
whole visual takes the level whose voxels best match it, by the 2D rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._planning import planned_requests_3d

SCREEN_HEIGHT_PX = 100.0
# Level 0 is 16^3 at unit spacing and level 1 is 8^3 at twice the spacing,
# so the levels switch at sqrt(1 * 2) level-0 voxels per pixel.
SWITCH_HEIGHT_WORLD = SCREEN_HEIGHT_PX * 2.0**0.5


def _add(controller, kind, store):
    scene = controller.add_scene(dim="3d", name=kind)
    if kind == "image":
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(),
            render_config=MultiscaleImageRenderConfig(block_size=8),
            single=MultiscaleImageSingleAppearance(color_map="viridis"),
        )
    else:
        visual = controller.add_labels_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(),
            render_config=MultiscaleLabelRenderConfig(block_size=8),
        )
    controller.add_canvas(scene_id=scene.id)
    return scene, visual


def _planned_levels(controller, scene, visual, view_height_world, lod_bias=1.0):
    """The set of levels (0 is the finest) an orthographic plan asks for."""
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    requests = planned_requests_3d(
        gfx,
        # Inside one level-0 diagonal of the volume, where the fallback
        # distance bands plan the finest level.
        camera_pos_world=np.array([8.0, 8.0, 20.0]),
        frustum_corners_world=None,
        fov_y_rad=0.0,
        screen_height_px=SCREEN_HEIGHT_PX,
        lod_bias=lod_bias,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
        view_height_world=view_height_world,
    )
    assert requests, f"nothing planned at view_height_world={view_height_world}"
    return {request.scale_index for request in requests}


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_orthographic_3d_takes_one_level_from_the_pixel_size(
    controller, multiscale_image_store, multiscale_labels_store, kind
):
    store = multiscale_image_store if kind == "image" else multiscale_labels_store
    scene, visual = _add(controller, kind, store)
    controller.fit_camera(scene.id)

    # Zoomed in: a pixel is a fraction of a level-0 voxel.
    assert _planned_levels(controller, scene, visual, 16.0) == {0}
    # Either side of the transition.
    assert _planned_levels(controller, scene, visual, SWITCH_HEIGHT_WORLD * 0.99) == {0}
    assert _planned_levels(controller, scene, visual, SWITCH_HEIGHT_WORLD * 1.01) == {1}
    # Zoomed out: a pixel covers many level-0 voxels.
    assert _planned_levels(controller, scene, visual, 1600.0) == {1}


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_orthographic_3d_lod_bias_is_coarser_when_higher(
    controller, multiscale_image_store, multiscale_labels_store, kind
):
    store = multiscale_image_store if kind == "image" else multiscale_labels_store
    scene, visual = _add(controller, kind, store)
    controller.fit_camera(scene.id)

    # One level-0 voxel per pixel: the bias alone decides.
    assert _planned_levels(controller, scene, visual, 100.0, lod_bias=1.0) == {0}
    assert _planned_levels(controller, scene, visual, 100.0, lod_bias=2.0) == {1}
    assert _planned_levels(controller, scene, visual, 200.0, lod_bias=0.5) == {0}


async def test_3d_request_carries_the_world_extent_only_at_zero_fov(
    controller, multiscale_image_store
):
    scene, _ = _add(controller, "image", multiscale_image_store)
    controller.fit_camera(scene.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    selection = controller._selections_for_scene(scene.id)[canvas_id]
    camera = view._camera

    request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
    assert request.fov_y_rad > 0
    assert request.world_extent == (0.0, 0.0)

    camera.fov = 0.0
    request = view.capture_reslicing_request(scene.dims.to_state(), selection, None)
    assert request.fov_y_rad == 0.0
    # The test canvas is square, so the view shows the camera's larger side
    # both ways.
    side = max(camera.width, camera.height)
    assert request.world_extent == pytest.approx((side, side))
