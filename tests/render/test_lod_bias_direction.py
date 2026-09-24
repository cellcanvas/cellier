"""A larger ``lod_bias`` never picks a finer level in 3D, for images and labels.

``VisualRenderConfig.lod_bias`` is documented as "higher is coarser", and 2D
already behaves that way.  The 3D image planner divides its distance
thresholds by the bias; the 3D labels planner multiplied, so for labels a
higher bias picked *finer* levels.  This pins both 3D planners to the same
direction at one camera, voxel by voxel.
"""

from __future__ import annotations

from itertools import pairwise

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

SHAPE = (16, 16, 16)  # level 0; level 1 is 8^3 at twice the spacing
# Geometric sweep: at small biases the whole volume is level 0, at large
# ones level 1, with a mix in between.
BIASES = [2.0 ** (k / 2) for k in range(-2, 15)]


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


def _finest_level_per_voxel(controller, scene, visual, lod_bias) -> np.ndarray:
    """The finest level any planned brick draws at each level-0 voxel.

    ``SHAPE``-shaped; ``-1`` where nothing was planned.
    """
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    requests = planned_requests_3d(
        gfx,
        # Off the +z face, looking at the volume from a distance where the
        # level-1 threshold (2 * focal / bias) sweeps through the bricks.
        camera_pos_world=np.array([8.0, 8.0, 40.0]),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=200.0,
        lod_bias=lod_bias,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )
    assert requests, f"nothing planned at lod_bias={lod_bias}"
    finest = np.full(SHAPE, np.iinfo(np.int64).max, dtype=np.int64)
    for request in requests:
        factor = 2**request.scale_index
        box = []
        # ``(start, stop)`` per axis in the level's voxels, halo included.
        for (start, stop), size in zip(request.axis_selections, SHAPE, strict=True):
            box.append(slice(max(start * factor, 0), min(stop * factor, size)))
        region = finest[tuple(box)]
        np.minimum(region, request.scale_index, out=region)
    finest[finest == np.iinfo(np.int64).max] = -1
    return finest


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_a_larger_lod_bias_never_selects_a_finer_level_in_3d(
    controller, multiscale_image_store, multiscale_labels_store, kind
):
    store = multiscale_image_store if kind == "image" else multiscale_labels_store
    scene, visual = _add(controller, kind, store)
    controller.fit_camera(scene.id)

    levels = [
        _finest_level_per_voxel(controller, scene, visual, bias) for bias in BIASES
    ]
    # Every voxel is drawn at every bias.
    assert all((lv >= 0).all() for lv in levels)
    for (lo_bias, lo), (hi_bias, hi) in pairwise(zip(BIASES, levels, strict=True)):
        finer = hi < lo
        assert not finer.any(), (
            f"{kind}: lod_bias {hi_bias:.3g} drew {int(finer.sum())} voxels finer "
            f"than lod_bias {lo_bias:.3g}"
        )
    # The sweep is not vacuous: it goes from all fine to all coarse.
    assert (levels[0] == 0).all()
    assert (levels[-1] == 1).all()
