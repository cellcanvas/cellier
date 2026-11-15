"""A brick-shader ray reaches data more than 512 LUT cells away.

Both brick shaders advance one level-0 LUT cell per loop iteration in every
render mode, at any drawn level.  The loop was capped at a fixed 512
iterations, so on a long grid a ray stopped at the 512th cell and anything
past it was neither drawn nor pickable, with no warning: a wall seen down a
600-cell bar vanished from cell 512 on, and at cell 511 only its middle drew
(the slanted rays near its edges spend iterations crossing y and z faces).
The cap is now the most cells a ray can visit, the sum of the grid's sides.

The bar is 24 x 32 x 9600 voxels at ``block_size=16``: 600 cells on x.  A
1-degree camera 100 voxels before the near end looks down the bar at a wall
of ones filling the cross-section in cell 590; the bricks are planned for
that view.  Picks at the wall's centre and near its four corners must land on
the wall, at the finest level and with the coarse level drawn.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.convenience.capture import _wait_until_loaded
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests.render._affine import (
    bind,
    canvas_view,
    load,
    multiscale_store,
    project,
    render,
)
from tests.render.test_labels_pick_face import _viewer

DISPLAYED = ("z", "y", "x")
LENGTH = 9600
BLOCK_SIZE = 16
WALL_CELL = 590
WALL_X0 = WALL_CELL * BLOCK_SIZE + BLOCK_SIZE // 4
BUDGETS = {"gpu_budget_bytes": 128 * 1024**2, "gpu_budget_bytes_2d": 16 * 1024**2}
KINDS = [
    ("labels", None),
    ("image", "iso"),
    ("image", "mip"),
    ("image", "attenuated_mip"),
]
#: The wall's centre and four points inset from its corners, ``(y, z)``.
PICK_POINTS = [(15.5, 11.5), (3.0, 3.0), (3.0, 20.0), (28.0, 3.0), (28.0, 20.0)]


def _add(viewer, family, mode, data, tmp_path, force_level):
    store = multiscale_store(tmp_path, "bar", data)
    transform = bind(viewer, store, np.eye(4))
    if family == "labels":
        return viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                force_level=force_level, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(**BUDGETS, block_size=BLOCK_SIZE),
            transform=transform,
        )
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(force_level=force_level),
        render_config=MultiscaleImageRenderConfig(**BUDGETS, block_size=BLOCK_SIZE),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode=mode, iso_threshold=0.5
        ),
        transform=transform,
    )


@pytest.mark.parametrize("force_level", [1, 2])
@pytest.mark.parametrize(("family", "mode"), KINDS)
def test_a_wall_past_512_cells_is_drawn_and_picked(tmp_path, family, mode, force_level):
    dtype = np.int32 if family == "labels" else np.float32
    data = np.zeros((24, 32, LENGTH), dtype=dtype)
    data[:, :, WALL_X0 : WALL_X0 + 3] = 1
    viewer = _viewer(DISPLAYED)
    try:
        visual = _add(viewer, family, mode, data, tmp_path, force_level)
        load(viewer)
        camera = canvas_view(viewer).camera
        camera.fov = 1.0
        camera.world.position = (-100.0, 15.5, 11.5)
        camera.look_at((float(WALL_X0), 15.5, 11.5))
        camera.depth_range = (1.0, 3.0 * LENGTH)

        async def _replan() -> None:
            viewer.controller.reslice_all()
            await _wait_until_loaded(viewer.controller)

        asyncio.run(_replan())
        _frame, renderer, camera = render(viewer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        # The coarse level's voxels are two wide, so its wall reaches a voxel
        # further than the level-0 one.
        x_high = WALL_X0 + (3 if force_level == 1 else 4)
        missed = []
        for y, z in PICK_POINTS:
            row, col = project(camera, np.array([z, y, WALL_X0 + 1.0]))
            info = renderer.get_pick_info((col + 0.5, row + 0.5))
            hit = info.get("world_object")
            coord = None if hit is None else gfx_visual.pick_data_coordinate(hit, info)
            if coord is None or not WALL_X0 <= coord[0] <= x_high:
                missed.append(((y, z), None if coord is None else coord[0]))
        assert not missed, (
            f"picks at (y, z) that did not land on the wall at x = {WALL_X0}: {missed}"
        )
    finally:
        viewer.controller.close()
