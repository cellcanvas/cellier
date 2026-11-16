"""A 3D labels surface pick reads the label it landed on, on every face.

B12: the pick buffer holds each axis in 14 bits and truncates, so a hit on a
voxel face (the bisected crossing of a background cell and a label cell) came
back up to ``dim / 16383`` voxels low.  On a low face that is outside the
label, and the consumer's ``floor`` read the background cell in front of the
surface; on a high face it landed inside by luck.  The brick shader now steps
a little way along the ray before encoding.

The block's six faces are all hit from the eight octant directions, and every
pick must decode to a voxel that holds the label.  Level 0 is forced, so the
drawn surface is the level-0 one the controller reads.

Both labels shaders are covered.  The in-memory one (``label_volume.wgsl``)
first stepped along the ray, which read background on about 0.3% of the hits on
a ray grazing a block edge; it now clamps into the foreground cell its DDA
names.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import _level_voxel_index
from cellier.convenience import Viewer
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.visuals import MultiscaleLabelRenderConfig, MultiscaleLabelsAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    P0,
    add_landmark,
    bind,
    block_volume,
    canvas_view,
    load,
    multiscale_store,
    render,
)

FAMILIES = ("labels", "labels_multiscale")
ORDERS = [("z", "y", "x"), ("x", "z", "y"), ("y", "x", "z")]
#: Octant directions plus the two horizontal axes (a camera looking along the
#: up vector cannot be aimed with ``look_at``).
DIRECTIONS = [
    np.array(d, dtype=float) / np.linalg.norm(d)
    for d in (
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
    )
]  # fmt: skip
DISTANCE = 18.0
PICKS_PER_VIEW = 40


def _viewer(displayed: tuple[str, ...]) -> Viewer:
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=displayed, gui="offscreen"
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    return viewer


def _pick_every_view(viewer, visual, displayed, check, centre=P0) -> int:
    """Pick drawn pixels from every direction; ``check(...)`` judges each one.

    *centre* is the block's centre in ``(z, y, x)``.  Returns the number of picks
    made.
    """
    view = canvas_view(viewer)
    gfx_visual = viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )
    # Render axis k is the k-th last displayed axis.
    target = np.array(
        [np.asarray(centre)["zyx".index(name)] for name in reversed(displayed)]
    )
    rng = np.random.default_rng(0)
    n_picks = 0
    for direction in DIRECTIONS:
        camera = view.camera
        camera.world.position = tuple(target + DISTANCE * direction)
        camera.look_at(tuple(target))
        frame, renderer, _ = render(viewer)
        rows, cols = np.nonzero(frame[..., :3].astype(float).max(axis=-1) > 20)
        if rows.size == 0:
            continue
        for index in rng.choice(rows.size, size=min(PICKS_PER_VIEW, rows.size)):
            info = renderer.get_pick_info((cols[index] + 0.5, rows[index] + 0.5))
            hit = info.get("world_object")
            coord = gfx_visual.pick_data_coordinate(hit, info)
            assert coord is not None
            full = viewer.controller._promote_pick_coordinate(
                coord,
                displayed_axes=tuple("zyx".index(name) for name in displayed),
                slice_indices={},
                world_ndim=3,
                hit_visual_id=visual.id,
            )
            n_picks += 1
            drawn_level = getattr(gfx_visual, "pick_drawn_level", None)
            if drawn_level is None and hasattr(gfx_visual, "pick_drawn_levels"):
                # An image: ``((key, level), ...)`` for its drawn channels.
                ((_, image_level),) = gfx_visual.pick_drawn_levels(hit, coord)
                level = image_level
            else:
                level = drawn_level and drawn_level(hit, coord)
            check(direction, level, full)
    assert n_picks >= 100, "too few drawn pixels to test"
    return n_picks


@pytest.mark.parametrize("displayed", ORDERS, ids="".join)
@pytest.mark.parametrize("family", FAMILIES)
def test_surface_picks_read_the_label_on_every_face(tmp_path, family, displayed):
    viewer = _viewer(displayed)
    try:
        visual = add_landmark(viewer, family, np.eye(4), tmp_path)
        load(viewer)
        labels = block_volume(np.int32)
        wrong = []

        def check(direction, level, full):
            # An in-memory node has one level and reports none.
            assert level == (None if family == "labels" else 1)
            voxel = tuple(int(v) for v in np.floor(np.asarray(full)))
            if labels[voxel] != 1:
                wrong.append((direction.round(2).tolist(), full))

        n_picks = _pick_every_view(viewer, visual, displayed, check)
        assert not wrong, (
            f"{len(wrong)} of {n_picks} picks read background: {wrong[:5]}"
        )
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("displayed", ORDERS, ids="".join)
def test_coarse_surface_picks_read_the_label_of_the_drawn_level(tmp_path, displayed):
    """With level 2 drawn, the pick names level 2 and its voxel holds the label.

    A coarse voxel covers two level-0 voxels per axis, so the coarse block
    reaches a voxel past the level-0 one on its high faces: the level-0 voxel
    under such a hit is background, and the label on screen is in level 2.
    """
    viewer = _viewer(displayed)
    try:
        data = block_volume(np.int32)
        store = multiscale_store(tmp_path, "landmark", data)
        visual = viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                force_level=2, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, np.eye(4)),
        )
        load(viewer)
        coarse = data[::2, ::2, ::2]
        wrong = []
        level0_misses = 0

        def check(direction, level, full):
            nonlocal level0_misses
            assert level == 2
            index = _level_voxel_index(full, store, level)
            if coarse[index] != 1:
                wrong.append((direction.round(2).tolist(), full, index))
            voxel = tuple(int(v) for v in np.floor(np.asarray(full)))
            level0_misses += int(data[voxel] != 1)

        n_picks = _pick_every_view(viewer, visual, displayed, check)
        assert not wrong, f"{len(wrong)} of {n_picks} picks: {wrong[:5]}"
        # The reason the read moved: level 0 alone would have missed some.
        assert level0_misses > 0
    finally:
        viewer.controller.close()
