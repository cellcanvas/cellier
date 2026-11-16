"""A pick reports the value of the pyramid level that was drawn at the hit.

A hit on a coarse level is on that level's surface, a voxel or more away from
the level-0 one, so the value under the cursor is the one the drawn level
holds, for labels and images, in 2D and 3D.  An in-memory visual has one level.

The controller part (a hit's levels name the level each channel is read at) uses
the small pyramid of ``test_pick_values_multiscale``; the render part draws a
block at a forced coarse level and checks what every pick reads.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.controller import _level_voxel_index
from cellier.events import ImagePickEvent, LabelsPickEvent
from cellier.render.render_manager import (
    _ImageDisplayedDataCoord,
    _LabelsDisplayedDataCoord,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    P0,
    bind,
    block_volume,
    canvas_view,
    drawn,
    load,
    multiscale_store,
    new_viewer,
    nudge_2d_camera,
)
from tests.render._pick import PickFrame
from tests.render.test_labels_pick_face import _pick_every_view, _viewer
from tests.render.test_pick_values_multiscale import (
    _composite_image,
    _pyramid,
    _raw,
)


@pytest.fixture
def level_labelled_store(tmp_path):
    """A pyramid whose every voxel holds ``100 + the level's extent``."""

    def _fill(arr):
        arr[...] = 100 + arr.shape[-1]

    return _pyramid(tmp_path, False, _fill, dtype="int32")


# -- the controller reads each channel at its level -----------------------------


async def test_a_composite_channel_is_read_at_the_level_it_is_drawn_at(
    controller, drive_reslice, tmp_path
):
    def _fill(arr):
        x = np.arange(arr.shape[-1], dtype=arr.dtype)
        for channel in range(arr.shape[0]):
            arr[channel] = 10.0 * (channel + 1) + x

    store = _pyramid(tmp_path, True, _fill)
    scene, visual, canvas_id, received = _composite_image(controller, store, "2d")
    pick = _ImageDisplayedDataCoord(
        displayed_data_coord=(3.5, 2.5),
        collapsed_data_indices=None,
        channel_index=1,
        drawn_channels=(0, 1),
        drawn_levels=((0, 1), (1, 2)),
    )
    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            pick,
            action="press",
            camera="2d",
            gesture_id=uuid4(),
        )
    )
    await drive_reslice(controller)

    (event,) = received
    # Channel 0 at level 1 (x = 3 holds 13); channel 1 at level 2, where the
    # point is in coarse voxel 1 (holds 21).
    assert event.pick_info.channel_values == {0: 13.0, 1: 21.0}


async def test_a_single_image_is_read_at_the_drawn_level(
    controller, drive_reslice, tmp_path
):
    def _fill(arr):
        arr[...] = np.arange(arr.shape[-1], dtype=arr.dtype)

    store = _pyramid(tmp_path, False, _fill)
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"),
        displayed_axes=("z", "y", "x"),
        name="scene",
    )
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())
    for levels, expected in (((), 5.0), (((0, 2),), 2.0)):
        pick = _ImageDisplayedDataCoord(
            displayed_data_coord=(5.5, 2.5, 3.5), drawn_levels=levels
        )
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                visual.id,
                pick,
                action="press",
                camera="3d",
                gesture_id=uuid4(),
            )
        )
        await drive_reslice(controller)
        assert received[-1].pick_info.channel_values == {0: expected}


async def test_a_2d_labels_hit_is_read_at_the_drawn_level(
    controller, drive_reslice, level_labelled_store
):
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        name="scene",
    )
    visual = controller.add_labels_multiscale(
        level_labelled_store,
        scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, LabelsPickEvent, received.append, owner_id=uuid4())
    for level, expected in ((None, 108), (2, 104)):
        pick = _LabelsDisplayedDataCoord(
            displayed_data_coord=(3.5, 2.5), drawn_level=level
        )
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                visual.id,
                pick,
                action="press",
                camera="2d",
                gesture_id=uuid4(),
            )
        )
        await drive_reslice(controller)
        assert received[-1].pick_info.value == expected


# -- 3D: a block drawn at level 2 -------------------------------------------------


def _coarse_image(viewer, tmp_path, render_mode):
    store = multiscale_store(tmp_path, "image", block_volume(np.float32))
    visual = viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(force_level=2),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        single=MultiscaleImageSingleAppearance(
            color_map="gray",
            clim=(0.0, 1.0),
            render_mode=render_mode,
            iso_threshold=0.5,
        ),
        transform=bind(viewer, store, np.eye(4)),
    )
    return store, visual


@pytest.mark.parametrize("render_mode", ["iso", "mip"])
@pytest.mark.parametrize("displayed", [("z", "y", "x"), ("y", "x", "z")], ids="".join)
def test_a_3d_image_hit_names_the_level_drawn(tmp_path, render_mode, displayed):
    """Every pick names level 2 and the voxel of level 2 under it is the block."""
    viewer = _viewer(displayed)
    try:
        data = block_volume(np.float32)
        store, visual = _coarse_image(viewer, tmp_path, render_mode)
        load(viewer)
        coarse = data[::2, ::2, ::2]
        wrong, level0_misses = [], 0

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
        if render_mode == "iso":
            # The reason the read moved: level 0 alone would have missed some.
            assert level0_misses > 0
    finally:
        viewer.controller.close()


# -- 2D: a block drawn at level 2 -------------------------------------------------


@pytest.mark.parametrize("family", ["image_multiscale", "labels_multiscale"])
def test_a_2d_hit_names_the_level_drawn_and_reads_what_is_drawn(tmp_path, family):
    """Drawn pixels read as the block at level 2; level 0 would read some wrong."""
    from tests.render._affine import SIZE, project, render

    viewer = new_viewer("2d")
    try:
        is_labels = family.startswith("labels")
        data = block_volume(np.int32 if is_labels else np.float32)
        store = multiscale_store(tmp_path, family, data)
        if is_labels:
            visual = viewer.add_labels_multiscale(
                store,
                appearance=MultiscaleLabelsAppearance(force_level=2),
                render_config=MultiscaleLabelRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                transform=bind(viewer, store, np.eye(4)),
            )
        else:
            visual = viewer.add_image_multiscale(
                store,
                appearance=MultiscaleImageAppearance(force_level=2),
                render_config=MultiscaleImageRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                single=MultiscaleImageSingleAppearance(
                    color_map="gray", clim=(0.0, 1.0), render_mode="mip"
                ),
                transform=bind(viewer, store, np.eye(4)),
            )
        load(viewer, slice_z=float(P0[0]))
        view = canvas_view(viewer)
        nudge_2d_camera(view.camera)
        frame, renderer, camera = render(viewer)
        pick_frame = PickFrame(renderer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        plane = data[int(P0[0])]
        coarse = data[::2, ::2, ::2]
        ys, xs = np.nonzero(plane)
        low = project(camera, np.array([P0[0], ys.min() - 2, xs.min() - 2]))
        high = project(camera, np.array([P0[0], ys.max() + 3, xs.max() + 3]))
        r0, r1 = sorted((int(low[0]), int(high[0])))
        c0, c1 = sorted((int(low[1]), int(high[1])))
        picked = disagree = level0_disagree = 0
        for row in range(max(r0, 0), min(r1, SIZE), 7):
            for col in range(max(c0, 0), min(c1, SIZE), 7):
                info = pick_frame.info((col + 0.5, row + 0.5))
                hit = info.get("world_object")
                if hit is None:
                    continue
                coord = gfx_visual.pick_data_coordinate(hit, info)
                levels = (
                    [(0, gfx_visual.pick_drawn_level(hit, coord))]
                    if is_labels
                    else gfx_visual.pick_drawn_levels(hit, coord)
                )
                ((_, level),) = levels
                assert level == 2
                # The displayed axes are (x, y); the data axes (z, y, x), with z
                # the slice plane.
                index = _level_voxel_index((P0[0] + 0.5, coord[1], coord[0]), store, 2)
                picked += 1
                is_drawn = bool(drawn(frame)[row, col])
                disagree += int((coarse[index] != 0) != is_drawn)
                vx, vy = int(np.floor(coord[0])), int(np.floor(coord[1]))
                level0_disagree += int((plane[vy, vx] != 0) != is_drawn)
        assert picked >= 20
        # The level that was drawn agrees with the screen at every pick.
        assert disagree == 0, f"{disagree} of {picked} picks disagree with the screen"
        if not is_labels:
            # An image hits undrawn pixels too, and level 0 misses some of them:
            # the reason the read moved.
            assert level0_disagree > 0
    finally:
        viewer.controller.close()
