"""A labels pick does not name a level the frame it picked did not draw.

A 3D labels hit is a pixel of the last drawn frame, and its level is looked up
in the CPU brick table.  The table is rewritten by a commit round, which the
frame that draws it follows; between the two (a timer commit, or another canvas
of the scene) the table describes a frame that has not been drawn.  The level
read then was the new one for a coordinate drawn at the old one, and the label
read at it was wrong on a third to a half of the picks.  The visual now records
the table's version at each frame it draws and gives no level while the table
has changed since, so the read falls back to level 0.
"""

from __future__ import annotations

import numpy as np

from cellier.visuals import MultiscaleLabelRenderConfig, MultiscaleLabelsAppearance
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    P0,
    bind,
    block_volume,
    canvas_view,
    load,
    multiscale_store,
)
from tests.render._pick import PickFrame
from tests.render.test_labels_pick_face import _viewer

#: An octant view of the block.
DIRECTION_FOR_VIEW = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def _picks(view, gfx_visual, count: int = 40):
    """``(hit object, decoded coordinate)`` of drawn pixels of the last frame."""
    frame = PickFrame(view._renderer)
    rows, cols = np.nonzero(frame.values)
    # The target is in physical pixels, ``info`` takes logical ones.
    scale = frame.values.shape[1] / frame._logical_size[0]
    rng = np.random.default_rng(0)
    picks = []
    for index in rng.choice(rows.size, size=min(count, rows.size), replace=False):
        info = frame.info(((cols[index] + 0.5) / scale, (rows[index] + 0.5) / scale))
        hit = info.get("world_object")
        coord = None if hit is None else gfx_visual.pick_data_coordinate(hit, info)
        if coord is not None:
            picks.append((hit, coord))
    return picks


def test_a_table_rewritten_after_the_draw_gives_no_level(tmp_path):
    displayed = ("z", "y", "x")
    viewer = _viewer(displayed)
    try:
        data = block_volume(np.int32)
        store = multiscale_store(tmp_path, "labels", data)
        visual = viewer.add_labels_multiscale(
            store,
            appearance=MultiscaleLabelsAppearance(
                force_level=2, ray_steps_per_voxel=4.0
            ),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=16),
            transform=bind(viewer, store, np.eye(4)),
        )
        load(viewer)
        view = canvas_view(viewer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        canvas_id = view.canvas_id
        target = np.array([P0["zyx".index(name)] for name in reversed(displayed)])
        view.camera.world.position = tuple(target + 18.0 * DIRECTION_FOR_VIEW)
        view.camera.look_at(tuple(target))

        # Drawn at level 2: the table is the one the frame drew with.
        view._draw_frame()
        picks = _picks(view, gfx_visual)
        assert len(picks) >= 20
        assert {gfx_visual.pick_drawn_level(h, c, canvas_id) for h, c in picks} == {2}

        # The plan changes and the finer bricks are committed, with no frame
        # drawn in between: the table no longer says what that frame drew.
        visual.appearance.force_level = 1
        load(viewer)
        assert {gfx_visual.pick_drawn_level(h, c, canvas_id) for h, c in picks} == {
            None
        }
        # Read as it is, the table names the new level: the wrong answer.
        assert {gfx_visual.pick_drawn_level(h, c) for h, c in picks} == {1}

        # The next frame draws the table, and the answer is the new level.
        view._draw_frame()
        fresh = _picks(view, gfx_visual)
        assert {gfx_visual.pick_drawn_level(h, c, canvas_id) for h, c in fresh} == {1}
    finally:
        viewer.controller.close()


def test_a_2d_image_table_rewritten_after_the_draw_gives_no_level(tmp_path):
    """The same for a 2D tile table, which a multiscale image also keeps."""
    from cellier.visuals import (
        MultiscaleImageAppearance,
        MultiscaleImageRenderConfig,
        MultiscaleImageSingleAppearance,
    )
    from tests.render._affine import new_viewer

    viewer = new_viewer("2d")
    try:
        store = multiscale_store(tmp_path, "image", block_volume(np.float32))
        visual = viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(force_level=2),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="mip"
            ),
            transform=bind(viewer, store, np.eye(4)),
        )
        load(viewer, slice_z=float(P0[0]))
        view = canvas_view(viewer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        canvas_id = view.canvas_id

        def levels(picks, *args):
            return {gfx_visual.pick_drawn_levels(h, c, *args)[0][1] for h, c in picks}

        view._draw_frame()
        picks = _picks(view, gfx_visual)
        assert len(picks) >= 20
        assert levels(picks, canvas_id) == {2}

        visual.appearance.force_level = 1
        load(viewer, slice_z=float(P0[0]))
        assert levels(picks, canvas_id) == {None}
        assert levels(picks) == {1}

        view._draw_frame()
        assert levels(_picks(view, gfx_visual), canvas_id) == {1}
    finally:
        viewer.controller.close()
