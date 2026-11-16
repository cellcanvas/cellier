"""A 3D pick decodes with the layout of the frame it was drawn in.

The brick shaders write a pick position in fields sized by the dataset size
(`cellier.render._pick_layout`).  When the dataset size changes, here because
the displayed axes of a 4D store change, the visual builds a new volume node,
with a new layout, before the next frame is drawn.  Until that frame, the
pick buffer still holds the old frame.

Each node keeps the layout it was built with, and the pick buffer names the
node that drew each pixel, so the old frame's picks still decode with the old
layout.  The data coordinate is read from the node's own decode
(``voxel_pos``), not from the visual's current sizes, so it stays the
coordinate the frame was drawn at.  The render manager then drops the pick,
because the old node is no longer the visual's node: no pick event names a
position decoded with the wrong layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data import MultiscaleZarrDataStore
from cellier.scene import BackgroundAppearance
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import P0, SHAPE, bind, canvas_view, load
from tests.render._pick import PickFrame
from tests.render.conftest import _write_multiscale_zarr

#: Frames of the 4D store: shorter than z, so showing ``t`` changes the size.
N_T = 12


def _store(tmp_path, dtype) -> MultiscaleZarrDataStore:
    """A ``(t, z, y, x)`` two-level store with a 3-voxel block at ``P0``."""
    data = np.zeros((N_T, *SHAPE), dtype=dtype)
    z, y, x = P0.astype(int)
    data[:, z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    coarse = data[:, ::2, ::2, ::2]
    root = tmp_path / "series"
    root.mkdir()

    def fill(arr):
        arr[...] = data if arr.shape == data.shape else coarse

    _write_multiscale_zarr(
        root,
        levels=[("s0", data.shape), ("s1", coarse.shape)],
        fill=fill,
        dtype=str(data.dtype),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="series",
    )


@pytest.mark.parametrize("family", ["image", "labels"])
def test_a_pick_read_after_a_resize_keeps_the_layout_it_was_drawn_with(
    tmp_path, family
):
    viewer = Viewer(
        [("t", "time"), *spatial_axes("z", "y", "x")],
        displayed_axes=("z", "y", "x"),
        gui="offscreen",
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0.0, 0.0, 0.0, 1.0))
    try:
        store = _store(tmp_path, np.int32 if family == "labels" else np.float32)
        transform = bind(viewer, store, np.eye(5))
        if family == "labels":
            visual = viewer.add_labels_multiscale(
                store,
                render_config=MultiscaleLabelRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                transform=transform,
            )
        else:
            visual = viewer.add_image_multiscale(
                store,
                render_config=MultiscaleImageRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                single=MultiscaleImageSingleAppearance(
                    render_mode="iso", clim=(0.0, 1.0), iso_threshold=0.5
                ),
                transform=transform,
            )
        load(viewer)
        view = canvas_view(viewer)
        gfx_visual = viewer.controller._render_manager._scenes[
            viewer.scene.id
        ].get_visual(visual.id)
        # Render axes are (x, y, z); the block centre is P0 (z, y, x).
        target = P0[::-1]
        view.camera.world.position = tuple(target + 18.0 * np.ones(3) / np.sqrt(3.0))
        view.camera.look_at(tuple(target))
        view._draw_frame()

        old_node = gfx_visual._inner_node_3d
        old_layout = old_node.pick_layout
        assert old_layout.dataset_size == (40, 32, 24)
        frame = PickFrame(view._renderer)
        rows, cols = np.nonzero(frame.values)
        scale = frame.values.shape[1] / frame._logical_size[0]
        rng = np.random.default_rng(0)
        positions = [
            ((cols[i] + 0.5) / scale, (rows[i] + 0.5) / scale)
            for i in rng.choice(rows.size, size=min(40, rows.size), replace=False)
        ]
        before = []
        for pos in positions:
            info = frame.info(pos)
            if info.get("world_object") is old_node:
                before.append((pos, gfx_visual.pick_data_coordinate(old_node, info)))
        assert len(before) >= 20
        # The hits are on the block, in the voxels it occupies.
        for _pos, coord in before:
            assert np.all(np.abs(np.floor(coord) - target) <= 1.0), coord

        # Show (t, y, x): the dataset size along z changes from 24 to 12, and
        # the visual builds a new node with a new layout.  No frame is drawn.
        viewer.scene.dims.selection.displayed_axes = (0, 2, 3)
        load(viewer)
        new_node = gfx_visual._inner_node_3d
        assert new_node is not old_node
        assert new_node.pick_layout.dataset_size == (40, 32, N_T)
        assert new_node.pick_layout != old_layout

        # The old frame's picks still name the old node and decode with its
        # layout, to the coordinates they were drawn at.
        for pos, coord in before:
            info = frame.info(pos)
            assert info["world_object"] is old_node
            assert gfx_visual.pick_data_coordinate(old_node, info) == coord
            # Mapped with the visual's current sizes (the new node's), as the
            # decode was before, z would be scaled by 12 / 24.
            n = np.asarray(info["norm_pos"]) / new_node._norm_size + 0.5
            stale = n * new_node._dataset_size
            assert not np.allclose(stale, coord)
            # The old node is no longer the visual's, so no event is made.
            assert (
                viewer.controller._render_manager._extract_pick_details(
                    viewer.scene.id, old_node, info, view.canvas_id
                )
                is None
            )
    finally:
        viewer.controller.close()
