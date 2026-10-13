"""The 2D paint cache is a grid of slots, not one column of them.

One column made the texture ``paint_max_tiles * block_size`` pixels tall,
which the default 512 tiles pushed over the device limit for any block size
above 32.
"""

from __future__ import annotations

import numpy as np
import pytest
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)


def _add_labels(controller, store, render_config):
    scene = controller.add_scene(displayed_axes=("y", "x"), name="scene")
    visual = controller.add_labels_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=render_config,
    )
    controller.add_canvas(scene_id=scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, gfx_visual


async def test_block_size_64_with_default_paint_max_tiles_draws(
    controller, render_scene, reslice, multiscale_labels_store
):
    """The default ``paint_max_tiles`` fits the device with 64-pixel tiles."""
    render_config = MultiscaleLabelRenderConfig(block_size=64)
    assert render_config.paint_max_tiles == 512
    scene, gfx_visual = _add_labels(controller, multiscale_labels_store, render_config)
    await reslice(controller, scene.id)

    max_dim = get_shared().device.limits["max-texture-dimension-2d"]
    height, width = gfx_visual._t_paint_cache.data.shape[:2]
    assert (height, width) == (23 * 64, 23 * 64)
    assert max(height, width) <= max_dim

    frame = render_scene(controller, scene.id)
    assert np.count_nonzero(frame[..., 3]) > 0


async def test_shader_reads_a_slot_off_the_first_row_and_column(
    controller, render_scene, reslice, multiscale_labels_store
):
    """A tile painted into slot 3 of a 2 x 2 grid shows as one solid block."""
    scene, gfx_visual = _add_labels(
        controller,
        multiscale_labels_store,
        MultiscaleLabelRenderConfig(block_size=8, paint_max_tiles=4),
    )
    await reslice(controller, scene.id)

    def paint(rows: np.ndarray, cols: np.ndarray) -> None:
        voxels = np.stack([np.zeros_like(rows), rows, cols], axis=1)
        values = np.full(rows.shape[0], 5.0, dtype=np.float32)
        assert gfx_visual.patch_paint_texture(voxels, values, (1, 2)) == 1

    # One voxel in each of three tiles takes slots 0, 1 and 2.
    for row, col in [(0, 0), (0, 8), (8, 0)]:
        paint(np.array([row]), np.array([col]))
    before = render_scene(controller, scene.id)

    # The whole of tile (1, 1) goes to slot 3: grid row 1, column 1.
    rows, cols = np.meshgrid(np.arange(8, 16), np.arange(8, 16), indexing="ij")
    paint(rows.ravel(), cols.ravel())
    assert gfx_visual._paint_slot_manager.get((1, 1)) == 3
    after = render_scene(controller, scene.id)

    # Opaque pixels only: the labels are drawn with a soft halo round them.
    changed = np.any(before != after, axis=-1) & (after[..., 3] == 255)
    # A wrong slot address would show the single voxels of slots 0..2 or
    # nothing; the right one fills a solid square, half the data on a side.
    # Measured by the longest run per axis: a halo pixel on a neighbouring
    # label may change too, which a bounding box would count.
    width = int(changed.sum(axis=1).max())
    height = int(changed.sum(axis=0).max())
    assert abs(height - width) <= 2
    assert min(height, width) > before.shape[0] // 4
    assert 0.95 * height * width <= changed.sum() <= 1.05 * height * width


async def test_paint_cache_over_the_device_limit_raises(
    controller, multiscale_labels_store
):
    """A grid too large for the device is refused with the fitting maximum."""
    max_dim = get_shared().device.limits["max-texture-dimension-2d"]
    block_size = 64
    most = (max_dim // block_size) ** 2
    render_config = MultiscaleLabelRenderConfig(
        block_size=block_size, paint_max_tiles=most + 1
    )
    with pytest.raises(ValueError, match=f"paint_max_tiles to at most {most}"):
        _add_labels(controller, multiscale_labels_store, render_config)
