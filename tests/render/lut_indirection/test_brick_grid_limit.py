"""The 3D brick grid is bounded by the device's 3D texture limit, not 255.

The LUT has one texel per level-0 brick.  Its uint8 channels hold a cache slot
and a level, so the grid may be longer than 255 bricks, but not longer than a
3D texture can be (2048 per axis under the default WebGPU limits).  A grid
past that limit raises when the visual is built, with a message that says
which ``block_size`` fits; pygfx itself would only fail at the first draw,
leaving an empty canvas.
"""

from __future__ import annotations

import numpy as np
import pytest
from pygfx.renderers.wgpu.engine.shared import get_shared

from cellier.render.lut_indirection import BlockLayout3D, LutIndirectionManager3D
from cellier.render.lut_indirection._lut_indirection_manager_3d import (
    build_lut_texture,
)
from cellier.visuals import (
    MultiscaleImageRenderConfig,
    MultiscaleLabelRenderConfig,
)
from tests.render._affine import bind, multiscale_store
from tests.render.test_labels_pick_face import _viewer


@pytest.fixture
def limit() -> int:
    return int(get_shared().device.limits["max-texture-dimension-3d"])


def test_a_grid_longer_than_255_bricks_builds():
    layout = BlockLayout3D(volume_shape=(4, 8, 300), block_size=1)
    manager = LutIndirectionManager3D(layout, n_levels=1)
    assert manager.lut_data.shape == (4, 8, 300, 4)
    assert manager.brick_max_data.shape == (4, 8, 300)


def test_a_grid_at_the_texture_limit_builds(limit):
    lut, _tex = build_lut_texture(BlockLayout3D((1, 1, limit), block_size=1))
    assert lut.shape == (1, 1, limit, 4)


def test_a_grid_past_the_texture_limit_raises(limit):
    layout = BlockLayout3D(volume_shape=(3, 1, 2 * limit + 2), block_size=2)
    with pytest.raises(ValueError, match=r"Increase block_size to at least 3\.") as e:
        build_lut_texture(layout)
    assert f"2 x 1 x {limit + 1} bricks" in str(e.value)
    assert f"limit of {limit} bricks per axis" in str(e.value)


@pytest.mark.parametrize("family", ["image", "labels"])
def test_adding_a_volume_past_the_limit_raises(tmp_path, limit, family):
    """The error reaches the caller that adds the visual, not the first draw."""
    data = np.zeros((2, 2, 2 * limit + 2), dtype=np.int32)
    store = multiscale_store(tmp_path, "long", data)
    viewer = _viewer(("z", "y", "x"))
    try:
        transform = bind(viewer, store, np.eye(4))
        if family == "labels":
            config = MultiscaleLabelRenderConfig(block_size=2)
            add = viewer.add_labels_multiscale
        else:
            config = MultiscaleImageRenderConfig(block_size=2)
            add = viewer.add_image_multiscale
        with pytest.raises(ValueError, match="3D texture limit"):
            add(store, render_config=config, transform=transform)
    finally:
        viewer.controller.close()
