"""A 3D isosurface image pick reads the voxel it landed on, on every face.

The same bug as ``test_labels_pick_face``: the pick buffer holds each axis in
14 bits and truncates, so a hit on a voxel face came back up to ``dim / 16383``
voxels low, out of the voxel on a low face.  An isosurface of a 0/1 block at
threshold 0.5 sits exactly on the block's faces (linear interpolation crosses
0.5 halfway between a 0 and a 1), so it is the case that shows it.  Both image
shaders now clamp the reported point into the interior of its voxel.

Every pick on the block must read the block's value.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data import ImageMemoryStore
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import bind, block_volume, load, multiscale_store
from tests.render.test_labels_pick_face import ORDERS, _pick_every_view, _viewer

ISO = 0.5
FAMILIES = ("image", "image_multiscale")


def _add_image(viewer, family, tmp_path):
    data = block_volume(np.float32)
    if family == "image":
        store = ImageMemoryStore(data=data, name="landmark")
        return viewer.add_image(
            store,
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="iso", iso_threshold=ISO
            ),
            transform=bind(viewer, store, np.eye(4)),
        )
    store = multiscale_store(tmp_path, "landmark", data)
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(force_level=0),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode="iso", iso_threshold=ISO
        ),
        transform=bind(viewer, store, np.eye(4)),
    )


@pytest.mark.parametrize("displayed", ORDERS, ids="".join)
@pytest.mark.parametrize("family", FAMILIES)
def test_iso_picks_read_the_block_on_every_face(tmp_path, family, displayed):
    viewer = _viewer(displayed)
    try:
        visual = _add_image(viewer, family, tmp_path)
        load(viewer)
        data = block_volume(np.float32)
        wrong = []

        def check(direction, level, full):
            voxel = tuple(int(v) for v in np.floor(np.asarray(full)))
            if data[voxel] != 1.0:
                wrong.append((direction.round(2).tolist(), full))

        n_picks = _pick_every_view(viewer, visual, displayed, check)
        assert not wrong, (
            f"{len(wrong)} of {n_picks} picks read background: {wrong[:5]}"
        )
    finally:
        viewer.controller.close()
