"""The multiscale visuals upload each level's translation (plan v2, Phase 4).

``offset_k`` in the block-scales uniform holds level ``k``'s translation in
level-0 voxels, shader order.  No shader reads it until Phases 5-6; this
checks the image and labels visuals, 2D and 3D, fill it from the store.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import ProgressiveLoadingConfig
from cellier.visuals._image import MultiscaleImageRenderConfig
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)

SHAPES = [(16, 32, 32), (8, 16, 8)]
#: Level 1 scale (z, y, x) and an in-contract translation, level-0 voxels.
SCALE = (2.0, 2.0, 4.0)
TRANSLATION = (0.5, 0.0, 2.0)


def _store(root, dtype) -> MultiscaleZarrDataStore:
    for index, shape in enumerate(SHAPES):
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
            },
            create=True,
            dtype=dtype,
            shape=shape,
        ).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), SCALE],
        level_translations=[(0.0, 0.0, 0.0), TRANSLATION],
        name="offsets",
    )


@pytest.mark.parametrize("kind", ["image", "labels"])
@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_visuals_upload_level_offsets(controller, reslice, tmp_path, kind, dim):
    scene = controller.add_scene(dim=dim, name=f"offsets-{kind}-{dim}")
    loading = ProgressiveLoadingConfig(backstop=False)
    if kind == "image":
        visual = controller.add_image_multiscale(
            data=_store(tmp_path, ts.float32),
            scene_id=scene.id,
            render_config=MultiscaleImageRenderConfig(block_size=8, loading=loading),
        )
    else:
        visual = controller.add_labels_multiscale(
            data=_store(tmp_path, ts.int32),
            scene_id=scene.id,
            appearance=MultiscaleLabelsAppearance(),
            render_config=MultiscaleLabelRenderConfig(block_size=8, loading=loading),
        )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    owners = [gfx] if kind == "labels" else [s for s in gfx._slots if s is not None]

    buffers = []
    for owner in owners:
        name = "_brick_scales_buffer" if dim == "3d" else "_block_scales_buffer_2d"
        buffer = getattr(owner, name, None)
        if buffer is not None:
            buffers.append(buffer.data)
    assert buffers
    # Shader order reverses data order; 2D shows the displayed (y, x) pair.
    expected = TRANSLATION[::-1] if dim == "3d" else TRANSLATION[1:][::-1]
    for data in buffers:
        np.testing.assert_array_equal(data["offset_1"][: len(expected)], 0.0)
        np.testing.assert_array_equal(data["offset_2"][: len(expected)], expected)
