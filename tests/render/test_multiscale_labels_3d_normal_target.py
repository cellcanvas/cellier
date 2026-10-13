"""3D multiscale labels draw in every render mode with the normal target.

Ambient occlusion installs the ``normal`` render target, which sets
``write_normal`` on every shader.  The multiscale labels shader then needs a
surface normal in every mode, but ``label_object_normal`` was only compiled
for the modes that shade with it, so ``flat_categorical`` called a function
that was not there and the shader did not compile.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.render._cellier_blender import NORMAL_TARGET
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._planes import shoot

SHAPES = [(16, 16, 16), (8, 8, 8)]
CUBE_LO, CUBE_HI = 5, 10  # inclusive, level-0 voxels, every axis
LABEL = 3

RENDER_MODES = get_args(
    MultiscaleLabelsAppearance.model_fields["render_mode"].annotation
)


def _store(root) -> MultiscaleZarrDataStore:
    level0 = np.zeros(SHAPES[0], dtype=np.int32)
    level0[CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1] = LABEL
    for index, data in enumerate((level0, level0[::2, ::2, ::2])):
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
            "metadata": {
                "shape": list(data.shape),
                "data_type": "int32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(data.shape)},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(np.ascontiguousarray(data)).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        name="labels-3d-normal-target",
    )


def test_every_render_mode_is_covered():
    assert set(RENDER_MODES) >= {
        "iso_categorical",
        "flat_categorical",
        "gradient_debug",
        "smooth_iso",
    }


@pytest.mark.parametrize("render_mode", RENDER_MODES)
async def test_labels_draw_with_the_normal_target(
    controller, reslice, tmp_path, render_mode
):
    scene = controller.add_scene(
        displayed_axes=("z", "y", "x"), name=f"normal-{render_mode}"
    )
    controller.add_labels_multiscale(
        data=_store(tmp_path),
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=0, render_mode=render_mode),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    # A shader that does not compile raises out of the draw.
    shot = shoot(controller, scene, size=(128, 128), blender_targets=[NORMAL_TARGET])

    if render_mode != "plane":  # no render plane was added, so nothing is cut
        assert (shot.frame[..., 3] > 0).any(), "the label cube drew no pixel"
