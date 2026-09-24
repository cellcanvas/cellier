"""``ray_steps_per_voxel``: the 3D ray-march step density of multiscale visuals.

Both 3D brick shaders take ``ray_steps_per_voxel`` samples per voxel of the
drawn level, counted along the ray (plan v2, D6).  Slabs one voxel thick,
seen face-on, show the density directly: at one sample per voxel every ray
lands a sample in each; at one per two voxels a ray can step over one.  The
setting is a live appearance field, so the test changes it between renders.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts
from pydantic import ValidationError

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS

SHAPES = [(32, 32, 32), (16, 16, 16)]
SCALES = [tuple(s0 / sk for s0, sk in zip(SHAPES[0], shape)) for shape in SHAPES]
#: Two slabs, each one voxel thick, at adjacent level-0 planes and side by
#: side in x.  Samples two voxels apart land in exactly one of two adjacent
#: voxels, so at 0.5 samples per voxel exactly one slab draws, whatever the
#: sampling phase.  (A single slab is hit all or nothing, depending on where
#: the brick entry puts the samples -- which Phase 6 of plan v2 moved.)
SLAB_Z = (15, 16)


def _write_slab(root, dtype: str, value) -> MultiscaleZarrDataStore:
    for index, shape in enumerate(SHAPES):
        arr = np.zeros(shape, dtype=dtype)
        if index == 0:
            arr[SLAB_Z[0], 4:-4, 4:16] = value
            arr[SLAB_Z[1], 4:-4, 16:-4] = value
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root / f"s{index}")},
            "metadata": {
                "shape": list(shape),
                "data_type": dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(shape)},
                },
            },
            "create": True,
            "delete_existing": True,
        }
        ts.open(spec).result()[...].write(arr).result()
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{index}" for index in range(len(SHAPES))],
        level_scales=SCALES,
        level_translations=[tuple((s - 1.0) / 2.0 for s in scale) for scale in SCALES],
        name="slab",
    )


def _add_visual(controller, scene_id, root, kind):
    loading = ProgressiveLoadingConfig(backstop=False)
    if kind == "labels":
        return controller.add_labels_multiscale(
            data=_write_slab(root, "int32", 1),
            scene_id=scene_id,
            appearance=MultiscaleLabelsAppearance(
                force_level=1, render_mode="flat_categorical"
            ),
            render_config=MultiscaleLabelRenderConfig(
                **SMALL_BUDGETS, block_size=8, loading=loading
            ),
        )
    return controller.add_image_multiscale(
        data=_write_slab(root, "float32", 1.0),
        scene_id=scene_id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(
            **SMALL_BUDGETS, block_size=8, loading=loading
        ),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode="mip"
        ),
    )


@pytest.mark.parametrize("kind", ["labels", "image"])
async def test_density_is_live_and_sets_what_a_ray_can_miss(
    controller, render_scene, offscreen_renderer, reslice, tmp_path, kind
):
    """Face-on, both one-voxel slabs draw in full at 1 sample per voxel.

    At 0.5 the step is two voxels: labels rays (no jitter) all step over the
    same one of the two slabs; MIP rays sample each at most half-way into a
    voxel, so the slabs dim.  Going back to 1.0 redraws the first frame
    exactly: the value is a live uniform, not a rebuild.
    """
    scene = controller.add_scene(dim="3d", name=f"slab-{kind}")
    visual = _add_visual(controller, scene.id, tmp_path, kind)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    render_scene(controller, scene.id)  # hides the background
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    canvas = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas._get_scene_fn(scene.id)
    # Near-orthographic, looking along the slab's normal (data z).
    canvas.camera.fov = 1
    canvas.camera.show_object(gfx_scene, view_dir=(0, 0, -1), up=(0, 1, 0))

    frames = {}
    for density in (1.0, 0.5, 2.0):
        visual.appearance.ray_steps_per_voxel = density
        frames[density] = offscreen_renderer(gfx_scene, canvas.camera, (256, 256))
    visual.appearance.ray_steps_per_voxel = 1.0
    again = offscreen_renderer(gfx_scene, canvas.camera, (256, 256))

    materials = (
        [gfx.material_3d]
        if kind == "labels"
        else [slot.material_3d for slot in gfx._slots if slot.material_3d]
    )
    assert materials
    assert all(m.ray_steps_per_voxel == 1.0 for m in materials)

    def brightness(frame):
        return float(frame[..., :3].astype(np.float64).sum())

    full = brightness(frames[1.0])
    assert full > 0
    assert brightness(frames[2.0]) == pytest.approx(full, rel=0.02)
    assert brightness(frames[0.5]) < 0.75 * full
    np.testing.assert_array_equal(again, frames[1.0])


def test_density_bounds():
    for model in (MultiscaleImageAppearance, MultiscaleLabelsAppearance):
        assert model().ray_steps_per_voxel == 1.0
        for bad in (0.49, 8.01):
            with pytest.raises(ValidationError):
                model(ray_steps_per_voxel=bad)
