"""3D multiscale labels: a pick just inside a label's edge names that label.

``plans/multiscale_level_transform_v2.md``, R3 (3D).  Before Phase 6 the 3D
labels shader drew every voxel half a level-0 voxel low on each axis, while
the pick decode names the voxel in the centre convention, so a click just
inside a label's silhouette on its low side reported the background behind
it.  Draw and pick go through one renderer without anti-aliasing, so the
drawn pixel is exactly what the shader wrote.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorstore as ts

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.visuals import ProgressiveLoadingConfig
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS

SHAPES = [(16, 16, 16), (8, 8, 8)]
CUBE_LO, CUBE_HI = 5, 10  # inclusive, level-0 voxels, every axis
LABEL = 3
SIZE = (256, 256)


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
        name="labels-3d-pick",
    )


#: ``(name, view direction, up)``: down z and down x, so the edges tested
#: cover all three data axes, low and high sides.
_VIEWS = [
    ("down z", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("down x", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
]


@pytest.mark.parametrize("render_mode", ["iso_categorical", "flat_categorical"])
@pytest.mark.parametrize(("name", "view_dir", "up"), _VIEWS)
async def test_a_pick_just_inside_the_silhouette_names_the_label(
    controller, render_scene, reslice, tmp_path, render_mode, name, view_dir, up
):
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    scene = controller.add_scene(dim="3d", name=f"pick-{name}-{render_mode}")
    visual = controller.add_labels_multiscale(
        data=_store(tmp_path),
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(force_level=1, render_mode=render_mode),
        render_config=MultiscaleLabelRenderConfig(
            **SMALL_BUDGETS,
            block_size=8,
            loading=ProgressiveLoadingConfig(backstop=False),
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    render_scene(controller, scene.id)  # hides the background
    canvas_view = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas_view._get_scene_fn(scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    camera = canvas_view.camera
    camera.fov = 1
    camera.show_object(gfx_scene, view_dir=view_dir, up=up)

    canvas = RenderCanvas(size=SIZE, pixel_ratio=1)
    renderer = gfx.WgpuRenderer(canvas)
    renderer.pixel_scale = 1
    renderer.ppaa = "none"
    canvas.request_draw(lambda: renderer.render(gfx_scene, camera))
    drawn = np.asarray(canvas.draw())[..., 3] > 0
    assert drawn.sum() > 1000, f"{name}: the cube did not draw"

    # One pixel inside the silhouette: drawn pixels whose neighbour one step
    # further out is empty, on every side.
    rows, cols = np.nonzero(drawn)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    mid_r, mid_c = (r0 + r1) // 2, (c0 + c1) // 2
    targets = {
        "left": (mid_r, c0 + 1),
        "right": (mid_r, c1 - 1),
        "top": (r0 + 1, mid_c),
        "bottom": (r1 - 1, mid_c),
    }
    level0 = np.zeros(SHAPES[0], dtype=np.int32)
    level0[CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1, CUBE_LO : CUBE_HI + 1] = LABEL
    wrong = {}
    for side, (r, c) in targets.items():
        assert drawn[r, c], (name, side)
        info = renderer.get_pick_info((c + 0.5, r + 0.5))
        coord = gfx_visual.pick_data_coordinate(info.get("world_object"), info)
        assert coord is not None, (name, side)
        # pygfx (x, y, z) reverses onto data (z, y, x); [i, i + 1) convention.
        index = tuple(int(np.floor(v)) for v in tuple(coord)[::-1])
        if level0[index] != LABEL:
            wrong[side] = index
    assert not wrong, f"{name} / {render_mode}: picks outside the label {wrong}"
