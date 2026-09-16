"""An image whose slice misses the data draws nothing (design 3.2).

Rendered end to end, in memory and multiscale, 2D and 3D.  Each case slices
inside the data first (pixels drawn), moves the slider past the last plane
(nothing drawn), and comes back (drawn again).  Labels are not covered here:
they keep clamping.
"""

from __future__ import annotations

import numpy as np

from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests.render.conftest import _write_multiscale_zarr

_TZYX = [("t", "time"), *spatial_axes("z", "y", "x")]


def _drawn(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame[..., 3]))


def _bright(arr: np.ndarray) -> None:
    arr[...] = 1.0


async def _assert_hides_and_returns(
    controller, scene, visual, render_scene, reslice, axis, inside, outside
):
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)

    controller.update_slice_indices(scene.id, {axis: inside})
    await reslice(controller, scene.id)
    assert _drawn(render_scene(controller, scene.id)) > 0
    assert gfx._slice_empty is False

    controller.update_slice_indices(scene.id, {axis: outside})
    await reslice(controller, scene.id)
    assert gfx._slice_empty is True
    assert _drawn(render_scene(controller, scene.id)) == 0

    controller.update_slice_indices(scene.id, {axis: inside})
    await reslice(controller, scene.id)
    assert gfx._slice_empty is False
    assert _drawn(render_scene(controller, scene.id)) > 0


async def test_in_memory_2d(controller, render_scene, reslice):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_image(
        data=ImageMemoryStore(data=np.ones((4, 16, 16), dtype=np.float32)),
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 2.0, 3.5
    )
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 2.0, -10.0
    )


async def test_in_memory_3d(controller, render_scene, reslice):
    scene = controller.add_scene(dim="3d", coordinate_system=_TZYX, name="scene")
    appearance = InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0))
    visual = controller.add_image(
        data=ImageMemoryStore(data=np.ones((3, 16, 16, 16), dtype=np.float32)),
        scene_id=scene.id,
        appearance=appearance,
    )
    controller.add_canvas(scene_id=scene.id)
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 1.0, -3.0
    )


def _multiscale_store(tmp_path, level_shapes, scales, translations):
    _write_multiscale_zarr(
        tmp_path,
        levels=[(f"s{i}", shape) for i, shape in enumerate(level_shapes)],
        fill=_bright,
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=[f"s{i}" for i in range(len(level_shapes))],
        level_scales=scales,
        level_translations=translations,
        name="store",
    )


async def test_multiscale_2d(controller, render_scene, reslice, tmp_path):
    store = _multiscale_store(
        tmp_path,
        [(16, 16, 16), (8, 8, 8)],
        [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    )
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        render_config=MultiscaleImageRenderConfig(block_size=8),
        single=MultiscaleImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 8.0, 15.5
    )
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 8.0, 40.0
    )


async def test_multiscale_3d(controller, render_scene, reslice, tmp_path):
    store = _multiscale_store(
        tmp_path,
        [(3, 16, 16, 16), (3, 8, 8, 8)],
        [(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        [(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
    )
    scene = controller.add_scene(dim="3d", coordinate_system=_TZYX, name="scene")
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(),
        render_config=MultiscaleImageRenderConfig(block_size=8),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    await _assert_hides_and_returns(
        controller, scene, visual, render_scene, reslice, 0, 1.0, 7.0
    )
