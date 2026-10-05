"""A ray-marched volume shades each pixel once.

The proxy box of a ray-marched volume covers most pixels twice, with a front
and a back face, and every fragment marches the whole ray from the near
plane.  Drawing both faces is invisible while the material writes depth (the
second fragment mostly fails the depth test) and doubles the result when it
does not: a composite of overlapping channels adds each channel twice, and
translucent labels blend twice.  The pipelines cull front faces, and the
vertex shader keeps the winding under a mirroring transform, so the back
faces are the ones kept and the box still draws with the camera inside it.

Each case draws a uniform volume with a material that does not write depth
and compares the centre pixel with a stored value.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.scene import spatial_axes
from cellier.visuals import (
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    ProgressiveLoadingConfig,
)
from cellier.visuals._image import MultiscaleImageRenderConfig
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests._v2 import bound
from tests.render.conftest import _write_multiscale_zarr

_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]
#: Every voxel of both channels.  Each channel maps it to 0.25 of its colour;
#: two shaded fragments per pixel would add up to 0.5.
VALUE = 0.25
#: Labels opacity.  Blended once it is the frame's alpha; twice gives 0.75.
OPACITY = 0.5

#: Stored results: the centre pixel of the frame, uint8, drawn over a hidden
#: background.  Image: the red and the green channel, each at 0.25 (both
#: faces drawn gave 90).  Labels: the frame's alpha (both faces gave 143).
EXPECTED_IMAGE_RGB = (64, 64, 0)
EXPECTED_LABELS_ALPHA = 64
TOLERANCE = 3

VIEWS = ["outside", "inside", "mirrored_outside", "mirrored_inside"]


def _image(controller, tmp_path, mirrored):
    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))],
        fill=lambda arr: arr.fill(VALUE),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="czyx",
    )
    scene = controller.add_scene(coordinate_system=_CZYX, dim="3d", name="scene")
    controller.add_image_multiscale(
        store,
        scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(
            **SMALL_BUDGETS,
            block_size=8,
            loading=ProgressiveLoadingConfig(backstop=False),
        ),
        transform=_mirror(controller, scene, store, 4) if mirrored else None,
        channel_axis=0,
        composite=True,
        channels={
            0: MultiscaleImageChannelAppearance(
                color_map="red", clim=(0.0, 1.0), render_mode="mip"
            ),
            1: MultiscaleImageChannelAppearance(
                color_map="green", clim=(0.0, 1.0), render_mode="mip"
            ),
        },
    )
    return scene


def _labels_appearance(cls, **fields):
    return cls(
        **fields,
        render_mode="flat_categorical",
        opacity=OPACITY,
        depth_write=False,
        transparency_mode="blend",
    )


def _labels_multiscale(controller, tmp_path, mirrored):
    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=lambda arr: arr.fill(3),
        dtype="int32",
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        name="labels",
    )
    scene = controller.add_scene(dim="3d", name="scene")
    appearance = _labels_appearance(MultiscaleLabelsAppearance, force_level=1)
    controller.add_labels_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=appearance,
        render_config=MultiscaleLabelRenderConfig(
            **SMALL_BUDGETS,
            block_size=8,
            loading=ProgressiveLoadingConfig(backstop=False),
        ),
        transform=_mirror(controller, scene, store, 3) if mirrored else None,
    )
    return scene


def _labels_memory(controller, tmp_path, mirrored):
    store = LabelMemoryStore(data=np.full((16, 16, 16), 3, dtype=np.int32))
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels(
        data=store,
        scene_id=scene.id,
        appearance=_labels_appearance(InMemoryLabelsAppearance),
        transform=_mirror(controller, scene, store, 3) if mirrored else None,
    )
    return scene


def _mirror(controller, scene, store, ndim):
    """A data -> world transform that mirrors the last axis in place."""
    scale = [1.0] * (ndim - 1) + [-1.0]
    translation = [0.0] * (ndim - 1) + [15.0]
    return bound(controller, scene.id, store, scale, translation)


BUILDERS = {
    "image": _image,
    "labels_multiscale": _labels_multiscale,
    "labels_memory": _labels_memory,
}


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("kind", list(BUILDERS))
async def test_each_pixel_is_shaded_once(
    controller, render_scene, offscreen_renderer, reslice, tmp_path, kind, view
):
    scene = BUILDERS[kind](controller, tmp_path, view.startswith("mirrored"))
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    render_scene(controller, scene.id)  # hides the background, fits the camera
    canvas = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas._get_scene_fn(scene.id)
    camera = canvas.camera
    if view.endswith("inside"):
        # At the centre of the 16-voxel box, looking off-axis at a wall.
        camera.local.position = (7.5, 7.5, 7.5)
        camera.look_at((20.0, 11.0, 10.0))
        camera.depth_range = (0.1, 100.0)

    frame = offscreen_renderer(gfx_scene, camera, (64, 64))
    centre = frame[32, 32].astype(int)

    if kind == "image":
        assert centre[:3] == pytest.approx(EXPECTED_IMAGE_RGB, abs=TOLERANCE)
    else:
        assert centre[3] == pytest.approx(EXPECTED_LABELS_ALPHA, abs=TOLERANCE)


#: Stored result: bricks resident in the 3D block cache after one reslice
#: with frustum culling on.  The 16-voxel volume is a 2x2x2 grid of 8-voxel
#: bricks at level 1; the image caches that grid once per channel.
EXPECTED_RESIDENT_BRICKS = {"image": 16, "labels_multiscale": 8}


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("kind", list(EXPECTED_RESIDENT_BRICKS))
async def test_frustum_cull_selects_bricks_under_mirroring(
    controller, reslice, tmp_path, kind, mirrored
):
    """Frustum culling keeps the same bricks under a mirroring transform.

    Regression: the frustum corners are mapped into data space before the
    planes are built, and a negative-determinant transform reversed their
    winding, so every plane faced outward and no brick was selected.
    """
    scene = BUILDERS[kind](controller, tmp_path, mirrored)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)

    (gfx_visual,) = controller._render_manager._scenes[scene.id]._visuals.values()
    owners = gfx_visual.slots if kind == "image" else (gfx_visual,)
    resident = sum(
        len(owner._block_cache_3d.tile_manager.tilemap)
        for owner in owners
        if owner._block_cache_3d is not None
    )
    assert resident == EXPECTED_RESIDENT_BRICKS[kind]
