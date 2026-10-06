"""A ray-marched volume marches its rays away from the camera, however rolled.

The brick and label volume shaders build each ray from a point on the near
plane and one on the far plane, and swap the two for a camera that mirrors
an axis.  They used to detect that from the product of the camera matrix's
diagonal, which is also negative for some pure rotations (a camera looking
along z and rolled a quarter turn, about): the ray then ran from the far
plane towards the camera and the first hit was the far side of the data.
The orbit controller keeps the camera level and never gets there; a rotation
set in code does.

The volume is two labels stacked along z, seen from the low-z side.  The
centre pixel is compared with the stored colour of the near label.
"""

from __future__ import annotations

import numpy as np
import pylinalg as la
import pytest

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.data.label._label_memory_store import LabelMemoryStore
from cellier.visuals import InMemoryLabelsAppearance
from cellier.visuals._labels import (
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

NEAR_LABEL = 3  # low z
FAR_LABEL = 7  # high z
#: Stored results: the flat colour of each label, RGB uint8.
NEAR_COLOUR = (245, 31, 253)
FAR_COLOUR = (226, 64, 222)
TOLERANCE = 3

#: Camera rotations, as (axis, degrees).  All look along +z, or within a few
#: degrees of it.  The diagonal product is +1 for the first and negative or
#: zero (to rounding) for the others; the determinant is +1 for all.
ROTATIONS = {
    "level": ((1.0, 0.0, 0.0), 180.0),
    "rolled_170": ((1.0, 1.0, 0.0), 170.0),
    "rolled_190": ((1.0, 1.0, 0.0), 190.0),
    "rolled_180": ((1.0, 1.0, 0.0), 180.0),
}


def _fill(arr: np.ndarray) -> None:
    arr[: arr.shape[0] // 2] = NEAR_LABEL
    arr[arr.shape[0] // 2 :] = FAR_LABEL


def _labels_memory(controller, tmp_path):
    labels = np.empty((16, 16, 16), dtype=np.int32)
    _fill(labels)
    scene = controller.add_scene(dim="3d", name="scene")
    controller.add_labels(
        data=LabelMemoryStore(data=labels),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(render_mode="flat_categorical"),
    )
    return scene


def _labels_multiscale(controller, tmp_path):
    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=_fill,
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
    controller.add_labels_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleLabelsAppearance(
            render_mode="flat_categorical", force_level=1
        ),
        render_config=MultiscaleLabelRenderConfig(
            **SMALL_BUDGETS,
            block_size=8,
        ),
    )
    return scene


BUILDERS = {
    "labels_memory": _labels_memory,
    "labels_multiscale": _labels_multiscale,
}


@pytest.mark.parametrize("rotation", list(ROTATIONS))
@pytest.mark.parametrize("kind", list(BUILDERS))
async def test_the_near_label_is_drawn_whatever_the_roll(
    controller, render_scene, offscreen_renderer, reslice, tmp_path, kind, rotation
):
    scene = BUILDERS[kind](controller, tmp_path)
    controller.add_canvas(scene_id=scene.id)
    await reslice(controller, scene.id)
    render_scene(controller, scene.id)  # hides the background
    canvas = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]
    gfx_scene = canvas._get_scene_fn(scene.id)
    camera = canvas.camera
    axis, degrees = ROTATIONS[rotation]
    quaternion = la.quat_from_axis_angle(
        np.asarray(axis) / np.linalg.norm(axis), np.deg2rad(degrees)
    )
    forward = la.vec_transform_quat((0.0, 0.0, -1.0), quaternion)
    camera.local.rotation = quaternion
    camera.local.position = np.full(3, 7.5) - 40.0 * forward
    camera.depth_range = (1.0, 200.0)

    frame = offscreen_renderer(gfx_scene, camera, (64, 64))

    centre = frame[32, 32, :3].astype(int)
    assert centre == pytest.approx(NEAR_COLOUR, abs=TOLERANCE)
    assert centre != pytest.approx(FAR_COLOUR, abs=TOLERANCE)
