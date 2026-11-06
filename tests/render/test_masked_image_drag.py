"""The extent clamp under a camera drag (masked image design v2, MI-P16, 6.4).

A signed distance mask whose pyramid is floor halved, so each coarse level
covers less than the finest; frustum culling and a small mask budget, so a
drag draws the coarse backstop outside the settled frustum.  Without the
clamp a read past a coarse level's end finds the brick's zero padding and
draws far structures; with it, nothing past each yaw's settled silhouette.
The clamp's control is the material switch the template variable has for
tests only.

Apart from ``test_masked_image_surface.py`` because a drag scene is the
slowest work in the masked tests (up to two minutes a scene on a loaded
Windows runner), and pytest-xdist's ``--dist loadfile`` runs a file on one
worker: here it shares no worker with the rest of the surface tests.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pylinalg as la
import pytest
from scipy import ndimage as ndi

from cellier.convenience.capture import _wait_for_slicer
from cellier.render.shaders._masked_volume_brick import MaskedVolumeBrickMaterial
from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
from tests.render._affine import render
from tests.render._masked import (
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    pyramid_store,
)

#: Position_6's mask halved: 75 x 143 x 143 voxels, four levels.
DRAG_SHAPE = (75, 143, 143)
DRAG_LEVELS = 4
DRAG_SIZE = 384
DRAG_YAWS = [6.0 * k for k in range(1, 9)]
#: Settled silhouettes are dilated by this, as in the design's evidence.
DRAG_MARGIN_PX = 25


def _signed_distance(shape) -> np.ndarray:
    """An ellipsoid's signed distance, clamped to +-12, negative outside."""
    centre = (np.array(shape) - 1) / 2.0
    grid = np.indices(shape, dtype=np.float32)
    semi = np.array([24.0, 46.0, 44.0], np.float32)
    r = np.sqrt(sum(((grid[i] - centre[i]) / semi[i]) ** 2 for i in range(3)))
    return np.clip((1.0 - r) * semi.min(), -12.0, 12.0).astype(np.float32)


def _drag(
    tmp_path, iso: float, round_up: bool, clamps: tuple[bool, ...]
) -> dict[bool, tuple[int, int]]:
    """``clamp -> (frames with far pixels, most far pixels in a frame)``.

    One yaw drag per clamp setting, in one viewer.  Frustum culling on and a
    small mask budget, so a settled view holds fine bricks in its frustum and
    the coarse backstop elsewhere.  Each yaw's settled silhouette is kept,
    dilated, drawn with the clamp; then for each clamp setting, from the
    first view, the camera yaws inside ``camera_interaction`` (nothing
    reslices) and each frame's drawn pixels outside that yaw's silhouette
    are counted.  Settled frames without the clamp can already show far
    structures, so silhouettes of their own would hide some of the
    control's far pixels: against the clamped ones it counts more (the most
    in a frame 14,943 against 1,952 at iso 0 on Metal, in the same 8 frames).
    """
    rng = np.random.default_rng(0)
    mask = pyramid_store(
        tmp_path, "mask", _signed_distance(DRAG_SHAPE), DRAG_LEVELS, round_up
    )
    value = pyramid_store(
        tmp_path,
        "value",
        (0.5 + 0.5 * rng.random(DRAG_SHAPE)).astype(np.float32),
        DRAG_LEVELS,
        round_up,
    )
    viewer = new_viewer("3d")
    controller = viewer.controller
    try:
        model = controller.add_masked_image(
            mask,
            value,
            viewer.scene.id,
            appearance=MaskedImageAppearance(frustum_cull=True),
            single=MaskedImageSingleAppearance(
                color_map="viridis", clim=(0.0, 1.0), iso_threshold=iso
            ),
            render_config=masked_config(
                gpu_budget_bytes=6 * 1024**2, value_gpu_budget_bytes=64 * 1024**2
            ),
        )
        material = gfx_visual(viewer, model.id)._slots[0].material_3d
        assert isinstance(material, MaskedVolumeBrickMaterial)
        material.extent_clamp = True
        load(viewer, size=(DRAG_SIZE, DRAG_SIZE))
        canvas_id = controller.get_canvas_ids(viewer.scene.id)[0]
        base = controller.get_camera_state(canvas_id)._replace(zoom=2.0)
        centre = np.array([(n - 1) / 2.0 for n in DRAG_SHAPE[::-1]])

        def yawed(degrees: float):
            turn = la.quat_from_axis_angle(np.array(base.up), np.radians(degrees))
            position = la.vec_transform_quat(np.array(base.position) - centre, turn)
            rotation = la.quat_mul(turn, np.array(base.rotation))
            return base._replace(
                position=tuple(float(v) for v in position + centre),
                rotation=tuple(float(v) for v in rotation),
            )

        def settled(degrees: float) -> np.ndarray:
            controller.set_camera_state(canvas_id, yawed(degrees))
            asyncio.run(_wait_for_slicer(controller))
            return render(viewer, DRAG_SIZE)[0]

        def drawn(frame: np.ndarray) -> np.ndarray:
            return frame[..., :3].max(axis=-1) > 8

        silhouettes = {
            yaw: ndi.binary_dilation(drawn(settled(yaw)), iterations=DRAG_MARGIN_PX)
            for yaw in DRAG_YAWS
        }
        out = {}
        for clamp in clamps:
            material.extent_clamp = clamp
            settled(0.0)
            bad, worst = 0, 0
            with controller.camera_interaction(canvas_id):
                for yaw in DRAG_YAWS:
                    controller.set_camera_state(canvas_id, yawed(yaw))
                    far = int(
                        (drawn(render(viewer, DRAG_SIZE)[0]) & ~silhouettes[yaw]).sum()
                    )
                    bad += far > 0
                    worst = max(worst, far)
            out[clamp] = (bad, worst)
        return out
    finally:
        controller.close()


@pytest.fixture(scope="module")
def drag(tmp_path_factory):
    """``drag(iso, round_up)``: :func:`_drag` of both clamp settings, cached.

    One viewer per scene: the silhouettes are drawn once and both settings
    drag from them.  A rounded-up pyramid is only dragged with the clamp.
    """
    cache: dict[tuple[float, bool], dict[bool, tuple[int, int]]] = {}

    def get(iso: float, round_up: bool = False) -> dict[bool, tuple[int, int]]:
        if (iso, round_up) not in cache:
            root = tmp_path_factory.mktemp(f"drag_{iso}_{round_up}")
            clamps = (True,) if round_up else (True, False)
            cache[iso, round_up] = _drag(root, iso, round_up, clamps)
        return cache[iso, round_up]

    return get


@pytest.mark.parametrize("iso", [0.0, -4.0])
def test_a_camera_drag_draws_nothing_past_a_coarse_level_end(drag, iso):
    assert drag(iso)[True] == (0, 0)


@pytest.mark.parametrize("iso", [0.0, -4.0])
def test_without_the_clamp_the_drag_draws_far_structures(drag, iso):
    """The control: the zero padding in the missing strip draws surfaces."""
    bad, worst = drag(iso)[False]
    assert bad >= 4 and worst > 500, (bad, worst)


@pytest.mark.parametrize("clamp", [True, False])
def test_a_positive_threshold_never_reaches_the_zero_padding(drag, clamp):
    """The second control: zero is below +4, so the strip draws nothing either way."""
    assert drag(4.0)[clamp] == (0, 0)


def test_a_pyramid_rounded_up_is_clean_with_the_clamp(drag):
    """Rounded up, each level covers the extent; the clamp keeps it clean.

    Without the clamp this pyramid draws a few far pixels (4 in 2 frames
    here): a rounded-up level's last voxel centre can still stop short of
    the finest extent (level 3 ends at 73.5 of 74.5), and the half voxel
    past it blends in the zero padding.  The design's prototype saw none on
    Position_6.
    """
    assert drag(-4.0, round_up=True)[True] == (0, 0)


def test_settled_frames_barely_change_with_the_clamp(tmp_path):
    """At rest the clamp changes at most a few pixels (6.4)."""
    mask = pyramid_store(tmp_path, "mask", _signed_distance(DRAG_SHAPE), DRAG_LEVELS)
    value = pyramid_store(
        tmp_path, "value", np.full(DRAG_SHAPE, 0.7, np.float32), DRAG_LEVELS
    )
    viewer = new_viewer("3d")
    frames = []
    try:
        model = viewer.controller.add_masked_image(
            mask,
            value,
            viewer.scene.id,
            single=MaskedImageSingleAppearance(
                color_map="viridis", clim=(0.0, 1.0), iso_threshold=-4.0
            ),
            render_config=masked_config(
                gpu_budget_bytes=64 * 1024**2, value_gpu_budget_bytes=64 * 1024**2
            ),
        )
        material = gfx_visual(viewer, model.id)._slots[0].material_3d
        material.extent_clamp = True
        load(viewer, size=(DRAG_SIZE, DRAG_SIZE))
        for clamp in (True, False):
            material.extent_clamp = clamp
            frames.append(render(viewer, DRAG_SIZE)[0][..., :3].astype(np.int64))
    finally:
        viewer.controller.close()
    differing = (np.abs(frames[0] - frames[1]).max(axis=-1) > 8).sum()
    assert differing <= 10, differing
