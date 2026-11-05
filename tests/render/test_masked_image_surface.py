"""The MaskedImage's 3D surface (masked image design v2, Phase 4; 6.2, 6.4).

The pair is the synthetic example's (``tests/render/_masked.py``): the mask
is an ellipsoid's distance map and the value its bands of ``BAND_WIDTH``
voxels, so the band a surface shows is known from the threshold, and the
band a projection along the normal shows from the threshold, the thickness
and the placement.  Each band is one primary of a nearest colormap, read as
a pixel's brightest channel (Phong shading scales a colour and adds a grey
highlight).

The extent clamp (MI-P16) is checked on a signed distance mask whose pyramid
is floor halved, so each coarse level covers less than the finest.  Its
control switches the clamp off through the material, the switch the
template variable has for this purpose only.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pylinalg as la
import pytest
from scipy import ndimage as ndi

from cellier.convenience.capture import _wait_for_slicer
from cellier.render.shaders._masked_volume_brick import (
    MaskedVolumeBrickMaterial,
    instantiate_by_text,
    value_access_source,
)
from cellier.render.visuals._masked_image import surface_interval
from cellier.visuals import MaskedImageAppearance, MaskedImageSingleAppearance
from tests.render._affine import render
from tests.render._masked import (
    band_colormap,
    band_counts,
    band_pair,
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    pyramid_store,
    store,
    value_to_mask_matrix,
)

SIDE = 64
SIZE = 256


def _band_viewer(tmp_path, kind, value_scale=1, **single):
    """A 3D viewer of the band pair, loaded; ``(viewer, model)``."""
    mask_data, value_data = band_pair(SIDE, value_scale)
    viewer = new_viewer("3d")
    mask = store(kind, tmp_path, "mask", mask_data)
    value = store(kind, tmp_path, "value", value_data)
    # An in-memory value at twice the resolution is 512 bricks of 16.
    value_budget = 48 if kind == "memory" and value_scale == 2 else 16
    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        value_to_mask=value_to_mask_matrix(value_scale),
        single=MaskedImageSingleAppearance(
            color_map=band_colormap(), clim=(0.5, 3.5), **single
        ),
        render_config=masked_config(value_gpu_budget_bytes=value_budget * 1024**2),
    )
    load(viewer, size=(SIZE, SIZE))
    return viewer, model


# -- the value at the surface (6.2) ----------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "lazy"])
@pytest.mark.parametrize("value_scale", [1, 2])
@pytest.mark.parametrize(("iso", "band"), [(4.0, 1), (12.0, 2)])
def test_the_surface_shows_its_band(tmp_path, kind, value_scale, iso, band):
    """Every drawn pixel of the surface at ``iso`` is the band ``iso`` lies in.

    On one grid and on a value at twice the mask's resolution, its own grid.
    """
    viewer, _model = _band_viewer(tmp_path, kind, value_scale, iso_threshold=iso)
    try:
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts.sum() > 500, counts
        assert counts[band - 1] == counts.sum(), counts
    finally:
        viewer.controller.close()


# -- the projection along the normal (6.2) ---------------------------------------


@pytest.mark.parametrize(
    ("op", "placement", "iso", "thickness", "bands"),
    [
        # Inward from iso 4 over 8 voxels: distances 4 to 12, bands 1 and 2.
        ("max", "inward", 4.0, 8.0, {2}),
        ("min", "inward", 4.0, 8.0, {1}),
        ("mean", "inward", 4.0, 8.0, {1, 2}),
        # Outward from iso 4 over 3: distances 1 to 4, band 1 only.
        ("max", "outward", 4.0, 3.0, {1}),
        # Outward from iso 13 over 8: distances 5 to 13.
        ("min", "outward", 13.0, 8.0, {1}),
        ("max", "outward", 13.0, 8.0, {2}),
        # Centred on iso 13 over 8: distances 9 to 17, band 2 at least.
        ("min", "centred", 13.0, 8.0, {2}),
    ],
)
def test_the_projection_reduces_along_the_inward_normal(
    tmp_path, op, placement, iso, thickness, bands
):
    """The max, min or mean over the placed interval gives the bands it spans.

    The cases pin the sign of the normal (a max inward is deeper, a max
    outward is not) and that the thickness is in world units.
    """
    viewer, _model = _band_viewer(
        tmp_path,
        "memory",
        iso_threshold=iso,
        surface_thickness=thickness,
        surface_projection=op,
        surface_placement=placement,
    )
    try:
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts.sum() > 500, counts
        shown = {band + 1 for band in range(3) if counts[band] > 0}
        assert shown == bands, counts
    finally:
        viewer.controller.close()


def test_a_projection_follows_the_thickness_when_it_changes(tmp_path):
    """The single appearance reaches the material: a max deepens as T grows."""
    viewer, model = _band_viewer(tmp_path, "lazy", iso_threshold=4.0)
    try:
        assert band_counts(render(viewer, SIZE)[0])[1] == 0
        model.single.surface_thickness = 8.0
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts[1] == counts.sum() > 0, counts
        model.single.surface_projection = "min"
        counts = band_counts(render(viewer, SIZE)[0])
        assert counts[0] == counts.sum() > 0, counts
    finally:
        viewer.controller.close()


@pytest.mark.parametrize(
    ("placement", "expected"),
    [("inward", (0.0, 6.0)), ("centred", (-3.0, 3.0)), ("outward", (-6.0, 0.0))],
)
def test_surface_interval(placement, expected):
    assert surface_interval(6.0, placement) == expected


# -- the value side's copy of the lookup -------------------------------------------


def test_the_value_lookup_is_the_mask_lookup_renamed():
    """No mask binding or accessor survives; the masked clamp is not copied."""
    text = instantiate_by_text(value_access_source())
    code = "\n".join(line.split("//")[0] for line in text.splitlines())
    for name in ("t_cache", "s_cache", "u_vol_params", "u_block_scales"):
        assert f" {name}" not in code and f"({name}" not in code, name
    assert "fn value_sample_atlas(" in code
    assert "mask_extent_clamp" not in code
    assert "$$" not in code


def test_instantiate_by_text_refuses_a_missing_accessor():
    with pytest.raises(ValueError, match="not found"):
        instantiate_by_text("fn get_lod_scale() {}")


# -- the extent clamp (MI-P16, 6.4) -----------------------------------------------


def _strip_viewer(tmp_path, clamp: bool):
    """A mask inside everywhere, 39 voxels on x, forced to its coarsest level.

    Floor halved twice, the coarsest level covers 36 of the 39 voxels on x.
    The value is 1 from x = 37 on, 0 before.
    """
    from cellier.data import ImageMemoryStore

    shape = (24, 24, 39)
    mask = pyramid_store(tmp_path, "mask", np.full(shape, 10.0, np.float32), 3)
    value_data = np.zeros(shape, np.float32)
    value_data[..., 37:] = 1.0
    value = ImageMemoryStore(data=value_data, name="value")
    viewer = new_viewer("3d")
    import cmap

    model = viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        appearance=MaskedImageAppearance(force_level=3),
        single=MaskedImageSingleAppearance(
            color_map=cmap.Colormap(["red", "lime"], interpolation="nearest"),
            clim=(0.0, 1.0),
            iso_threshold=5.0,
        ),
        render_config=masked_config(),
    )
    gfx_visual(viewer, model.id)._slots[0].material_3d.extent_clamp = clamp
    load(viewer, size=(SIZE, SIZE))
    view = viewer.controller._render_manager._canvases[
        viewer.controller.get_canvas_ids(viewer.scene.id)[0]
    ]
    view.camera.show_object(
        view._get_scene_fn(viewer.scene.id), view_dir=(-1, 0, 0), up=(0, 0, 1)
    )
    return viewer


@pytest.mark.parametrize("clamp", [True, False])
def test_a_read_past_a_coarse_level_end_repeats_its_last_voxel(tmp_path, clamp):
    """Seen from +x, the face lies in the missing strip of the coarsest level.

    With the clamp the strip repeats the level's last voxel, so the surface
    is the box's face at x = 38.5, where the value is 1 (lime).  Without it
    the strip reads the brick's zero padding, the surface moves in to about
    x = 35.5, where the value is 0 (red): the control.
    """
    viewer = _strip_viewer(tmp_path, clamp)
    try:
        frame = render(viewer, SIZE)[0]
        rgb = frame[..., :3].astype(np.int64)
        drawn = rgb.max(axis=-1) > 8
        lime = int((np.argmax(rgb, axis=-1)[drawn] == 1).sum())
        if clamp:
            assert drawn.sum() > 10_000
            assert lime == drawn.sum()
        else:
            assert lime < 0.01 * drawn.sum()
    finally:
        viewer.controller.close()


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


def _drag(tmp_path, iso: float, clamp: bool, round_up: bool) -> tuple[int, int]:
    """``(frames with far pixels, most far pixels in a frame)`` of a yaw drag.

    Frustum culling on and a small mask budget, so a settled view holds
    fine bricks in its frustum and the coarse backstop elsewhere.  Each
    yaw's settled silhouette is kept, dilated; then from the first view the
    camera yaws inside ``camera_interaction`` (nothing reslices) and each
    frame's drawn pixels outside that yaw's silhouette are counted.
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
        material.extent_clamp = clamp
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
        return bad, worst
    finally:
        controller.close()


@pytest.mark.parametrize("iso", [0.0, -4.0])
def test_a_camera_drag_draws_nothing_past_a_coarse_level_end(tmp_path, iso):
    assert _drag(tmp_path, iso, clamp=True, round_up=False) == (0, 0)


@pytest.mark.parametrize("iso", [0.0, -4.0])
def test_without_the_clamp_the_drag_draws_far_structures(tmp_path, iso):
    """The control: the zero padding in the missing strip draws surfaces."""
    bad, worst = _drag(tmp_path, iso, clamp=False, round_up=False)
    assert bad >= 4 and worst > 500, (bad, worst)


@pytest.mark.parametrize("clamp", [True, False])
def test_a_positive_threshold_never_reaches_the_zero_padding(tmp_path, clamp):
    """The second control: zero is below +4, so the strip draws nothing either way."""
    assert _drag(tmp_path, 4.0, clamp=clamp, round_up=False) == (0, 0)


def test_a_pyramid_rounded_up_is_clean_with_the_clamp(tmp_path):
    """Rounded up, each level covers the extent; the clamp keeps it clean.

    Without the clamp this pyramid draws a few far pixels (4 in 2 frames
    here): a rounded-up level's last voxel centre can still stop short of
    the finest extent (level 3 ends at 73.5 of 74.5), and the half voxel
    past it blends in the zero padding.  The design's prototype saw none on
    Position_6.
    """
    assert _drag(tmp_path, -4.0, clamp=True, round_up=True) == (0, 0)


def test_settled_frames_barely_change_with_the_clamp(tmp_path):
    """At rest the clamp changes at most a few pixels (6.4)."""
    frames = []
    for clamp in (True, False):
        mask = pyramid_store(
            tmp_path / str(clamp), "mask", _signed_distance(DRAG_SHAPE), DRAG_LEVELS
        )
        value = pyramid_store(
            tmp_path / str(clamp),
            "value",
            np.full(DRAG_SHAPE, 0.7, np.float32),
            DRAG_LEVELS,
        )
        viewer = new_viewer("3d")
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
            material.extent_clamp = clamp
            load(viewer, size=(DRAG_SIZE, DRAG_SIZE))
            frames.append(render(viewer, DRAG_SIZE)[0][..., :3].astype(np.int64))
        finally:
            viewer.controller.close()
    differing = (np.abs(frames[0] - frames[1]).max(axis=-1) > 8).sum()
    assert differing <= 10, differing
