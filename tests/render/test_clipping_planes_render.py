"""Clipping planes on screen: every visual family, in 3D and in 2D.

A plane assigned to ``visual.clipping_planes`` is compared with the same
visual drawn from a store whose clipped voxels were zeroed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.render import _clipping as h

SIZE = 192
#: Pixels allowed to differ off the edge band, as a fraction of what is drawn.
TOLERANCE = 0.005


async def _scenes(kind, controller, reslice, tmp_path, planes, dim):
    data = h.data_for(kind)
    masked = np.where(h.kept_mask(planes), data, 0).astype(data.dtype)
    out = []
    for tag, volume in (("clip", data), ("ref", masked)):
        scene = controller.add_scene(dim=dim, name=f"{kind}_{tag}")
        visual, store = h.add_visual(
            kind, controller, scene.id, volume, tmp_path, f"{kind}_{tag}"
        )
        controller.add_canvas(scene_id=scene.id)
        out.append((scene, visual, store))
    return out


@pytest.mark.parametrize("plane_name", list(h.PLANES))
@pytest.mark.parametrize("kind", h.KINDS)
async def test_a_volume_is_cut_at_the_plane(
    kind, plane_name, controller, offscreen_renderer, reslice, tmp_path
):
    planes = h.PLANES[plane_name]
    (clip, visual, store), (ref, _, _) = await _scenes(
        kind, controller, reslice, tmp_path, planes, "3d"
    )
    for scene in (clip, ref):
        await reslice(controller, scene.id)
        # Fit again: an in-memory visual has no bounds until it has loaded.
        controller.fit_camera(scene.id)
    visual.clipping_planes = h.clipping_planes(store, planes)
    await reslice(controller, clip.id)
    clip_gfx, camera = h.gfx_scene(controller, clip.id)
    ref_gfx, _ = h.gfx_scene(controller, ref.id)

    is_mip = kind.endswith("mip")
    axis_aligned = plane_name != "oblique"
    for view_name, place in h.camera_views(camera):
        if view_name == "inside" and not (is_mip and axis_aligned):
            # From inside a voxel fills much of the frame, so the reference's
            # voxel staircase and its own shading are what is compared.
            continue
        place()
        clipped = offscreen_renderer(clip_gfx, camera, (SIZE, SIZE)).copy()
        reference = offscreen_renderer(ref_gfx, camera, (SIZE, SIZE)).copy()
        if kind == "image_memory_mip":
            h.hide_colormap_zero(clipped)
            h.hide_colormap_zero(reference)
        result = h.compare(clipped, reference)
        allowed = TOLERANCE * max(result["reference_px"], 1)
        assert result["reference_px"] > 200, (view_name, result)
        assert result["silhouette_off_edge"] <= allowed, (view_name, result)
        # Colour is compared where the reference's colour means the same
        # thing.  A MIP has no shading.  A lit cut face takes the plane's
        # normal (D21); the reference's face is lit by a normal estimated
        # from masked data, which agrees for a face along the voxel grid
        # seen from the fitted view and not otherwise.
        if is_mip:
            assert result["colour_off_edge"] <= allowed, (view_name, result)
        elif axis_aligned and view_name == "fit":
            assert result["colour_off_edge"] <= 4 * allowed, (view_name, result)


@pytest.mark.parametrize("z", [8, 16, 24])
@pytest.mark.parametrize(
    "kind",
    ["image_memory_mip", "image_multiscale_mip", "labels_memory", "labels_multiscale"],
)
async def test_a_slice_is_cut_at_the_clip_line(
    kind, z, controller, offscreen_renderer, reslice, tmp_path
):
    planes = h.PLANES["oblique"]
    (clip, visual, store), (ref, _, _) = await _scenes(
        kind, controller, reslice, tmp_path, planes, "2d"
    )
    visual.clipping_planes = h.clipping_planes(store, planes)
    for scene in (clip, ref):
        controller.update_slice_indices(scene.id, {0: z * h.Z_SCALE})
        await reslice(controller, scene.id)
        controller.fit_camera(scene.id)
        await reslice(controller, scene.id)
    clip_gfx, camera = h.gfx_scene(controller, clip.id)
    ref_gfx, _ = h.gfx_scene(controller, ref.id)
    clipped = offscreen_renderer(clip_gfx, camera, (SIZE, SIZE)).copy()
    reference = offscreen_renderer(ref_gfx, camera, (SIZE, SIZE)).copy()
    if kind.startswith("image"):
        # An image draws its zeros.
        h.hide_colormap_zero(clipped)
        h.hide_colormap_zero(reference)
    result = h.compare(clipped, reference)
    allowed = TOLERANCE * max(result["reference_px"], 1)
    assert result["reference_px"] > 200, result
    assert result["clipped_px"] > 200, result
    assert result["silhouette_off_edge"] <= allowed, result
    assert result["colour_off_edge"] <= allowed, result
