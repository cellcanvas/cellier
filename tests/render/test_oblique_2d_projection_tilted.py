"""Oblique 2D design v3, Phase 7: slab projection in ``TILTED`` views.

``plans/oblique_2d_raster_design_v3.md`` 4.2.  A ray along the projection
axis that crosses several data axes (``SLICE_SHIFT`` and ``OBLIQUE`` views)
is sampled one drawn-level voxel apart (Euclidean, K6), anchored on the
slice plane, over the half-open window ``[s - h, s + h)``; with nearest
sampling a sample counts when the centre of the voxel it reads is inside.

The reference here is that rule evaluated on the CPU from the transform and
the data -- the design's own definition, not another renderer (memory: no
cross-library tests).
"""

from __future__ import annotations

import asyncio
import warnings

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.render import RenderManagerConfig
from cellier.render._config import SchedulerConfig
from cellier.render.scheduling import PlanMode
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.scene.dims import spatial_axes
from cellier.transform import (
    DepthAlignment,
    classify_projection,
    ray_step,
    slice_footprint,
)
from tests.render._affine import (
    LINEARS,
    SHAPE,
    SIZE,
    bind,
    load,
    new_viewer,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render._pick import PickFrame
from tests.render.test_oblique_2d_brick_route import _pixel_world, _random_volume
from tests.render.test_oblique_2d_projection import FAMILIES, _add, _gfx

_EPS = 1e-3


def _tilted_reference(camera, matrix, data, slice_z, half, op, steps=1.0):
    """4.2's rule on the CPU: ``(expected, valid)`` per pixel.

    Rays along world z through each pixel centre; samples one step apart
    (``ray_step``), anchored on the slice; nearest voxels; a sample counts
    when it is in the data box and its voxel's centre depth is in
    ``[-h, h)``.  Pixels with any sample within 0.02 voxel of a rounding
    tie, or a centre within ``_EPS`` of a window face, are not valid.
    """
    linear, translation = matrix[:3, :3], matrix[:3, 3]
    inverse = np.linalg.inv(linear)
    direction = inverse @ np.array([1.0, 0.0, 0.0])
    dt = ray_step(direction, steps_per_voxel=steps)
    xy = _pixel_world(camera)
    shape = np.asarray(data.shape)
    n = int(np.ceil(half / dt)) + 2
    values, tie = [], np.zeros(xy.shape[:2], bool)
    counted = np.zeros(xy.shape[:2], bool)
    for j in range(-n, n + 1):
        z = slice_z + j * dt
        world = np.stack([np.full(xy.shape[:2], z), xy[..., 1], xy[..., 0]], -1)
        p = (world - translation) @ inverse.T
        inside = np.all((p >= -0.5) & (p <= shape - 0.5), axis=-1)
        voxel = np.floor(p + 0.5)
        tie |= inside & np.any(np.abs(p - np.floor(p) - 0.5) < 0.02, axis=-1)
        centre_depth = (voxel @ linear.T + translation)[..., 0] - slice_z
        tie |= inside & (
            (np.abs(centre_depth + half) < _EPS) | (np.abs(centre_depth - half) < _EPS)
        )
        keep = inside & (centre_depth >= -half) & (centre_depth < half)
        index = tuple(
            np.clip(voxel[..., a].astype(int), 0, shape[a] - 1) for a in range(3)
        )
        v = np.where(keep, data[index], np.nan)
        values.append(v)
        counted |= keep
    stack = np.stack(values)
    with warnings.catch_warnings():
        # A ray with no counted sample: NaN, and not valid below.
        warnings.simplefilter("ignore", RuntimeWarning)
        if op == "mip":
            expected = np.nanmax(stack, axis=0)
        elif op == "minip":
            expected = np.nanmin(stack, axis=0)
        else:
            expected = np.nanmean(stack, axis=0)
    return expected, counted & ~tie


# ---------------------------------------------------------------------------
# The rule against its CPU evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("op", ["mip", "minip", "mean"])
@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_a_tilted_projection_follows_the_tilted_ray_rule(tmp_path, family, op, name):
    matrix = world_matrix(LINEARS[name])
    data = _random_volume(50)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op=op, matrix=matrix)
        centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
        slice_z, half = float(centre[0]), 4.0
        load(viewer, slice_z=slice_z, half_thickness=half)
        controller = viewer.controller
        displayed = viewer.scene.dims.selection.displayed_axes
        geometry = classify_projection(visual.transform, displayed, 0)
        assert geometry.depth_alignment is DepthAlignment.TILTED
        assert _gfx(viewer, visual).slab.along_axis is False
        frame, _, camera = render(viewer)
        expected, valid = _tilted_reference(camera, matrix, data, slice_z, half, op)
        assert np.count_nonzero(valid) > 10_000
        got = frame[..., 0].astype(float)
        wrong = np.abs(got - expected)[valid] > 1.0
        # The rule exactly, but for a few rays whose float32 sample lands on
        # the other side of a voxel edge than the float64 reference's.
        assert np.count_nonzero(wrong) <= 0.002 * np.count_nonzero(valid)
        assert controller.draw_status(visual.id).drawable
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_bright_landmark_in_the_slab_wins_at_its_projected_pixel(tmp_path, family):
    matrix = world_matrix(LINEARS["rot_zx"])
    data = np.full(SHAPE, 16.0, np.float32)
    mark = np.array([10, 15, 21])
    data[tuple(mark)] = 250.0
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, tmp_path, data, op="mip", matrix=matrix)
        point = to_world(matrix, mark.astype(float))
        load(viewer, slice_z=float(point[0]) - 2.0, half_thickness=3.0)
        frame, _, camera = render(viewer)
        ndc = camera.camera_matrix @ np.r_[point[::-1], 1.0]
        row = int((1.0 - ndc[1]) / 2.0 * SIZE)
        col = int((ndc[0] + 1.0) / 2.0 * SIZE)
        assert frame[row, col, 0] >= 249
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tilted_mip_pick_returns_the_winning_sample(tmp_path, family):
    """D10's pick test on ``rot_zx`` (O12)."""
    matrix = world_matrix(LINEARS["rot_zx"])
    data = np.full(SHAPE, 16.0, np.float32)
    mark = np.array([12, 16, 19])
    data[tuple(mark)] = 250.0
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, tmp_path, data, op="mip", matrix=matrix)
        point = to_world(matrix, mark.astype(float))
        slice_z = float(point[0]) + 2.5
        load(viewer, slice_z=slice_z, half_thickness=4.0)
        _, renderer, camera = render(viewer)
        ndc = camera.camera_matrix @ np.r_[point[::-1], 1.0]
        row, col = (1.0 - ndc[1]) / 2.0 * SIZE, (ndc[0] + 1.0) / 2.0 * SIZE
        info = PickFrame(renderer).info((float(col), float(row)))
        manager = viewer.controller._render_manager
        details = manager._extract_pick_details(
            viewer.scene.id, info["world_object"], info
        )
        coordinate = viewer.controller._promote_pick_coordinate(
            details.displayed_data_coord,
            displayed_axes=viewer.scene.dims.selection.displayed_axes,
            slice_indices={0: slice_z},
            world_ndim=3,
            hit_visual_id=visual.id,
            collapsed_data_indices=details.collapsed_data_indices,
        )
        # The winning sample reads the landmark's voxel: its index, within
        # the half-voxel a sample may sit off the voxel's centre.
        assert np.all(np.abs(np.asarray(coordinate) - (mark + 0.5)) <= 0.75)
    finally:
        viewer.controller.close()


def test_a_shear_lands_each_depth_at_its_shifted_x(tmp_path):
    """``SLICE_SHIFT``: world x reads data x + 0.5 z, so a voxel at depth k
    shows ``0.5 k`` further along x."""
    matrix = world_matrix(LINEARS["shear_x_by_z"])
    data = np.full(SHAPE, 16.0, np.float32)
    marks = [np.array([z, 16, 18]) for z in (8, 10, 12, 14)]
    for mark in marks:
        data[tuple(mark)] = 250.0
    viewer = new_viewer("2d")
    try:
        visual = _add(
            viewer, "image_multiscale", tmp_path, data, op="mip", matrix=matrix
        )
        assert viewer.controller.view_classification(visual.id).alignment.value == (
            "slice_shift"
        )
        centre = to_world(matrix, np.array([11.0, 16.0, 18.0]))
        load(viewer, slice_z=float(centre[0]), half_thickness=5.0)
        assert _gfx(viewer, visual).slice_axes is not None
        frame, _, camera = render(viewer)
        for mark in marks:
            point = to_world(matrix, mark.astype(float))
            ndc = camera.camera_matrix @ np.r_[point[::-1], 1.0]
            row = int((1.0 - ndc[1]) / 2.0 * SIZE)
            col = int((ndc[0] + 1.0) / 2.0 * SIZE)
            assert frame[row, col, 0] >= 249, mark
    finally:
        viewer.controller.close()


def test_steps_per_voxel_leaves_an_along_axis_reduction_exact(tmp_path):
    """Two samples per voxel on a ray along z read every voxel twice: the
    MIP is the same voxel-centre reduction."""
    data = _random_volume(51)
    viewer = new_viewer("2d")
    try:
        _add(
            viewer,
            "image_multiscale",
            tmp_path,
            data,
            op="mip",
            projection_steps_per_voxel=2.0,
        )
        load(viewer, slice_z=10.0, half_thickness=3.0)
        frame, _, camera = render(viewer)
        expected, valid = _tilted_reference(
            camera, world_matrix(np.eye(3)), data, 10.0, 3.0, "mip", steps=2.0
        )
        np.testing.assert_array_equal(
            expected[valid],
            data[7:13].max(0)[
                tuple(
                    np.clip(
                        np.floor(_pixel_world(camera)[..., axis] + 0.5).astype(int),
                        0,
                        n - 1,
                    )
                    for axis, n in ((1, SHAPE[1]), (0, SHAPE[2]))
                )
            ][valid],
        )
        got = frame[..., 0].astype(float)
        assert np.count_nonzero(np.abs(got - expected)[valid] > 1.0) == 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# A thickness drag (D13): reuse, the 2D rule, coverage
# ---------------------------------------------------------------------------


def test_a_thickness_drag_replans_without_rereading_resident_bricks(
    tmp_path, monkeypatch
):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

    reads: list[int] = []
    original = MultiscaleZarrDataStore.get_data

    async def spy(self, request):
        reads.append(request.scale_index)
        return await original(self, request)

    monkeypatch.setattr(MultiscaleZarrDataStore, "get_data", spy)
    data = np.tile(_random_volume(52), (3, 4, 4))[:64, :128, :160]
    matrix = world_matrix(rot(0, 2, 20.0))
    viewer = new_viewer("2d")
    try:
        _add(viewer, "image_multiscale", tmp_path, data, op="mip", matrix=matrix)
        load(viewer, slice_z=32.0, half_thickness=6.0)
        first = reads.count(0)
        assert first > 8
        reads.clear()
        load(viewer, slice_z=32.0, half_thickness=7.0)
        # One more voxel of slab reaches few new bricks; nothing resident is
        # read again.
        assert reads.count(0) < first // 2
    finally:
        viewer.controller.close()


def _slab_coverage(viewer, visual, frame, camera, data_shape) -> float:
    from tests.render.test_oblique_2d_motion import _coverage

    controller = viewer.controller
    classification = controller.view_classification(visual.id)
    rendered = controller._scene_rendered(viewer.scene.id)
    selection = viewer.scene.dims.to_selection(*rendered).with_planes_at_slice(
        [axis for axis in classification.sliced_world_axes if axis != 0]
    )
    polygon = slice_footprint(
        data_shape, classification, selection, projection_world_axis=0
    )[::-1, ::-1]
    return _coverage(frame, camera, polygon)


def test_a_thickness_drag_follows_the_2d_rule_with_full_coverage(tmp_path, monkeypatch):
    """With the flag on, each tick plans the backstop only; every tick still
    draws at least 95% of the slab's footprint (U11, 8.4 at test scale)."""
    from tests.render.test_oblique_2d_motion import _pyramid, _settle

    modes: list[PlanMode] = []
    original = GFXMultiscaleImageVisual.plan

    def spy(self, request, config, mode=PlanMode.FULL):
        modes.append(mode)
        return original(self, request, config, mode)

    monkeypatch.setattr(GFXMultiscaleImageVisual, "plan", spy)
    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=("y", "x"),
        render_config=RenderManagerConfig(scheduler=SchedulerConfig(dims_settle_s=60)),
        gui="offscreen",
    )
    from cellier.scene._background import BackgroundAppearance

    viewer.background = BackgroundAppearance(mode="uniform", color=(0, 0, 0, 1))
    try:
        data = np.tile(_random_volume(53), (2, 4, 4))[:40, :128, :160] + 16.0
        store = _pyramid(tmp_path, "p", data.astype(np.float32), [(1, 2, 2)])
        from cellier.visuals import (
            MultiscaleImageAppearance,
            MultiscaleImageRenderConfig,
            MultiscaleImageSingleAppearance,
        )
        from tests._gpu_budget import SMALL_BUDGETS

        visual = viewer.add_image_multiscale(
            store,
            appearance=MultiscaleImageAppearance(
                coarsest_while_moving_2d=True,
                slab_projection="mip",
                projection_axes=("z",),
            ),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
            single=MultiscaleImageSingleAppearance(
                color_map="gray", clim=(0.0, 255.0), render_mode="mip"
            ),
            transform=bind(viewer, store, world_matrix(rot(0, 2, 25.0))),
        )
        load(viewer, slice_z=20.0, half_thickness=2.0)
        controller = viewer.controller
        coverages = []
        modes.clear()

        async def _drag() -> int:
            for half in (3.0, 4.0, 5.0):
                controller.update_thickness(
                    viewer.scene.id, {0: half}, interactive=True
                )
                await _settle(viewer)
                frame, _, camera = render(viewer)
                coverages.append(
                    _slab_coverage(viewer, visual, frame, camera, data.shape)
                )
            return len(modes)

        n_during = asyncio.run(_drag())
        assert set(modes[:n_during]) == {PlanMode.BACKSTOP_ONLY}
        assert min(coverages) >= 0.95, coverages
    finally:
        viewer.controller.close()
