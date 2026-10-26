"""Oblique 2D design v3, Phase 4: in-memory images and labels on the brick route.

``plans/oblique_2d_raster_design_v3.md`` 5.1 (U1), 5.4 (I-iii+, U8, U3) and
D6.  A tilted 2D view of an in-memory visual is drawn by an inner multiscale
visual over a two-level brick pyramid (``InMemoryBrickSource``): the array,
and a coarse level built on a worker at add.

Expectations are analytic (the transform, the camera) or properties of the
data -- never another library's rendering (memory: no cross-library tests).
"""

from __future__ import annotations

import asyncio
import threading
import time
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.convenience.capture import _load_data
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.data._in_memory_bricks import (
    COARSE_FACTOR,
    InMemoryBrickSource,
    downsampled_axes,
)
from cellier.data.image._image_requests import ChunkRequest
from cellier.render.scheduling import ChunkClass, PlanMode
from cellier.render.visuals._image_memory import GFXImageMemoryVisual
from cellier.render.visuals._label_memory import GFXLabelMemoryVisual
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    ImageVisual,
    InMemoryImageRenderConfig,
    InMemoryImageSingleAppearance,
    InMemoryLabelRenderConfig,
    InMemoryLabelsAppearance,
    LabelMemoryVisual,
)
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests.render._affine import (
    LANDMARK_PX,
    LINEARS,
    P0,
    SHAPE,
    SIZE,
    bind,
    canvas_view,
    landmark_error,
    load,
    new_viewer,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render._pick import PickFrame
from tests.render.test_oblique_2d_brick_route import (
    _expected_voxels,
    _random_volume,
    _wrong_pixels,
)
from tests.render.test_oblique_2d_labels import _IDS, _grey_colours, _random_labels

_BUDGET = 64 * 1024**2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add(
    viewer: Viewer,
    family: str,
    matrix: np.ndarray,
    data: np.ndarray | None = None,
    *,
    moving_2d: bool = False,
    render_order: int = 0,
    force_level: int | None = None,
):
    if family == "image":
        data = _random_volume() if data is None else data
        store = ImageMemoryStore(data=data, name="mem")
        return viewer.add_image(
            store,
            appearance=InMemoryImageAppearance(
                interpolation="nearest",
                render_order=render_order,
                coarsest_while_moving_2d=moving_2d,
                force_level=force_level,
            ),
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 255.0), render_mode="mip"
            ),
            render_config=InMemoryImageRenderConfig(
                gpu_budget_bytes=_BUDGET, block_size=16
            ),
            transform=bind(viewer, store, matrix),
        )
    data = _random_labels() if data is None else data
    store = LabelMemoryStore(data=data, name="mem")
    return viewer.add_labels(
        store,
        appearance=InMemoryLabelsAppearance(
            colormap_mode="direct",
            color_dict=_grey_colours(),
            render_order=render_order,
            coarsest_while_moving_2d=moving_2d,
            force_level=force_level,
        ),
        render_config=InMemoryLabelRenderConfig(
            gpu_budget_bytes=_BUDGET, block_size=16
        ),
        transform=bind(viewer, store, matrix),
    )


def _gfx(viewer: Viewer, visual):
    return viewer.controller._render_manager._scenes[viewer.scene.id].get_visual(
        visual.id
    )


def _active_node(viewer: Viewer, visual):
    manager = viewer.controller._render_manager._scenes[viewer.scene.id]
    return manager.get_active_node(visual.id)


def _store_of(viewer: Viewer, visual):
    return viewer.controller._model.data.stores[UUID(visual.data_store_id)]


def _built(viewer: Viewer, visual) -> None:
    """Wait for the coarse level, then load (the build's landing reslices)."""

    async def _go() -> None:
        await _gfx(viewer, visual).brick_source.wait_built()
        await asyncio.sleep(0)
        await _load_data(viewer)

    asyncio.run(_go())


def _slice_at(matrix: np.ndarray, fraction: float) -> float:
    centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
    corner = to_world(matrix, np.zeros(3))
    return float(corner[0] + fraction * 2.0 * (centre[0] - corner[0]))


@pytest.fixture
def plans(monkeypatch):
    """Record ``(visual id, mode, desired sets)`` for every in-memory plan."""
    calls: list = []
    for cls in (GFXImageMemoryVisual, GFXLabelMemoryVisual):
        original = cls.plan

        def spy(self, request, config, mode=PlanMode.FULL, _original=original):
            desired = _original(self, request, config, mode)
            calls.append((self.visual_model_id, mode, desired))
            return desired

        monkeypatch.setattr(cls, "plan", spy)
    return calls


FAMILIES = ["image", "labels"]


# ---------------------------------------------------------------------------
# Parity with the multiscale route (Phases 2-3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_tilted_in_memory_view_is_on_the_brick_route(tmp_path, family):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, world_matrix(LINEARS["rot_zx"]))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        assert gfx_visual.slice_axes == (0, 1, 2)
        assert gfx_visual.chunked_now
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        assert viewer.controller.draw_status(visual.id).drawable
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_an_in_memory_landmark_lands_where_the_transform_puts_it(
    tmp_path, family, name
):
    assert landmark_error(family, name, "2d", tmp_path) < LANDMARK_PX


@pytest.mark.parametrize("name", ["rot_zx", "general"])
@pytest.mark.parametrize("fraction", [0.3, 0.7])
def test_every_in_memory_pixel_reads_its_own_voxel(tmp_path, name, fraction):
    matrix = world_matrix(LINEARS[name])
    data = _random_volume(7)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image", matrix, data)
        slice_z = _slice_at(matrix, fraction)
        load(viewer, slice_z=slice_z)
        _built(viewer, visual)
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z)
        assert np.count_nonzero(valid) > 10_000
        assert _wrong_pixels(frame, data, voxel, valid) == 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("name", ["rot_zx", "general"])
def test_every_in_memory_labels_pixel_shows_its_label(tmp_path, name):
    matrix = world_matrix(LINEARS[name])
    data = _random_labels(8)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "labels", matrix, data)
        slice_z = _slice_at(matrix, 0.5)
        load(viewer, slice_z=slice_z)
        _built(viewer, visual)
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z)
        index = tuple(np.clip(voxel[..., a], 0, SHAPE[a] - 1) for a in range(3))
        grey = frame[..., 0].astype(int)
        assert np.count_nonzero(valid) > 10_000
        assert np.count_nonzero(np.abs(grey - data[index])[valid] > 1) == 0
        drawn = frame[..., :3].max(axis=-1) > 0
        nearest = np.abs(grey[drawn][:, None] - _IDS[None, :]).min(axis=1)
        assert np.all(nearest <= 1)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_the_camera_never_clips_an_in_memory_slice_node(tmp_path, family):
    matrix = world_matrix(rot(0, 2, 30.0))
    data = np.tile(_random_volume(1), (12, 1, 1))[:256]
    if family == "labels":
        data = np.tile(_random_labels(1), (12, 1, 1))[:256]
    viewer = new_viewer("2d")
    try:
        _add(viewer, family, matrix, data)
        far = to_world(matrix, np.array([255.0, 0.0, 0.0]))
        corner = to_world(matrix, np.zeros(3))
        slice_z = float(corner[0] + 0.98 * (far[0] - corner[0]))
        load(viewer, slice_z=slice_z)
        canvas_id = viewer.controller.get_canvas_ids(viewer.scene.id)[0]
        viewer.controller.set_camera_depth_range(canvas_id, (-20.0, 20.0))
        asyncio.run(_load_data(viewer))
        frame, _, camera = render(viewer)
        _voxel, valid = _expected_voxels(camera, matrix, slice_z, data.shape)
        assert np.count_nonzero(valid) > 10_000
        assert np.all(frame[..., :3].max(axis=-1)[valid] > 0)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_in_memory_picks_land_on_the_voxel_under_the_cursor(tmp_path, family):
    matrix = world_matrix(LINEARS["general"])
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, matrix)
        controller = viewer.controller
        manager = controller._render_manager
        rng = np.random.default_rng(11)
        for mark in rng.integers(6, np.asarray(SHAPE) - 6, size=(4, 3)):
            point = to_world(matrix, mark.astype(float))
            load(viewer, slice_z=float(point[0]))
            _, renderer, camera = render(viewer)
            ndc = camera.camera_matrix @ np.r_[point[::-1], 1.0]
            row, col = (1.0 - ndc[1]) / 2.0 * SIZE, (ndc[0] + 1.0) / 2.0 * SIZE
            info = PickFrame(renderer).info((float(col), float(row)))
            assert info["world_object"] is not None
            details = manager._extract_pick_details(
                viewer.scene.id, info["world_object"], info
            )
            coordinate = controller._promote_pick_coordinate(
                details.displayed_data_coord,
                displayed_axes=viewer.scene.dims.selection.displayed_axes,
                slice_indices={0: float(point[0])},
                world_ndim=3,
                hit_visual_id=visual.id,
                collapsed_data_indices=details.collapsed_data_indices,
            )
            assert np.all(np.abs(np.floor(coordinate) - mark) <= 1)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_a_transform_jump_moves_an_in_memory_visual_both_ways(tmp_path, family):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        coordinator = viewer.controller._render_manager._slice_coordinator
        assert not gfx_visual.chunked_now
        assert gfx_visual.brick_inner is None

        async def _assign(matrix) -> None:
            visual.transform = bind(viewer, _store_of(viewer, visual), matrix)
            await _load_data(viewer)

        asyncio.run(_assign(world_matrix(LINEARS["rot_zx"])))
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        assert coordinator._visual_caches.get(visual.id)
        assert render(viewer)[0][..., :3].max() > 0
        asyncio.run(_assign(world_matrix(np.eye(3))))
        assert not gfx_visual.chunked_now
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        # Off the route its atlas is no longer the scheduler's.
        assert not coordinator._visual_caches.get(visual.id)
        assert render(viewer)[0][..., :3].max() > 0
        # And back again: the kept inner visual draws.
        asyncio.run(_assign(world_matrix(LINEARS["rot_zx"])))
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("moving_2d", [True, False])
def test_an_in_memory_transform_drag_follows_the_2d_rule(tmp_path, plans, moving_2d):
    viewer = new_viewer("2d")
    try:
        visual = _add(
            viewer, "image", world_matrix(rot(0, 2, 10.0)), moving_2d=moving_2d
        )
        load(viewer, slice_z=float(P0[0]))
        _built(viewer, visual)
        plans.clear()
        store = _store_of(viewer, visual)

        async def _drag() -> tuple[int, int]:
            with viewer.controller.transform_interaction(visual.id):
                for degrees in (15.0, 20.0):
                    visual.transform = bind(
                        viewer, store, world_matrix(rot(0, 2, degrees))
                    )
                    await asyncio.sleep(0)
                n = len(plans)
            # The release plans synchronously; loading may plan again.
            return n, len(plans)

        n_during, n_released = asyncio.run(_drag())
        modes = [mode for vid, mode, _ in plans if vid == visual.id]
        assert n_during
        if moving_2d:
            assert set(modes[:n_during]) == {PlanMode.BACKSTOP_ONLY}
            assert modes[n_during:n_released] == [PlanMode.FULL]
        else:
            assert set(modes) == {PlanMode.FULL}
            assert n_released == n_during
    finally:
        viewer.controller.close()


def test_an_in_memory_drag_through_aligned_stays_on_the_route(tmp_path):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "labels", world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        store = _store_of(viewer, visual)
        seen: list[bool] = []

        async def _drag() -> None:
            with viewer.controller.transform_interaction(visual.id):
                for degrees in (10.0, 0.0):
                    visual.transform = bind(
                        viewer, store, world_matrix(rot(0, 2, degrees))
                    )
                    seen.append(gfx_visual.chunked_now)
            await _load_data(viewer)
            seen.append(gfx_visual.chunked_now)

        asyncio.run(_drag())
        assert seen == [True, True, False]
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Preparation, the build and target-only planning (5.1, U8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_a_visual_that_never_takes_the_route_allocates_no_atlas(tmp_path, family):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, family, world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        render(viewer)
        assert _gfx(viewer, visual).brick_inner is None
    finally:
        viewer.controller.close()


def test_before_the_build_lands_the_plan_is_target_only(tmp_path, plans, monkeypatch):
    gate = threading.Event()
    original = InMemoryBrickSource._build_coarse

    def _slow_build(self) -> None:
        gate.wait(10.0)
        original(self)

    monkeypatch.setattr(InMemoryBrickSource, "_build_coarse", _slow_build)
    reads: list[int] = []
    original_read = InMemoryBrickSource.get_data

    async def _spy_read(self, request):
        reads.append(request.scale_index)
        return await original_read(self, request)

    monkeypatch.setattr(InMemoryBrickSource, "get_data", _spy_read)
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image", world_matrix(LINEARS["rot_zx"]))
        start = time.perf_counter()
        load(viewer, slice_z=float(P0[0]))
        # No read waited for the build.
        assert time.perf_counter() - start < 5.0
        source = _gfx(viewer, visual).brick_source
        assert not source.built
        desired = [d for vid, _, sets in plans if vid == visual.id for d in sets]
        assert desired
        backstop = np.uint8(ChunkClass.BACKSTOP)
        assert all(not np.any(d.cls == backstop) for d in desired)
        assert reads and set(reads) == {0}
        assert render(viewer)[0][..., :3].max() > 0
        plans.clear()
        gate.set()
        _built(viewer, visual)
        assert source.built
        after = [d for vid, _, sets in plans if vid == visual.id for d in sets]
        assert after and any(np.any(d.cls == backstop) for d in after)
    finally:
        gate.set()
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Camera reslice per view (D6)
# ---------------------------------------------------------------------------


def test_a_pan_replans_a_tilted_in_memory_view_and_not_an_aligned_one(tmp_path, plans):
    viewer = new_viewer("2d")
    try:
        tilted = _add(viewer, "image", world_matrix(LINEARS["rot_zx"]))
        aligned = _add(viewer, "labels", world_matrix(np.eye(3)))
        load(viewer, slice_z=float(P0[0]))
        controller = viewer.controller
        assert controller._camera_sensitive(tilted)
        assert not controller._camera_sensitive(aligned)
        plans.clear()

        async def _pan() -> None:
            camera = canvas_view(viewer).camera
            x, y, z = camera.local.position
            camera.local.position = (x + 5.0, y, z)
            controller._run_camera_reslice(viewer.scene.id)
            await _load_data(viewer)

        asyncio.run(_pan())
        planned = {vid for vid, _, _ in plans}
        assert tilted.id in planned
        assert aligned.id not in planned
    finally:
        viewer.controller.close()


def test_a_scrub_ending_while_the_camera_moves_holds_back_a_tilted_target(
    tmp_path, monkeypatch
):
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image", world_matrix(LINEARS["rot_zx"]), moving_2d=True)
        load(viewer, slice_z=float(P0[0]))
        controller = viewer.controller
        monkeypatch.setattr(controller, "_scene_camera_moving", lambda scene_id: True)

        async def _scrub() -> None:
            with controller.dims_interaction(viewer.scene.id):
                controller.update_slice_indices(viewer.scene.id, {0: float(P0[0]) + 1})
                await asyncio.sleep(0)

        asyncio.run(_scrub())
        assert visual.id in controller._camera_handoff.get(viewer.scene.id, set())
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Contents changes (U3)
# ---------------------------------------------------------------------------


def test_a_contents_change_reaches_the_tilted_view_at_both_levels(
    tmp_path, monkeypatch
):
    viewer = new_viewer("2d")
    try:
        data = np.tile(_random_volume(9), (2, 2, 2))[:40, :64, :80].copy()
        matrix = world_matrix(rot(0, 2, 25.0))
        visual = _add(viewer, "image", matrix, data)
        centre = np.array([20.0, 32.0, 40.0])
        slice_z = float(to_world(matrix, centre)[0])
        load(viewer, slice_z=slice_z)
        _built(viewer, visual)
        gfx_visual = _gfx(viewer, visual)
        source = gfx_visual.brick_source
        store = _store_of(viewer, visual)
        reads: list[tuple[int, tuple]] = []
        original_read = InMemoryBrickSource.get_data

        async def _spy_read(self, request):
            reads.append((request.scale_index, request.axis_selections))
            return await original_read(self, request)

        monkeypatch.setattr(InMemoryBrickSource, "get_data", _spy_read)
        # A paint-stroke-sized change: a 20-voxel ball set to 1.
        z, y, x = np.ogrid[: data.shape[0], : data.shape[1], : data.shape[2]]
        ball = (z - 20) ** 2 + (y - 32) ** 2 + (x - 40) ** 2 <= 10**2
        region = tuple(
            (float(lo), float(hi)) for lo, hi in ((10, 31), (22, 43), (30, 51))
        )
        before = source._coarse.copy()

        async def _paint() -> float:
            start = time.perf_counter()
            store.data[ball] = 1.0
            store.notify_changed("contents", regions=[region])
            await _load_data(viewer)
            return time.perf_counter() - start

        elapsed = asyncio.run(_paint())
        # Only the bricks the region touches were read again, at both levels.
        assert reads
        for level, selections in reads:
            scale = COARSE_FACTOR if level == 1 else 1
            for (lo, hi), sel in zip(region, selections, strict=True):
                if isinstance(sel, tuple):
                    assert sel[0] * scale <= hi and sel[1] * scale >= lo
        assert {level for level, _ in reads} <= {0, 1}
        # The coarse blocks the ball covers were recomputed.
        assert not np.array_equal(before, source._coarse)
        np.testing.assert_allclose(source._coarse[5, 8, 10], 1.0)
        # The next tilted frame shows the new values where the ball is.
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z, data.shape)
        assert _wrong_pixels(frame, data, voxel, valid) == 0
        print(
            f"contents change to tilted frame: {elapsed * 1000:.1f} ms, "
            f"coarse recompute {source.recompute_seconds * 1000:.2f} ms"
        )
    finally:
        viewer.controller.close()


def test_a_stale_coarse_block_is_never_served(tmp_path):
    """A coarse read overlapping a block being recomputed waits for it."""
    data = np.zeros((16, 16, 16), np.float32)
    store = ImageMemoryStore(data=data, name="s")
    from cellier.data._axes import data_axes_from_world, level_coordinate_systems
    from cellier.scene.dims import world_coordinate_system

    world = world_coordinate_system(spatial_axes("z", "y", "x"))
    store.data_coordinate_systems = level_coordinate_systems(
        store.id, data_axes_from_world(world, 3, {}), 1, "s"
    )
    source = InMemoryBrickSource(store, labels=False, downsample=(True, True, True))

    async def _go() -> np.ndarray:
        await source.wait_built()
        gate = threading.Event()
        original = source._recompute

        def _slow(box):
            gate.wait(10.0)
            original(box)

        source._recompute = _slow
        store.data[:] = 8.0
        source.contents_changed([((0.0, 16.0),) * 3])
        request = ChunkRequest(uuid4(), uuid4(), 1, ((0, 4), (0, 4), (0, 4)))
        read = asyncio.ensure_future(source.get_data(request))
        await asyncio.sleep(0.05)
        assert not read.done()
        gate.set()
        return await read

    out = asyncio.run(_go())
    np.testing.assert_allclose(out, 8.0)


# ---------------------------------------------------------------------------
# The coarse level (D9)
# ---------------------------------------------------------------------------


def _source(data, *, labels, downsample=None):
    store = (LabelMemoryStore if labels else ImageMemoryStore)(data=data, name="s")
    from cellier.data._axes import data_axes_from_world, level_coordinate_systems
    from cellier.scene.dims import world_coordinate_system

    world = world_coordinate_system(spatial_axes("z", "y", "x"))
    store.data_coordinate_systems = level_coordinate_systems(
        store.id, data_axes_from_world(world, data.ndim, {}), 1, "s"
    )
    source = InMemoryBrickSource(
        store,
        labels=labels,
        downsample=(True,) * data.ndim if downsample is None else downsample,
    )
    asyncio.run(source.wait_built())
    return source


def test_the_labels_coarse_level_holds_only_ids_in_the_data():
    data = _random_labels(12, shape=(23, 30, 41))
    source = _source(data, labels=True)
    assert set(np.unique(source._coarse)) <= set(np.unique(data))
    assert source._coarse.shape == tuple(-(-n // 4) for n in data.shape)


@pytest.mark.parametrize("labels", [False, True])
def test_a_landmark_lands_at_the_same_place_on_both_levels(labels):
    """The level transform states the reduction exactly, edges included."""
    shape = (23, 30, 41)  # no length a multiple of 4
    transform_offset = 1.0 if labels else 1.5
    for point in ((1, 5, 9), (21, 29, 40), (12, 17, 37)):
        data = np.zeros(shape, np.int32 if labels else np.float32)
        z, y, x = point
        if labels:
            data[z, y, x] = 7
        else:
            data[
                max(0, z - 1) : z + 2, max(0, y - 1) : y + 2, max(0, x - 1) : x + 2
            ] = 1
        source = _source(data, labels=labels)
        coarse = source._coarse
        if labels:
            hits = np.argwhere(coarse == 7)
            if not len(hits):
                continue  # a stride may drop a single voxel; that is the rule
            world = hits * 4 + transform_offset
            assert np.all(np.abs(world - np.asarray(point)).max(axis=1) < 4 / 2 + 1)
        else:
            weights = coarse / coarse.sum()
            centre = (np.argwhere(coarse > 0) * 4 + transform_offset).T @ weights[
                coarse > 0
            ]
            assert np.all(np.abs(centre - np.asarray(point)) < 4 / 2)
    # And the transform itself says so.
    matrix = np.asarray(source.level_transforms[1].matrix)
    np.testing.assert_allclose(np.diag(matrix)[:3], 4.0)
    np.testing.assert_allclose(matrix[:3, 3], transform_offset)


def test_the_block_mean_averages_only_voxels_that_exist():
    data = np.arange(5, dtype=np.float32).reshape(5, 1, 1) * np.ones((5, 1, 1))
    source = _source(data, labels=False, downsample=(True, False, False))
    np.testing.assert_allclose(source._coarse[:, 0, 0], [1.5, 4.0])


def test_time_and_discrete_axes_are_not_downsampled(tmp_path):
    from tests.render._affine import new_tzyx_viewer

    viewer = new_tzyx_viewer("2d")
    try:
        store = ImageMemoryStore(data=np.zeros((4, 8, 8, 8), np.float32), name="t")
        visual = viewer.add_image(store, transform=bind(viewer, store, np.eye(5)))
        world = viewer.scene.dims.world_coordinate_system
        flags = downsampled_axes(
            visual.transform, world, store.data_coordinate_systems[0]
        )
        assert flags == (False, True, True, True)
        assert _gfx(viewer, visual).brick_source.level_shapes[1] == (4, 2, 2, 2)
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The orthoviewer (0b item 7) and the model (U1)
# ---------------------------------------------------------------------------


def test_the_orthoviewer_on_a_tilted_in_memory_volume(tmp_path):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        store = LabelMemoryStore(data=_random_labels(13), name="labels")
        visuals = ortho.add_labels(
            store,
            appearance=InMemoryLabelsAppearance(),
            render_config=InMemoryLabelRenderConfig(
                gpu_budget_bytes=_BUDGET, block_size=16
            ),
        )
        scene = next(iter(ortho.scenes.values()))
        ortho.controller._ensure_data_coordinate_systems(scene.id, store)
        from cellier.transform import AffineTransform

        transform = AffineTransform.from_matrix(
            world_matrix(LINEARS["general"]),
            store.data_coordinate_systems[0],
            scene.dims.world_coordinate_system,
        )
        for visual in visuals.values():
            visual.transform = transform
        controller = ortho.controller
        drawn = {
            panel: controller.draw_status(visual.id).drawable
            for panel, visual in visuals.items()
        }
        assert drawn == {"xy": True, "xz": True, "yz": True, "vol": True}
        gfx = {
            panel: controller._render_manager._scenes[
                controller._visual_to_scene[visual.id]
            ].get_visual(visual.id)
            for panel, visual in visuals.items()
        }
        on_route = [panel for panel, g in gfx.items() if g.chunked_now]
        assert sorted(on_route) == ["xy", "xz", "yz"]
        # Each panel's copy has its own source and atlas.
        sources = {id(g.brick_source) for g in gfx.values()}
        assert len(sources) == 4
        atlas = sum(
            g.brick_inner._block_cache_3d.cache_tex.nbytes
            for g in gfx.values()
            if g.brick_inner is not None
        )
        coarse = sum(
            g.brick_source._coarse.nbytes
            for g in gfx.values()
            if g.brick_source is not None and g.brick_source._coarse is not None
        )
        print(f"orthoviewer: atlases {atlas / 2**20:.1f} MiB, coarse {coarse} B")
        inners = [g.brick_inner for g in gfx.values() if g.brick_inner is not None]
    finally:
        ortho.controller.close()
    # Close frees every atlas and coarse level.
    assert all(inner._block_cache_3d is None for inner in inners)
    assert all(g.brick_source is None for g in gfx.values())


@pytest.mark.parametrize("cls", [ImageVisual, LabelMemoryVisual])
def test_the_new_model_fields_round_trip(cls):
    kwargs = {"name": "v", "data_store_id": "store"}
    if cls is LabelMemoryVisual:
        kwargs["appearance"] = InMemoryLabelsAppearance(
            coarsest_while_moving_2d=True, settled_lod_bias=2.0, force_level=2
        )
        kwargs["render_config"] = InMemoryLabelRenderConfig(block_size=16)
    else:
        kwargs["appearance"] = InMemoryImageAppearance(
            coarsest_while_moving_2d=True, settled_lod_bias=2.0, force_level=2
        )
        kwargs["render_config"] = InMemoryImageRenderConfig(block_size=16)
    visual = cls(**kwargs)
    again = cls.model_validate_json(visual.model_dump_json())
    assert again.appearance.coarsest_while_moving_2d
    assert again.appearance.settled_lod_bias == 2.0
    assert again.appearance.force_level == 2
    assert again.render_config.block_size == 16
    assert again.plans_coarse_while_moving(2)
    assert not again.plans_coarse_while_moving(3)
    # A model saved before the fields existed still loads, with the defaults.
    old = visual.model_dump(mode="json")
    old.pop("render_config")
    for field in ("coarsest_while_moving_2d", "settled_lod_bias", "force_level"):
        old["appearance"].pop(field)
    legacy = cls.model_validate(old)
    assert legacy.render_config == type(kwargs["render_config"])()
    assert legacy.appearance.force_level is None


def test_an_extent_change_rebuilds_the_brick_pyramid(tmp_path):
    """5.4.4: a new shape rebuilds the source, and the tilted view draws it."""
    matrix = world_matrix(LINEARS["rot_zx"])
    viewer = new_viewer("2d")
    try:
        visual = _add(viewer, "image", matrix)
        slice_z = _slice_at(matrix, 0.5)
        load(viewer, slice_z=slice_z)
        gfx_visual = _gfx(viewer, visual)
        old_source = gfx_visual.brick_source
        store = _store_of(viewer, visual)
        data = _random_volume(14)
        data = np.concatenate([data, data[:, :, :8]], axis=2)  # wider in x

        async def _grow() -> None:
            store.data = data
            await _load_data(viewer)
            await gfx_visual.brick_source.wait_built()
            await _load_data(viewer)

        asyncio.run(_grow())
        assert gfx_visual.brick_source is not old_source
        assert gfx_visual.brick_source.level_shapes[0] == data.shape
        assert _active_node(viewer, visual) is gfx_visual.slice_node
        frame, _, camera = render(viewer)
        voxel, valid = _expected_voxels(camera, matrix, slice_z, data.shape)
        assert np.count_nonzero(valid) > 10_000
        assert _wrong_pixels(frame, data, voxel, valid) == 0
    finally:
        viewer.controller.close()
