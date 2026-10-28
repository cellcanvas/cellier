"""Oblique 2D design v3, Phase 3: motion on the brick route.

``plans/oblique_2d_raster_design_v3.md`` 5.1 (preparation), 5.2.3 (sticky
routes, no handoff), 5.3.7 (the 2D rule) and D7 (``transform_interaction``).
A tilted 2D view of a multiscale visual follows the 2D motion rule on every
drag -- a dims scrub, or a transform drag inside ``transform_interaction``
-- never the 3D plane-drag hold.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.convenience.capture import _load_data
from cellier.data import MultiscaleZarrDataStore
from cellier.events import TransformInteractionEvent
from cellier.render import RenderManagerConfig
from cellier.render._config import SchedulerConfig
from cellier.render.scheduling import PlanMode
from cellier.render.visuals._image import GFXMultiscaleImageVisual
from cellier.render.visuals._label_multiscale import GFXMultiscaleLabelVisual
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform, slice_footprint
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._affine import (
    LINEARS,
    P0,
    SHAPE,
    bind,
    load,
    multiscale_store,
    render,
    rot,
    to_world,
    world_matrix,
)
from tests.render.conftest import _write_multiscale_zarr
from tests.render.test_oblique_2d_brick_route import _pixel_world, _random_volume

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _viewer(settle_s: float = 60.0, displayed=("y", "x")) -> Viewer:
    """A 2D viewer whose drags never end on stillness inside a test."""
    from cellier.scene._background import BackgroundAppearance

    viewer = Viewer(
        spatial_axes("z", "y", "x"),
        displayed_axes=displayed,
        render_config=RenderManagerConfig(
            scheduler=SchedulerConfig(dims_settle_s=settle_s)
        ),
        gui="offscreen",
    )
    viewer.background = BackgroundAppearance(mode="uniform", color=(0, 0, 0, 1))
    return viewer


def _add_image(
    viewer: Viewer,
    store,
    matrix: np.ndarray,
    *,
    moving_2d: bool = True,
    force_level: int | None = 0,
):
    return viewer.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(
            force_level=force_level,
            interpolation="nearest",
            coarsest_while_moving_2d=moving_2d,
        ),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        single=MultiscaleImageSingleAppearance(
            color_map="gray", clim=(0.0, 255.0), render_mode="mip"
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


async def _settle(viewer: Viewer) -> None:
    """Load to quiescence without waiting on an open drag's stillness timer."""
    controller = viewer.controller
    manager = controller._render_manager
    for _ in range(20):
        await asyncio.sleep(0)
        tasks = list(manager._slicer._tasks.values())
        if tasks:
            await asyncio.gather(*tasks)
        await manager.scheduler.drain(commit=True)
        if all(task.done() for task in manager._slicer._tasks.values()):
            return


@pytest.fixture
def plan_modes(monkeypatch):
    """Record ``(visual id, plan mode, node axes)`` for every multiscale plan."""
    calls: list[tuple[UUID, PlanMode, tuple]] = []
    for cls in (GFXMultiscaleImageVisual, GFXMultiscaleLabelVisual):
        original = cls.plan

        def spy(self, request, config, mode=PlanMode.FULL, _original=original):
            displayed = tuple(request.dims_state.selection.displayed_axes)
            calls.append((self.visual_model_id, mode, self._node_axes(displayed)))
            return _original(self, request, config, mode)

        monkeypatch.setattr(cls, "plan", spy)
    return calls


def _modes(calls, visual) -> list[PlanMode]:
    return [mode for vid, mode, _ in calls if vid == visual.id]


def _rotated(viewer: Viewer, visual, degrees: float) -> AffineTransform:
    return bind(viewer, _store_of(viewer, visual), world_matrix(rot(0, 2, degrees)))


# ---------------------------------------------------------------------------
# transform_interaction (D7)
# ---------------------------------------------------------------------------


def test_a_transform_drag_is_announced_once_at_each_end(tmp_path):
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(np.eye(3)),
        )
        load(viewer, slice_z=float(P0[0]))
        events: list = []
        viewer.controller._outgoing_events.subscribe(
            TransformInteractionEvent, events.append, owner_id=uuid4()
        )
        controller = viewer.controller

        async def _drag() -> None:
            with controller.transform_interaction(visual.id):
                assert controller.transform_interaction_state(visual.id) == "idle"
                for degrees in (5.0, 10.0, 15.0):
                    visual.transform = _rotated(viewer, visual, degrees)
                    await _settle(viewer)
                assert controller.transform_interaction_state(visual.id) == "active"
            await _settle(viewer)

        asyncio.run(_drag())
        assert [(e.phase, e.reason) for e in events] == [
            ("start", None),
            ("end", "release"),
        ]
        assert all(e.visual_id == visual.id for e in events)
        # Outside a scope a change is a jump: no drag starts.
        events.clear()

        async def _jump() -> None:
            visual.transform = _rotated(viewer, visual, 20.0)
            await _settle(viewer)

        asyncio.run(_jump())
        assert [e.phase for e in events if e.phase == "start"] == []
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("moving_2d", [True, False])
def test_a_transform_drag_follows_the_2d_rule(tmp_path, plan_modes, moving_2d):
    """5.3.7: with the 2D flag on, each tick plans the backstop only and the
    release plans the target once; with it off, every tick plans in full."""
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(rot(0, 2, 10.0)),
            moving_2d=moving_2d,
        )
        load(viewer, slice_z=float(P0[0]))
        plan_modes.clear()

        async def _drag() -> None:
            with viewer.controller.transform_interaction(visual.id):
                for degrees in (15.0, 20.0, 25.0):
                    visual.transform = _rotated(viewer, visual, degrees)
                    await _settle(viewer)
                during = list(_modes(plan_modes, visual))
            await _settle(viewer)
            return during

        during = asyncio.run(_drag())
        after = _modes(plan_modes, visual)[len(during) :]
        assert during
        if moving_2d:
            assert set(during) == {PlanMode.BACKSTOP_ONLY}
            assert after == [PlanMode.FULL]
        else:
            assert set(during) == {PlanMode.FULL}
            assert after == []
        assert all(axes == (0, 1, 2) for _, _, axes in plan_modes)
    finally:
        viewer.controller.close()


def test_a_dims_scrub_of_a_tilted_view_follows_the_2d_rule(tmp_path, plan_modes):
    viewer = _viewer()
    try:
        matrix = world_matrix(LINEARS["rot_zx"])
        visual = _add_image(
            viewer, multiscale_store(tmp_path, "s", _random_volume()), matrix
        )
        centre = to_world(matrix, (np.asarray(SHAPE, float) - 1) / 2)
        load(viewer, slice_z=float(centre[0]))
        plan_modes.clear()
        controller = viewer.controller

        async def _scrub() -> int:
            with controller.dims_interaction(viewer.scene.id):
                for offset in (1.0, 2.0, 3.0):
                    controller.update_slice_indices(
                        viewer.scene.id, {0: float(centre[0] + offset)}
                    )
                    await _settle(viewer)
                n = len(_modes(plan_modes, visual))
            await _settle(viewer)
            return n

        n_during = asyncio.run(_scrub())
        modes = _modes(plan_modes, visual)
        assert set(modes[:n_during]) == {PlanMode.BACKSTOP_ONLY}
        assert modes[n_during:] == [PlanMode.FULL]
    finally:
        viewer.controller.close()


def test_the_plane_drag_hold_never_applies_in_a_2d_scene(tmp_path):
    """K11: a 2D scene never holds a multiscale plan for a plane drag."""
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(LINEARS["rot_zx"]),
        )
        load(viewer, slice_z=float(P0[0]))
        assert visual.appearance.coarsest_while_moving_3d
        assert not viewer.controller._plane_drag_holds_plan(visual)
    finally:
        viewer.controller.close()


def test_a_3d_view_keeps_its_plan_through_a_transform_drag(tmp_path, plan_modes):
    """D7: a 3D view of a multiscale visual with the 3D flag on plans nothing
    during the drag (its bricks move with the node), and in full after."""
    viewer = _viewer(displayed=("z", "y", "x"))
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(np.eye(3)),
        )
        load(viewer)
        plan_modes.clear()

        async def _drag() -> int:
            with viewer.controller.transform_interaction(visual.id):
                for degrees in (5.0, 10.0):
                    visual.transform = _rotated(viewer, visual, degrees)
                    await _settle(viewer)
                n = len(_modes(plan_modes, visual))
            await _settle(viewer)
            return n

        assert asyncio.run(_drag()) == 0
        assert _modes(plan_modes, visual) == [PlanMode.FULL]
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Sticky routes, no handoff (5.2.3, D8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["image", "labels"])
def test_a_drag_through_aligned_stays_on_the_brick_route(tmp_path, plan_modes, family):
    """From aligned, through tilted and back: the slice node until release."""
    viewer = _viewer()
    try:
        data = _random_volume()
        if family == "image":
            visual = _add_image(
                viewer, multiscale_store(tmp_path, "s", data), world_matrix(np.eye(3))
            )
        else:
            store = multiscale_store(tmp_path, "l", data.astype(np.int32))
            visual = viewer.add_labels_multiscale(
                store,
                appearance=MultiscaleLabelsAppearance(
                    force_level=0, coarsest_while_moving_2d=True
                ),
                render_config=MultiscaleLabelRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                transform=bind(viewer, store, world_matrix(np.eye(3))),
            )
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        scene = viewer.controller._render_manager.get_scene(viewer.scene.id)
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        seen: list[tuple[bool, bool]] = []

        def _record() -> None:
            # Exactly one of the visual's two nodes is in the scene graph.
            in_scene = [
                node is not None and node.parent is scene
                for node in (
                    gfx_visual.slice_node or gfx_visual.node_3d,
                    gfx_visual.node_2d,
                )
            ]
            assert sum(in_scene) == 1
            seen.append((gfx_visual.slice_axes is not None, in_scene[0]))

        async def _drag() -> None:
            with viewer.controller.transform_interaction(visual.id):
                for degrees in (10.0, 20.0, 10.0, 0.0):
                    plan_modes.clear()
                    visual.transform = _rotated(viewer, visual, degrees)
                    _record()
                    if degrees == 10.0 and len(seen) == 1:
                        # The switch's own reslice plans the slice node.
                        assert any(
                            vid == visual.id and axes == (0, 1, 2)
                            for vid, _, axes in plan_modes
                        )
                    await _settle(viewer)
                    assert render(viewer)[0][..., :3].max() > 0
            await _settle(viewer)
            _record()

        asyncio.run(_drag())
        assert seen == [(True, True)] * 4 + [(False, False)]
        assert _active_node(viewer, visual) is gfx_visual.node_2d
        assert render(viewer)[0][..., :3].max() > 0
    finally:
        viewer.controller.close()


def test_a_jump_through_aligned_switches_at_once(tmp_path):
    """Outside a scope there is no stickiness (5.2.3)."""
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(rot(0, 2, 10.0)),
        )
        load(viewer, slice_z=float(P0[0]))
        assert _gfx(viewer, visual).slice_axes == (0, 1, 2)

        async def _jump() -> None:
            visual.transform = _rotated(viewer, visual, 0.0)
            await _load_data(viewer)

        asyncio.run(_jump())
        assert _gfx(viewer, visual).slice_axes is None
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# Atlas preparation (5.1, U12)
# ---------------------------------------------------------------------------


def _atlas_on_gpu(gfx_visual) -> bool:
    slots = getattr(gfx_visual, "slots", None) or (gfx_visual,)
    caches = [slot._block_cache_3d for slot in slots]
    return all(
        cache is not None and cache.cache_tex._wgpu_object is not None
        for cache in caches
    )


def test_a_visual_that_never_tilts_allocates_no_atlas(tmp_path):
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(np.eye(3)),
        )
        load(viewer, slice_z=float(P0[0]))
        render(viewer)
        assert all(slot._block_cache_3d is None for slot in _gfx(viewer, visual).slots)
    finally:
        viewer.controller.close()


def test_opening_a_transform_drag_prepares_the_atlas(tmp_path):
    viewer = _viewer()
    try:
        visual = _add_image(
            viewer,
            multiscale_store(tmp_path, "s", _random_volume()),
            world_matrix(np.eye(3)),
        )
        load(viewer, slice_z=float(P0[0]))
        gfx_visual = _gfx(viewer, visual)
        node_2d = gfx_visual.node_2d
        source = uuid4()
        viewer.controller.begin_transform_interaction(visual.id, source_id=source)
        try:
            assert _atlas_on_gpu(gfx_visual)
            # Prepared off the scene graph: today's node still draws.
            assert _active_node(viewer, visual) is node_2d
            assert gfx_visual.slice_axes is None
            assert render(viewer)[0][..., :3].max() > 0
        finally:
            viewer.controller.end_transform_interaction(visual.id, source_id=source)
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("family", ["image", "labels"])
def test_a_visual_added_tilted_has_its_atlas_at_add(tmp_path, family):
    viewer = _viewer()
    try:
        matrix = world_matrix(LINEARS["rot_zx"])
        if family == "image":
            visual = _add_image(
                viewer, multiscale_store(tmp_path, "s", _random_volume()), matrix
            )
        else:
            store = multiscale_store(tmp_path, "l", _random_volume().astype(np.int32))
            visual = viewer.add_labels_multiscale(
                store,
                appearance=MultiscaleLabelsAppearance(force_level=0),
                render_config=MultiscaleLabelRenderConfig(
                    **SMALL_BUDGETS, block_size=16
                ),
                transform=bind(viewer, store, matrix),
            )
        load(viewer, slice_z=float(P0[0]))
        assert _atlas_on_gpu(_gfx(viewer, visual))
    finally:
        viewer.controller.close()


def test_a_zero_filled_atlas_needs_no_upload():
    """Canary for the one pygfx internal of ``allocate_zeroed_texture``."""
    import pygfx as gfx
    import wgpu
    from pygfx.resources._base import resource_update_registry

    from cellier.render._gpu_lifetime import allocate_zeroed_texture

    texture = gfx.Texture(np.zeros((8, 8, 8), np.float32), dim=3, format="1xf4")
    allocate_zeroed_texture(texture)
    assert texture._wgpu_object is not None
    assert texture._wgpu_usage & wgpu.TextureUsage.TEXTURE_BINDING
    assert texture._gfx_get_chunk_descriptions() == []
    assert texture not in resource_update_registry._syncable
    # A later write uploads as usual.
    texture.update_range((0, 0, 0), (2, 2, 2))
    assert texture._gfx_get_chunk_descriptions()
    allocate_zeroed_texture(texture)  # already on the GPU: nothing happens


# ---------------------------------------------------------------------------
# Motion: backstop first, reuse, coverage, non-pow2
# ---------------------------------------------------------------------------


def _pyramid(
    root: Path, name: str, data: np.ndarray, factors
) -> MultiscaleZarrDataStore:
    """A zarr pyramid of *data*; *factors* is each coarse level's stride per axis."""
    root = root / name
    root.mkdir(parents=True, exist_ok=True)
    levels = [data] + [data[tuple(slice(None, None, f) for f in fs)] for fs in factors]
    scales = [(1.0, 1.0, 1.0)] + [tuple(float(f) for f in fs) for fs in factors]
    shifts = [(0.0, 0.0, 0.0)] + [tuple((f - 1) / 2.0 for f in fs) for fs in factors]
    by_shape = {level.shape: level for level in levels}

    def fill(arr):
        arr[...] = by_shape[arr.shape]

    _write_multiscale_zarr(
        root,
        levels=[(f"s{i}", level.shape) for i, level in enumerate(levels)],
        fill=fill,
        dtype=str(data.dtype),
    )
    # Strided levels sample voxel 0, f, 2f ...: level voxel k is level-0 voxel
    # f k, so the translation is 0, not the mean's (f - 1) / 2.
    del shifts
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=[f"s{i}" for i in range(len(levels))],
        level_scales=scales,
        level_translations=[(0.0, 0.0, 0.0)] * len(levels),
        name=name,
    )


def test_the_backstop_lands_first_on_a_switch(tmp_path):
    from cellier.events import BackstopCompleteEvent, ResliceCompletedEvent

    viewer = _viewer()
    try:
        data = np.tile(_random_volume(2), (2, 4, 4))[:48, :128, :160]
        store = _pyramid(tmp_path, "p", data, [(2, 2, 2), (4, 4, 4)])
        visual = _add_image(viewer, store, world_matrix(np.eye(3)), force_level=None)
        load(viewer, slice_z=20.0)
        order: list[str] = []
        owner = uuid4()
        bus = viewer.controller._outgoing_events
        bus.subscribe(
            BackstopCompleteEvent, lambda e: order.append("backstop"), owner_id=owner
        )
        bus.subscribe(
            ResliceCompletedEvent,
            lambda e: order.append("done") if e.visual_id == visual.id else None,
            owner_id=owner,
        )

        async def _jump() -> None:
            visual.transform = _rotated(viewer, visual, 20.0)
            await _load_data(viewer)

        asyncio.run(_jump())
        # The switch's plan completes its backstop before anything else.
        assert order and order[0] == "backstop"
    finally:
        viewer.controller.close()


def test_a_one_voxel_tick_rereads_few_bricks(tmp_path, monkeypatch):
    """Bricks survive a one-voxel scrub: a tick reads at most one new layer."""
    viewer = _viewer()
    try:
        data = np.tile(_random_volume(3), (3, 4, 4))[:64, :128, :160]
        store = _pyramid(tmp_path, "p", data, [(2, 2, 2)])
        matrix = world_matrix(rot(0, 2, 20.0))
        visual = _add_image(viewer, store, matrix)
        reads: list = []
        original = type(store).get_data

        async def spy(self, request):
            reads.append(request.scale_index)
            return await original(self, request)

        monkeypatch.setattr(type(store), "get_data", spy)
        slice_z = 30.0
        load(viewer, slice_z=slice_z)
        first = reads.count(0)
        assert first > 8
        reads.clear()
        load(viewer, slice_z=slice_z + 1.0)
        # A 16-voxel brick layer the plane crosses: at most a quarter of the
        # bricks on a 20 degree plane are new after a one-voxel move.
        assert reads.count(0) <= first // 4
        assert _gfx(viewer, visual).slice_axes is not None
    finally:
        viewer.controller.close()


def _coverage(frame, camera, polygon) -> float:
    """Share of the footprint's pixels (one-pixel border excluded) drawn."""
    xy = _pixel_world(camera, frame.shape[0])
    pixel = float(camera.width) / frame.shape[0] / float(camera.zoom)
    inside = np.ones(xy.shape[:2], dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge = end - start
        cross = edge[0] * (xy[..., 1] - start[1]) - edge[1] * (xy[..., 0] - start[0])
        inside &= cross > 1.5 * pixel * np.linalg.norm(edge)
    assert np.count_nonzero(inside) > 1000
    drawn = frame[..., :3].max(axis=-1) > 0
    return float(np.count_nonzero(drawn & inside) / np.count_nonzero(inside))


def test_mid_drag_coverage_with_z_never_downsampled(tmp_path):
    """U11 / 8.4 at test scale: every tick of a transform drag from 0 degrees
    draws at least 95% of the footprint, on a pyramid that keeps z (K11)."""
    viewer = _viewer()
    try:
        data = np.tile(_random_volume(4), (2, 4, 4))[:40, :128, :160]
        store = _pyramid(tmp_path, "p", data, [(1, 2, 2), (1, 4, 4)])
        visual = _add_image(viewer, store, world_matrix(np.eye(3)), force_level=None)
        slice_z = 20.0
        load(viewer, slice_z=slice_z)
        controller = viewer.controller
        coverages: list[float] = []

        async def _drag() -> None:
            with controller.transform_interaction(visual.id):
                for degrees in (5.0, 10.0, 15.0, 20.0):
                    visual.transform = _rotated(viewer, visual, degrees)
                    await _settle(viewer)
                    frame, _, camera = render(viewer)
                    classification = controller.view_classification(visual.id)
                    rendered = controller._scene_rendered(viewer.scene.id)
                    selection = viewer.scene.dims.to_selection(
                        *rendered
                    ).with_planes_at_slice(classification.sliced_world_axes)
                    polygon = slice_footprint(data.shape, classification, selection)[
                        ::-1, ::-1
                    ]
                    coverages.append(_coverage(frame, camera, polygon))
            await _settle(viewer)

        asyncio.run(_drag())
        assert min(coverages) >= 0.95, coverages
    finally:
        viewer.controller.close()


def test_a_non_pow2_pyramid_draws_its_coarse_level_on_the_slice(tmp_path):
    """The non-pow2 cell-to-brick rule on the route: a level of stride 3,
    forced (``force_level`` is 1-indexed: 2), shows level-1 voxel
    ``round(p / 3)`` at every pixel."""
    from tests.render.test_oblique_2d_brick_route import _expected_voxels

    viewer = _viewer()
    try:
        data = np.tile(_random_volume(5), (2, 2, 2))[:42, :60, :75]
        store = _pyramid(tmp_path, "p", data, [(3, 3, 3)])
        matrix = world_matrix(LINEARS["rot_zx"])
        visual = _add_image(viewer, store, matrix, force_level=2)
        slice_z = float(to_world(matrix, np.asarray(data.shape, float) / 2)[0])
        load(viewer, slice_z=slice_z)
        assert _gfx(viewer, visual).slice_axes is not None
        frame, _, camera = render(viewer)
        coarse = data[::3, ::3, ::3]
        # Level-0 voxel centres to level-1 voxels: index = round(p / 3).
        voxel0, _ = _expected_voxels(camera, matrix, slice_z, data.shape)
        xy = _pixel_world(camera)
        world = np.stack([np.full(xy.shape[:2], slice_z), xy[..., 1], xy[..., 0]], -1)
        p = (world - matrix[:3, 3]) @ np.linalg.inv(matrix[:3, :3]).T
        q = p / 3.0
        tie = np.any(np.abs(q - np.floor(q) - 0.5) < 0.02 / 3.0, axis=-1)
        voxel1 = np.floor(q + 0.5).astype(int)
        inside = np.all((voxel1 >= 0) & (voxel1 < np.asarray(coarse.shape)), axis=-1)
        inside &= np.all((voxel0 >= 0) & (voxel0 < np.asarray(data.shape)), axis=-1)
        valid = inside & ~tie
        assert np.count_nonzero(valid) > 10_000
        index = tuple(np.clip(voxel1[..., a], 0, coarse.shape[a] - 1) for a in range(3))
        grey = frame[..., 0].astype(float)
        assert np.count_nonzero(np.abs(grey - coarse[index])[valid] > 2.0) == 0
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The orthoviewer forwards the scope (5.2.4)
# ---------------------------------------------------------------------------


def test_the_orthoviewer_opens_the_drag_on_every_panel(tmp_path):
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        store = multiscale_store(tmp_path, "s", _random_volume())
        visuals = ortho.add_image_multiscale(
            store,
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        )
        controller = ortho.controller
        with ortho.transform_interaction(visuals):
            assert all(
                controller._transform_interaction_driver.scope_open(v.id)
                for v in visuals.values()
            )
        assert not any(
            controller._transform_interaction_driver.scope_open(v.id)
            for v in visuals.values()
        )
    finally:
        ortho.controller.close()
