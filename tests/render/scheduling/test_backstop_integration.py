"""The backstop through a real controller and store (design v3, 5.9, Phase 4).

The fixture is a 16^3 volume with two levels and ``force_level=1``, so the
target is level 1 (store scale 0) and the backstop is level 2 (scale 1).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from cellier.render._backstop import backstop_cap
from cellier.render.scheduling import ChunkClass, PlanMode
from cellier.visuals import MultiscaleImageSingleAppearance, ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

TARGET_SCALE, BACKSTOP_SCALE = 0, 1
TARGET_LEVEL, BACKSTOP_LEVEL = 1, 2


def _add(controller, store, dim="3d", loading=None, **render):
    scene = controller.add_scene(dim=dim, name="scene")
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=TARGET_LEVEL),
        render_config=MultiscaleImageRenderConfig(
            block_size=8,
            loading=loading or ProgressiveLoadingConfig(),
            **{**SMALL_BUDGETS, **render},
        ),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx


def _residency(gfx, dim):
    slot = gfx.slots[0]
    return slot.residency_3d() if dim == "3d" else slot.residency_2d()


def _lut_levels(gfx, dim) -> np.ndarray:
    """The level every LUT cell draws (0 = nothing)."""
    slot = gfx.slots[0]
    if dim == "3d":
        return slot._lut_manager_3d.lut_data[..., 3]
    return slot._lut_manager_2d.lut_data[..., 2]


class _Gate:
    """Holds target reads until opened; backstop reads go straight through."""

    def __init__(self, monkeypatch, store) -> None:
        self.reads: list = []
        self.open = asyncio.Event()
        original = type(store).get_data
        gate = self

        async def gated(self_store, request):
            gate.reads.append(request)
            if request.scale_index == TARGET_SCALE:
                await gate.open.wait()
            return await original(self_store, request)

        monkeypatch.setattr(type(store), "get_data", gated)


def _backstop_landed(scheduler, cache_id) -> bool:
    """Commit what has arrived (as a drawn frame would); is the backstop in?"""
    scheduler.commit_round()
    progress = scheduler.progress(cache_id)
    return progress.resident_backstop == progress.needed_backstop > 0


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.002)


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_the_backstop_is_read_first_and_drawn_while_the_target_loads(
    controller, multiscale_image_store, monkeypatch, dim
):
    gate = _Gate(monkeypatch, multiscale_image_store)
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim=dim)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    scheduler = controller._render_manager.scheduler
    cache_id = _residency(gfx, dim).cache_id

    await _until(lambda: _backstop_landed(scheduler, cache_id))
    # Every backstop read was issued before any target read.
    scales = [r.scale_index for r in gate.reads]
    n_backstop = scheduler.progress(cache_id).needed_backstop
    assert n_backstop > 0
    assert scales[:n_backstop] == [BACKSTOP_SCALE] * n_backstop
    # The whole view is drawn, blurry, while the target is held back.
    levels = _lut_levels(gfx, dim)
    assert (levels == BACKSTOP_LEVEL).all()

    gate.open.set()
    await drain_loading(controller)
    progress = scheduler.progress(cache_id)
    assert progress.resident_target == progress.needed_target > 0
    # The target covers the backstop wherever it has data.
    assert (_lut_levels(gfx, dim) == TARGET_LEVEL).all()


async def test_a_slice_move_shows_the_new_slice_blurry_not_the_old_one_sharp(
    controller, multiscale_image_store, monkeypatch
):
    gate = _Gate(monkeypatch, multiscale_image_store)
    gate.open.set()
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim="2d")
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    assert (_lut_levels(gfx, "2d") == TARGET_LEVEL).all()

    gate.open.clear()
    controller.update_slice_indices(scene.id, {0: 6.0})
    scheduler = controller._render_manager.scheduler

    def new_backstop_drawn() -> bool:
        scheduler.commit_round()
        drawn = gfx._block_cache_2d.tile_manager.tilemap
        # Level 2 reads half the planes: z = 6 is its plane 3.
        return (BACKSTOP_LEVEL, ((0, 3),)) in {(k.level, k.slice_coord) for k in drawn}

    await _until(new_backstop_drawn)
    # The old plane's fine tiles are still resident and painted underneath,
    # but the new plane's backstop is the foreground and covers them.
    tilemap = gfx._block_cache_2d.tile_manager.tilemap
    assert (TARGET_LEVEL, ((0, 0),)) in {(k.level, k.slice_coord) for k in tilemap}
    assert (_lut_levels(gfx, "2d") == BACKSTOP_LEVEL).all()

    gate.open.set()
    await drain_loading(controller)
    assert (_lut_levels(gfx, "2d") == TARGET_LEVEL).all()
    tilemap = gfx._block_cache_2d.tile_manager.tilemap
    assert {k.slice_coord for k in tilemap if k.level == TARGET_LEVEL} == {((0, 6),)}


@pytest.mark.parametrize("dim", ["3d", "2d"])
async def test_with_the_backstop_off_only_the_target_loads(
    controller, multiscale_image_store, monkeypatch, dim
):
    gate = _Gate(monkeypatch, multiscale_image_store)
    gate.open.set()
    scene, _visual, gfx = _add(
        controller,
        multiscale_image_store,
        dim=dim,
        loading=ProgressiveLoadingConfig(backstop=False),
    )
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(
        _residency(gfx, dim).cache_id
    )
    assert progress.needed_backstop == 0
    assert {r.scale_index for r in gate.reads} == {TARGET_SCALE}


async def test_changing_only_loading_reslices_and_keeps_the_atlas(
    controller, multiscale_image_store
):
    scene, visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    scheduler = controller._render_manager.scheduler
    cache_id = _residency(gfx, "3d").cache_id
    atlas = gfx.slots[0]._block_cache_3d
    assert scheduler.progress(cache_id).needed_backstop > 0

    visual.render_config = visual.render_config.model_copy(
        update={"loading": ProgressiveLoadingConfig(backstop=False)}
    )
    await drain_loading(controller)

    assert _residency(gfx, "3d").cache_id == cache_id
    assert gfx.slots[0]._block_cache_3d is atlas
    assert scheduler.progress(cache_id).needed_backstop == 0


async def test_changing_another_render_config_field_warns(
    controller, multiscale_image_store, caplog
):
    _scene, visual, _gfx = _add(controller, multiscale_image_store)
    with caplog.at_level(logging.WARNING, logger="cellier.render.cache"):
        visual.render_config = visual.render_config.model_copy(
            update={"gpu_budget_bytes": 2 * 1024**2}
        )
    assert any("only 'loading' applies" in r.getMessage() for r in caplog.records)


async def test_a_backstop_only_plan_holds_only_the_backstop(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    coordinator = controller._render_manager._slice_coordinator
    seen: list = []
    original = coordinator.submit

    def spy(request, visual_configs):
        seen.append((request, visual_configs))
        return original(request, visual_configs)

    monkeypatch.setattr(coordinator, "submit", spy)
    controller.reslice_all()
    await drain_loading(controller)
    request, configs = seen[-1]

    (full,) = gfx.plan(request, configs[gfx.visual_model_id], PlanMode.FULL)
    (backstop_only,) = gfx.plan(
        request, configs[gfx.visual_model_id], PlanMode.BACKSTOP_ONLY
    )
    assert (full.cls == ChunkClass.TARGET).any()
    assert len(backstop_only.keys) > 0
    assert (backstop_only.cls == ChunkClass.BACKSTOP).all()
    np.testing.assert_array_equal(
        backstop_only.keys, full.keys[full.cls == ChunkClass.BACKSTOP]
    )


async def test_the_backstop_cap_truncates_and_logs_once(
    controller, multiscale_image_store, caplog
):
    # A level-1 backstop (8 bricks, the same as the target) on a small
    # atlas: the cap keeps a few nearest first, the rest stay target.
    loading = ProgressiveLoadingConfig(backstop_level=1, backstop_max_slot_fraction=0.5)
    scene, _visual, gfx = _add(
        controller,
        multiscale_image_store,
        loading=loading,
        gpu_budget_bytes=32 * 10**3 * 4,
    )
    controller.fit_camera(scene.id)
    with caplog.at_level(logging.INFO, logger="cellier.render.cache"):
        controller.reslice_all()
        await drain_loading(controller)
        controller.reslice_all()
        await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(
        _residency(gfx, "3d").cache_id
    )
    cap = backstop_cap(loading, _residency(gfx, "3d").n_slots)
    assert 0 < cap < 8
    assert progress.needed_backstop == cap
    assert progress.truncated_backstop == 8 - cap
    lines = [r for r in caplog.records if "backstop_capped" in r.getMessage()]
    assert len(lines) == 1


async def test_each_composite_channel_gets_its_own_backstop(
    controller, tmp_path, monkeypatch
):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from cellier.scene import spatial_axes
    from cellier.visuals import MultiscaleImageChannelAppearance
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0,) * 4, (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0,) * 4, (0.0, 0.5, 0.5, 0.5)],
    )
    gate = _Gate(monkeypatch, store)
    gate.open.set()
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        appearance=MultiscaleImageAppearance(force_level=TARGET_LEVEL),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        channel_axis=0,
        composite=True,
        channels={
            0: MultiscaleImageChannelAppearance(color_map="red"),
            1: MultiscaleImageChannelAppearance(color_map="green"),
        },
        single=MultiscaleImageSingleAppearance(color_map="grays"),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    scheduler = controller._render_manager.scheduler
    for index in gfx._drawn.values():
        progress = scheduler.progress(gfx.slots[index].residency_3d().cache_id)
        assert progress.resident_backstop == progress.needed_backstop > 0
    backstop_channels = {
        r.axis_selections[0] for r in gate.reads if r.scale_index == BACKSTOP_SCALE
    }
    assert backstop_channels == {0, 1}
