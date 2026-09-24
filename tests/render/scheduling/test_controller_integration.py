"""The chunk scheduler wired into the viewer: 3D multiscale loads (design v3).

Routing, cache registration, retire / remove, invalidation, frame-paced
commits and the reslice events, through a real controller and store.
"""

from __future__ import annotations

import asyncio

import numpy as np

from cellier.render.scheduling import Tier
from cellier.visuals import MultiscaleImageSingleAppearance
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading


def _add(controller, store, dim="3d"):
    scene = controller.add_scene(dim=dim, name="scene")
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx


def _cache_id(gfx) -> int:
    return gfx.slots[0].residency_3d().cache_id


def _visible(scheduler, cache_id) -> int:
    reg = scheduler.core.registry(cache_id)
    return int((reg.tier == Tier.VISIBLE).sum())


def _count_reads(monkeypatch, store) -> list:
    reads: list = []
    original = type(store).get_data

    async def counting(self, request):
        reads.append(request)
        return await original(self, request)

    monkeypatch.setattr(type(store), "get_data", counting)
    return reads


async def test_a_3d_load_goes_through_the_scheduler(
    controller, multiscale_image_store, monkeypatch
):
    reads = _count_reads(monkeypatch, multiscale_image_store)
    scene, _visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    scheduler = controller._render_manager.scheduler
    cache_id = _cache_id(gfx)
    progress = scheduler.progress(cache_id)
    assert progress.needed_target > 0
    assert progress.needed_backstop > 0
    assert progress.resident_target == progress.needed_target
    assert progress.resident_backstop == progress.needed_backstop
    needed = progress.needed_target + progress.needed_backstop
    assert len(reads) == needed
    # Nothing went through the slicer for this visual.
    assert not controller._render_manager._slicer._tasks
    # The backstop stays drawn under the target once it is complete.
    assert len(gfx._block_cache_3d.tile_manager.tilemap) == needed


async def test_scene_ready_fires_once_all_bricks_are_resident(
    controller, multiscale_image_store
):
    scene, _visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    fired: list[int] = []

    def on_ready() -> None:
        progress = controller._render_manager.scheduler.progress(_cache_id(gfx))
        fired.append(progress.resident_target - progress.needed_target)

    controller.on_scene_ready(scene.id, on_ready)
    await drain_loading(controller)
    await asyncio.sleep(0)
    assert fired == [0]


async def test_coalesced_reslices_answer_every_announcement(
    controller, multiscale_image_store
):
    """Two reslices in one loop turn are one pass; both readiness callbacks fire."""
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    fired: list[str] = []
    controller.reslice_scene(scene.id, on_ready=lambda: fired.append("first"))
    controller.reslice_scene(scene.id, on_ready=lambda: fired.append("second"))
    await drain_loading(controller)
    await asyncio.sleep(0)
    assert sorted(fired) == ["first", "second"]


async def test_hiding_a_visual_retires_its_atlas(controller, multiscale_image_store):
    scene, visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    scheduler = controller._render_manager.scheduler
    cache_id = _cache_id(gfx)
    progress = scheduler.progress(cache_id)
    resident = progress.resident_target + progress.resident_backstop
    assert _visible(scheduler, cache_id) == resident > 0

    controller.set_visual_visible(visual.id, False)
    controller.reslice_all()
    await drain_loading(controller)

    assert _visible(scheduler, cache_id) == 0
    # Residents stay (RECENT), so showing it again costs nothing.
    reg = scheduler.core.registry(cache_id)
    assert int((reg.slot >= 0).sum()) == resident


async def test_switching_to_2d_retires_the_3d_atlas(controller, tmp_path):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (16, 16, 16)), ("s1", (8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    store = MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    )
    scene, _visual, gfx = _add(controller, store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    scheduler = controller._render_manager.scheduler
    cache_id = _cache_id(gfx)
    assert _visible(scheduler, cache_id) > 0

    controller.set_displayed_axes(scene.id, (1, 2))
    controller.reslice_all()
    await drain_loading(controller)

    assert _visible(scheduler, cache_id) == 0
    assert len(gfx._block_cache_2d.tile_manager.tilemap) > 0


async def test_removing_a_visual_removes_its_atlas(controller, multiscale_image_store):
    scene, visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    scheduler = controller._render_manager.scheduler
    cache_id = _cache_id(gfx)

    controller.remove_visual(visual.id)

    assert not scheduler.core.is_registered(cache_id)
    assert scheduler.core.cache_ids == []


async def test_a_store_change_refetches_what_it_touched(
    controller, multiscale_image_store, monkeypatch
):
    reads = _count_reads(monkeypatch, multiscale_image_store)
    scene, _visual, gfx = _add(controller, multiscale_image_store)
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    n_first = len(reads)

    # A change in one corner of level 0: only the bricks over it reload.
    multiscale_image_store.notify_changed(
        "contents", regions=[((0.0, 2.0), (0.0, 2.0), (0.0, 2.0))]
    )
    await drain_loading(controller)
    assert 0 < len(reads) - n_first < n_first

    # No region: everything reloads.
    multiscale_image_store.notify_changed("contents")
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(_cache_id(gfx))
    assert progress.resident_target == progress.needed_target


async def test_a_drawn_frame_commits_its_scene(controller, multiscale_image_store):
    """The canvas's before_draw hook runs a commit round for its own scene."""
    scene, _visual, gfx = _add(controller, multiscale_image_store)
    render_manager = controller._render_manager
    scheduler = render_manager.scheduler
    rounds: list[object] = []
    original = scheduler.commit_round

    def spy(scene_arg=None, *args):
        rounds.append(scene_arg)
        return original(scene_arg, *args)

    scheduler.commit_round = spy  # the hook looks it up at call time
    controller.fit_camera(scene.id)
    controller.reslice_all()
    for _ in range(200):
        await asyncio.sleep(0.005)
        if scheduler.core.oldest_arrival() is not None:
            break
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    canvas = render_manager._canvases[canvas_id].widget
    canvas.force_draw()
    assert scene.id in rounds
    assert len(gfx._block_cache_3d.tile_manager.tilemap) > 0
    await drain_loading(controller)


async def test_an_undrawn_channel_retires_its_atlas(controller, tmp_path):
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
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes("z", "y", "x")],
        dim="3d",
        name="scene",
    )
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
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
    ids = {key: gfx.slots[i].residency_3d().cache_id for key, i in gfx._drawn.items()}
    assert all(_visible(scheduler, cid) > 0 for cid in ids.values())

    visual.channels[1].visible = False
    controller.reslice_all()
    await drain_loading(controller)

    assert _visible(scheduler, ids[0]) > 0
    assert _visible(scheduler, ids[1]) == 0
    assert np.all(scheduler.core.registry(ids[1]).tier != Tier.VISIBLE)


# -- 2D ------------------------------------------------------------------------


def _cache_id_2d(gfx) -> int:
    return gfx.slots[0].residency_2d().cache_id


async def test_a_2d_load_goes_through_the_scheduler(
    controller, multiscale_image_store, monkeypatch
):
    reads = _count_reads(monkeypatch, multiscale_image_store)
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim="2d")
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    scheduler = controller._render_manager.scheduler
    progress = scheduler.progress(_cache_id_2d(gfx))
    assert progress.needed_target > 0
    assert progress.needed_backstop > 0
    assert progress.resident_target == progress.needed_target
    needed = progress.needed_target + progress.needed_backstop
    assert len(reads) == needed
    assert not controller._render_manager._slicer._tasks
    tilemap = gfx._block_cache_2d.tile_manager.tilemap
    assert len(tilemap) == needed
    # Every tile is on the plane the slider selects: z = 0 at level 1 (the
    # target) and at level 2 (the backstop).
    assert {(key.level, key.slice_coord) for key in tilemap} == {
        (1, ((0, 0),)),
        (2, ((0, 0),)),
    }


async def test_a_2d_slice_move_ends_with_only_the_new_plane_drawn(
    controller, multiscale_image_store
):
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim="2d")
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    controller.update_slice_indices(scene.id, {0: 6.0})
    await drain_loading(controller)

    tilemap = gfx._block_cache_2d.tile_manager.tilemap
    assert len(tilemap) > 0
    # force_level=1 is the finest level, which reads plane 6 itself; the
    # backstop (level 2, half the planes) reads plane 3.
    assert {(key.level, key.slice_coord) for key in tilemap} == {
        (1, ((0, 6),)),
        (2, ((0, 3),)),
    }
    # The old plane stays resident (RECENT), so going back costs no reads.
    reg = controller._render_manager.scheduler.core.registry(_cache_id_2d(gfx))
    assert int((reg.slot >= 0).sum()) == 2 * len(tilemap)


async def test_switching_to_3d_retires_the_2d_atlas(controller, multiscale_image_store):
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim="2d")
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    scheduler = controller._render_manager.scheduler
    cache_id = _cache_id_2d(gfx)
    assert _visible(scheduler, cache_id) > 0

    controller.set_displayed_axes(scene.id, (0, 1, 2))
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    assert _visible(scheduler, cache_id) == 0
    assert _visible(scheduler, _cache_id(gfx)) > 0


async def test_a_store_change_refetches_the_2d_tiles_it_touched(
    controller, multiscale_image_store, monkeypatch
):
    reads = _count_reads(monkeypatch, multiscale_image_store)
    scene, _visual, gfx = _add(controller, multiscale_image_store, dim="2d")
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)
    n_first = len(reads)

    # A change on another plane touches nothing drawn.
    multiscale_image_store.notify_changed(
        "contents", regions=[((10.0, 12.0), (0.0, 16.0), (0.0, 16.0))]
    )
    await drain_loading(controller)
    assert len(reads) == n_first

    # A change in one corner of the drawn plane reloads the tiles over it.
    multiscale_image_store.notify_changed(
        "contents", regions=[((0.0, 1.0), (0.0, 2.0), (0.0, 2.0))]
    )
    await drain_loading(controller)
    assert 0 < len(reads) - n_first < n_first
    progress = controller._render_manager.scheduler.progress(_cache_id_2d(gfx))
    assert progress.resident_target == progress.needed_target
