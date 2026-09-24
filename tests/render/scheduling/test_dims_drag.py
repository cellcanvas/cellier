"""``dims_drag`` and the dims settle debounce (design v3 5.10, Phase 5).

The fixture is a 16^3 volume with two levels and ``force_level=1``: the
target is level 1 and the backstop level 2.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from cellier.render.scheduling import ChunkClass
from cellier.visuals import MultiscaleImageSingleAppearance, ProgressiveLoadingConfig
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

DRAG = ProgressiveLoadingConfig(dims_drag="backstop")


def _tzyx_store(tmp_path):
    """Two timepoints of the 16^3, two-level volume: a collapsed t in 3D."""
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
    from tests.render.conftest import _write_multiscale_zarr

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (8, 16, 16, 16)), ("s1", (8, 8, 8, 8))],
        fill=lambda a: a.__setitem__(slice(None), 1.0),
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0,) * 4, (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0,) * 4, (0.0, 0.5, 0.5, 0.5)],
    )


def _add(controller, store, dim="2d", loading=DRAG, name="scene"):
    if len(store.level_shapes[0]) == 4:
        from cellier.scene import spatial_axes

        scene = controller.add_scene(
            coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
            dim=dim,
            name=name,
        )
    else:
        scene = controller.add_scene(dim=dim, name=name)
    visual = controller.add_image_multiscale(
        data=store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(
            **SMALL_BUDGETS, block_size=8, loading=loading
        ),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
        ),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx


def _cache_id(gfx, dim="2d") -> int:
    slot = gfx.slots[0]
    return (slot.residency_2d() if dim == "2d" else slot.residency_3d()).cache_id


def _wanted_classes(controller, cache_id) -> set[int]:
    """Classes of the records the latest pass wanted (``VISIBLE``)."""
    from cellier.render.scheduling import Tier

    reg = controller._render_manager.scheduler.core.registry(cache_id)
    return {int(c) for c in reg.cls[reg.tier == Tier.VISIBLE]}


async def _loaded(controller, scene) -> None:
    controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)


def _settle_s(controller) -> float:
    return controller._render_manager.config.scheduler.dims_settle_s


def _count_full_passes(monkeypatch, controller) -> list:
    """Record the plan mode of every chunked plan, per call."""
    from cellier.render.scene_manager import SceneManager

    modes: list = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        modes.extend(cfg.plan_mode for cfg in visual_configs.values())
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return modes


# -- the config -------------------------------------------------------------------


def test_eager_is_the_default() -> None:
    assert ProgressiveLoadingConfig().dims_drag == "eager"


def test_backstop_drag_without_a_backstop_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs backstop=True"):
        ProgressiveLoadingConfig(backstop=False, dims_drag="backstop")
    # Off with eager is fine: the target alone.
    ProgressiveLoadingConfig(backstop=False, dims_drag="eager")


# -- the controller ---------------------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_a_tick_plans_the_backstop_and_the_settle_the_target(
    controller, tmp_path, dim
):
    scene, _visual, gfx = _add(controller, _tzyx_store(tmp_path), dim=dim)
    await _loaded(controller, scene)
    cache_id = _cache_id(gfx, dim)
    assert _wanted_classes(controller, cache_id) == {
        int(ChunkClass.BACKSTOP),
        int(ChunkClass.TARGET),
    }

    controller.update_slice_indices(scene.id, {0: 6.0})
    await asyncio.sleep(0)  # the pass
    await asyncio.sleep(0)
    assert _wanted_classes(controller, cache_id) == {int(ChunkClass.BACKSTOP)}
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_target == 0

    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)
    progress = controller._render_manager.scheduler.progress(cache_id)
    assert progress.needed_target > 0
    assert progress.resident_target == progress.needed_target
    assert int(ChunkClass.TARGET) in _wanted_classes(controller, cache_id)


async def test_ticks_restart_the_settle_and_it_plans_once(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    await _loaded(controller, scene)
    modes = _count_full_passes(monkeypatch, controller)

    settle = _settle_s(controller)
    for z in (2.0, 4.0, 6.0, 8.0):
        controller.update_slice_indices(scene.id, {0: z})
        await asyncio.sleep(settle / 3)  # faster than the settle
    assert modes and all(m.name == "BACKSTOP_ONLY" for m in modes)

    await asyncio.sleep(settle * 2)
    await drain_loading(controller)
    full = [m for m in modes if m.name == "FULL"]
    assert len(full) == 1  # one settle, one canvas


async def test_eager_ticks_plan_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, _visual, _gfx = _add(
        controller, multiscale_image_store, loading=ProgressiveLoadingConfig()
    )
    await _loaded(controller, scene)
    modes = _count_full_passes(monkeypatch, controller)
    controller.update_slice_indices(scene.id, {0: 6.0})
    await drain_loading(controller)
    assert modes and all(m.name == "FULL" for m in modes)
    assert not controller._dims_settle_tasks


async def test_a_displayed_axes_change_plans_in_full_and_drops_the_settle(
    controller, tmp_path, monkeypatch
):
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
    scene, _visual, _gfx = _add(controller, store)
    await _loaded(controller, scene)
    controller.update_slice_indices(scene.id, {0: 6.0})
    assert scene.id in controller._dims_settle_tasks

    modes = _count_full_passes(monkeypatch, controller)
    controller.set_displayed_axes(scene.id, (0, 2))
    assert scene.id not in controller._dims_settle_tasks
    await drain_loading(controller)
    assert modes and all(m.name == "FULL" for m in modes)


async def test_other_visuals_in_the_scene_still_plan_in_full(
    controller, multiscale_image_store, monkeypatch
):
    scene, drag_visual, _gfx = _add(controller, multiscale_image_store)
    eager_visual = controller.add_image_multiscale(
        data=multiscale_image_store,
        scene_id=scene.id,
        appearance=MultiscaleImageAppearance(force_level=1),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        single=MultiscaleImageSingleAppearance(color_map="viridis"),
    )
    await _loaded(controller, scene)
    from cellier.render.scene_manager import SceneManager

    seen: dict = {}
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for visual_id, cfg in visual_configs.items():
            seen.setdefault(visual_id, set()).add(cfg.plan_mode.name)
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    controller.update_slice_indices(scene.id, {0: 6.0})
    assert seen[drag_visual.id] == {"BACKSTOP_ONLY"}
    assert seen[eager_visual.id] == {"FULL"}
    await asyncio.sleep(_settle_s(controller) * 2)
    await drain_loading(controller)


async def test_removing_the_scene_or_closing_cancels_the_settle(
    controller, multiscale_image_store
):
    scene, _visual, _gfx = _add(controller, multiscale_image_store)
    other, _v2, _g2 = _add(controller, multiscale_image_store, name="other")
    await _loaded(controller, scene)
    controller.update_slice_indices(scene.id, {0: 6.0})
    controller.update_slice_indices(other.id, {0: 6.0})
    task = controller._dims_settle_tasks[scene.id]

    controller.remove_scene(scene.id)
    await asyncio.sleep(0)
    assert task.cancelled()
    assert scene.id not in controller._dims_settle_tasks

    other_task = controller._dims_settle_tasks[other.id]
    controller.close()
    await asyncio.sleep(0)
    assert other_task.cancelled()
    assert not controller._dims_settle_tasks
