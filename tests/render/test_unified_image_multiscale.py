"""The unified multiscale image visual (unified image design 3.8, 3.9).

A multiscale image keeps a pool of per-channel slots.  The pyramid is planned
once, with the first drawn slot, and materialized per channel; every tile key
carries its own channel index, so a slot keeps its cache across mode
switches.  These tests drive the visual through the controller and the real
slicer, then read back pixels where the claim is about what is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.scene import spatial_axes
from cellier.visuals import (
    MultiscaleImageAppearance,
    MultiscaleImageChannelAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import MultiscaleImageRenderConfig
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]


@pytest.fixture
def czyx_store(tmp_path):
    """A 2-level CZYX pyramid: channel 0 fills the left half, channel 1 the right."""
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

    def _fill(arr: np.ndarray) -> None:
        width = arr.shape[-1]
        arr[0, ..., : width // 2] = 1.0
        arr[1, ..., width // 2 :] = 1.0

    _write_multiscale_zarr(
        tmp_path,
        levels=[("s0", (2, 16, 16, 16)), ("s1", (2, 8, 8, 8))],
        fill=_fill,
    )
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=["s0", "s1"],
        level_scales=[(1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 2.0)],
        level_translations=[(0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.5, 0.5)],
        name="czyx",
    )


def _add(controller, store, *, dim, composite=True, channels=None, **appearance):
    scene = controller.add_scene(coordinate_system=_CZYX, dim=dim, name="scene")
    if channels is None:
        channels = {
            0: MultiscaleImageChannelAppearance(color_map="red", clim=(0.0, 1.0)),
            1: MultiscaleImageChannelAppearance(color_map="green", clim=(0.0, 1.0)),
        }
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        appearance=MultiscaleImageAppearance(**appearance),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        channel_axis=0,
        composite=composite,
        channels=channels,
        single=MultiscaleImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene_id=scene.id)
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    return scene, visual, gfx_visual


def _tile_channels(slot, mode: str) -> set[int]:
    """The channel index in every slice key cached on *slot* for *mode*."""
    cache = slot._block_cache_3d if mode == "3d" else slot._block_cache_2d
    channels = set()
    for key in cache.tile_manager.tilemap:
        channels.update(value for axis, value in key.slice_coord if axis == 0)
    return channels


def _drawn_alpha(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame[..., 3]))


def _count(frame: np.ndarray, channel: int) -> int:
    """Pixels bright in one RGB channel."""
    return int(np.count_nonzero(frame[..., channel] > 128))


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def test_the_pool_has_max_channels_slots(controller, czyx_store):
    _scene, visual, gfx_visual = _add(controller, czyx_store, dim="2d")
    assert len(gfx_visual.slots) == visual.max_channels


def test_a_multiscale_image_without_a_channel_axis_has_one_slot(
    controller, multiscale_image_store
):
    scene = controller.add_scene(dim="2d", name="scene")
    visual = controller.add_image_multiscale(
        multiscale_image_store,
        scene.id,
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    assert len(gfx_visual.slots) == 1


# ---------------------------------------------------------------------------
# Composite and single, in 2D and 3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_composite_caches_each_channel_in_its_own_slot(
    controller, reslice, czyx_store, dim
):
    scene, _visual, gfx_visual = _add(controller, czyx_store, dim=dim, force_level=1)

    await reslice(controller, scene.id)

    assert set(gfx_visual._drawn) == {0, 1}
    for key, index in gfx_visual._drawn.items():
        assert _tile_channels(gfx_visual.slots[index], dim) == {key}


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_single_mode_draws_the_slider_channel(
    controller, reslice, czyx_store, dim
):
    scene, _visual, gfx_visual = _add(
        controller, czyx_store, dim=dim, composite=False, force_level=1
    )
    controller.update_slice_indices(scene.id, {0: 1.0})

    await reslice(controller, scene.id)

    assert list(gfx_visual._drawn) == [1]
    (index,) = gfx_visual._drawn.values()
    assert _tile_channels(gfx_visual.slots[index], dim) == {1}


async def test_overlapping_3d_channels_stop_writing_depth(
    controller, reslice, czyx_store
):
    """The first volume's hit depth would otherwise clip the next into speckle."""
    scene, visual, gfx_visual = _add(controller, czyx_store, dim="3d", force_level=1)
    await reslice(controller, scene.id)

    for index in gfx_visual._drawn.values():
        material = gfx_visual.slots[index].material_3d
        assert material.depth_write is False
        assert material.depth_test is True

    visual.channels[1].visible = False
    await reslice(controller, scene.id)
    (index,) = gfx_visual._drawn.values()
    assert gfx_visual.slots[index].material_3d.depth_write is True


async def test_a_hidden_channel_is_not_drawn(
    controller, reslice, render_scene, czyx_store
):
    """Channel 0 draws red, channel 1 green; hiding 1 removes only the green."""
    scene, visual, gfx_visual = _add(controller, czyx_store, dim="2d")
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)
    assert _count(frame, 1) > 0
    red = _count(frame, 0)
    assert red > 0

    visual.channels[1].visible = False
    await reslice(controller, scene.id)
    frame = render_scene(controller, scene.id)

    assert list(gfx_visual._drawn) == [0]
    assert _count(frame, 1) == 0
    # Only the seam between the halves may change, by antialiasing.
    assert abs(_count(frame, 0) - red) <= red // 50


async def test_an_empty_composite_draws_nothing(
    controller, reslice, render_scene, czyx_store
):
    scene, _visual, gfx_visual = _add(controller, czyx_store, dim="2d", channels={})

    await reslice(controller, scene.id)

    assert gfx_visual._drawn == {}
    assert _drawn_alpha(render_scene(controller, scene.id)) == 0


# ---------------------------------------------------------------------------
# Level of detail and cache reuse
# ---------------------------------------------------------------------------


async def test_force_level_reaches_every_drawn_slot(controller, reslice, czyx_store):
    scene, _visual, gfx_visual = _add(controller, czyx_store, dim="3d", force_level=1)

    await reslice(controller, scene.id)

    for index in gfx_visual._drawn.values():
        slot = gfx_visual.slots[index]
        levels = {key.level for key in slot._block_cache_3d.tile_manager.tilemap}
        # The target is level 1; the coarsest level is the backstop under it.
        assert levels - {slot.n_levels} == {1}


async def test_a_mode_switch_reuses_the_channel_cache(controller, reslice, czyx_store):
    """Composite -> single on a channel already cached fetches nothing new."""
    scene, visual, gfx_visual = _add(controller, czyx_store, dim="3d", force_level=1)
    await reslice(controller, scene.id)
    slot_for_one = gfx_visual._slot_for_key[1]

    controller.update_slice_indices(scene.id, {0: 1.0})
    controller.set_image_composite(visual.id, False)
    await reslice(controller, scene.id)

    assert gfx_visual._drawn == {1: slot_for_one}
    slot = gfx_visual.slots[slot_for_one]
    assert slot._last_plan_stats["total_required"] > 0
    scheduler = controller._render_manager.scheduler
    desired, new = scheduler.core.pass_stats(slot.residency_3d().cache_id)
    stats = slot._last_plan_stats
    assert desired == stats["n_backstop"] + stats["n_target"]
    assert new == 0


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


async def test_remove_visual_closes_it_and_forgets_every_atlas(
    controller, reslice, czyx_store
):
    scene, visual, gfx_visual = _add(controller, czyx_store, dim="2d")
    await reslice(controller, scene.id)
    scheduler = controller._render_manager.scheduler
    atlases = set(gfx_visual.residencies())
    assert atlases
    assert atlases <= set(scheduler.core.cache_ids)

    controller.remove_visual(visual.id)

    # Every slot's atlas is gone from the scheduler, so nothing lands later.
    assert not atlases & set(scheduler.core.cache_ids)
    assert gfx_visual.slots == ()
    assert gfx_visual.node_2d.children == ()


# ---------------------------------------------------------------------------
# The GPU budget
# ---------------------------------------------------------------------------


def _cache_bytes(gfx_visual, index: int) -> int:
    """The 3D brick cache's capacity on one slot, in bytes."""
    info = gfx_visual.slots[index]._block_cache_3d.info
    return info.n_slots * info.padded_block_size**3 * 4


@pytest.mark.parametrize("n_channels", [1, 2])
def test_the_budget_splits_between_channels_not_pool_slots(
    controller, czyx_store, n_channels
):
    """A slot the visual cannot draw must not take budget from one it can.

    The pool is always ``max_channels`` slots so a channel can be added
    later, but sizing each slot's cache by the pool made a single-channel
    image's cache a quarter of the budget -- it had the whole budget before
    the pool existed.  Compared against the one-channel case rather than in
    bytes: ``grid_side`` is the floored cube root of the slot budget, so
    capacity is quantised and never an exact fraction.
    """
    capacities = {}
    for channels in (1, n_channels):
        scene = controller.add_scene(coordinate_system=_CZYX, dim="3d")
        visual = controller.add_image_multiscale(
            czyx_store,
            scene.id,
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
            channel_axis=0,
            composite=True,
            channels={
                index: MultiscaleImageChannelAppearance() for index in range(channels)
            },
        )
        gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
        assert len(gfx_visual.slots) == visual.max_channels  # pool unchanged
        capacities[channels] = _cache_bytes(gfx_visual, 0)

    # One channel gets the whole budget; more channels share it, so a slot
    # never grows as channels are added, and never shrinks by the pool size.
    assert capacities[n_channels] <= capacities[1]
    if n_channels > 1:
        # Within a tenth of an even split: ``grid_side`` is the floored cube
        # root of the slot budget, so the share is quantised, never exact.
        # The pool-of-4 bug gave a quarter of this.
        assert capacities[n_channels] * n_channels >= capacities[1] * 0.9


def test_an_image_without_channels_keeps_the_whole_budget(
    controller, multiscale_image_store, czyx_store
):
    """No channel axis, or none configured, is sized like a lone image."""
    scene = controller.add_scene(dim="3d", name="plain")
    plain = controller.add_image_multiscale(
        multiscale_image_store,
        scene.id,
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
    )
    plain_gfx = controller._render_manager._scenes[scene.id].get_visual(plain.id)

    channel_scene = controller.add_scene(coordinate_system=_CZYX, dim="3d")
    channelled = controller.add_image_multiscale(
        czyx_store,
        channel_scene.id,
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
        channel_axis=0,
        channels={0: MultiscaleImageChannelAppearance()},
    )
    channel_gfx = controller._render_manager._scenes[channel_scene.id].get_visual(
        channelled.id
    )

    assert _cache_bytes(channel_gfx, 0) == _cache_bytes(plain_gfx, 0)
