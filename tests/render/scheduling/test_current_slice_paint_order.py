"""2D: loaded tiles of the slice on screen stay above the backstop.

Through a real controller and store
(``plans/2d_current_slice_paint_order.md``).  The fixture is a 64^2 plane,
8 deep, with four levels and ``block_size=8``: an 8 x 8 grid of base cells,
level 4 (the backstop) one tile.  ``force_level`` picks each plan's target,
so the canvas size does not.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore
from cellier.scene import spatial_axes
from cellier.visuals import (
    MultiscaleImageSingleAppearance,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from cellier.visuals._image import (
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr, drain_loading

BACKSTOP_LEVEL = 4
SCALES = (1.0, 2.0, 4.0, 8.0)
ZYX = [(8, 64, 64), (4, 32, 32), (2, 16, 16), (1, 8, 8)]
KINDS = ["image", "labels"]


def _store(tmp_path, kind: str = "image", n_channels: int = 0):
    lead = (n_channels,) if n_channels else ()
    _write_multiscale_zarr(
        tmp_path,
        levels=[(f"s{i}", (*lead, *shape)) for i, shape in enumerate(ZYX)],
        fill=lambda a: a.__setitem__(slice(None), 1),
        dtype="int32" if kind == "labels" else "float32",
    )
    ones = (1.0,) * len(lead)
    zeros = (0.0,) * len(lead)
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(tmp_path),
        scale_names=[f"s{i}" for i in range(len(ZYX))],
        level_scales=[(*ones, s, s, s) for s in SCALES],
        level_translations=[(*zeros, *((s - 1) / 2,) * 3) for s in SCALES],
    )


def _add(controller, store, kind: str = "image", force_level: int = 1):
    """A 2D scene on *store*: ``(scene, visual, residency, canvas_id)``."""
    channels = len(store.level_shapes[0]) == 4
    scene = controller.add_scene(
        **(
            {"coordinate_system": [("c", "channel"), *spatial_axes("z", "y", "x")]}
            if channels
            else {}
        ),
        displayed_axes=("y", "x"),
        name="scene",
    )
    if kind == "labels":
        visual = controller.add_labels_multiscale(
            store,
            scene.id,
            MultiscaleLabelsAppearance(force_level=force_level),
            render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS, block_size=8),
        )
    else:
        visual = controller.add_image_multiscale(
            data=store,
            scene_id=scene.id,
            appearance=MultiscaleImageAppearance(force_level=force_level),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=8),
            single=MultiscaleImageSingleAppearance(
                color_map="viridis", clim=(0.0, 1.0), render_mode="mip"
            ),
            **({"channel_axis": 0, "composite": False} if channels else {}),
        )
    controller.add_canvas(scene_id=scene.id)
    # Every plan below is asked for; a camera move makes none of its own.
    controller.camera_reslice_enabled = False
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    slot = gfx if kind == "labels" else gfx.slots[0]
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    return scene, visual, slot.residency_2d(), canvas_id


def _levels(residency) -> np.ndarray:
    """The level every LUT cell draws (0 = nothing)."""
    return residency.lut_manager.lut_data[..., 2].copy()


class _Gate:
    """Holds the reads of the scales in ``hold`` until ``open`` is set."""

    def __init__(self, monkeypatch, store) -> None:
        self.hold: set[int] = set()
        self.held = 0
        self.open = asyncio.Event()
        original = type(store).get_data
        gate = self

        async def gated(self_store, request):
            if request.scale_index in gate.hold:
                gate.held += 1
                await gate.open.wait()
            return await original(self_store, request)

        monkeypatch.setattr(type(store), "get_data", gated)

    def close(self, *scales: int) -> None:
        self.hold, self.held = set(scales), 0
        self.open.clear()


async def _until(predicate, timeout_s: float = 5.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.002)


async def _load(controller) -> None:
    controller.reslice_all()
    await drain_loading(controller)


async def _held(controller, gate) -> None:
    """Plan, wait for the held reads to be asked for, and commit a frame."""
    controller.reslice_all()
    await _until(lambda: gate.held > 0)
    controller._render_manager.scheduler.commit_round()


# -- the current slice ids ---------------------------------------------------


async def test_the_ids_follow_the_slice_and_not_the_camera(controller, tmp_path):
    scene, _visual, residency, canvas_id = _add(controller, _store(tmp_path))
    assert residency.current_slice_ids is None
    controller.fit_camera(scene.id)
    await _load(controller)
    # Plane 0 of every level is one selection, so one id.
    plane_0 = residency.find((0, None, None))
    assert residency.current_slice_ids.tolist() == [plane_0] * 4

    state = controller.get_camera_state(canvas_id)
    controller.set_camera_state(canvas_id, state._replace(extent=(16.0, 16.0)))
    await _load(controller)
    assert residency.current_slice_ids.tolist() == [plane_0] * 4

    # z = 1 is level 1's plane 1; the coarser levels still read their plane 0.
    controller.update_slice_indices(scene.id, {0: 1.0})
    await _load(controller)
    plane_1 = residency.find((1, None, None))
    assert plane_1 not in (-1, plane_0)
    assert residency.current_slice_ids.tolist() == [plane_1, plane_0, plane_0, plane_0]

    # z = 5: planes 5, 2, 1 and 0.  Only the target and the backstop are
    # planned, so level 2's plane has no id, and none is made for it; level
    # 3's is the id level 1 made at z = 1.
    controller.update_slice_indices(scene.id, {0: 5.0})
    await _load(controller)
    plane_5 = residency.find((5, None, None))
    assert residency.current_slice_ids.tolist() == [plane_5, -1, plane_1, plane_0]
    assert residency.find((2, None, None)) == -1
    assert len(residency._selections) == 3


async def test_a_channel_change_changes_the_id_of_every_level(controller, tmp_path):
    store = _store(tmp_path, n_channels=2)
    scene, visual, _residency, _canvas_id = _add(controller, store)
    gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    controller.fit_camera(scene.id)
    await _load(controller)

    def drawn(channel: int):
        """The atlas of the one drawn channel, and its ids."""
        assert list(gfx._drawn) == [channel]
        residency = gfx.slots[gfx._drawn[channel]].residency_2d()
        ids = residency.current_slice_ids
        assert (ids == residency.find((channel, 0, None, None))).all()
        return residency, {residency.selection(int(sid)) for sid in ids}

    before, selections_before = drawn(0)
    controller.update_slice_indices(scene.id, {0: 1.0})
    await _load(controller)
    after, selections_after = drawn(1)
    # No level's selection is one the other channel read, in either atlas.
    assert selections_before.isdisjoint(selections_after)
    # One selection per channel was interned, whichever atlas drew it.
    atlases = {id(r): r for r in (before, after)}.values()
    assert sum(len(r._selections) for r in atlases) == 2


# -- a camera move keeps what is loaded --------------------------------------


@pytest.mark.parametrize("kind", KINDS)
async def test_a_zoom_out_keeps_the_loaded_tiles_above_the_backstop(
    controller, tmp_path, monkeypatch, kind
):
    store = _store(tmp_path, kind)
    gate = _Gate(monkeypatch, store)
    scene, visual, residency, canvas_id = _add(controller, store, kind, force_level=3)
    # Wide: the whole slice at level 3.
    controller.fit_camera(scene.id)
    await _load(controller)
    wide = controller.get_camera_state(canvas_id)
    assert (_levels(residency) == 3).all()

    # Zoomed in: the middle of the slice at level 1.
    visual.appearance.force_level = 1
    controller.set_camera_state(canvas_id, wide._replace(extent=(16.0, 16.0)))
    await _load(controller)
    close = _levels(residency)
    assert close[3:5, 3:5].tolist() == [[1, 1], [1, 1]]

    # Zoomed out to the level between them, its tiles held back.
    gate.close(1)
    visual.appearance.force_level = 2
    controller.set_camera_state(canvas_id, wide)
    await _held(controller, gate)
    levels = _levels(residency)
    # No cell is at the backstop: the wide view's tiles cover the slice and
    # the close view's tiles the middle of it.
    assert set(np.unique(levels).tolist()) == {1, 3}
    np.testing.assert_array_equal(levels == 1, close == 1)

    gate.open.set()
    await drain_loading(controller)
    assert (_levels(residency) == 2).all()


@pytest.mark.parametrize("kind", KINDS)
async def test_a_slice_move_keeps_no_tile_of_the_old_slice(
    controller, tmp_path, monkeypatch, kind
):
    """The dims twin: the new slice blurry, not the old one sharp.

    From z = 0 to z = 1 the coarser levels still read their plane 0, which
    is the selection (and the id) of the old level-1 tiles; they are still
    another slice's.
    """
    store = _store(tmp_path, kind)
    gate = _Gate(monkeypatch, store)
    scene, _visual, residency, _canvas_id = _add(controller, store, kind)
    controller.fit_camera(scene.id)
    await _load(controller)
    assert (_levels(residency) == 1).all()

    gate.close(0)
    controller.update_slice_indices(scene.id, {0: 1.0})
    await _held(controller, gate)
    assert (_levels(residency) == BACKSTOP_LEVEL).all()

    gate.open.set()
    await drain_loading(controller)
    assert (_levels(residency) == 1).all()
    drawn = residency.block_cache.tile_manager.tilemap
    assert {k.slice_coord for k in drawn if k.level == 1} == {((0, 1),)}


@pytest.mark.parametrize("kind", KINDS)
async def test_a_slice_move_keeps_a_coarse_tile_whose_plane_did_not_change(
    controller, tmp_path, monkeypatch, kind
):
    store = _store(tmp_path, kind)
    gate = _Gate(monkeypatch, store)
    scene, visual, residency, _canvas_id = _add(controller, store, kind, force_level=2)
    controller.fit_camera(scene.id)
    await _load(controller)
    assert (_levels(residency) == 2).all()

    # z = 1 at level 1: level 2 still reads its plane 0, loaded above.
    gate.close(0)
    visual.appearance.force_level = 1
    controller.update_slice_indices(scene.id, {0: 1.0})
    await _held(controller, gate)
    assert (_levels(residency) == 2).all()

    gate.open.set()
    await drain_loading(controller)
    assert (_levels(residency) == 1).all()
