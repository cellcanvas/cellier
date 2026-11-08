"""A MaskedImage's two sides plan, load and close together.

The mask side is the drawn visual; the value side is an inner visual whose
node is never in the scene.  Both load through the chunk scheduler, both paint
only the current slice's bricks, and close frees both by refcount.
"""

from __future__ import annotations

import asyncio
import gc
import time
import weakref
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cellier.render.visuals._masked_image import _MaskSlot, _ValueSlot
from cellier.transform import AffineTransform
from cellier.visuals import MaskedImageSingleAppearance, MultiscaleImageRenderConfig
from tests._gpu_budget import SMALL_BUDGETS
from tests.render._masked import (
    delay,
    gfx_visual,
    load,
    masked_config,
    new_viewer,
    random_pair,
    store,
)

if TYPE_CHECKING:
    from cellier.render.block_cache._image_residency import ImageResidency3D


KINDS = ("memory", "lazy")


def _add(viewer, mask, value, **fields):
    return viewer.controller.add_masked_image(
        mask,
        value,
        viewer.scene.id,
        render_config=fields.pop("render_config", masked_config()),
        single=MaskedImageSingleAppearance(iso_threshold=0.5, clim=(0.0, 1.0)),
        **fields,
    )


def _grid_bricks(shape, block=16) -> int:
    return int(np.prod([-(-n // block) for n in shape]))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dim", ["3d", "2d"])
def test_both_atlases_fill(tmp_path, kind, dim):
    """Each side registers its own atlas and fills it; progress counts both."""
    mask_data, value_data = random_pair()
    viewer = new_viewer(dim)
    try:
        mask = store(kind, tmp_path, "mask", mask_data)
        value = store(kind, tmp_path, "value", value_data)
        model = _add(viewer, mask, value)
        load(viewer, slice_z=11.0 if dim == "2d" else None)
        visual = gfx_visual(viewer, model.id)
        if dim == "2d":
            # Forced onto the brick route, the normal the sliced space axis.
            assert visual.slice_axes == (0, 1, 2)
        residencies = visual.residencies()
        assert len(residencies) == 2
        mask_cache = visual._slots[0].residency_3d()
        value_cache = visual.value_side._slots[0].residency_3d()
        assert set(residencies) == {mask_cache.cache_id, value_cache.cache_id}
        # No 2D tile atlas on either side.
        assert visual._slots[0].residency_2d() is None
        assert visual.value_side._slots[0].residency_2d() is None
        coordinator = viewer.controller._render_manager._slice_coordinator
        progress = coordinator.visual_progress(model.id)
        assert progress is not None and progress.complete
        core = coordinator._scheduler.core
        per_cache = [core.progress(cid) for cid in residencies]
        assert progress.needed_target == sum(p.needed_target for p in per_cache)
        expected = _grid_bricks((1 if dim == "2d" else 24, 32, 40))
        for residency in (mask_cache, value_cache):
            drawn = list(residency.block_cache.tile_manager.tilemap)
            # The finest level is painted whole: every brick of the view.
            finest = [key for key in drawn if key.level == 1]
            assert len(finest) == expected, (kind, dim, len(finest))
        # The value side's node is not in the scene.
        assert visual.value_side.node_3d.parent is None
        status = viewer.controller.draw_status(model.id)
        assert status.drawable
    finally:
        viewer.controller.close()


def test_value_side_reads_only_the_value_store(tmp_path):
    mask_data, value_data = random_pair()
    viewer = new_viewer("3d")
    try:
        mask = store("lazy", tmp_path, "mask", mask_data)
        value = store("lazy", tmp_path, "value", value_data)
        reads = {"mask": 0, "value": 0}
        for name, data_store in (("mask", mask), ("value", value)):
            original = data_store.get_data

            async def counted(request, _name=name, _original=original):
                reads[_name] += 1
                return await _original(request)

            object.__setattr__(data_store, "get_data", counted)
        _add(viewer, mask, value)
        load(viewer)
        assert reads["mask"] > 0
        assert reads["value"] > 0
    finally:
        viewer.controller.close()


def test_masked_residencies_paint_the_current_slice_only(tmp_path):
    """The flag is the masked visual's; a plain image keeps the default."""
    mask_data, value_data = random_pair()
    viewer = new_viewer("3d")
    try:
        mask = store("memory", tmp_path, "mask", mask_data)
        value = store("memory", tmp_path, "value", value_data)
        model = _add(viewer, mask, value)
        plain = viewer.add_image_multiscale(
            store("lazy", tmp_path, "plain", mask_data),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        )
        load(viewer)
        masked = gfx_visual(viewer, model.id).residencies().values()
        assert [r.current_slice_only for r in masked] == [True, True]
        plain_residencies = gfx_visual(viewer, plain.id).residencies().values()
        assert [r.current_slice_only for r in plain_residencies] == [False]
    finally:
        viewer.controller.close()


# -- the value side's placement ----------------------------------------------


def _fields(spaces) -> dict:
    """What P0-17 compared: every field, the visual system by axes and name."""
    return {
        "data": spaces.data,
        "world": spaces.world,
        "rendered": [a.name for a in spaces.rendered.axes],
        "retained_axes": spaces.retained_axes,
        "data_to_world_axes": dict(spaces.data_to_world_axes),
        "data_levels": spaces.data_levels,
        "visual_axes": [a.name for a in spaces.visual.axes],
        "rendered_to_world": np.asarray(spaces.rendered_to_world.matrix).tolist(),
        "world_to_rendered": np.asarray(spaces.world_to_rendered.matrix).tolist(),
        "level_transforms": [
            np.asarray(t.matrix).tolist() for t in spaces.level_transforms
        ],
        "coupling": np.asarray(spaces.coupling).tolist(),
        "alignment": spaces.classification.alignment,
        "classified_retained": spaces.classification.retained_data_axes,
        "effective": np.asarray(
            spaces.classification.effective_transform.matrix
        ).tolist(),
    }


def _rotated_scaled() -> np.ndarray:
    """Value to mask: 2x finer, rotated 20 degrees in (y, x), shifted."""
    angle = np.deg2rad(20.0)
    cos, sin = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[1:3, 1:3] = [[cos, -sin], [sin, cos]]
    matrix[:3, :3] *= 0.5
    matrix[:3, 3] = (1.0, -2.0, 3.0)
    return matrix


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("view", ["3d", "tilted_2d"])
def test_value_spaces_match_the_value_placed_as_an_image(tmp_path, kind, view):
    """The seam's fields equal the controller's own for the composed transform.

    A plain multiscale image visual of the value store, placed by
    ``transform @ value_to_mask``, is what the prototype's ghost id was:
    every field must agree.  In 3D a wrong value transform does not change
    the frame, so the fields are the check.
    """
    from cellier.convenience.capture import _ensure_canvases
    from tests.render._affine import rot, world_matrix

    mask_data, value_data = random_pair()
    viewer = new_viewer("3d" if view == "3d" else "2d")
    try:
        controller = viewer.controller
        mask = store(kind, tmp_path, "mask", mask_data)
        value = store(kind, tmp_path, "value", value_data)
        controller._ensure_data_coordinate_systems(viewer.scene.id, mask)
        world = viewer.scene.dims.world_coordinate_system
        tilt = rot(0, 2, 30.0) if view == "tilted_2d" else np.eye(3)
        transform = AffineTransform.from_matrix(
            world_matrix(tilt), mask.data_coordinate_systems[0], world
        )
        model = _add(
            viewer, mask, value, transform=transform, value_to_mask=_rotated_scaled()
        )
        composed = AffineTransform.from_matrix(
            np.asarray(transform.matrix) @ np.asarray(model.value_to_mask.matrix),
            value.data_coordinate_systems[0],
            world,
        )
        twin = viewer.add_image_multiscale(
            value,
            transform=composed,
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        )
        _ensure_canvases(viewer, (256, 256))
        seam = controller.value_render_spaces(model.id)
        reference = controller.render_spaces(twin.id)
        assert seam is not None and reference is not None
        got, want = _fields(seam), _fields(reference)
        if view == "tilted_2d":
            route = controller._brick_route_axes(model.id)
            assert route is not None
            assert controller._brick_route_axes(twin.id) == route
            # The slice node's rendered system and embedding are the mask's;
            # each routed visual registers its own over the same world axes.
            mask_spaces = controller.render_spaces(model.id)
            assert seam.rendered is mask_spaces.rendered
            for name in ("rendered_to_world", "world_to_rendered"):
                assert got[name] == _fields(mask_spaces)[name]
        for name in want:
            assert got[name] == want[name], name
        # The seam registers no second visual.
        assert set(controller._visual_to_scene) == {model.id, twin.id}
    finally:
        viewer.controller.close()


def test_value_space_is_forgotten_on_removal(tmp_path):
    mask_data, value_data = random_pair()
    viewer = new_viewer("3d")
    try:
        controller = viewer.controller
        model = _add(
            viewer,
            store("memory", tmp_path, "mask", mask_data),
            store("memory", tmp_path, "value", value_data),
        )
        load(viewer)
        assert model.id in controller._value_spaces
        controller.remove_visual(model.id)
        assert model.id not in controller._value_spaces
    finally:
        viewer.controller.close()


# -- close -------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_close_frees_both_atlases_by_refcount(tmp_path, kind):
    mask_data, value_data = random_pair()
    viewer = new_viewer("3d")
    model = _add(
        viewer,
        store(kind, tmp_path, "mask", mask_data),
        store(kind, tmp_path, "value", value_data),
    )
    load(viewer)
    visual = gfx_visual(viewer, model.id)
    refs = [
        weakref.ref(visual._slots[0]._block_cache_3d),
        weakref.ref(visual.value_side._slots[0]._block_cache_3d),
        weakref.ref(visual.value_side),
    ]
    del visual
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        viewer.controller.close()
        assert [ref() is None for ref in refs] == [True, True, True]
    finally:
        if was_enabled:
            gc.enable()


# -- only the current time point is drawn -------------------------------------


def _time_data(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((2, *(16, 32, 32))).astype(np.float32)


def _painted_times(residency: ImageResidency3D) -> set[int]:
    return {
        int(dict(key.slice_coord)[0])
        for key in residency.block_cache.tile_manager.tilemap
    }


@pytest.mark.parametrize("delayed", ["mask", "value"])
@pytest.mark.parametrize("rule", [True, False])
def test_neither_lut_names_the_old_time_point_after_a_step(
    tmp_path, monkeypatch, delayed, rule
):
    """Walk every loop tick after a step: no LUT paints a t = 0 brick.

    The control (the rule off) must paint the old time point at some tick,
    or the test proves nothing.
    """
    if not rule:
        monkeypatch.setattr(_MaskSlot, "_current_slice_only", False)
        monkeypatch.setattr(_ValueSlot, "_current_slice_only", False)
    viewer = new_viewer("3d", time=True)
    try:
        controller = viewer.controller
        stores = {
            "mask": store("lazy", tmp_path, "mask", _time_data(1)),
            "value": store("lazy", tmp_path, "value", _time_data(2)),
        }
        model = _add(viewer, stores["mask"], stores["value"])
        viewer.set_slice_positions({0: 0.0})
        load(viewer)
        visual = gfx_visual(viewer, model.id)
        residencies = list(visual.residencies().values())
        assert all(_painted_times(r) == {0} for r in residencies)
        delay(stores[delayed], 0.12)
        seen: list[set[int]] = []

        async def _step() -> None:
            from cellier.convenience.capture import _wait_until_loaded

            viewer.set_slice_positions({0: 1.0})
            controller.reslice_all()
            await asyncio.sleep(0)
            waiter = asyncio.ensure_future(_wait_until_loaded(controller))
            start = time.perf_counter()
            while not waiter.done() and time.perf_counter() - start < 20.0:
                controller._render_manager.scheduler.commit_round(viewer.scene.id)
                for residency in visual.residencies().values():
                    seen.append(_painted_times(residency))
                await asyncio.sleep(0.015)
            await waiter

        asyncio.run(_step())
        final = [_painted_times(r) for r in visual.residencies().values()]
        assert final == [{1}, {1}]
        old_painted = any(0 in times for times in seen)
        if rule:
            assert not old_painted
        else:
            assert old_painted, "the control painted no old brick: the test is blind"
    finally:
        viewer.controller.close()


# -- refusals that need a placed visual ---------------------------------------


def test_paint_controller_is_refused(tmp_path):
    mask_data, value_data = random_pair()
    viewer = new_viewer("2d")
    try:
        model = _add(
            viewer,
            store("memory", tmp_path, "mask", mask_data),
            store("memory", tmp_path, "value", value_data),
        )
        canvas_id = None
        with pytest.raises(TypeError, match="painting either is not supported"):
            viewer.controller.add_paint_controller(model.id, canvas_id)
    finally:
        viewer.controller.close()


def test_an_in_memory_side_that_grows_past_its_budget_draws_nothing(tmp_path):
    """A live store cannot be refused, so the view draws nothing."""
    from cellier.data import ImageMemoryStore

    mask_data, value_data = random_pair(shape=(16, 16, 16))
    viewer = new_viewer("3d")
    try:
        controller = viewer.controller
        value = ImageMemoryStore(data=value_data, name="value")
        # 16 MiB at 16-voxel bricks with overlap 3 holds 511 slots.
        model = _add(viewer, store("memory", tmp_path, "mask", mask_data), value)
        load(viewer)
        assert controller.draw_status(model.id).drawable
        value.data = np.zeros((16 * 9, 16 * 8, 16 * 8), np.float32)
        load(viewer)
        status = controller.draw_status(model.id)
        assert not status.drawable
        assert "in-memory value image" in status.reason
        assert "value_gpu_budget_bytes" in status.reason
        value.data = value_data
        load(viewer)
        assert controller.draw_status(model.id).drawable
    finally:
        viewer.controller.close()


def test_a_2d_view_slicing_two_space_axes_is_refused():
    """The slice node's normal is the one sliced space axis.

    A world of four space axes, two of them sliced: no single normal, so
    the view is refused rather than drawn as a plain image by the tile path.
    """
    from cellier.convenience import Viewer
    from cellier.data import ImageMemoryStore
    from cellier.scene.dims import spatial_axes

    rng = np.random.default_rng(0)
    viewer = Viewer(
        spatial_axes("w", "z", "y", "x"), displayed_axes=("y", "x"), gui="offscreen"
    )
    try:
        mask, value = (
            ImageMemoryStore(data=rng.random((2, 16, 16, 16), np.float32), name=name)
            for name in ("mask", "value")
        )
        model = _add(viewer, mask, value)
        load(viewer)
        status = viewer.controller.draw_status(model.id)
        assert not status.drawable
        assert "slices 2 space axes" in status.reason
    finally:
        viewer.controller.close()
