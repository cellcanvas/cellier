"""In-memory image and labels on the chunk scheduler.

Each slot of an in-memory image (the labels visual has one) owns a
whole-slice cache per mode (``LevelResidency``): a plan wants the full slice
or volume of every drawn channel, a key a cache holds reads nothing, and
what is drawn follows the visibility rule.  Driven through a real controller
with an offscreen canvas; frames are drawn by hand, as a canvas would draw
them.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.render._spaces import node_follows_slice, node_matrix
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageChannelAppearance
from tests.render.conftest import drain_loading

#: World ``(c, y, x)``: a 2D scene displays ``(y, x)`` and slices ``c``.
_CYX = [("c", "channel"), *spatial_axes("y", "x")]
_ZYX = spatial_axes("z", "y", "x")


class Reads:
    """Count a store's whole-slice reads; optionally slow or gate them."""

    def __init__(self, store, *, delay: float = 0.0) -> None:
        self.calls: list[tuple] = []
        self.delay = delay
        self.gate: threading.Event | None = None
        original = store.read_padded

        def read_padded(request):
            self.calls.append(tuple(request.axis_selections))
            if self.gate is not None:
                self.gate.wait(5.0)
            if self.delay:
                time.sleep(self.delay)
            return original(request)

        object.__setattr__(store, "read_padded", read_padded)

    @property
    def n(self) -> int:
        return len(self.calls)


class Rig:
    """One scene, one offscreen canvas, frames drawn on request."""

    def __init__(self, world, displayed) -> None:
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        self.scene = self.controller.add_scene(
            displayed_axes=displayed, coordinate_system=world, name="scene"
        )
        self.controller.add_canvas(self.scene.id, canvas_size=(64, 64))
        canvas_id = self.controller.get_canvas_ids(self.scene.id)[0]
        self.view = self.controller.get_canvas_view(canvas_id)

    def gfx(self, visual):
        return self.controller._render_manager._scenes[self.scene.id].get_visual(
            visual.id
        )

    def frame(self) -> None:
        """Draw one frame: the canvas's ``before_draw`` runs the commit round."""
        self.view._canvas.draw()

    async def load(self) -> None:
        self.controller.reslice_all()
        await drain_loading(self.controller)

    async def frames_for(self, seconds: float, probe, hz: float = 60.0) -> list:
        """Draw frames at *hz* for *seconds*; return ``probe()`` after each."""
        out = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.frame()
            out.append(probe())
            await asyncio.sleep(1.0 / hz)
        return out

    def close(self) -> None:
        self.controller.close()


def _held_key(residency):
    """The request key the residency holds, or ``None``."""
    held = residency.held.get(0)
    return None if held is None else residency.item_of(held[0])[1]


def _volume(shape, dtype=np.float32) -> np.ndarray:
    """Every voxel set to its leading index plus one (never zero)."""
    data = np.empty(shape, dtype=dtype)
    for i in range(shape[0]):
        data[i] = i + 1
    return data


# -- modes ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "composite", "labels"])
async def test_switching_2d_3d_and_back_reads_nothing(kind):
    """Each mode has its own cache per slot, so a switch back reads nothing."""
    if kind == "composite":
        world = [("c", "channel"), *_ZYX]
        rig = Rig(world, ("y", "x"))
        store = ImageMemoryStore(data=_volume((2, 4, 8, 8)))
        rig.controller.add_image(
            store,
            rig.scene.id,
            channel_axis=0,
            composite=True,
            channels={i: InMemoryImageChannelAppearance() for i in range(2)},
        )
        to_3d, to_2d = (1, 2, 3), (2, 3)
    else:
        rig = Rig(_ZYX, ("y", "x"))
        if kind == "image":
            store = ImageMemoryStore(data=_volume((4, 8, 8)))
            rig.controller.add_image(store, rig.scene.id)
        else:
            store = LabelMemoryStore(data=_volume((4, 8, 8), np.int32))
            rig.controller.add_labels(store, rig.scene.id)
        to_3d, to_2d = (0, 1, 2), (1, 2)
    n_channels = 2 if kind == "composite" else 1
    try:
        await rig.load()
        reads = Reads(store)

        rig.controller.update_displayed_axes(rig.scene.id, to_3d)
        await rig.load()
        assert reads.n == n_channels  # the first visit to 3D reads once
        rig.controller.update_displayed_axes(rig.scene.id, to_2d)
        await rig.load()
        rig.controller.update_displayed_axes(rig.scene.id, to_3d)
        await rig.load()
        assert reads.n == n_channels
    finally:
        rig.close()


# -- channels ----------------------------------------------------------------------


async def test_a_slot_given_another_channel_never_shows_the_old_one():
    """Composite: a slot reassigned to channel 2 is hidden until 2 lands."""
    rig = Rig(_CYX, ("y", "x"))
    store = ImageMemoryStore(data=_volume((3, 8, 8)))
    channels = {i: InMemoryImageChannelAppearance() for i in range(2)}
    visual = rig.controller.add_image(
        store,
        rig.scene.id,
        channel_axis=0,
        composite=True,
        channels=channels,
        max_channels=2,
    )
    gfx = rig.gfx(visual)
    try:
        await rig.load()
        slot_of_one = gfx._slot_for_key[1]
        reads = Reads(store)
        reads.gate = threading.Event()

        rig.controller.remove_channel(visual.id, 1)
        rig.controller.add_channel(visual.id, 2, InMemoryImageChannelAppearance())
        await asyncio.sleep(0.01)
        assert gfx._slot_for_key[2] == slot_of_one
        slot = gfx.slots[slot_of_one]
        rig.frame()
        assert slot.node_2d.visible is False  # never channel 1's slice

        reads.gate.set()
        await drain_loading(rig.controller)
        rig.frame()
        assert slot.node_2d.visible is True
        assert np.all(slot.node_2d.geometry.grid.data == 3.0)
    finally:
        rig.close()


async def test_a_channel_scrub_in_single_mode_draws_no_blank_frame():
    """Slot 0 follows the slider and keeps its slice while the next loads."""
    rig = Rig(_CYX, ("y", "x"))
    store = ImageMemoryStore(data=_volume((8, 32, 32)))
    visual = rig.controller.add_image(store, rig.scene.id, channel_axis=0)
    gfx = rig.gfx(visual)
    try:
        await rig.load()
        Reads(store, delay=0.04)
        node = gfx.slots[0].node_2d

        async def scrub() -> None:
            for i in range(30):
                rig.controller.update_slice_indices(rig.scene.id, {0: float(i % 8)})
                await asyncio.sleep(1 / 30)

        task = asyncio.ensure_future(scrub())
        visible: list[bool] = []
        while not task.done():
            rig.frame()
            visible.append(bool(node.visible))
            await asyncio.sleep(1 / 60)
        await drain_loading(rig.controller)
        rig.frame()

        assert len(visible) > 30
        assert all(visible)
        assert np.all(node.geometry.grid.data == float(29 % 8 + 1))
        assert all(not slot.node_2d.visible for slot in gfx.slots[1:])
    finally:
        rig.close()


# -- placement -------------------------------------------------------------------


@pytest.mark.parametrize("family", ["image", "labels"])
async def test_a_slice_shift_view_places_the_node_by_the_shown_slice(family):
    """In a SLICE_SHIFT view the node always sits where the shown slice belongs."""
    from cellier.convenience.capture import _ensure_canvases
    from tests.render._affine import LINEARS, bind, new_viewer, world_matrix

    viewer = new_viewer("2d")
    matrix = world_matrix(LINEARS["shear_x_by_z"])
    rng = np.random.default_rng(1)
    if family == "image":
        store = ImageMemoryStore(data=(rng.random((24, 32, 40)) + 1).astype(np.float32))
        model = viewer.add_image(store, transform=bind(viewer, store, matrix))
    else:
        store = LabelMemoryStore(
            data=(rng.random((24, 32, 40)) * 4 + 1).astype(np.int32)
        )
        model = viewer.add_labels(store, transform=bind(viewer, store, matrix))
    _ensure_canvases(viewer, (64, 64))
    controller = viewer.controller
    scene_id = viewer.scene.id
    gfx = controller._render_manager._scenes[scene_id].get_visual(model.id)
    view = controller.get_canvas_view(controller.get_canvas_ids(scene_id)[0])
    residency = next(
        r
        for r in gfx.residencies().values()
        if r
        is (gfx.slots[0].residency["2d"] if family == "image" else gfx._residency["2d"])
    )
    node = gfx.slots[0].node_2d if family == "image" else gfx._inner_node_2d
    try:
        viewer.set_slice_positions({0: 2.0})
        controller.reslice_all()
        await drain_loading(controller)
        assert node_follows_slice(gfx._spaces)
        Reads(store, delay=0.03)

        async def scrub() -> None:
            for i in range(30):
                viewer.set_slice_positions({0: 2.0 + i * 0.5})
                await asyncio.sleep(1 / 30)

        task = asyncio.ensure_future(scrub())
        checked = mismatched = 0
        while not task.done():
            view._canvas.draw()
            key = _held_key(residency)
            if node.visible and key is not None:
                collapsed = dict(key[1])
                expected = node_matrix(
                    gfx._spaces,
                    gfx._transform,
                    {
                        a: float(collapsed.get(a, 0.0))
                        for a in gfx._spaces.collapsed_axes
                    },
                )
                checked += 1
                if not np.allclose(gfx.node_2d.local.matrix, expected, atol=1e-5):
                    mismatched += 1
            await asyncio.sleep(1 / 60)
        await drain_loading(controller)
        assert checked > 10
        assert mismatched == 0
    finally:
        controller.close()


# -- the brick route -----------------------------------------------------------------


@pytest.mark.parametrize("family", ["image", "labels"])
async def test_a_return_from_the_brick_route_reads_nothing(family):
    """Leaving the route keeps the whole-slice caches; one route at a time."""
    from cellier.convenience.capture import _ensure_canvases
    from tests.render._affine import LINEARS, P0, bind, new_viewer, world_matrix
    from tests.render.test_oblique_2d_in_memory import _add

    viewer = new_viewer("2d")
    visual = _add(viewer, family, world_matrix(np.eye(3)))
    _ensure_canvases(viewer, (64, 64))
    controller = viewer.controller
    gfx = controller._render_manager._scenes[viewer.scene.id].get_visual(visual.id)
    from uuid import UUID

    store = controller._model.data.stores[UUID(visual.data_store_id)]
    whole = gfx.slots[0].node_2d if family == "image" else gfx._inner_node_2d
    try:
        viewer.set_slice_positions({0: float(P0[0])})
        controller.reslice_all()
        await drain_loading(controller)
        if gfx.brick_source is not None:
            await gfx.brick_source.wait_built()
        reads = Reads(store)

        visual.transform = bind(viewer, store, world_matrix(LINEARS["rot_zx"]))
        controller.reslice_all()
        await drain_loading(controller)
        assert gfx.on_brick_route
        assert whole.parent is None or not _in_scene(whole)
        visual.transform = bind(viewer, store, world_matrix(np.eye(3)))
        controller.reslice_all()
        before = reads.n
        await drain_loading(controller)
        assert not gfx.on_brick_route
        assert reads.n == before  # the return reads nothing
        assert whole.visible and _in_scene(whole)
        assert not _in_scene(gfx.slice_node)
    finally:
        controller.close()


def _in_scene(node) -> bool:
    while node is not None:
        if type(node).__name__ == "Scene":
            return True
        node = node.parent
    return False


# -- painting ------------------------------------------------------------------------


async def test_a_paint_stroke_never_blanks_and_ends_with_every_dab():
    """Each dab invalidates; the held slice stays drawn until its re-read lands."""
    rig = Rig(_ZYX, ("y", "x"))
    store = LabelMemoryStore(data=np.zeros((2, 64, 64), dtype=np.int32))
    store.data[:, 10:20, 10:20] = 3
    visual = rig.controller.add_labels(store, rig.scene.id)
    gfx = rig.gfx(visual)
    node = gfx._inner_node_2d
    try:
        await rig.load()
        visible: list[bool] = []
        for i in range(30):
            store.data[0, 30:34, i : i + 2] = 5
            store.notify_changed("contents", regions=[((0, 1), (30, 34), (i, i + 2))])
            rig.frame()
            visible.append(bool(node.visible))
            await asyncio.sleep(1 / 60)
        await drain_loading(rig.controller)
        rig.frame()

        assert all(visible)
        drawn = node.geometry.grid.data[..., 0]
        np.testing.assert_array_equal(drawn, store.data[0])
    finally:
        rig.close()


# -- idle visuals ------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_an_empty_slice_hides_and_retires_then_returns_without_a_read(kind):
    rig = Rig(_ZYX, ("y", "x"))
    if kind == "image":
        store = ImageMemoryStore(data=_volume((4, 8, 8)))
        visual = rig.controller.add_image(store, rig.scene.id)
    else:
        store = LabelMemoryStore(data=_volume((4, 8, 8), np.int32))
        visual = rig.controller.add_labels(store, rig.scene.id)
    # A second visual keeps the scene's extent wide, so the slider can leave
    # the first one's data.
    rig.controller.add_image(
        ImageMemoryStore(data=np.zeros((20, 8, 8), dtype=np.float32)), rig.scene.id
    )
    gfx = rig.gfx(visual)
    node = gfx.slots[0].node_2d if kind == "image" else gfx._inner_node_2d
    coordinator = rig.controller._render_manager._slice_coordinator
    try:
        rig.controller.update_slice_indices(rig.scene.id, {0: 2.0})
        await rig.load()
        assert node.visible
        reads = Reads(store)

        rig.controller.update_slice_indices(rig.scene.id, {0: 12.0})
        await rig.load()
        assert not node.visible
        assert set(gfx.residencies()) <= coordinator._retired

        rig.controller.update_slice_indices(rig.scene.id, {0: 2.0})
        await rig.load()
        assert node.visible
        assert reads.n == 0
    finally:
        rig.close()


@pytest.mark.parametrize("kind", ["image", "labels"])
async def test_a_hidden_visual_shown_again_reads_nothing(kind):
    rig = Rig(_ZYX, ("y", "x"))
    if kind == "image":
        store = ImageMemoryStore(data=_volume((4, 8, 8)))
        visual = rig.controller.add_image(store, rig.scene.id)
    else:
        store = LabelMemoryStore(data=_volume((4, 8, 8), np.int32))
        visual = rig.controller.add_labels(store, rig.scene.id)
    gfx = rig.gfx(visual)
    node = gfx.slots[0].node_2d if kind == "image" else gfx._inner_node_2d
    coordinator = rig.controller._render_manager._slice_coordinator
    try:
        await rig.load()
        reads = Reads(store)

        visual.appearance.visible = False
        await rig.load()
        assert set(gfx.residencies()) <= coordinator._retired
        visual.appearance.visible = True
        await rig.load()
        rig.frame()

        assert reads.n == 0
        assert node.visible
    finally:
        rig.close()
