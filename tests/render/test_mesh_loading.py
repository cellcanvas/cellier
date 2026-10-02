"""The mesh on the chunk scheduler, end to end through the controller.

``plans/mesh_refactor_v3.md`` Phase 3 (L1-L6, M5).  The rule under test
everywhere: **a mesh never draws a position other than the slider's.**  While
a new position loads, the mesh is not drawn.

A frame here is a real frame of an offscreen canvas: the scheduler's commit
round, then the per-canvas ``prepare_draw``, then the render.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals._mesh_memory import MeshFlatAppearance

N_T = 6
_TZYX = [("t", "time"), *spatial_axes("z", "y", "x")]


def _series_store(n_t: int = N_T) -> MeshMemoryStore:
    """One triangle per timepoint: face ``t`` is the mesh at time ``t``."""
    positions = []
    for t in range(n_t):
        positions += [[t, 1, 1, 1 + t], [t, 1, 9, 1 + t], [t, 9, 1, 5 + t]]
    indices = np.arange(3 * n_t, dtype=np.int32).reshape(n_t, 3)
    return MeshMemoryStore(
        positions=np.array(positions, dtype=np.float32), indices=indices, name="series"
    )


def _static_store() -> MeshMemoryStore:
    """A mesh with no ``t`` axis: the ``t`` slider does not change its key."""
    return MeshMemoryStore(
        positions=np.array([[1, 1, 1], [5, 1, 1], [1, 5, 1], [1, 1, 5]], np.float32),
        indices=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32),
        name="static",
    )


class Reads:
    """Every mesh read, optionally held until the test lets it land."""

    def __init__(self, monkeypatch) -> None:
        self.started: list[tuple[str, object]] = []
        self.in_flight: dict[str, int] = {}
        self.max_in_flight: dict[str, int] = {}
        self.hold = False
        self.fail: BaseException | None = None
        self._gates: list[tuple[str, object, asyncio.Future]] = []
        inner = MeshMemoryStore.get_data
        reads = self

        async def get_data(store, request):
            name = store.name
            reads.started.append((name, request))
            reads.in_flight[name] = reads.in_flight.get(name, 0) + 1
            reads.max_in_flight[name] = max(
                reads.max_in_flight.get(name, 0), reads.in_flight[name]
            )
            try:
                if reads.hold:
                    gate = asyncio.get_running_loop().create_future()
                    reads._gates.append((name, request, gate))
                    await gate
                if reads.fail is not None:
                    raise reads.fail
                return await inner(store, request)
            finally:
                reads.in_flight[name] -= 1

        monkeypatch.setattr(MeshMemoryStore, "get_data", get_data)

    def count(self, name: str) -> int:
        return sum(1 for started, _ in self.started if started == name)

    def held(self) -> int:
        return len(self._gates)

    def release(self, index: int = 0) -> None:
        """Let one held read run."""
        _name, _request, gate = self._gates.pop(index)
        gate.set_result(None)

    def release_all(self) -> None:
        while self._gates:
            self.release()


class Rig:
    """A 3D ``tzyx`` scene with one offscreen canvas."""

    def __init__(self, dim: str = "3d") -> None:
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        self.scene = self.controller.add_scene(
            dim=dim, coordinate_system=_TZYX, name="scene"
        )
        self.controller.add_canvas(self.scene.id, canvas_size=(120, 90))
        self.canvas_id = self.controller.get_canvas_ids(self.scene.id)[0]
        self.view = self.controller.get_canvas_view(self.canvas_id)
        self.scheduler = self.controller._render_manager.scheduler

    def add_mesh(self, store, **appearance):
        return self.controller.add_mesh(
            data=store,
            scene_id=self.scene.id,
            appearance=MeshFlatAppearance(**appearance),
            name=store.name,
        )

    def gfx(self, visual):
        scenes = self.controller._render_manager._scenes
        return scenes[self.scene.id].get_visual(visual.id)

    def frame(self) -> None:
        self.view._canvas.draw()

    def drawn(self, visual) -> list[int] | None:
        """The faces the next frame draws; ``None`` when the mesh is not drawn."""
        self.frame()
        return _drawn_now(self.gfx(visual))

    def set_t(self, t: float, *, interactive: bool = False) -> None:
        self.controller.update_slice_indices(
            self.scene.id, {0: float(t)}, interactive=interactive
        )

    async def settle(self, timeout: float = 10.0) -> None:
        """Wait, drawing frames, until nothing is loading."""
        async with asyncio.timeout(timeout):
            while True:
                await asyncio.sleep(0)
                deferred = self.controller._deferred_reslice_tasks()
                if deferred:
                    await asyncio.gather(*deferred, return_exceptions=True)
                    continue
                self.frame()
                if self.scheduler.idle():
                    return
                await asyncio.sleep(0.002)

    async def turns(self, n: int = 5) -> None:
        """Let the loop run a few iterations (a pass, a read's start)."""
        for _ in range(n):
            await asyncio.sleep(0)

    async def until(self, condition, timeout: float = 10.0) -> None:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.002)


def _drawn_now(gfx) -> list[int] | None:
    nodes = gfx._levels[0]
    if not (gfx.node_3d.visible and nodes.mesh_3d.visible) and not (
        gfx.node_2d.visible and nodes.group_2d.visible
    ):
        return None
    if nodes.holds == "2d":
        # A section: the faces lying in the plane, and those it crosses.
        fill = nodes.fill_face_ids if nodes.fill.visible else None
        outline = nodes.outline_face_ids if nodes.outline.visible else None
        faces = np.concatenate(
            [
                np.zeros(0, dtype=np.int64) if ids is None else ids[ids >= 0]
                for ids in (fill, outline)
            ]
        )
        return sorted({int(face) for face in faces})
    faces = nodes.original_face_indices
    if faces is None:
        return list(range(len(nodes.mesh_3d.geometry.indices.data)))
    return np.asarray(faces).tolist()


@pytest.fixture
def reads(monkeypatch) -> Reads:
    return Reads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig()
    yield rig
    rig.controller.close()


# -- loading and the display rule ---------------------------------------------


async def test_a_mesh_loads_and_draws_its_position(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    assert rig.drawn(series) == [0]

    rig.set_t(3)
    await rig.settle()
    assert rig.drawn(series) == [3]


async def test_a_static_mesh_is_not_reread_on_an_unrelated_axis(rig, reads):
    """Its key does not change, so nothing is read, and it still completes."""
    static = rig.add_mesh(_static_store())
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count("static") == 1
    assert rig.drawn(static) == [0, 1, 2, 3]

    ready: list[int] = []
    for t in range(1, N_T):
        rig.set_t(t)
        # Never hidden: the plan's key is the one it holds.
        assert rig.drawn(static) == [0, 1, 2, 3]
    rig.controller.reslice_scene(rig.scene.id, on_ready=lambda: ready.append(1))
    await rig.settle()

    assert reads.count("static") == 1
    assert ready == [1]
    assert rig.drawn(series) == [N_T - 1]


async def test_no_frame_draws_a_position_the_slider_left(rig, reads):
    """With a held read: hidden from the tick until that position commits."""
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    rig.set_t(2)
    # The frame after the tick, before the pass is even applied.
    assert rig.drawn(series) is None
    await rig.turns()
    assert reads.held() == 1
    for _ in range(3):
        assert rig.drawn(series) is None
        await rig.turns()

    reads.release()
    await rig.settle()
    assert rig.drawn(series) == [2]


async def test_one_fine_read_in_flight_and_stale_arrivals_are_not_drawn(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    frames: list[tuple[float, list[int] | None]] = []

    def record() -> None:
        position = rig.scene.dims.selection.slice_indices[0]
        frames.append((float(position), rig.drawn(series)))

    for t in (1, 2, 3, 4):
        rig.set_t(t, interactive=True)
        record()
        await rig.turns()
        record()
    # One read at a time: the first tick's, and nothing else has started.
    assert reads.held() == 1
    assert reads.max_in_flight["series"] == 1

    # The stale read lands; the next (the newest position's) starts.
    reads.release()
    await rig.until(lambda: reads.held() == 1)
    record()
    reads.release()
    reads.hold = False
    await rig.settle()
    record()

    assert reads.max_in_flight["series"] == 1
    for position, faces in frames:
        assert faces is None or faces == [int(position)], frames
    assert frames[-1] == (4.0, [4])
    # Positions 2 and 3 were never read: the newest request won.
    read_ts = [
        round(float(request.region.bounding_box().min_coordinate[0]))
        for name, request in reads.started
        if name == "series"
    ]
    assert read_ts == [0, 1, 4]


async def test_reads_landing_out_of_order_never_draw_another_position(
    rig, reads, monkeypatch
):
    """Two reads of one cache in flight (the cap lifted), landing newest first."""
    from cellier.render import _level_residency
    from cellier.render.scheduling import CachePolicy

    monkeypatch.setattr(
        _level_residency.LevelResidency,
        "policy",
        CachePolicy(resource="compute", retry_max_attempts=1, retry_on_pass=False),
    )
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    rig.set_t(1)
    await rig.turns()
    rig.set_t(2)
    await rig.turns()
    assert reads.held() == 2

    reads.release(1)  # position 2 first
    await rig.until(lambda: rig.drawn(series) is not None)
    assert rig.drawn(series) == [2]
    reads.release()  # then the stale position 1
    reads.hold = False
    await rig.settle()
    assert rig.drawn(series) == [2]


async def test_an_appearance_change_or_a_camera_move_does_not_hide(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    rig.controller.fit_camera(rig.scene.id)
    before = reads.count("series")

    rig.controller.update_appearance_field(series.id, "opacity", 0.5)
    assert rig.drawn(series) == [0]
    rig.controller.update_appearance_field(series.id, "color", (1.0, 0.0, 0.0, 1.0))
    assert rig.drawn(series) == [0]

    state = rig.controller.get_canvas_view(rig.canvas_id).capture_camera_state()
    rig.controller.fit_camera(rig.scene.id)
    rig.controller.set_camera_state(rig.canvas_id, state)
    assert rig.drawn(series) == [0]
    await rig.settle()
    assert reads.count("series") == before


async def test_a_level_that_leaves_the_plan_is_released_and_read_again(rig, reads):
    """D9: no revisit guarantee."""
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    fine = rig.gfx(series)._levels[0]
    assert not fine.is_empty

    rig.set_t(1)
    # Released when the plan is made: the arrays are let go at once.
    assert fine.is_empty
    await rig.settle()
    rig.set_t(0)
    await rig.settle()

    assert rig.drawn(series) == [0]
    assert reads.count("series") == 3


async def test_on_ready_during_a_scrub_fires_once_with_the_final_position(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    ready: list[list[int] | None] = []
    for t in (1, 2, 3):
        rig.set_t(t)
        await rig.turns()
    rig.controller.reslice_scene(
        rig.scene.id, on_ready=lambda: ready.append(_drawn_now(rig.gfx(series)))
    )
    await rig.turns()
    assert ready == []
    reads.hold = False
    reads.release_all()
    await rig.settle()

    # Once, and the picture it announces is already built (K5).
    assert ready == [[3]]


async def test_a_capture_mid_scrub_shows_the_final_position(rig, reads):
    series = rig.add_mesh(_series_store(), color=(1.0, 0.0, 0.0, 1.0))
    rig.controller.reslice_all()
    await rig.settle()
    rig.controller.fit_camera(rig.scene.id)

    reads.hold = True
    for t in (3, 2, 1):
        rig.set_t(t)
        await rig.turns()

    def red(frame) -> int:
        return int(((frame[..., 0] > 150) & (frame[..., 1] < 90)).sum())

    # A capture taken now draws no mesh: no position but the slider's.
    blank = rig.controller.screenshot(rig.canvas_id)
    reads.hold = False
    reads.release_all()
    await rig.settle()
    shot = rig.controller.screenshot(rig.canvas_id)

    assert red(blank) == 0
    assert red(shot) > 0
    assert _drawn_now(rig.gfx(series)) == [1]
    # A capture canvas leaves no record behind on the visual.
    assert set(rig.gfx(series)._drawn) <= {rig.canvas_id}


# -- lifecycle (L6) -----------------------------------------------------------


async def test_a_store_change_mid_read_never_draws_the_earlier_data(rig, reads):
    store = _series_store()
    series = rig.add_mesh(store)
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    rig.set_t(1)
    await rig.turns()
    assert reads.held() == 1
    # Move every vertex: an "extent" change.  The read in flight is stale.
    moved = store.positions.copy()
    moved[:, 3] += 100.0
    store.positions = moved
    assert rig.drawn(series) is None
    reads.release()
    await rig.turns(20)
    assert rig.drawn(series) is None or _first_x(rig.gfx(series)) >= 100.0
    reads.hold = False
    reads.release_all()
    await rig.settle()

    assert rig.drawn(series) == [1]
    assert _first_x(rig.gfx(series)) >= 100.0


def _first_x(gfx) -> float:
    return float(gfx._levels[0].mesh_3d.geometry.positions.data[0, 0])


async def test_a_contents_change_rereads_without_a_new_plan(rig, reads):
    store = _series_store()
    series = rig.add_mesh(store)
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count("series") == 1

    # Reverse each face's winding: values change, the extent does not.
    store.indices = store.indices[:, ::-1].copy()
    assert rig.drawn(series) is None
    await rig.settle()

    assert reads.count("series") == 2
    assert rig.drawn(series) == [0]


async def test_removing_the_visual_mid_read_leaves_the_scheduler_idle(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()

    reads.hold = True
    rig.set_t(1)
    await rig.turns()
    assert reads.held() == 1
    rig.controller.remove_visual(series.id)
    reads.hold = False
    reads.release_all()
    await rig.settle()

    assert rig.scheduler.idle()
    assert rig.scheduler.core.cache_ids == []


async def test_a_hidden_mesh_reads_nothing(rig, reads):
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count("series") == 1

    rig.controller.update_appearance_field(series.id, "visible", False)
    assert rig.drawn(series) is None
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count("series") == 1

    # Shown at the same position: what it held is still there.
    rig.controller.update_appearance_field(series.id, "visible", True)
    assert rig.drawn(series) == [0]
    await rig.settle()
    assert reads.count("series") == 1

    # Hidden, moved, shown: it loads the new position.
    rig.controller.update_appearance_field(series.id, "visible", False)
    rig.set_t(2)
    await rig.settle()
    assert reads.count("series") == 1
    rig.controller.update_appearance_field(series.id, "visible", True)
    assert rig.drawn(series) is None
    await rig.settle()
    assert reads.count("series") == 2
    assert rig.drawn(series) == [2]


async def test_a_failing_read_is_attempted_once_per_key(rig, reads, caplog):
    series = rig.add_mesh(_series_store())
    reads.fail = RuntimeError("the kernel raised")
    ready: list[int] = []
    with caplog.at_level(logging.WARNING, logger="cellier"):
        rig.controller.reslice_scene(rig.scene.id, on_ready=lambda: ready.append(1))
        await rig.settle()
        # Later reslices of the same position do not run it again.
        for _ in range(3):
            rig.controller.reslice_all()
            await rig.settle()

    assert reads.count("series") == 1
    assert ready == [1]  # a given-up read still completes
    assert rig.drawn(series) is None
    assert any("read failed" in record.getMessage() for record in caplog.records)
    progress = rig.controller._render_manager.loading_progress(series.id)
    assert progress.failed == 1

    # A new key is a new attempt.
    reads.fail = None
    rig.set_t(1)
    await rig.settle()
    assert reads.count("series") == 2
    assert rig.drawn(series) == [1]


async def test_a_transform_change_that_keeps_the_region_reads_nothing(rig, reads):
    from tests._v2 import bound

    store = _series_store()
    series = rig.add_mesh(store)
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count("series") == 1

    # A shift along x: the region constrains t only, so the key is the same.
    shifted = bound(
        rig.controller, rig.scene.id, store, (1, 1, 1, 1), (0.0, 0.0, 0.0, 25.0)
    )
    rig.controller.set_visual_transform(series.id, shifted)
    assert rig.drawn(series) == [0]
    await rig.settle()
    assert reads.count("series") == 1

    # A shift along t moves the region in data space: a new key, a read.
    shifted_t = bound(
        rig.controller, rig.scene.id, store, (1, 1, 1, 1), (-2.0, 0.0, 0.0, 0.0)
    )
    rig.controller.set_visual_transform(series.id, shifted_t)
    await rig.settle()
    assert reads.count("series") == 2
    assert rig.drawn(series) == [2]


async def test_fit_camera_is_the_same_hidden_empty_and_shown(rig, reads):
    """M5: the fit frames the store's extent, whatever the mesh holds."""
    series = rig.add_mesh(_series_store())

    def fitted():
        rig.controller.fit_camera(rig.scene.id)
        state = rig.view.capture_camera_state()
        return np.concatenate(
            [np.asarray(state.position), [state.extent[0], state.extent[1]]]
        )

    not_loaded = fitted()
    rig.controller.reslice_all()
    await rig.settle()
    shown = fitted()
    # A position with no mesh at all (between timepoints): empty.
    rig.set_t(2.5)
    await rig.settle()
    assert rig.drawn(series) is None
    empty = fitted()
    rig.controller.update_appearance_field(series.id, "visible", False)
    hidden = fitted()

    np.testing.assert_allclose(shown, not_loaded)
    np.testing.assert_allclose(empty, not_loaded)
    np.testing.assert_allclose(hidden, not_loaded)


async def test_a_2d_view_loads_through_the_same_path(reads):
    rig = Rig(dim="2d")
    try:
        store = MeshMemoryStore(
            positions=np.array(
                [
                    [0, 4, 1, 1],
                    [0, 4, 9, 1],
                    [0, 4, 1, 9],
                    [0, 7, 1, 1],
                    [0, 7, 9, 1],
                    [0, 7, 1, 9],
                ],
                dtype=np.float32,
            ),
            indices=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
            name="planes",
        )
        mesh = rig.add_mesh(store)
        rig.controller.update_slice_indices(rig.scene.id, {0: 0.0, 1: 4.0})
        await rig.settle()
        assert rig.drawn(mesh) == [0]
        assert rig.gfx(mesh)._levels[0].holds == "2d"

        rig.controller.update_slice_indices(rig.scene.id, {1: 7.0})
        assert rig.drawn(mesh) is None
        await rig.settle()
        assert rig.drawn(mesh) == [1]
    finally:
        rig.controller.close()


async def test_a_mesh_scrub_leaves_an_image_beside_it_alone(rig, reads, monkeypatch):
    """The image's reads are the same with and without the mesh scrub."""
    image_reads: list[object] = []
    inner = ImageMemoryStore.get_data

    async def get_data(store, request):
        image_reads.append(request)
        return await inner(store, request)

    monkeypatch.setattr(ImageMemoryStore, "get_data", get_data)
    rig.controller.add_image(
        ImageMemoryStore(data=np.zeros((N_T, 8, 8, 8), dtype=np.float32)),
        rig.scene.id,
    )

    async def scrub() -> int:
        image_reads.clear()
        for t in range(N_T):
            rig.set_t(t)
            await rig.turns()
        await rig.settle()
        return len(image_reads)

    rig.controller.reslice_all()
    await rig.settle()
    alone = await scrub()
    series = rig.add_mesh(_series_store())
    rig.controller.reslice_all()
    await rig.settle()
    with_mesh = await scrub()

    assert with_mesh == alone
    assert rig.drawn(series) == [N_T - 1]


async def test_progress_is_reported_for_a_mesh(rig, reads):
    """What "Data fetch status" reads: one target, in flight, then resident."""
    from cellier.events._events import ResliceProgressEvent

    series = rig.add_mesh(_series_store())
    events: list[ResliceProgressEvent] = []
    rig.controller._outgoing_events.subscribe(
        ResliceProgressEvent, events.append, entity_id=series.id, owner_id=series.id
    )
    reads.hold = True
    rig.controller.reslice_all()
    await rig.turns()
    loading = rig.controller._render_manager.loading_progress(series.id)
    assert (loading.needed_target, loading.resident_target) == (1, 0)
    assert (loading.needed_backstop, loading.in_flight) == (0, 1)
    assert not loading.complete

    reads.hold = False
    reads.release_all()
    await rig.settle()
    done = rig.controller._render_manager.loading_progress(series.id)
    assert (done.needed_target, done.resident_target, done.in_flight) == (1, 1, 0)
    assert done.complete
    # Progress events are flushed once per loop iteration.
    await rig.turns()
    assert events and events[-1].progress.complete


async def test_pick_mapping_on_the_identity_and_the_indexed_path(rig, reads):
    """A pick names the store's face, whichever path the read took."""
    static = rig.add_mesh(_static_store())
    series = rig.add_mesh(_series_store())
    rig.set_t(3)
    await rig.settle()

    # Identity path: every face of the static mesh is drawn, in order.
    whole = rig.gfx(static)
    nodes = whole._levels[0]
    assert nodes.original_face_indices is None
    assert whole.decode_pick(nodes.mesh_3d, {"face_index": 2}).face_index == 2

    # Indexed path: the one drawn face is the store's face 3.
    sliced = rig.gfx(series)
    nodes = sliced._levels[0]
    assert sliced.decode_pick(nodes.mesh_3d, {"face_index": 0}).face_index == 3
    # The scene manager finds the visual from the child that was hit.
    scene_manager = rig.controller._render_manager._scenes[rig.scene.id]
    assert scene_manager.get_visual_id_for_node(nodes.mesh_3d) == series.id
