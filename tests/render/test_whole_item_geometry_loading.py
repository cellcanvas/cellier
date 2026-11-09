"""In-memory geometry on the chunk scheduler, end to end through the controller.

The points, lines and graph visuals load their whole filtered slice
through a single-level ``LevelResidency``:

- an unchanged request reads nothing, and a scrub keeps at most two reads
  in flight per visual;
- the previous result stays drawn while the next one loads, and a
  superseded arrival of a compatible key is drawn too; an incompatible
  one (another placement) is not;
- a contents change keeps the old result drawn until the re-read lands,
  and a filter that raised is read again after the next change;
- progress is reported, and a failed read still completes.

A frame here is a real frame of an offscreen canvas: the scheduler's commit
round, then the render.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data.graph._graph_memory_store import GraphMemoryStore
from cellier.data.lines._lines_memory_store import LinesMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance

N_T = 6
_TZYX = [("t", "time"), *spatial_axes("z", "y", "x")]


# -- the three families --------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    name: str
    store_cls: type
    slice_strategy: str = "mask"

    @property
    def id(self) -> str:
        return self.name if self.name != "graph" else f"graph-{self.slice_strategy}"

    def make_store(self) -> Any:
        """One element per timepoint: element ``t`` lies at time ``t``."""
        if self.name == "points":
            positions = [[t, 0, 1, 1 + t] for t in range(N_T)]
            return PointsMemoryStore(
                positions=np.array(positions, np.float32), name="series"
            )
        if self.name == "lines":
            positions = []
            for t in range(N_T):
                positions += [[t, 0, 1, 1 + t], [t, 0, 5, 1 + t]]
            return LinesMemoryStore(
                positions=np.array(positions, np.float32), name="series"
            )
        positions = []
        edges = []
        for t in range(N_T):
            positions += [[t, 0, 1, 1 + t], [t, 0, 5, 1 + t]]
            edges.append([2 * t, 2 * t + 1])
        return GraphMemoryStore(
            positions=np.array(positions, np.float32),
            edges=np.array(edges, np.int64),
            slice_strategy=self.slice_strategy,
            name="series",
        )

    def add(self, controller, scene_id, store):
        if self.name == "points":
            return controller.add_points(
                data=store, scene_id=scene_id, appearance=PointsMarkerAppearance()
            )
        if self.name == "lines":
            return controller.add_lines(
                data=store, scene_id=scene_id, appearance=LinesMemoryAppearance()
            )
        return controller.add_graph(data=store, scene_id=scene_id)

    def drawn(self, gfx) -> list[int] | None:
        """The store elements the visual draws; ``None`` when it draws nothing."""
        if not gfx.node.visible:
            return None
        if self.name == "points":
            if gfx.node.material is gfx._empty_material:
                return None
            return sorted(np.asarray(gfx._original_indices).tolist())
        if self.name == "lines":
            if gfx.node.material is gfx._empty_material:
                return None
            return sorted(np.asarray(gfx._original_edge_indices).tolist())
        if gfx.node_points.material is gfx._empty_node_material:
            return None
        return sorted(np.asarray(gfx._original_node_rows).tolist())[::2]

    def element_of(self, t: int) -> list[int]:
        """What drawing time *t* shows (graph: its first node)."""
        return [2 * t] if self.name == "graph" else [t]


POINTS = Kind("points", PointsMemoryStore)
KINDS = [
    POINTS,
    Kind("lines", LinesMemoryStore),
    Kind("graph", GraphMemoryStore, "mask"),
    Kind("graph", GraphMemoryStore, "roi"),
]


class Reads:
    """Every read of one store class, optionally held until the test lets it."""

    def __init__(self, monkeypatch, store_cls) -> None:
        self.started: list[Any] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.hold = False
        self.fail: BaseException | None = None
        self._gates: list[asyncio.Future] = []
        inner = store_cls.get_data
        reads = self

        async def get_data(store, request):
            reads.started.append(request)
            reads.in_flight += 1
            reads.max_in_flight = max(reads.max_in_flight, reads.in_flight)
            try:
                if reads.hold:
                    gate = asyncio.get_running_loop().create_future()
                    reads._gates.append(gate)
                    await gate
                if reads.fail is not None:
                    raise reads.fail
                return await inner(store, request)
            finally:
                reads.in_flight -= 1

        monkeypatch.setattr(store_cls, "get_data", get_data)

    @property
    def count(self) -> int:
        return len(self.started)

    def held(self) -> int:
        return len(self._gates)

    def release(self, index: int = 0) -> None:
        self._gates.pop(index).set_result(None)

    def release_all(self) -> None:
        while self._gates:
            self.release()


class Rig:
    """A ``tzyx`` scene with one offscreen canvas and one geometry visual."""

    def __init__(self, kind: Kind, dim: str = "2d") -> None:
        self.kind = kind
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        self.scene = self.controller.add_scene(
            displayed_axes=("z", "y", "x") if dim == "3d" else ("y", "x"),
            coordinate_system=_TZYX,
            name="scene",
        )
        self.controller.add_canvas(self.scene.id, canvas_size=(120, 90))
        canvas_id = self.controller.get_canvas_ids(self.scene.id)[0]
        self.view = self.controller.get_canvas_view(canvas_id)
        self.scheduler = self.controller._render_manager.scheduler
        self.store = kind.make_store()
        self.visual = kind.add(self.controller, self.scene.id, self.store)

    @property
    def gfx(self):
        scenes = self.controller._render_manager._scenes
        return scenes[self.scene.id].get_visual(self.visual.id)

    def drawn(self) -> list[int] | None:
        self.view._canvas.draw()
        return self.kind.drawn(self.gfx)

    def set_t(self, t: float) -> None:
        self.controller.update_slice_indices(self.scene.id, {0: float(t)})

    async def settle(self, timeout: float = 10.0) -> None:
        async with asyncio.timeout(timeout):
            while True:
                await asyncio.sleep(0)
                deferred = self.controller._deferred_reslice_tasks()
                if deferred:
                    await asyncio.gather(*deferred, return_exceptions=True)
                    continue
                self.view._canvas.draw()
                if self.scheduler.idle():
                    return
                await asyncio.sleep(0.002)

    async def turns(self, n: int = 5) -> None:
        for _ in range(n):
            await asyncio.sleep(0)

    async def until(self, condition, timeout: float = 10.0) -> None:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.002)


@pytest.fixture(params=KINDS, ids=lambda kind: kind.id)
def kind(request) -> Kind:
    return request.param


@pytest.fixture
def rig(kind):
    rig = Rig(kind)
    yield rig
    rig.controller.close()


@pytest.fixture
def reads(monkeypatch, kind) -> Reads:
    return Reads(monkeypatch, kind.store_cls)


# -- loading ---------------------------------------------------------------------


@pytest.mark.parametrize("dim", ["2d", "3d"])
async def test_the_visual_loads_through_the_scheduler(kind, monkeypatch, dim):
    reads = Reads(monkeypatch, kind.store_cls)
    rig = Rig(kind, dim)
    try:
        rig.controller.reslice_all()
        await rig.settle()
        assert reads.count == 1
        cache_ids = list(rig.gfx.residencies())
        assert all(rig.scheduler.core.is_registered(cid) for cid in cache_ids)
        assert rig.drawn() == kind.element_of(0)
    finally:
        rig.controller.close()


async def test_an_unchanged_request_reads_nothing(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    for _ in range(3):
        rig.controller.reslice_all()
        await rig.settle()
    assert reads.count == 1
    rig.set_t(2)
    await rig.settle()
    rig.set_t(2)
    await rig.settle()
    assert reads.count == 2
    assert rig.drawn() == rig.kind.element_of(2)


async def test_a_scrub_keeps_two_reads_in_flight_and_never_blanks(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    reads.hold = True
    frames: list[list[int] | None] = []
    for t in (1, 2, 3, 4, 5):
        rig.set_t(t)
        frames.append(rig.drawn())
        await rig.turns()
        frames.append(rig.drawn())
    assert reads.held() == 2
    assert reads.max_in_flight == 2
    # The previous result stays drawn while the next loads.
    assert all(frame is not None for frame in frames)
    reads.hold = False
    reads.release_all()
    await rig.settle()
    assert rig.drawn() == rig.kind.element_of(5)


async def test_a_superseded_compatible_arrival_is_drawn(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    reads.hold = True
    rig.set_t(1)
    await rig.turns()
    rig.set_t(2)
    await rig.turns()
    assert reads.held() == 2
    reads.release(0)  # position 1 lands after the slider left it
    await rig.until(lambda: rig.drawn() == rig.kind.element_of(1))
    reads.release(0)
    await rig.until(lambda: rig.drawn() == rig.kind.element_of(2))


async def test_an_incompatible_arrival_is_not_drawn(rig, reads):
    """A 2D read landing after a switch to 3D places nothing."""
    rig.controller.reslice_all()
    await rig.settle()
    reads.hold = True
    rig.set_t(1)
    await rig.turns()
    assert reads.held() == 1
    rig.controller.update_displayed_axes(rig.scene.id, (1, 2, 3))
    rig.controller.reslice_all()
    await rig.turns()
    assert rig.drawn() is None  # another placement: nothing is kept
    reads.release(0)  # the 2D read lands
    await rig.turns(20)
    assert rig.drawn() is None
    reads.hold = False
    reads.release_all()
    await rig.settle()
    assert rig.drawn() == rig.kind.element_of(1)


# -- store changes -----------------------------------------------------------------


async def test_a_contents_change_keeps_the_old_result_until_the_reread(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.count == 1
    reslices: list = []
    original = rig.controller.reslice_visual
    rig.controller.reslice_visual = lambda vid: (reslices.append(vid), original(vid))

    reads.hold = True
    rig.store.notify_changed("contents")
    await rig.turns()
    assert reads.held() == 1
    for _ in range(3):
        assert rig.drawn() == rig.kind.element_of(0)
        await rig.turns()
    reads.hold = False
    reads.release_all()
    await rig.settle()
    assert reads.count == 2
    assert rig.drawn() == rig.kind.element_of(0)
    assert reslices == []


async def test_a_burst_of_changes_reads_twice_at_most(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    for _ in range(20):
        rig.store.notify_changed("contents")
    await rig.settle()
    assert 2 <= reads.count <= 3
    assert rig.drawn() == rig.kind.element_of(0)


async def test_a_failed_read_completes_and_is_read_again_after_a_change(
    rig, reads, caplog
):
    reads.fail = RuntimeError("the store changed under the filter")
    ready: list[int] = []
    with caplog.at_level(logging.WARNING, logger="cellier"):
        rig.controller.reslice_scene(rig.scene.id, on_ready=lambda: ready.append(1))
        await rig.settle()
        for _ in range(2):
            rig.controller.reslice_all()
            await rig.settle()
    assert reads.count == 1  # one attempt
    assert ready == [1]  # a failed read still completes
    assert rig.drawn() is None
    progress = rig.controller._render_manager.loading_progress(rig.visual.id)
    assert progress.failed == 1

    reads.fail = None
    rig.store.notify_changed("contents")
    await rig.settle()
    assert reads.count == 2
    assert rig.drawn() == rig.kind.element_of(0)


# -- visibility, progress --------------------------------------------------------------


async def test_a_hidden_visual_is_retired_and_shown_again_reads_nothing(rig, reads):
    rig.controller.reslice_all()
    await rig.settle()
    rig.controller.update_appearance_field(rig.visual.id, "visible", False)
    await rig.settle()
    rig.controller.update_appearance_field(rig.visual.id, "visible", True)
    assert rig.drawn() == rig.kind.element_of(0)
    await rig.settle()
    assert reads.count == 1

    # A change while hidden is read once, when shown.
    rig.controller.update_appearance_field(rig.visual.id, "visible", False)
    await rig.settle()
    rig.store.notify_changed("contents")
    await rig.settle()
    rig.controller.update_appearance_field(rig.visual.id, "visible", True)
    assert rig.drawn() == rig.kind.element_of(0)  # Kept while it reloads
    await rig.settle()
    assert reads.count == 2
    assert rig.drawn() == rig.kind.element_of(0)


async def test_progress_is_reported(rig, reads):
    from cellier.events._events import ResliceProgressEvent

    events: list[ResliceProgressEvent] = []
    rig.controller._outgoing_events.subscribe(
        ResliceProgressEvent,
        events.append,
        entity_id=rig.visual.id,
        owner_id=rig.visual.id,
    )
    reads.hold = True
    rig.controller.reslice_all()
    await rig.turns()
    loading = rig.controller._render_manager.loading_progress(rig.visual.id)
    assert (loading.needed_target, loading.resident_target) == (1, 0)
    assert (loading.needed_backstop, loading.in_flight) == (0, 1)
    assert loading.backstop_complete
    assert not loading.complete
    reads.hold = False
    reads.release_all()
    await rig.settle()
    done = rig.controller._render_manager.loading_progress(rig.visual.id)
    assert (done.resident_target, done.in_flight) == (1, 0)
    assert done.complete
    await rig.turns()
    assert events and events[-1].progress.complete


async def test_the_first_load_holds_the_draw(rig, reads):
    """``awaits_data`` only before the first result (draw hold)."""
    reads.hold = True
    rig.controller.reslice_all()
    await rig.turns()
    assert rig.gfx.awaits_data
    reads.hold = False
    reads.release_all()
    await rig.settle()
    assert not rig.gfx.awaits_data
    reads.hold = True
    rig.set_t(3)
    await rig.turns()
    assert not rig.gfx.awaits_data  # kept: the previous result is drawn
    reads.hold = False
    reads.release_all()
    await rig.settle()


# -- graph trails --------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["mask", "roi"])
async def test_a_trail_change_keeps_the_previous_result_while_it_loads(
    monkeypatch, strategy
):
    """A trail is in the request key: a change reads again, kept meanwhile."""
    from cellier.visuals._graph_memory import TrailConfig

    kind = Kind("graph", GraphMemoryStore, strategy)
    reads = Reads(monkeypatch, GraphMemoryStore)
    rig = Rig(kind)
    try:
        rig.set_t(3)
        rig.controller.reslice_all()
        await rig.settle()
        assert rig.drawn() == [6]
        reads.hold = True
        rig.visual.trail = {0: TrailConfig(before=2.0, after=0.0)}
        await rig.turns()
        assert reads.held() == 1
        assert rig.drawn() == [6]  # kept while the trail loads
        reads.hold = False
        reads.release_all()
        await rig.settle()
        assert rig.drawn() == [2, 4, 6]
        assert reads.count == 2
    finally:
        rig.controller.close()
