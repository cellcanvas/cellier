"""A mesh with two levels of detail, through the controller.

``plans/mesh_refactor_v3.md`` Phase 6 (M3, M4's coarse child, L4).  A plan
asks for the coarse and the fine level together; the coarse one is on screen
first and the fine one replaces it, with no frame without the mesh between.
Frames are real frames of an offscreen canvas.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.data.mesh import MeshLevel, MeshMemoryStore, MultiscaleMeshStore
from cellier.events import MeshPickInfo
from cellier.visuals import (
    GeometryLodConfig,
    MeshFlatAppearance,
    MultiscaleMeshVisual,
)
from tests._meshes import uv_sphere
from tests.render.test_mesh_section import Rig2D

RED = (1.0, 0.0, 0.0, 1.0)
CENTRE = (16.0, 16.0, 16.0)
RADIUS = 10.0


class LevelReads:
    """Every read of a multiscale mesh, optionally held until released."""

    def __init__(self, monkeypatch) -> None:
        #: The level of each read, in the order they started.
        self.started: list[int] = []
        #: ``(store name, level)`` of each read, in the same order.
        self.started_by: list[tuple[str, int]] = []
        self.results: list = []
        self.hold = False
        self._gates: list[tuple[str, int, asyncio.Future]] = []
        inner = MultiscaleMeshStore.get_data
        reads = self

        async def get_data(store, request):
            level = int(request.scale_index)
            reads.started.append(level)
            reads.started_by.append((store.name, level))
            if reads.hold:
                gate = asyncio.get_running_loop().create_future()
                reads._gates.append((store.name, level, gate))
                await gate
            result = await inner(store, request)
            reads.results.append(result)
            return result

        monkeypatch.setattr(MultiscaleMeshStore, "get_data", get_data)

    def of(self, name: str) -> list[int]:
        """The levels read from the store called *name*, in order."""
        return [level for store, level in self.started_by if store == name]

    def clear(self) -> None:
        self.started.clear()
        self.started_by.clear()
        self.results.clear()

    def held(self) -> list[int]:
        """The levels of the reads waiting to be released."""
        return [level for _, level, _ in self._gates]

    def release(self, level: int, name: str | None = None) -> None:
        """Let the oldest held read of *level* (of store *name*) run."""
        for index, (store, held_level, gate) in enumerate(self._gates):
            if held_level == level and name in (None, store):
                del self._gates[index]
                gate.set_result(None)
                return
        raise AssertionError(f"no read of level {level} is held: {self.held()}")

    def release_all(self) -> None:
        self.hold = False
        while self._gates:
            self._gates.pop()[2].set_result(None)


@pytest.fixture
def reads(monkeypatch) -> LevelReads:
    return LevelReads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig2D(dim="3d")
    yield rig
    rig.controller.close()


@pytest.fixture
def rig_2d():
    rig = Rig2D()
    yield rig
    rig.controller.close()


def _level(n_lat: int, radius: float = RADIUS) -> MeshLevel:
    positions, indices = uv_sphere(radius, CENTRE, n_lat=n_lat, n_lon=2 * n_lat)
    return MeshLevel(positions=positions, indices=indices)


def _store(*n_lats: int, name: str = "sphere") -> MultiscaleMeshStore:
    return MultiscaleMeshStore(levels=[_level(n) for n in n_lats], name=name)


def _add(rig, store, lod=None, **kwargs):
    return rig.controller.add_multiscale_mesh(
        data=store,
        scene_id=rig.scene.id,
        appearance=MeshFlatAppearance(color=RED),
        name=store.name,
        lod=lod,
        **kwargs,
    )


def _drawn_level(rig, mesh) -> int | None:
    """The level the next frame draws; ``None`` when the mesh is not drawn."""
    rig.frame()
    return rig.gfx(mesh).drawn_level()


async def _landed(rig, mesh, level: int) -> None:
    """Wait until *level* holds what the newest plan asked of it."""
    residency = rig.gfx(mesh)._residency

    def holds() -> bool:
        rig.frame()
        return residency.is_drawable(level)

    await rig.until(holds)


# -- what is kept resident ----------------------------------------------------


async def test_the_finest_and_the_coarsest_are_resident_by_default(rig, reads):
    mesh = _add(rig, _store(12, 8, 4))
    assert isinstance(mesh, MultiscaleMeshVisual)
    gfx = rig.gfx(mesh)
    assert gfx.n_levels == 2
    assert gfx.resident_levels == (0, 2)

    rig.controller.reslice_all()
    await rig.settle()
    assert sorted(reads.started) == [0, 2]


async def test_coarse_level_names_the_level_kept(rig, reads):
    mesh = _add(rig, _store(12, 8, 4), lod=GeometryLodConfig(coarse_level=2))
    assert rig.gfx(mesh).resident_levels == (0, 1)

    rig.controller.reslice_all()
    await rig.settle()
    assert sorted(reads.started) == [0, 1]


def test_a_coarse_level_the_store_lacks_is_refused_and_nothing_is_added(rig):
    store = _store(12, 4)
    with pytest.raises(ValueError, match="has 2 levels"):
        _add(rig, store, lod=GeometryLodConfig(coarse_level=3))
    assert rig.scene.visuals == []
    assert store.id not in rig.controller._model.data.stores


async def test_a_store_of_one_level_loads_as_a_plain_mesh(rig, reads):
    mesh = _add(rig, _store(12))
    gfx = rig.gfx(mesh)
    assert gfx.resident_levels == (0,)

    rig.controller.reslice_all()
    await rig.settle()
    assert reads.started == [0]
    assert _drawn_level(rig, mesh) == 0


def test_a_multiscale_visual_needs_a_multiscale_store(rig):
    positions, indices = uv_sphere(RADIUS, CENTRE)
    store = MeshMemoryStore(positions=positions, indices=indices)
    model = MultiscaleMeshVisual(
        name="mesh",
        data_store_id=str(store.id),
        appearance=MeshFlatAppearance(),
    )
    with pytest.raises(TypeError, match="needs a MultiscaleMeshStore"):
        rig.controller.add_visual(rig.scene.id, model, data_store=store)


# -- coarse first, then fine --------------------------------------------------


@pytest.mark.parametrize("which", ["3d", "2d"])
async def test_coarse_is_drawn_first_and_fine_replaces_it(which, request, reads):
    rig = request.getfixturevalue("rig" if which == "3d" else "rig_2d")
    mesh = _add(rig, _store(24, 6))
    reads.hold = True
    rig.set_z(16.0)
    rig.controller.reslice_all()
    # The fit frames the store's extent, so it needs nothing loaded.
    rig.controller.fit_camera(rig.scene.id)
    await rig.turns()

    # Both reads start in the same round, the coarse one first (D24).
    assert reads.started == [1, 0]
    assert _drawn_level(rig, mesh) is None

    reads.release(1)
    await _landed(rig, mesh, 1)
    # The coarse level is on screen while the fine read is still out.
    assert reads.held() == [0]
    for _ in range(3):
        assert _drawn_level(rig, mesh) == 1
    assert rig.red().sum() > 0

    reads.release(0)
    residency = rig.gfx(mesh)._residency
    drawn = []
    async with asyncio.timeout(10.0):
        while not residency.is_drawable(0):
            drawn.append(_drawn_level(rig, mesh))
            await asyncio.sleep(0.002)
    drawn.append(_drawn_level(rig, mesh))
    # No frame without the mesh between coarse and fine, and never both.
    assert None not in drawn
    assert drawn[-1] == 0
    assert set(drawn) <= {0, 1}
    nodes = rig.gfx(mesh)._levels
    shown = [
        level
        for level, child in nodes.items()
        if child.mesh_3d.visible or child.group_2d.visible
    ]
    assert shown == [0]
    assert rig.red().sum() > 0
    reads.release_all()


async def test_a_fine_read_that_lands_first_is_drawn_directly(rig, reads):
    mesh = _add(rig, _store(24, 6))
    reads.hold = True
    rig.controller.reslice_all()
    await rig.turns()

    reads.release(0)
    await _landed(rig, mesh, 0)
    assert _drawn_level(rig, mesh) == 0

    reads.release(1)
    await _landed(rig, mesh, 1)
    assert _drawn_level(rig, mesh) == 0
    reads.release_all()


async def test_a_new_position_hides_both_levels_until_one_loads(rig_2d, reads):
    rig = rig_2d
    mesh = _add(rig, _store(24, 6))
    rig.set_z(16.0)
    rig.controller.reslice_all()
    await rig.settle()
    assert _drawn_level(rig, mesh) == 0

    reads.hold = True
    rig.set_z(20.0)
    await rig.turns()
    # Neither level may draw the plane the slider left.
    rig.controller._render_manager.config.draw_hold_ms = 0.0
    assert _drawn_level(rig, mesh) is None
    assert rig.gfx(mesh).awaits_data

    reads.release(1)
    await _landed(rig, mesh, 1)
    assert _drawn_level(rig, mesh) == 1
    # The coarse picture is a picture: nothing is awaited any more.
    assert not rig.gfx(mesh).awaits_data
    reads.release_all()
    await rig.settle()
    assert _drawn_level(rig, mesh) == 0


async def test_an_unchanged_position_reads_neither_level_again(rig, reads):
    mesh = _add(rig, _store(24, 6))
    rig.controller.reslice_all()
    await rig.settle()
    reads.started.clear()

    rig.controller.reslice_all()
    await rig.settle()

    assert reads.started == []
    assert _drawn_level(rig, mesh) == 0


async def test_a_store_change_reads_both_levels_again(rig, reads):
    store = _store(24, 6)
    mesh = _add(rig, store)
    rig.controller.reslice_all()
    await rig.settle()
    reads.started.clear()

    store.levels = [_level(20, radius=6.0), _level(6, radius=6.0)]
    await rig.settle()

    assert sorted(reads.started) == [0, 1]
    assert _drawn_level(rig, mesh) == 0
    assert len(rig.gfx(mesh)._levels[0].mesh_3d.geometry.indices.data) == len(
        store.levels[0].indices
    )


async def test_a_hidden_multiscale_mesh_reads_nothing(rig, reads):
    mesh = _add(rig, _store(24, 6))
    rig.controller.set_visual_visible(mesh.id, False)
    rig.controller.reslice_all()
    await rig.settle()
    assert reads.started == []

    rig.controller.set_visual_visible(mesh.id, True)
    await rig.settle()
    assert sorted(reads.started) == [0, 1]


# -- picks, bounds ------------------------------------------------------------


async def test_a_pick_reports_the_level_that_was_drawn(rig, reads):
    mesh = _add(rig, _store(24, 8, 6))
    rig.controller.reslice_all()
    await rig.settle()
    gfx = rig.gfx(mesh)
    fine, coarse = gfx._levels[0], gfx._levels[2]

    assert gfx.decode_pick(fine.mesh_3d, {"face_index": 3}) == MeshPickInfo(
        face_index=3, part="face", level=0
    )
    # The coarse child's faces are numbered in the coarse level.
    assert gfx.decode_pick(coarse.mesh_3d, {"face_index": 3}) == MeshPickInfo(
        face_index=3, part="face", level=2
    )
    details = rig.controller._render_manager._extract_pick_details(
        rig.scene.id, coarse.mesh_3d, {"face_index": 5}
    )
    assert details == MeshPickInfo(face_index=5, part="face", level=2)


async def test_a_section_pick_reports_the_level_too(rig_2d, reads):
    rig = rig_2d
    mesh = _add(rig, _store(24, 6))
    rig.set_z(16.0)
    rig.controller.reslice_all()
    await rig.settle()
    gfx = rig.gfx(mesh)
    coarse = gfx._levels[1]

    assert gfx.decode_pick(coarse.fill, {"face_index": 0}) == MeshPickInfo(
        face_index=None, part="fill", level=1
    )
    outline = gfx.decode_pick(coarse.outline, {"vertex_index": 4})
    assert (outline.part, outline.level) == ("outline", 1)
    # The face is one the plane crosses, in the coarse level's numbering.
    store = next(iter(rig.controller._model.data.stores.values()))
    level = store.levels[1]
    crossed = level.positions[level.indices[outline.face_index], 0]
    assert crossed.min() <= 16.0 <= crossed.max()


async def test_bounds_and_the_camera_fit_use_the_finest_level(rig, reads):
    # A coarse level that reaches further than the fine one.
    store = MultiscaleMeshStore(levels=[_level(24, RADIUS), _level(6, 1.6 * RADIUS)])
    mesh = _add(rig, store)
    gfx = rig.gfx(mesh)
    reads.hold = True
    rig.controller.reslice_all()
    await rig.turns()

    low, high = gfx._local_extent()
    fine = store.levels[0].positions[:, ::-1]
    np.testing.assert_allclose(low, fine.min(axis=0))
    np.testing.assert_allclose(high, fine.max(axis=0))

    def fit():
        rig.controller.fit_camera(rig.scene.id)
        return rig.view.capture_camera_state()

    nothing_loaded = fit()
    reads.release(0)
    await _landed(rig, mesh, 0)
    assert fit() == nothing_loaded
    reads.release_all()
    await rig.settle()


# -- the section at both levels -----------------------------------------------


async def test_an_open_coarse_level_draws_its_outline_only(rig_2d, reads):
    rig = rig_2d
    fine = _level(24)
    positions, indices = uv_sphere(RADIUS, CENTRE, n_lat=6, n_lon=12)
    # Drop the faces of one quarter turn: the coarse cut no longer closes.
    centroid = positions[indices].mean(axis=1)
    keep = ~((centroid[:, 1] > CENTRE[1]) & (centroid[:, 2] > CENTRE[2]))
    coarse = MeshLevel(positions=positions, indices=indices[keep])
    mesh = _add(rig, MultiscaleMeshStore(levels=[fine, coarse]))
    reads.hold = True
    rig.set_z(16.5)
    rig.controller.reslice_all()
    rig.controller.fit_camera(rig.scene.id)
    await rig.turns()

    reads.release(1)
    await _landed(rig, mesh, 1)
    assert _drawn_level(rig, mesh) == 1
    nodes = rig.gfx(mesh)._levels[1]
    assert nodes.outline.visible
    assert not nodes.fill.visible
    ring = rig.red().sum()
    assert ring > 0

    reads.release(0)
    await _landed(rig, mesh, 0)
    assert _drawn_level(rig, mesh) == 0
    fine_nodes = rig.gfx(mesh)._levels[0]
    assert fine_nodes.outline.visible and fine_nodes.fill.visible
    # The fine level's cut closes, so it is a disc and not a ring.
    assert rig.red().sum() > 2 * ring
    reads.release_all()


@pytest.mark.parametrize("displayed", [(1, 2), (0, 2), (0, 1)], ids=["yx", "zx", "zy"])
async def test_a_large_sphere_is_closed_in_every_2d_view_at_both_levels(
    displayed, rig_2d, reads
):
    """About 1M faces with a 100k coarse level: each cut is one closed loop."""
    rig = rig_2d
    fine, coarse = _level(500), _level(160)
    assert fine.n_faces > 990_000 and 95_000 < coarse.n_faces < 105_000
    mesh = _add(rig, MultiscaleMeshStore(levels=[fine, coarse]))
    rig.controller.set_displayed_axes(rig.scene.id, displayed)
    sliced = next(axis for axis in range(3) if axis not in displayed)
    reads.hold = True
    rig.controller.update_slice_indices(rig.scene.id, {sliced: 13.3})
    rig.controller.reslice_all()
    await rig.turns()

    reads.release(1)
    await _landed(rig, mesh, 1)
    assert _drawn_level(rig, mesh) == 1
    coarse_nodes = rig.gfx(mesh)._levels[1]
    assert coarse_nodes.fill.visible and coarse_nodes.outline.visible

    reads.release(0)
    await _landed(rig, mesh, 0)
    assert _drawn_level(rig, mesh) == 0
    fine_nodes = rig.gfx(mesh)._levels[0]
    assert fine_nodes.fill.visible and fine_nodes.outline.visible

    assert sorted(result.level for result in reads.results) == [0, 1]
    for result in reads.results:
        assert result.n_closed_loops == 1
        assert result.n_open_segments == 0
    reads.release_all()


# -- the viewer ---------------------------------------------------------------


def test_the_viewer_adds_a_multiscale_mesh_with_its_controls(qtbot):
    from cellier.convenience import Viewer
    from cellier.convenience.gui import MeshControlsConfig
    from cellier.convenience.layout._shared import appearance_specs
    from cellier.scene.dims import spatial_axes

    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    controls = MeshControlsConfig(appearance=True, section_controls=True)
    mesh = viewer.add_multiscale_mesh(
        _store(12, 4),
        MeshFlatAppearance(color=RED),
        lod=GeometryLodConfig(dims_drag="full"),
        controls=controls,
    )

    assert isinstance(mesh, MultiscaleMeshVisual)
    assert mesh.lod.dims_drag == "full"
    assert viewer.scene.visuals == [mesh]
    # The dock builds the same controls as for a plain mesh.
    kinds = [spec.kind for spec in appearance_specs(mesh, controls).specs]
    assert "mesh_section" in kinds
    viewer.controller.close()
