"""A multiscale mesh during a dims scrub, through the controller.

``plans/mesh_refactor_v3.md`` Phase 7 (I1, D23).  A scrub tick plans a
multiscale mesh coarse only; the scrub's end plans the finest.  A mesh the
scrub does not change keeps both levels and is drawn coarse while it lasts.
Scrubs are ticked with ``interactive=True`` or inside a scope; frames are
real frames of an offscreen canvas.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.data.mesh import MeshLevel, MultiscaleMeshStore
from cellier.visuals import GeometryLodConfig, MeshFlatAppearance, MeshSectionConfig
from tests._meshes import uv_sphere
from tests.render.test_mesh_loading import _TZYX, Rig
from tests.render.test_mesh_multiscale import LevelReads

N_T = 6
RED = (1.0, 0.0, 0.0, 1.0)
CENTRE = (16.0, 16.0, 16.0)
#: Faces per timepoint in the series' fine and coarse level.
FACES = {}


def _sphere(n_lat: int, radius: float = 10.0):
    return uv_sphere(radius, CENTRE, n_lat=n_lat, n_lon=2 * n_lat)


def _series_level(n_lat: int) -> MeshLevel:
    """One sphere per timepoint, ``(t, z, y, x)``; it grows with ``t``."""
    positions, indices, offset = [], [], 0
    for t in range(N_T):
        zyx, faces = _sphere(n_lat, radius=6.0 + t)
        column = np.full((len(zyx), 1), float(t), dtype=np.float32)
        positions.append(np.concatenate([column, zyx], axis=1))
        indices.append(faces + offset)
        offset += len(zyx)
    FACES[n_lat] = len(faces)
    return MeshLevel(
        positions=np.concatenate(positions), indices=np.concatenate(indices)
    )


def _series_store() -> MultiscaleMeshStore:
    return MultiscaleMeshStore(
        levels=[_series_level(12), _series_level(4)], name="series"
    )


def _static_store(name: str = "static") -> MultiscaleMeshStore:
    """No ``t`` column: the ``t`` slider does not change what it reads."""
    levels = [MeshLevel(positions=p, indices=i) for p, i in (_sphere(12), _sphere(4))]
    return MultiscaleMeshStore(levels=levels, name=name)


def _add(rig, store, scene=None, **kwargs):
    return rig.controller.add_multiscale_mesh(
        data=store,
        scene_id=(scene or rig.scene).id,
        appearance=MeshFlatAppearance(color=RED),
        name=store.name,
        **kwargs,
    )


def _drawn_level(rig, mesh) -> int | None:
    rig.frame()
    return rig.gfx(mesh).drawn_level()


def _drawn_timepoint(rig, mesh) -> int | None:
    """The timepoint of the series the next frame draws, from its faces."""
    level = _drawn_level(rig, mesh)
    if level is None:
        return None
    nodes = rig.gfx(mesh)._levels[level]
    faces = np.asarray(nodes.original_face_indices)
    per_timepoint = FACES[12 if level == 0 else 4]
    timepoints = set((faces // per_timepoint).tolist())
    assert len(timepoints) == 1
    return timepoints.pop()


def _spy_uploads(rig, mesh) -> list[int]:
    """The levels uploaded to *mesh* from now on."""
    gfx = rig.gfx(mesh)
    uploads: list[int] = []
    inner = gfx._residency._upload

    def upload(level, key, data):
        uploads.append(level)
        inner(level, key, data)

    gfx._residency._upload = upload
    return uploads


@pytest.fixture
def reads(monkeypatch) -> LevelReads:
    return LevelReads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig()
    yield rig
    rig.controller.close()


async def _loaded(rig, reads, *stores, **kwargs):
    meshes = [_add(rig, store, **kwargs) for store in stores]
    rig.controller.reslice_all()
    await rig.settle()
    reads.clear()
    return meshes


def test_the_scrub_opt_in_follows_dims_drag(rig):
    coarse = _add(rig, _static_store("a"))
    full = _add(rig, _static_store("b"), lod=GeometryLodConfig(dims_drag="full"))
    assert coarse.plans_coarse_on_scrub
    assert not full.plans_coarse_on_scrub


# -- a mesh the scrub changes -------------------------------------------------


async def test_a_scrub_reads_coarse_per_tick_and_fine_once_at_its_end(rig, reads):
    (series,) = await _loaded(rig, reads, _series_store())

    with rig.controller.dims_interaction(rig.scene.id):
        for t in (1, 2, 3):
            rig.set_t(t)
            await rig.until(lambda: rig.gfx(series)._residency.is_drawable(1))
            # Each tick's coarse picture, and never another position.
            assert _drawn_level(rig, series) == 1
            assert _drawn_timepoint(rig, series) == t
        assert reads.started == [1, 1, 1]
    await rig.settle()

    # The end reads the finest once, and the coarse level not again.
    assert reads.started == [1, 1, 1, 0]
    assert _drawn_level(rig, series) == 0
    assert _drawn_timepoint(rig, series) == 3


async def test_no_scrub_frame_draws_a_position_the_slider_left(rig, reads):
    (series,) = await _loaded(rig, reads, _series_store())
    rig.controller._render_manager.config.draw_hold_ms = 0.0
    seen = []

    with rig.controller.dims_interaction(rig.scene.id):
        for t in (1, 2, 3, 4, 3, 2):
            rig.set_t(t)
            for _ in range(4):
                seen.append((t, _drawn_timepoint(rig, series)))
                await asyncio.sleep(0.002)
    await rig.settle()

    assert all(drawn in (None, t) for t, drawn in seen)
    assert _drawn_timepoint(rig, series) == 2


async def test_a_jump_reads_coarse_and_fine_at_once(rig, reads):
    (series,) = await _loaded(rig, reads, _series_store())

    rig.set_t(4)
    await rig.settle()

    assert sorted(reads.started) == [0, 1]
    assert _drawn_level(rig, series) == 0
    assert _drawn_timepoint(rig, series) == 4


async def test_dims_drag_full_reads_both_levels_on_every_tick(rig, reads):
    (series,) = await _loaded(
        rig, reads, _series_store(), lod=GeometryLodConfig(dims_drag="full")
    )

    with rig.controller.dims_interaction(rig.scene.id):
        for t in (1, 2):
            rig.set_t(t)
            await rig.until(lambda: rig.gfx(series)._residency.is_drawable(0))
        assert sorted(reads.started) == [0, 0, 1, 1]
    await rig.settle()

    # Nothing was left for the end to load.
    assert sorted(reads.started) == [0, 0, 1, 1]


async def test_a_section_change_mid_scrub_reads_coarse_and_the_end_reads_fine():
    from tests.render.test_mesh_section import Rig2D

    rig = Rig2D()
    try:
        store = _static_store()
        mesh = _add(rig, store, section=MeshSectionConfig())
        rig.set_z(16.0)
        rig.controller.reslice_all()
        await rig.settle()
        # Patched by hand: the fixture's monkeypatch is not in scope here.
        started: list[int] = []
        inner = MultiscaleMeshStore.get_data

        async def get_data(self, request):
            started.append(int(request.scale_index))
            return await inner(self, request)

        MultiscaleMeshStore.get_data = get_data
        try:
            with rig.controller.dims_interaction(rig.scene.id):
                rig.set_z(17.0)
                await rig.until(lambda: rig.gfx(mesh)._residency.is_drawable(1))
                rig.controller.update_section_field(mesh.id, "fill", False)
                await rig.until(lambda: rig.gfx(mesh)._residency.is_drawable(1))
                assert started == [1, 1]
            await rig.settle()
            assert started == [1, 1, 0]
        finally:
            MultiscaleMeshStore.get_data = inner
        assert not rig.gfx(mesh)._levels[0].fill.visible
    finally:
        rig.controller.close()


async def test_a_coarse_read_starts_while_a_fine_read_is_in_flight(rig, reads):
    (series,) = await _loaded(rig, reads, _series_store())
    reads.hold = True
    rig.set_t(1)  # a jump: coarse and fine reads go out
    await rig.turns()
    reads.release(1)
    await rig.until(lambda: rig.gfx(series)._residency.is_drawable(1))
    assert reads.held() == [0]

    # A scrub resumes while the fine read of t = 1 is still out.
    rig.set_t(2, interactive=True)
    await rig.turns()
    assert reads.held() == [0, 1]

    reads.release(1)
    await rig.until(lambda: rig.gfx(series)._residency.is_drawable(1))
    # Back within one coarse read, with the slow fine read still out.
    assert reads.held() == [0]
    assert _drawn_timepoint(rig, series) == 2
    reads.release_all()
    await rig.settle()
    assert _drawn_level(rig, series) == 0
    assert _drawn_timepoint(rig, series) == 2


async def test_steps_apart_each_come_back_within_a_coarse_read(rig, reads):
    """Single steps slower than the settle time: each is its own scrub."""
    (series,) = await _loaded(rig, reads, _series_store())
    settle = rig.controller._render_manager.config.scheduler.dims_settle_s
    reads.hold = True

    for t in (1, 2, 3):
        rig.set_t(t, interactive=True)
        await rig.turns()
        reads.release(1)
        await rig.until(lambda: rig.gfx(series)._residency.is_drawable(1))
        assert _drawn_timepoint(rig, series) == t
        # The scrub ends by stillness and asks for the finest, held here.
        await asyncio.sleep(settle + 0.1)
        await rig.turns()
        assert _drawn_timepoint(rig, series) == t

    reads.release_all()
    await rig.settle()
    assert _drawn_level(rig, series) == 0
    assert _drawn_timepoint(rig, series) == 3


# -- a mesh the scrub does not change (D23) -----------------------------------


@pytest.mark.parametrize("draw", ["coarse", "full"])
async def test_a_static_mesh_reads_nothing_and_is_never_hidden(draw, rig, reads):
    (static,) = await _loaded(
        rig, reads, _static_store(), lod=GeometryLodConfig(dims_drag_draw=draw)
    )
    series = _add(rig, _series_store())
    rig.controller.reslice_all()
    await rig.settle()
    reads.clear()
    uploads = _spy_uploads(rig, static)
    # Every frame is rendered here.  A frame the draw hold skips (the series
    # beside it is loading) renders nothing, so what it would have drawn of
    # the static mesh is not a picture and is not looked at.
    rig.controller._render_manager.config.draw_hold_ms = 0.0
    during = []

    with rig.controller.dims_interaction(rig.scene.id):
        for t in (1, 2, 3):
            rig.set_t(t)
            for _ in range(3):
                during.append(_drawn_level(rig, static))
                await asyncio.sleep(0.002)
    # The frame after the end draws the finest again.
    after = _drawn_level(rig, static)
    await rig.settle()

    assert reads.of("static") == []
    assert uploads == []
    assert set(during) == ({1} if draw == "coarse" else {0})
    assert after == 0
    assert _drawn_level(rig, static) == 0
    # The series beside it did scrub.
    assert reads.of("series").count(1) == 3
    assert _drawn_level(rig, series) == 0


async def test_dims_drag_draw_applies_in_the_next_frame_with_no_read(rig, reads):
    (static,) = await _loaded(rig, reads, _static_store())
    resets = []
    inner_reset = rig.view._accum_pass.reset
    rig.view._accum_pass.reset = lambda: (resets.append(1), inner_reset())[1]

    with rig.controller.dims_interaction(rig.scene.id):
        rig.set_t(1)
        assert _drawn_level(rig, static) == 1
        resets.clear()
        assert _drawn_level(rig, static) == 1
        assert resets == []  # nothing changed: the history is kept

        config = rig.controller.set_lod_config(static.id, dims_drag_draw="full")
        assert config.dims_drag_draw == "full"
        assert static.lod is config
        # The frame that draws another level discards the accumulation.
        assert _drawn_level(rig, static) == 0
        assert resets

        rig.controller.set_lod_config(static.id, dims_drag_draw="coarse")
        assert _drawn_level(rig, static) == 1
    assert _drawn_level(rig, static) == 0
    await rig.settle()

    assert reads.started == []


async def test_a_capture_mid_scrub_draws_the_finest(rig, reads):
    (static,) = await _loaded(rig, reads, _static_store())
    rig.controller.fit_camera(rig.scene.id)
    gfx = rig.gfx(static)
    flags = []
    inner = gfx.prepare_draw

    def prepare_draw(canvas_id, camera_moving, dims_scrubbing):
        changed = inner(canvas_id, camera_moving, dims_scrubbing)
        flags.append((canvas_id == rig.canvas_id, dims_scrubbing, gfx.drawn_level()))
        return changed

    gfx.prepare_draw = prepare_draw

    with rig.controller.dims_interaction(rig.scene.id):
        rig.set_t(1)
        assert _drawn_level(rig, static) == 1
        flags.clear()
        shot = rig.controller.screenshot(rig.canvas_id)
        assert flags
        assert all(entry == (False, False, 0) for entry in flags)
        # The window still draws coarse after the screenshot.
        assert _drawn_level(rig, static) == 1
    assert ((shot[..., 0] > 150) & (shot[..., 1] < 90)).sum() > 0
    await rig.settle()


async def test_a_scrub_of_one_scene_leaves_another_scene_fine(rig, reads):
    """Two scenes on one store, as the panels of an ortho viewer are."""
    store = _static_store()
    other = rig.controller.add_scene(dim="3d", coordinate_system=_TZYX, name="other")
    rig.controller.add_canvas(other.id, canvas_size=(120, 90))
    other_view = rig.controller.get_canvas_view(
        rig.controller.get_canvas_ids(other.id)[0]
    )
    here = _add(rig, store)
    there = _add(rig, store, scene=other)
    rig.controller.reslice_all()
    await rig.settle()
    scenes = rig.controller._render_manager._scenes
    there_gfx = scenes[other.id].get_visual(there.id)

    with rig.controller.dims_interaction(rig.scene.id):
        rig.set_t(1)
        assert _drawn_level(rig, here) == 1
        other_view._canvas.draw()
        assert there_gfx.drawn_level() == 0
    await rig.settle()


# -- the setter ---------------------------------------------------------------


def test_set_lod_config_refuses_what_it_cannot_apply(rig):
    from cellier.data.mesh import MeshMemoryStore

    static = _add(rig, _static_store())
    with pytest.raises(ValueError, match="coarse_level cannot be changed"):
        rig.controller.set_lod_config(static.id, coarse_level=2)
    with pytest.raises(ValueError, match="Unknown GeometryLodConfig"):
        rig.controller.set_lod_config(static.id, speed="fast")
    with pytest.raises(ValueError):
        rig.controller.set_lod_config(static.id, dims_drag="sometimes")
    assert static.lod == GeometryLodConfig()

    positions, indices = _sphere(6)
    plain = rig.controller.add_mesh(
        data=MeshMemoryStore(positions=positions, indices=indices),
        scene_id=rig.scene.id,
        appearance=MeshFlatAppearance(),
    )
    with pytest.raises(TypeError, match="only a multiscale mesh"):
        rig.controller.set_lod_config(plain.id, dims_drag="full")


async def test_dims_drag_set_live_is_read_by_the_next_scrub(rig, reads):
    (series,) = await _loaded(rig, reads, _series_store())

    rig.controller.set_lod_config(series.id, dims_drag="full")
    with rig.controller.dims_interaction(rig.scene.id):
        rig.set_t(1)
        await rig.until(lambda: rig.gfx(series)._residency.is_drawable(0))
    await rig.settle()

    assert sorted(reads.started) == [0, 1]
