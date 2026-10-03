"""A multiscale mesh while the camera moves.

``plans/mesh_refactor_v3.md`` Phase 8 (I2).  In a 3D view the canvas whose
camera is moving draws the coarse level; both levels stay loaded, so nothing
is read or uploaded, and the finest is back in the frame after the motion
ends.  Headless, with the camera rig of ``test_camera_interaction.py``:
synthetic input and a fake clock.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from cellier.data.mesh import MeshLevel, MultiscaleMeshStore
from cellier.visuals import GeometryLodConfig, MeshFlatAppearance
from tests._meshes import uv_sphere
from tests.render.test_camera_interaction import NO_SETTLE, Rig, _Clock
from tests.render.test_mesh_multiscale import LevelReads

RED = (1.0, 0.0, 0.0, 1.0)


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    import pygfx.controllers._base as pygfx_controller_base

    clock = _Clock()
    monkeypatch.setattr(pygfx_controller_base, "perf_counter", clock)
    return clock


@pytest.fixture
def reads(monkeypatch) -> LevelReads:
    return LevelReads(monkeypatch)


@pytest.fixture
def make_rig(clock):
    rigs: list[Rig] = []

    def _make(**kwargs) -> Rig:
        rig = Rig(None, clock, with_image=False, **kwargs)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.controller.close()


def _store() -> MultiscaleMeshStore:
    levels = [
        MeshLevel(positions=positions, indices=indices)
        for positions, indices in (
            uv_sphere(10.0, (16.0, 16.0, 16.0), n_lat=n, n_lon=2 * n) for n in (16, 4)
        )
    ]
    return MultiscaleMeshStore(levels=levels, name="sphere")


class Drawn:
    """The level each rendered frame drew of a mesh, per canvas."""

    def __init__(self, rig: Rig, mesh) -> None:
        scenes = rig.controller._render_manager._scenes
        self.gfx = scenes[rig.scene.id].get_visual(mesh.id)
        #: ``(canvas index or None for a capture, level, changed)`` per frame.
        self.frames: list[tuple[int | None, int | None, bool]] = []
        self.uploads: list[int] = []
        inner = self.gfx.prepare_draw

        def prepare_draw(canvas_id, camera_moving, dims_scrubbing):
            changed = inner(canvas_id, camera_moving, dims_scrubbing)
            index = (
                rig.canvas_ids.index(canvas_id) if canvas_id in rig.canvas_ids else None
            )
            self.frames.append((index, self.gfx.drawn_level(), changed))
            return changed

        self.gfx.prepare_draw = prepare_draw
        inner_upload = self.gfx._residency._upload

        def upload(level, key, data):
            self.uploads.append(level)
            inner_upload(level, key, data)

        self.gfx._residency._upload = upload

    def levels(self, canvas: int | None = 0) -> list[int | None]:
        return [level for index, level, _ in self.frames if index == canvas]

    def clear(self) -> None:
        self.frames.clear()
        self.uploads.clear()


async def _loaded(rig: Rig, reads: LevelReads, lod=None, store=None):
    mesh = rig.controller.add_multiscale_mesh(
        data=store or _store(),
        scene_id=rig.scene.id,
        appearance=MeshFlatAppearance(color=RED),
        lod=lod,
    )
    await rig.settle_first_frames()
    drawn = Drawn(rig, mesh)
    reads.clear()
    return mesh, drawn


async def test_an_orbit_draws_coarse_while_it_moves_and_fine_after(make_rig, reads):
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)

    await rig.press()
    assert drawn.levels() in ([], [0])  # a press moves nothing
    drawn.clear()
    x = await rig.move(5)
    # Coarse from the first moved frame.
    assert drawn.levels() == [1] * 5
    assert rig.view.camera_moving is True

    drawn.clear()
    await rig.release(x)
    levels = drawn.levels()
    # Coarse through the damped tail, fine in the frame after it stops.
    assert levels[0] == 1 and levels[-1] == 0
    # Once fine is back it stays: no frame goes back to coarse.
    assert levels == sorted(levels, reverse=True)
    assert rig.view.camera_moving is False
    # Both levels were loaded all along.
    assert reads.started == []
    assert drawn.uploads == []


async def test_a_drag_held_still_shows_fine_after_the_settle_time(make_rig, reads):
    rig = make_rig(settle_s=0.05)
    _mesh, drawn = await _loaded(rig, reads)
    # Frames drawn without yielding, so a slow machine cannot settle early.
    x = rig.press_and_move_now(3, tail=True)
    assert drawn.levels()[-1] == 1

    for _ in range(400):
        await rig.run(1)
        if rig.events and rig.events[-1][1] == "end":
            break
        await asyncio.sleep(0.005)
    assert rig.events[-1] == (0, "end", "settle")
    await rig.run_until_idle()
    # The button is still down; the picture is the finest.
    assert drawn.levels()[-1] == 0

    drawn.clear()
    rig.controller._render_manager.config.camera.settle_threshold_s = NO_SETTLE
    x = await rig.move(2, start=x)
    assert drawn.levels() == [1, 1]
    await rig.release(x + 6.0)
    assert drawn.levels()[-1] == 0
    assert reads.started == []


async def test_accumulation_resets_in_the_frame_the_level_changes(make_rig, reads):
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)

    await rig.press()
    drawn.clear()
    x = await rig.move(4)
    await rig.release(x)

    changed = [(level, flag) for _, level, flag in drawn.frames]
    # The first coarse frame and the first fine frame, and no other.
    assert changed[0] == (1, True)
    assert next(entry for entry in changed if entry[0] == 0) == (0, True)
    assert sum(flag for _, flag in changed) == 2


async def test_orbiting_one_canvas_leaves_the_other_fine(make_rig, reads):
    rig = make_rig(n_canvases=2)
    _mesh, drawn = await _loaded(rig, reads)

    await rig.press(canvas=1)
    drawn.clear()
    x = await rig.move(4, canvas=1)
    # The still canvas is asked for a frame too: it must draw fine.
    rig.requested[0] = True
    await rig.run(1)

    assert set(drawn.levels(1)) == {1}
    assert drawn.levels(0) == [0]
    await rig.release(x, canvas=1)
    assert drawn.levels(1)[-1] == 0


async def test_a_screenshot_mid_orbit_draws_fine(make_rig, reads):
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)
    await rig.press()
    x = await rig.move(3)
    drawn.clear()

    shot = rig.controller.screenshot(rig.canvas_id)

    assert drawn.levels(None) and set(drawn.levels(None)) == {0}
    assert ((np.asarray(shot)[..., 0] > 150) & (np.asarray(shot)[..., 1] < 90)).any()
    # The window is still mid-orbit: its next frame draws coarse.
    x = await rig.move(1, start=x)
    assert drawn.levels(0) == [1]
    await rig.release(x)


async def test_fit_camera_mid_orbit_is_part_of_the_motion(make_rig, reads):
    """While the controller drives, a programmatic move is a tick of the motion.

    The tracker's rule (a move inside an open scope is a motion tick), so
    the canvas keeps drawing coarse until the motion ends.
    """
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)
    await rig.press()
    x = await rig.move(3)
    assert drawn.levels()[-1] == 1

    rig.controller.fit_camera(rig.scene.id)
    assert rig.view.camera_moving is True
    x = await rig.move(1, start=x)
    assert drawn.levels()[-1] == 1

    await rig.release(x)
    assert drawn.levels()[-1] == 0
    assert reads.started == []


async def test_a_jump_ends_the_motion_and_shows_fine(make_rig, reads):
    """A motion with no scope open (a scripted one) is ended by a jump."""
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)
    controller = rig.controller
    state = controller.get_camera_state(rig.canvas_id)
    moved = state._replace(position=tuple(p + 5.0 for p in state.position))
    controller.set_camera_state(rig.canvas_id, moved, interactive=True)
    assert rig.view.camera_moving is True
    rig.requested[0] = True
    await rig.run(1)
    assert drawn.levels()[-1] == 1

    controller.fit_camera(rig.scene.id)
    assert rig.view.camera_moving is False
    rig.requested[0] = True
    await rig.run_until_idle()

    assert drawn.levels()[-1] == 0
    assert reads.started == []


async def test_a_programmatic_move_never_draws_coarse(make_rig, reads):
    rig = make_rig()
    _mesh, drawn = await _loaded(rig, reads)

    state = rig.view.capture_camera_state()
    rig.controller.fit_camera(rig.scene.id)
    rig.controller.set_camera_state(rig.canvas_id, state)
    rig.requested[0] = True
    await rig.run_until_idle()

    assert set(drawn.levels()) <= {0}


async def test_panning_a_2d_view_never_switches_level(make_rig, reads):
    rig = make_rig(dim="2d")
    # Through the sphere's middle, so the 2D view has a section to draw.
    rig.controller.update_slice_indices(rig.scene.id, {0: 16.0})
    _mesh, drawn = await _loaded(rig, reads)

    await rig.press()
    drawn.clear()
    x = await rig.move(4)
    assert rig.view.camera_moving is True
    await rig.release(x)

    assert drawn.levels() and set(drawn.levels()) == {0}
    assert not any(flag for _, _, flag in drawn.frames)


async def test_camera_motion_full_never_switches_level(make_rig, reads):
    rig = make_rig()
    mesh, drawn = await _loaded(rig, reads, lod=GeometryLodConfig(camera_motion="full"))

    await rig.press()
    drawn.clear()
    x = await rig.move(3)
    assert set(drawn.levels()) == {0}

    # Set live, mid-orbit: the next frame draws coarse, with nothing read.
    rig.controller.set_lod_config(mesh.id, camera_motion="coarse")
    x = await rig.move(1, start=x)
    assert drawn.levels()[-1] == 1
    await rig.release(x)
    assert drawn.levels()[-1] == 0
    assert reads.started == []


async def test_with_the_coarse_level_given_up_an_orbit_draws_fine(
    make_rig, monkeypatch
):
    """A failed coarse read is not drawable, so the fine level is drawn."""
    inner = MultiscaleMeshStore.get_data

    async def get_data(store, request):
        if int(request.scale_index) == 1:
            raise OSError("no coarse level today")
        return await inner(store, request)

    monkeypatch.setattr(MultiscaleMeshStore, "get_data", get_data)
    rig = make_rig()
    mesh = rig.controller.add_multiscale_mesh(
        data=_store(),
        scene_id=rig.scene.id,
        appearance=MeshFlatAppearance(color=RED),
    )
    await rig.settle_first_frames()
    drawn = Drawn(rig, mesh)
    residency = drawn.gfx._residency
    assert residency.is_drawable(0) and not residency.is_drawable(1)

    await rig.press()
    x = await rig.move(4)
    assert rig.view.camera_moving is True
    await rig.release(x)

    assert drawn.levels() and set(drawn.levels()) == {0}
