"""The draw hold: a canvas keeps its picture while a hidden mesh loads.

``RenderManagerConfig.draw_hold_ms``.  After a dims change the display rule
hides a mesh until its read lands; the canvas skips frames for a short time
instead of drawing that gap.  Frames are real frames of an offscreen canvas.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from cellier.data.points._points_memory_store import PointsMemoryStore
from tests.render.test_mesh_loading import Reads
from tests.render.test_mesh_section import Rig2D, _load, _sphere


@pytest.fixture
def reads(monkeypatch) -> Reads:
    return Reads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig2D()
    yield rig
    rig.controller.close()


def _config(rig):
    return rig.controller._render_manager.config


async def _loaded(rig, hold_ms: float):
    _config(rig).draw_hold_ms = hold_ms
    mesh = rig.add(_sphere())
    await _load(rig)
    before = rig.red()
    assert before.sum() > 0
    return mesh, before


async def _turns(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def test_the_last_picture_is_kept_while_the_read_is_out(rig, reads):
    mesh, before = await _loaded(rig, 10_000.0)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()

    # The mesh is hidden (the display rule), and the frame is not drawn.
    assert rig.gfx(mesh).awaits_data
    np.testing.assert_array_equal(rig.red(), before)
    np.testing.assert_array_equal(rig.red(), before)

    reads.hold = False
    reads.release_all()
    await rig.settle()
    after = rig.red()
    assert not rig.gfx(mesh).awaits_data
    # The section at z = 22 is a smaller disc than the one at z = 16.
    assert 0 < after.sum() < before.sum()
    assert rig.view._hold_waiting is None


async def test_after_the_hold_time_the_frame_is_drawn_without_the_mesh(rig, reads):
    _mesh, before = await _loaded(rig, 20.0)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()

    np.testing.assert_array_equal(rig.red(), before)  # starts the clock
    await asyncio.sleep(0.05)
    assert rig.red().sum() == 0
    reads.hold = False
    reads.release_all()
    await rig.settle()


async def test_zero_turns_the_hold_off(rig, reads):
    await _loaded(rig, 0.0)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()

    assert rig.red().sum() == 0
    reads.hold = False
    reads.release_all()
    await rig.settle()


async def test_more_ticks_do_not_restart_the_clock(rig, reads):
    _mesh, before = await _loaded(rig, 40.0)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()
    np.testing.assert_array_equal(rig.red(), before)  # starts the clock

    await asyncio.sleep(0.03)
    rig.set_z(23.0)  # 30 ms in: a new tick, the same clock
    await _turns()
    await asyncio.sleep(0.03)
    assert rig.red().sum() == 0
    reads.hold = False
    reads.release_all()
    await rig.settle()


async def test_camera_input_ends_the_hold(rig, reads):
    await _loaded(rig, 10_000.0)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()
    assert rig.view._hold_waiting is not None

    rig.view._on_controller_input(SimpleNamespace(type="pointer_down", buttons=(1,)))

    assert rig.view._hold_waiting is None
    assert rig.red().sum() == 0
    reads.hold = False
    reads.release_all()
    await rig.settle()


async def test_a_visual_that_keeps_its_picture_starts_no_hold(rig):
    _config(rig).draw_hold_ms = 10_000.0
    coordinates = np.array([[16.0, 16.0, 16.0], [20.0, 12.0, 12.0]], dtype=np.float32)
    rig.controller.add_points(
        data=PointsMemoryStore(positions=coordinates, name="points"),
        scene_id=rig.scene.id,
        name="points",
    )
    rig.controller.reslice_all()
    await rig.settle()

    rig.set_z(20.0)
    await _turns()

    assert rig.view._hold_waiting is None


async def test_a_hidden_mesh_starts_no_hold(rig, reads):
    mesh, _before = await _loaded(rig, 10_000.0)
    rig.controller.set_visual_visible(mesh.id, False)
    reads.hold = True
    rig.set_z(22.0)
    await _turns()

    assert rig.view._hold_waiting is None
    reads.hold = False
    reads.release_all()
    await rig.settle()
