"""The "Data fetch status" indicator for a mesh.

``plans/mesh_refactor_v3.md`` Phase 9.  A mesh is not drawn while a new
position loads, and the indicator is what says that it is loading.  A mesh
reads a level whole, so the text names the level, not a chunk count.
"""

from __future__ import annotations

import pytest

from cellier.data.mesh import MeshMemoryStore
from cellier.events import LoadingProgress
from cellier.gui._loading import indicator_state
from cellier.gui.qt.visuals import QtLoadingIndicator
from cellier.scene.dims import spatial_axes
from tests._meshes import uv_sphere
from tests.render.test_mesh_multiscale import LevelReads, Rig2D, _add, _landed, _store
from tests.render.test_mesh_section import Reads

# -- the state, without a toolkit -----------------------------------------------


def test_the_text_names_the_level_that_is_loading():
    coarse = LoadingProgress(
        needed_backstop=1,
        needed_target=1,
        backstop_complete=False,
        complete=False,
    )
    assert indicator_state(coarse, levels=True).text == "Loading coarse level"
    fine = coarse._replace(resident_backstop=1, backstop_complete=True)
    state = indicator_state(fine, levels=True)
    assert (state.text, state.busy) == ("Loading fine level", True)
    done = fine._replace(resident_target=1, complete=True)
    state = indicator_state(done, levels=True)
    assert (state.text, state.busy, state.value) == ("Loaded", False, 1)


def test_a_mesh_with_one_level_is_just_loading():
    loading = LoadingProgress(needed_target=1, complete=False)
    assert indicator_state(loading, levels=True).text == "Loading"
    failed = loading._replace(failed=1, complete=True)
    assert indicator_state(failed, levels=True).text == "Loaded, 1 failed"


def test_a_coarse_only_plan_says_the_fine_level_waits():
    scrub = LoadingProgress(
        needed_backstop=1, resident_backstop=1, target_deferred=True
    )
    state = indicator_state(scrub, levels=True)
    assert state.text == "Coarse level ready. Fine on stop."
    assert state.busy and state.value == 0


def test_the_chunk_wording_is_unchanged_by_default():
    progress = LoadingProgress(needed_target=4, resident_target=1, complete=False)
    assert indicator_state(progress).text == "Detail: 1 / 4"


# -- through a load -------------------------------------------------------------


@pytest.fixture
def reads(monkeypatch) -> LevelReads:
    return LevelReads(monkeypatch)


@pytest.fixture
def rig():
    rig = Rig2D(dim="3d")
    yield rig
    rig.controller.close()


def _indicator(rig, mesh, qtbot) -> QtLoadingIndicator:
    controller = rig.controller
    indicator = QtLoadingIndicator(
        mesh.id, initial={mesh.id: controller.loading_progress(mesh.id)}, levels=True
    )
    qtbot.addWidget(indicator.widget)
    controller.connect_widget(
        indicator, subscription_specs=indicator.subscription_specs()
    )
    return indicator


async def test_the_indicator_follows_a_multiscale_mesh_as_it_loads(rig, reads, qtbot):
    mesh = _add(rig, _store(12, 4))
    indicator = _indicator(rig, mesh, qtbot)
    assert indicator.text == "Not loaded"

    reads.hold = True
    rig.controller.reslice_all()
    await rig.until(lambda: sorted(reads.held()) == [0, 1])
    await rig.until(lambda: indicator.text == "Loading coarse level")

    reads.release(1)
    await _landed(rig, mesh, 1)
    await rig.until(lambda: indicator.text == "Loading fine level")
    assert indicator.state.busy

    reads.release(0)
    await _landed(rig, mesh, 0)
    await rig.until(lambda: indicator.text == "Loaded")
    assert not indicator.state.busy


async def test_a_scrub_says_the_fine_level_waits(reads, qtbot):
    rig = Rig2D(dim="3d", axes=[("t", "time"), *spatial_axes("z", "y", "x")])
    try:
        mesh = _add(rig, _store(12, 4))
        rig.controller.reslice_all()
        await rig.settle()
        indicator = _indicator(rig, mesh, qtbot)
        assert indicator.text == "Loaded"

        with rig.scrub():
            rig.controller.update_slice_indices(rig.scene.id, {0: 1.0})
            await rig.until(
                lambda: indicator.text == "Coarse level ready. Fine on stop."
            )
        await rig.settle()
        await rig.until(lambda: indicator.text == "Loaded", timeout=3.0)
    finally:
        rig.controller.close()


async def test_a_plain_mesh_says_loading_then_loaded(rig, monkeypatch, qtbot):
    held = Reads(monkeypatch)
    positions, indices = uv_sphere(10.0, (16.0, 16.0, 16.0))
    mesh = rig.add(MeshMemoryStore(positions=positions, indices=indices, name="m"))
    indicator = _indicator(rig, mesh, qtbot)

    held.hold = True
    rig.controller.reslice_all()
    await rig.until(lambda: indicator.text == "Loading")

    held.hold = False
    held.release_all()
    await rig.settle()
    await rig.until(lambda: indicator.text == "Loaded", timeout=3.0)
