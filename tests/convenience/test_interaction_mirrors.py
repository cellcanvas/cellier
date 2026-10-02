"""``Viewer`` and ``OrthoViewer`` mirrors of the interaction API (design 4.8)."""

from __future__ import annotations

import numpy as np
import pytest
from cmap import Colormap

from cellier.convenience import OrthoViewer, Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageAppearance, InMemoryImageSingleAppearance

_WORLD = [("t", "time"), *spatial_axes("z", "y", "x")]


def _add_image(viewer) -> None:
    viewer.add_image(
        ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32)),
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map=Colormap("gray")),
    )


def _dims_events(viewer, scene) -> list[tuple]:
    events: list[tuple] = []
    viewer.controller.on_dims_interaction(
        scene.id,
        lambda event: events.append((event.phase, event.reason)),
        owner_id=viewer.controller._id,
    )
    return events


def _camera_events(viewer, scene) -> list[tuple]:
    events: list[tuple] = []
    viewer.controller.on_camera_interaction(
        scene.id,
        lambda event: events.append((event.phase, event.reason)),
        owner_id=viewer.controller._id,
    )
    return events


# -- Viewer -----------------------------------------------------------------------


async def test_viewer_slice_positions_jump_by_default():
    viewer = Viewer(_WORLD, dim="2d", gui="offscreen")
    events = _dims_events(viewer, viewer.scene)
    viewer.set_slice_positions({0: 2.0, 1: 3.0})
    assert viewer.scene.dims.selection.slice_indices[0] == 2.0
    assert viewer.scene.dims.selection.slice_indices[1] == 3.0
    assert events == []


async def test_viewer_dims_interaction_scrubs_and_releases():
    viewer = Viewer(_WORLD, dim="2d", gui="offscreen")
    events = _dims_events(viewer, viewer.scene)
    with viewer.dims_interaction():
        viewer.set_slice_positions({0: 1.0})
        viewer.set_slice_positions({0: 2.0})
        assert events == [("start", None)]
    assert events == [("start", None), ("end", "release")]

    viewer.set_slice_positions({0: 3.0}, interactive=True)
    assert events[-1] == ("start", None)
    viewer.controller.close()


async def test_viewer_camera_calls_use_its_canvas():
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    _add_image(viewer)
    with pytest.raises(ValueError, match="0 canvases"):
        viewer.get_camera_state()
    viewer.add_canvas()
    (canvas_id,) = viewer.canvases
    events = _camera_events(viewer, viewer.scene)

    viewer.fit_camera()
    state = viewer.get_camera_state()
    assert state == viewer.controller.get_camera_state(canvas_id)
    moved = state._replace(position=tuple(p + 2.0 for p in state.position))
    viewer.set_camera_state(moved)  # a jump
    assert viewer.get_camera_state() == moved
    assert events == []

    with viewer.camera_interaction():
        viewer.set_camera_state(state)
        assert events == [("start", None)]
        viewer.fit_camera()  # in scope: a tick, not a jump
        assert events == [("start", None)]
    assert events == [("start", None), ("end", "release")]

    viewer.set_camera_state(moved, canvas=canvas_id, interactive=True)
    assert events[-1] == ("start", None)
    with pytest.raises(ValueError, match="not one of this viewer's canvases"):
        viewer.set_camera_state(moved, canvas=viewer.scene.id)
    viewer.controller.close()


# -- OrthoViewer ------------------------------------------------------------------


async def test_ortho_slice_positions_move_every_panel():
    viewer = OrthoViewer(_WORLD)
    viewer.set_slice_positions({0: 2.0})
    assert all(
        scene.dims.selection.slice_indices[0] == 2.0 for scene in viewer.scenes.values()
    )
    viewer.axis_sync_enabled = False
    viewer.set_slice_positions({0: 3.0})
    assert all(
        scene.dims.selection.slice_indices[0] == 3.0 for scene in viewer.scenes.values()
    )


@pytest.mark.parametrize("linked", [True, False])
async def test_ortho_dims_interaction_scrubs_all_four_panels(linked):
    viewer = OrthoViewer(_WORLD, link_axes=linked)
    events = {key: _dims_events(viewer, scene) for key, scene in viewer.scenes.items()}
    with viewer.dims_interaction():
        viewer.set_slice_positions({0: 1.0})
        viewer.set_slice_positions({0: 2.0})
        assert all(log == [("start", None)] for log in events.values())
    assert all(log == [("start", None), ("end", "release")] for log in events.values())
    assert not viewer.controller._dims_driver.tasks()


async def test_ortho_interactive_slice_positions_scrub_all_four_panels():
    viewer = OrthoViewer(_WORLD)
    viewer.controller._render_manager.config.scheduler.dims_settle_s = 60.0
    viewer.set_slice_positions({0: 1.0}, interactive=True)
    assert {
        viewer.controller.dims_interaction_state(scene.id)
        for scene in viewer.scenes.values()
    } == {"active"}
    viewer.controller.close()


async def test_ortho_camera_calls_name_a_panel():
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    _add_image(viewer)
    with pytest.raises(ValueError, match="Unknown panel"):
        viewer.get_camera_state("front")
    with pytest.raises(ValueError, match="no canvas yet"):
        viewer.get_camera_state("xy")
    with pytest.raises(ValueError, match="Unknown panel"):
        viewer.fit_camera("front")
    viewer.fit_camera()  # no canvases yet: nothing to fit, nothing raised
