"""A paint session holds the camera still without touching the camera models.

The camera models' ``controller.enabled`` is the user's setting, and a saved
file records it.  A paint session used to write ``False`` there, so a file
saved during a session restored a camera that could not move, with no
session left to free it, and the session's end wrote ``True`` over a camera
the user had turned off.  Now the session holds a runtime lock on the
canvas view instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.data import LabelMemoryStore
from cellier.scene.dims import spatial_axes


@pytest.fixture
def make_viewer(qtbot):
    made = []

    def _make():
        viewer = Viewer(
            spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="offscreen"
        )
        viewer.controller.camera_reslice_enabled = False
        data = np.zeros((6, 16, 16), dtype=np.int32)
        labels = viewer.add_labels(LabelMemoryStore(data=data), name="labels")
        viewer.add_canvas()
        made.append(viewer)
        canvas_id = viewer.canvases[0]
        return viewer, labels, canvas_id

    yield _make
    for viewer in made:
        viewer.controller.close()


def _paint(viewer, labels, canvas_id):
    return viewer.controller.add_paint_controller(
        visual_id=labels.id,
        canvas_id=canvas_id,
        brush_value=1,
        brush_radius_voxels=1.0,
    )


def _models_enabled(viewer, canvas_id) -> dict[str, bool]:
    cameras = viewer.scene.canvases[canvas_id].cameras
    return {dim: camera.controller.enabled for dim, camera in cameras.items()}


def _live(viewer, canvas_id) -> dict[str, bool]:
    view = viewer.controller.get_canvas_view(canvas_id)
    return {"2d": view._controller_2d.enabled, "3d": view._controller_3d.enabled}


async def test_a_session_holds_the_camera_still_across_a_toggle(make_viewer):
    viewer, labels, canvas_id = make_viewer()
    paint = _paint(viewer, labels, canvas_id)

    assert _live(viewer, canvas_id) == {"2d": False, "3d": False}
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert _live(viewer, canvas_id) == {"2d": False, "3d": False}
    viewer.set_displayed_dimensions(("y", "x"))
    assert _live(viewer, canvas_id) == {"2d": False, "3d": False}
    # The user's setting is untouched throughout.
    assert _models_enabled(viewer, canvas_id) == {"2d": True, "3d": True}

    paint.commit()

    assert _live(viewer, canvas_id) == {"2d": True, "3d": False}
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert _live(viewer, canvas_id) == {"2d": False, "3d": True}


async def test_a_file_saved_during_a_session_restores_a_free_camera(
    make_viewer, tmp_path
):
    """B11: the session's hold is not saved."""
    viewer, labels, canvas_id = make_viewer()
    paint = _paint(viewer, labels, canvas_id)
    path = tmp_path / "viewer.json"

    viewer.to_file(path)
    paint.abort()

    loaded = Viewer.from_file(path, gui="offscreen")
    try:
        loaded_canvas = loaded.canvases[0]
        assert _models_enabled(loaded, loaded_canvas) == {"2d": True, "3d": True}
        view = loaded.controller.get_canvas_view(loaded_canvas)
        assert view._controller.enabled is True
    finally:
        loaded.controller.close()


async def test_a_camera_the_user_turned_off_stays_off_after_a_session(make_viewer):
    viewer, labels, canvas_id = make_viewer()
    viewer.controller.set_camera_controller_enabled(canvas_id, False)

    _paint(viewer, labels, canvas_id).commit()

    assert _models_enabled(viewer, canvas_id) == {"2d": False, "3d": False}
    assert _live(viewer, canvas_id) == {"2d": False, "3d": False}


async def test_turning_the_camera_on_during_a_session_applies_after_it(
    make_viewer,
):
    """The session wins while it runs; the user's setting after."""
    viewer, labels, canvas_id = make_viewer()
    viewer.controller.set_camera_controller_enabled(canvas_id, False)
    paint = _paint(viewer, labels, canvas_id)

    viewer.controller.set_camera_controller_enabled(canvas_id, True)
    assert _live(viewer, canvas_id)["2d"] is False

    paint.abort()
    assert _live(viewer, canvas_id)["2d"] is True


async def test_the_camera_moves_again_only_when_every_session_ends(make_viewer):
    viewer, labels, canvas_id = make_viewer()
    first = _paint(viewer, labels, canvas_id)
    second = _paint(viewer, labels, canvas_id)

    first.commit()
    assert _live(viewer, canvas_id)["2d"] is False

    second.commit()
    assert _live(viewer, canvas_id)["2d"] is True


async def test_ending_a_session_after_its_canvas_is_removed_is_harmless(
    make_viewer,
):
    viewer, labels, canvas_id = make_viewer()
    paint = _paint(viewer, labels, canvas_id)

    viewer.controller.remove_canvas(canvas_id)

    paint.abort()
