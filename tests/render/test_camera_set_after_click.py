"""A programmatic camera move is not undone by a click that came just before it.

A press and release on the canvas leave the pygfx camera controller with a
finished drag.  The controller keeps the camera state it saw at the press and
writes it back on its next tick, so a ``set_camera_state`` call between the
click and the next frame was undone: the camera went back to where it was at the
click.  Found while building the B12 pick sweep, where each pose was a
``set_camera_state`` after a run of synthetic presses.  The canvas now hands the
controllers the camera's state whenever the camera is moved programmatically.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cellier.data import ImageMemoryStore
from cellier.visuals import InMemoryImageSingleAppearance
from tests.render._affine import bind, canvas_view, load
from tests.render.test_labels_pick_face import _viewer


def _click(view, x: float, y: float, *, release: bool = True) -> None:
    kinds = [("pointer_down", (1,))] + ([("pointer_up", ())] if release else [])
    for kind, buttons in kinds:
        view._renderer.convert_event(
            {
                "event_type": kind,
                "x": x,
                "y": y,
                "button": 1,
                "buttons": buttons,
                "modifiers": (),
                "ntouches": 0,
                "touches": {},
                "time_stamp": time.time(),
            }
        )


@pytest.fixture
def viewer():
    viewer = _viewer(("z", "y", "x"))
    data = np.zeros((16, 16, 16), np.float32)
    data[4:8, 4:8, 4:8] = 1
    store = ImageMemoryStore(data=data, name="cube")
    viewer.add_image(
        store,
        single=InMemoryImageSingleAppearance(
            color_map="gray", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
        ),
        transform=bind(viewer, store, np.eye(4)),
    )
    load(viewer)
    try:
        yield viewer
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("release", [True, False], ids=["click", "held"])
def test_a_set_camera_state_after_a_click_is_kept(viewer, release):
    controller = viewer.controller
    canvas_id = controller.get_canvas_ids(viewer.scene.id)[0]
    view = canvas_view(viewer)
    start = controller.get_camera_state(canvas_id)

    _click(view, 100.0, 100.0, release=release)
    target = start._replace(position=(30.0, 12.0, 5.0))
    controller.set_camera_state(canvas_id, target)
    # What the next frame does with the controller.
    view._draw_frame()
    time.sleep(0.05)
    view._draw_frame()

    got = controller.get_camera_state(canvas_id).position
    np.testing.assert_allclose(got, target.position, atol=1e-3)
