"""An appearance change must ask the canvas for a frame.

Found by a user running ``examples/convenience/geometry_viewer_marimo.py``:
the ``visible`` checkbox appeared dead while every control next to it worked.
The model, the bus and the pygfx node were all correct -- ``node.visible``
really did become ``False``, and an offscreen render of the scene really did
change.  What did not happen was a **repaint**.

Nothing in the appearance path asked for one.  A reslice does, but only the
three ``_RESLICE_FIELDS`` trigger a reslice; every other appearance change
repaints the same data.  Fields that write a pygfx *material* tend to get a
frame anyway, because the dirtied resource reaches the canvas on its own --
which is why ``size``, ``color`` and the graph's ``node_visible`` looked fine.
``visible`` and ``aabb.enabled`` set plain scene-graph flags and got nothing,
so they were the two controls that looked broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.gui import build_canvas_widget
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.gui._axis_values import ContinuousAxisValues
from cellier.scene.dims import spatial_axes

_RANGES = dict.fromkeys(range(3), ContinuousAxisValues(min=0.0, max=4.0))


@pytest.fixture
def viewer_with_canvas(qtbot):
    """A points viewer with one canvas, and a per-canvas draw counter."""
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="qt"
    )
    viewer.controller.camera_reslice_enabled = False
    visual = viewer.add_points(
        PointsMemoryStore(
            positions=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        )
    )
    build_canvas_widget(viewer, _RANGES)

    draws: list = []
    for canvas_id, canvas_view in viewer.controller._render_manager._canvases.items():
        original = canvas_view.request_draw

        def _counting(*args, _original=original, _id=canvas_id, **kwargs):
            draws.append(_id)
            return _original(*args, **kwargs)

        canvas_view.request_draw = _counting

    return viewer, visual, draws


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visible", False),  # the field that surfaced this
        ("size", 9.0),
        ("opacity", 0.4),
        ("color", (0.1, 0.2, 0.3, 1.0)),
    ],
)
def test_an_appearance_change_requests_a_frame(viewer_with_canvas, field, value):
    viewer, visual, draws = viewer_with_canvas

    viewer.controller.update_appearance_field(visual.id, field, value)

    assert draws, f"writing {field!r} asked no canvas to redraw"


def test_an_aabb_change_requests_a_frame(viewer_with_canvas):
    """The bounding box is not an appearance field and needs its own path."""
    viewer, visual, draws = viewer_with_canvas

    viewer.controller.update_aabb_field(visual.id, "enabled", True)

    assert draws


def test_one_frame_per_change_not_one_per_canvas_visual(viewer_with_canvas):
    """A single field write must not fan out into a burst of frames.

    Sliders emit continuously while dragging, so a redraw per change is the
    budget; anything multiplied by it would be felt.
    """
    viewer, visual, draws = viewer_with_canvas

    viewer.controller.update_appearance_field(visual.id, "size", 7.0)

    assert len(draws) == 1


def test_a_visual_with_no_canvas_does_not_raise(qtbot):
    """The headless model-only path stays usable -- no canvas, no redraw."""
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui="qt"
    )
    visual = viewer.add_points(
        PointsMemoryStore(positions=np.array([[0, 0, 0]], dtype=np.float32))
    )

    viewer.controller.update_appearance_field(visual.id, "visible", False)
    viewer.controller.update_aabb_field(visual.id, "enabled", True)

    assert visual.appearance.visible is False
