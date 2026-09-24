"""The dock's image control follows its viewer between 2D and 3D.

``DimsChangedEvent`` fires only on a change and ``connect_widget`` does not
replay state, so the builders seed the control from the scene and the control
then follows the events (``gui._image_controls.display_seed``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience.gui._controls_config import InMemoryImageControlsConfig
from cellier.convenience.layout._shared import appearance_targets
from cellier.convenience.layout._walk import build_appearance_widgets
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes


def _image_control(gui, dim):
    from cellier.convenience._backend import backend_for

    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui, dim=dim)
    viewer.add_image(
        ImageMemoryStore(data=np.zeros((4, 8, 8), dtype=np.float32)),
        controls=InMemoryImageControlsConfig(
            appearance=["clim", "render_mode", "iso_threshold"]
        ),
    )
    (target,) = appearance_targets(viewer)
    image, *_rest = build_appearance_widgets(
        target.visual,
        target.config,
        viewer.controller,
        target.visual_ids,
        backend=backend_for(gui),
    )
    return viewer, image


@pytest.fixture(params=["qt", "anywidget"])
def gui(request):
    if request.param == "qt":
        request.getfixturevalue("qtbot")
    else:
        pytest.importorskip("anywidget")
    return request.param


@pytest.mark.parametrize(("dim", "n"), [("2d", 2), ("3d", 3)])
def test_the_control_starts_with_the_viewers_dimensionality(gui, dim, n):
    _viewer, image = _image_control(gui, dim)

    assert image.n_displayed_dimensions == n


def test_the_control_follows_the_viewer(gui):
    viewer, image = _image_control(gui, "2d")

    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert image.n_displayed_dimensions == 3

    viewer.set_displayed_dimensions(("y", "x"))
    assert image.n_displayed_dimensions == 2


def test_the_qt_render_mode_row_appears_in_3d(qtbot):
    viewer, image = _image_control("qt", "2d")
    render_mode = image._controls[("single", None, "render_mode")]
    assert render_mode.isHidden()

    viewer.set_displayed_dimensions(("z", "y", "x"))

    assert not render_mode.isHidden()
