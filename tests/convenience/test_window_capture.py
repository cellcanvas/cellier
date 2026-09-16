"""Tests for the Qt window composite (``screenshot_window``).

The thing worth asserting is the one that motivated the function: Qt's
``grab()`` cannot see a wgpu surface, so a plain grab returns a window whose
canvas rectangles are flat fill.  These check that the composite puts real
rendered pixels into those rectangles, in the right place.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.convenience import screenshot_window
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def window_with_canvas(qtbot, offscreen_gpu):
    """A ``QMainWindow`` holding a label and one live cellier canvas."""
    from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

    data = np.zeros((8, 16, 16), dtype=np.float32)
    data[...] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[None, None, :]

    controller = CellierController(gui="qt")
    controller.camera_reslice_enabled = False
    scene = controller.add_scene(dim="2d", name="scene")
    controller.add_image(
        data=ImageMemoryStore(data=data, name="gradient"),
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    canvas_widget = controller.add_canvas(scene_id=scene.id)

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel("a dock-ish label"))
    layout.addWidget(canvas_widget)
    window.setCentralWidget(central)
    window.resize(400, 400)
    qtbot.addWidget(window)
    window.show()

    controller.fit_camera(scene.id)
    yield window, controller, canvas_widget
    controller.close()


def _canvas_region(window, canvas_widget, composite):
    """Return the sub-array of *composite* covering *canvas_widget*."""
    from PySide6.QtCore import QPoint

    ratio = window.grab().devicePixelRatio()
    top_left = canvas_widget.mapTo(window, QPoint(0, 0))
    x = round(top_left.x() * ratio)
    y = round(top_left.y() * ratio)
    width = round(canvas_widget.width() * ratio)
    height = round(canvas_widget.height() * ratio)
    return composite[y : y + height, x : x + width]


def test_the_canvas_region_carries_rendered_pixels(window_with_canvas):
    """The composite fills in what ``grab()`` leaves blank.

    A bare ``window.grab()`` returns exactly one distinct colour over the
    canvas -- the wgpu surface is not part of Qt's paint pipeline.  The
    composite must do better than that by a wide margin.
    """
    window, controller, canvas_widget = window_with_canvas

    composite = screenshot_window(window, controller)
    region = _canvas_region(window, canvas_widget, composite)

    n_colors = len(np.unique(region.reshape(-1, 4), axis=0))
    assert n_colors > 20, f"the canvas region has only {n_colors} colours; blank?"


def test_a_plain_grab_really_is_blank_there(window_with_canvas):
    """The negative control: without the composite, that region is flat.

    Without this, the test above would still pass if Qt ever started capturing
    wgpu surfaces on its own -- and the composite would be dead code nobody
    noticed.
    """
    from PySide6.QtGui import QImage

    from cellier.convenience._window_capture import _qimage_to_rgba

    window, _controller, canvas_widget = window_with_canvas

    grabbed = _qimage_to_rgba(
        window.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    )
    region = _canvas_region(window, canvas_widget, grabbed)

    assert len(np.unique(region.reshape(-1, 4), axis=0)) <= 2


def test_the_chrome_survives_the_composite(window_with_canvas):
    """Pasting canvases must not wipe the widgets around them."""
    window, controller, canvas_widget = window_with_canvas

    composite = screenshot_window(window, controller)
    # The label sits above the canvas, so the strip over it is pure chrome.
    from PySide6.QtCore import QPoint

    ratio = window.grab().devicePixelRatio()
    canvas_top = round(canvas_widget.mapTo(window, QPoint(0, 0)).y() * ratio)
    chrome = composite[:canvas_top]

    assert chrome.size > 0
    assert len(np.unique(chrome.reshape(-1, 4), axis=0)) > 1


def test_the_composite_matches_the_grab_geometry(window_with_canvas):
    """The result is the window's own grab, same shape, physical pixels."""
    window, controller, _canvas_widget = window_with_canvas

    composite = screenshot_window(window, controller)
    pixmap = window.grab()

    assert composite.shape == (pixmap.height(), pixmap.width(), 4)
    assert composite.dtype == np.uint8


def test_save_writes_the_composite(window_with_canvas, tmp_path):
    """``save=`` writes the same pixels it returns."""
    imageio = pytest.importorskip("imageio.v3")
    window, controller, _canvas_widget = window_with_canvas
    path = tmp_path / "window.png"

    composite = screenshot_window(window, controller, save=path)

    assert np.array_equal(imageio.imread(path), composite)


def test_canvases_in_other_windows_are_left_alone(window_with_canvas, qtbot):
    """Only canvases belonging to the given window are composited."""
    from PySide6.QtWidgets import QMainWindow

    _window, controller, _canvas_widget = window_with_canvas
    other = QMainWindow()
    qtbot.addWidget(other)
    other.resize(120, 120)
    other.show()

    composite = screenshot_window(other, controller)

    # Nothing from the first window's canvas leaked into the second window.
    assert composite.shape[0] > 0
    assert len(np.unique(composite.reshape(-1, 4), axis=0)) <= 4
