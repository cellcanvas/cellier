"""Tests for capture at the convenience layer: ``Viewer`` and ``OrthoViewer``.

Addressing is what these cover.  ``Viewer`` disambiguates *viewpoints onto one
scene*; ``OrthoViewer`` selects *which of four scenes* to render, and composites
all four by default.  The pixels themselves are covered by
``tests/render/test_capture.py``.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.visuals._image_memory import InMemoryImageAppearance


@pytest.fixture
def volume_store() -> ImageMemoryStore:
    """A (16, 24, 32) volume with a bright cuboid in the middle."""
    data = np.zeros((16, 24, 32), dtype=np.float32)
    data[4:12, 6:18, 8:24] = 1.0
    return ImageMemoryStore(data=data, name="volume")


@pytest.fixture
def offscreen_viewer(volume_store, offscreen_gpu) -> Viewer:
    """A headless ``Viewer`` holding one image, with no canvas yet."""
    viewer = Viewer(axis_labels=("z", "y", "x"), gui="offscreen")
    viewer.add_image(
        data=volume_store,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    return viewer


@pytest.fixture
def offscreen_ortho(volume_store, offscreen_gpu) -> OrthoViewer:
    """A headless ``OrthoViewer`` holding one image across all four panels."""
    ortho = OrthoViewer(axis_labels=("z", "y", "x"), gui="offscreen")
    ortho.add_image(
        data=volume_store,
        appearance=InMemoryImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
    )
    return ortho


# ---------------------------------------------------------------------------
# Viewer addressing
# ---------------------------------------------------------------------------


def test_canvases_starts_empty_and_tracks_add_canvas(offscreen_viewer):
    """``canvases`` reports the scene's canvases in creation order."""
    assert offscreen_viewer.canvases == ()

    offscreen_viewer.add_canvas(canvas_size=(64, 64))
    offscreen_viewer.add_canvas(canvas_size=(64, 64))

    assert len(offscreen_viewer.canvases) == 2
    assert len(set(offscreen_viewer.canvases)) == 2


def test_a_single_canvas_needs_no_addressing(offscreen_viewer):
    """With one canvas, ``screenshot()`` uses it without being told to."""
    offscreen_viewer.add_canvas(canvas_size=(64, 48))

    frame = offscreen_viewer.screenshot(size=(80, 60))

    assert frame.shape == (60, 80, 4)


def test_several_canvases_raise_rather_than_guess(offscreen_viewer):
    """Silently taking ``canvases[0]`` is how the wrong view goes unnoticed."""
    offscreen_viewer.add_canvas(canvas_size=(64, 48))
    offscreen_viewer.add_canvas(canvas_size=(64, 48))

    with pytest.raises(ValueError, match="cannot choose one for you"):
        offscreen_viewer.screenshot()


def test_an_explicit_canvas_resolves_the_ambiguity(offscreen_viewer):
    """Naming a canvas from ``canvases`` is how you pick between them."""
    offscreen_viewer.add_canvas(canvas_size=(64, 48))
    offscreen_viewer.add_canvas(canvas_size=(100, 50))

    frame = offscreen_viewer.screenshot(canvas=offscreen_viewer.canvases[1])

    # Defaulted from the named canvas, so the second canvas's size is what
    # comes back -- proof the argument selected that viewpoint.
    assert frame.shape == (50, 100, 4)


def test_a_foreign_canvas_is_rejected(offscreen_viewer):
    """A canvas id from somewhere else raises instead of capturing something."""
    offscreen_viewer.add_canvas(canvas_size=(64, 48))

    with pytest.raises(ValueError, match="is not one of this viewer's canvases"):
        offscreen_viewer.screenshot(canvas=uuid4())


def test_save_writes_a_png(offscreen_viewer, tmp_path):
    """``save=`` writes the same pixels it returns."""
    imageio = pytest.importorskip("imageio.v3")
    offscreen_viewer.add_canvas(canvas_size=(64, 48))
    path = tmp_path / "shot.png"

    frame = offscreen_viewer.screenshot(size=(40, 30), save=path)

    assert path.exists()
    assert np.array_equal(imageio.imread(path), frame)


# ---------------------------------------------------------------------------
# OrthoViewer addressing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel", ["xy", "xz", "yz", "vol"])
def test_each_panel_captures_on_its_own(offscreen_ortho, panel):
    """Every panel key names a capturable scene."""
    frame = offscreen_ortho.screenshot(panel=panel, size=(50, 40))

    assert frame.shape == (40, 50, 4)


def test_an_unknown_panel_is_rejected(offscreen_ortho):
    """A typo names the four valid keys rather than failing obscurely."""
    with pytest.raises(ValueError, match="Unknown panel"):
        offscreen_ortho.screenshot(panel="xyz")


def test_the_default_is_a_two_by_two_grid(offscreen_ortho):
    """No panel means all four, composited at twice the per-panel size."""
    grid = offscreen_ortho.screenshot(size=(50, 40))

    assert grid.shape == (80, 100, 4)


def test_the_grid_is_laid_out_like_the_window(offscreen_ortho):
    """XY and XZ on the top row, YZ and the volume below.

    Asserted by capturing each panel separately and finding it in the
    quadrant ``build_ortho_grid_widget`` puts it in -- so a change to the
    arrangement shows up here rather than in a screenshot nobody compares.
    """
    size = (50, 40)
    width, height = size
    panels = {
        key: offscreen_ortho.screenshot(panel=key, size=size)
        for key in ("xy", "xz", "yz", "vol")
    }
    grid = offscreen_ortho.screenshot(size=size)

    assert np.array_equal(grid[:height, :width], panels["xy"])
    assert np.array_equal(grid[:height, width:], panels["xz"])
    assert np.array_equal(grid[height:, :width], panels["yz"])
    assert np.array_equal(grid[height:, width:], panels["vol"])


def test_the_grid_is_reproducible(offscreen_ortho):
    """The composite inherits the per-panel reproducibility contract."""
    first = offscreen_ortho.screenshot(size=(40, 30))
    second = offscreen_ortho.screenshot(size=(40, 30))

    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# The offscreen gui has no widgets
# ---------------------------------------------------------------------------


def test_layout_builders_refuse_an_offscreen_viewer(offscreen_viewer):
    """An offscreen viewer has nothing to embed, and the error says so."""
    from cellier.convenience.gui._canvas import build_canvas_widget

    with pytest.raises(ValueError, match="no embeddable widget"):
        build_canvas_widget(offscreen_viewer, {"z": (0, 1), "y": (0, 1), "x": (0, 1)})


def test_run_refuses_an_offscreen_viewer(offscreen_viewer):
    """``run`` dispatches on ``viewer.gui``, and points at ``screenshot()``.

    ``show`` and ``launch`` are Qt entry points by definition and reject a
    non-Qt viewer on their own terms; ``run`` is the one that chooses, so it
    is the one that has to name what an offscreen viewer should do instead.
    """
    from cellier.convenience import Layout, run

    with pytest.raises(ValueError, match="no window to show"):
        run(offscreen_viewer, Layout(center=None))
