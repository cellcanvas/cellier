"""Hiding a visual, end to end through the render stack.

Reported repeatedly against ``geometry_viewer_marimo.py`` and
``multiscale_image_viewer.py``, on **both** toolkits and for every visual
type: unchecking ``Visible`` appears to do nothing, and afterwards the canvas
seems frozen -- until you drag the camera, at which point the visual
disappears.

The model, the bus, the pygfx node and the redraw were all correct, and the
tests here cover that. The defect was one layer further down, in
``CanvasView``'s ``TemporalAccumulationPass``: it blends each frame into a
history texture whose ``reset()`` was called from exactly one place, guarded
by "did the camera move". A content change left the history in place, so the
hidden visual lingered as a ghost -- and dragging the camera cleared it, which
is what made this look like a redraw bug. Fixed by
``CanvasView.invalidate_accumulation``; see
``plans/temporal_accumulation_fix.md``.

**Read the paths these tests use carefully.** ``render_scene`` builds a
*fresh* offscreen renderer per call, with no effect passes and no history, so
it can prove the scene graph is right and is structurally **incapable** of
observing the accumulation bug. Two earlier versions of this module claimed
more than that setup could show. The tests that speak to the real defect are
the two at the bottom, which assert against ``CanvasView``'s own pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.layout._walk import render_dock
from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.visuals._mesh_memory import MeshFlatAppearance

_POS = np.array([[1, 1, 1], [5, 1, 1], [1, 5, 1], [1, 1, 5]], dtype=np.float32)


def _mesh_scene(controller, *, with_second_visual: bool = False):
    scene = controller.add_scene(dim="3d", name="s")
    visual = controller.add_mesh(
        MeshMemoryStore(
            positions=_POS,
            indices=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32),
        ),
        scene.id,
        MeshFlatAppearance(),
        "mesh",
    )
    if with_second_visual:
        controller.add_points(
            PointsMemoryStore(positions=_POS * 1.5), scene.id, None, "reference"
        )
    controller.add_canvas(scene_id=scene.id)
    return scene, visual


@pytest.mark.asyncio
async def test_hiding_removes_the_visual_and_showing_restores_it_exactly(
    controller, reslice, render_scene
):
    """The scene-graph round trip is exact, so nothing is stuck or lost.

    Compared against a frame rendered while hidden from the start rather than
    against a fixed colour count: the scene carries a background, so "empty"
    is whatever the background paints, not a single colour.
    """
    scene, visual = _mesh_scene(controller)
    controller.update_appearance_field(visual.id, "visible", False)
    await reslice(controller, scene.id)
    never_shown = render_scene(controller, scene.id)

    controller.update_appearance_field(visual.id, "visible", True)
    shown = render_scene(controller, scene.id)
    assert not np.array_equal(shown, never_shown), "sanity: the visual renders"

    controller.update_appearance_field(visual.id, "visible", False)
    # Nothing of the visual survives -- identical to never having shown it.
    assert np.array_equal(never_shown, render_scene(controller, scene.id))

    controller.update_appearance_field(visual.id, "visible", True)
    assert np.array_equal(shown, render_scene(controller, scene.id))


@pytest.mark.asyncio
async def test_a_second_visual_stays_when_the_first_is_hidden(
    controller, reslice, render_scene
):
    """With something else in the scene, hiding one visual leaves the rest.

    Not how the example is built -- it gives each panel a single visual on
    purpose -- but worth pinning separately: hiding must scope to the visual
    that was hidden and not disturb its neighbours.
    """
    scene, visual = _mesh_scene(controller, with_second_visual=True)
    await reslice(controller, scene.id)
    both = render_scene(controller, scene.id)

    controller.update_appearance_field(visual.id, "visible", False)
    only_reference = render_scene(controller, scene.id)

    assert not np.array_equal(both, only_reference)
    # Still something on screen: the reference survived.
    assert (
        len(np.unique(only_reference.reshape(-1, only_reference.shape[-1]), axis=0)) > 1
    )


@pytest.mark.asyncio
async def test_other_controls_still_reach_a_hidden_visual(
    controller, reslice, render_scene
):
    """Edits made while hidden are not lost -- they show up on re-showing.

    "Changing other appearance parameters has no effect" is true of the
    *picture* while the visual is hidden, and must not be true of the model.
    """
    scene, visual = _mesh_scene(controller)
    await reslice(controller, scene.id)
    render_scene(controller, scene.id)

    controller.update_appearance_field(visual.id, "visible", False)
    controller.update_appearance_field(visual.id, "opacity", 0.25)
    assert visual.appearance.opacity == pytest.approx(0.25)

    controller.update_appearance_field(visual.id, "visible", True)
    recoloured = render_scene(controller, scene.id)

    controller.update_appearance_field(visual.id, "opacity", 1.0)
    assert not np.array_equal(recoloured, render_scene(controller, scene.id))


@pytest.mark.asyncio
async def test_nothing_blocks_the_canvas_when_no_visual_is_visible(controller, reslice):
    """The question this module exists to answer, asked of the real draw path.

    Rendering the scene directly (as the tests above do) proves the *scene* is
    right, not that a frame was ever asked for.  This drives the canvas's own
    ``_draw_frame`` and counts both halves: a hide requests a draw, the
    renderer runs with nothing visible, and a re-show requests another.  If
    anything ever gates the pipeline on having a visible visual, this fails.
    """
    scene, visual = _mesh_scene(controller)
    await reslice(controller, scene.id)

    canvas_view = controller._render_manager._canvases[
        controller.get_canvas_ids(scene.id)[0]
    ]

    draws: list[int] = []
    renders: list[int] = []
    original_request = canvas_view.request_draw
    original_render = canvas_view._renderer.render
    canvas_view.request_draw = lambda *a, **k: (
        draws.append(1),
        original_request(*a, **k),
    )[1]
    canvas_view._renderer.render = lambda *a, **k: (
        renders.append(1),
        original_render(*a, **k),
    )[1]

    def _run(action) -> tuple[int, int]:
        draws.clear()
        renders.clear()
        action()
        canvas_view._draw_frame()
        return len(draws), len(renders)

    hide_draws, hide_renders = _run(
        lambda: controller.update_appearance_field(visual.id, "visible", False)
    )
    assert hide_draws == 1, "hiding must ask the canvas for a frame"
    assert hide_renders == 1, "the renderer must still run with nothing visible"

    # ...and the pipeline keeps working while nothing is visible.
    _idle_draws, idle_renders = _run(lambda: None)
    assert idle_renders == 1

    edit_draws, edit_renders = _run(
        lambda: controller.update_appearance_field(visual.id, "opacity", 0.5)
    )
    assert edit_draws == 1, "an edit made while hidden still asks for a frame"
    assert edit_renders == 1

    show_draws, show_renders = _run(
        lambda: controller.update_appearance_field(visual.id, "visible", True)
    )
    assert show_draws == 1
    assert show_renders == 1


@pytest.mark.asyncio
async def test_the_qt_visible_checkbox_hides_and_restores_the_picture(
    qtbot, controller, reslice, render_scene
):
    """The whole Qt chain, driven the way a user drives it.

    Everything else here writes the model through the controller.  This finds
    the ``QCheckBox`` the appearance dock actually built and toggles *that*,
    so the widget, the bus, the render visual and the frame are all covered by
    one assertion -- and both directions are, since a model-side write has to
    move the checkbox back.
    """
    from PySide6.QtWidgets import QCheckBox, QWidget

    from cellier.convenience import Viewer
    from cellier.convenience.gui import MeshControlsConfig
    from cellier.convenience.layout._spec import AppearanceControls
    from cellier.data.mesh._mesh_memory_store import MeshMemoryStore
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    viewer = Viewer(("z", "y", "x"), dim="3d", gui="qt")
    viewer.controller.camera_reslice_enabled = False
    visual = viewer.add_mesh(
        MeshMemoryStore(
            positions=_POS,
            indices=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32),
        ),
        appearance=MeshFlatAppearance(color=(1.0, 0.2, 0.2, 1.0)),
        controls=MeshControlsConfig(appearance=["visible"]),
    )
    viewer.controller.add_canvas(scene_id=viewer.scene.id)
    dock = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])
    await reslice(viewer.controller, viewer.scene.id)

    # The checkbox carries no text of its own: its row's label names it
    # (``plans/label_ownership_unification.md``), so find the row first.
    from cellier.gui.qt.visuals._chrome import LABELLED_ROW_OBJECT_NAME

    row = next(
        candidate
        for candidate in dock.findChildren(QWidget)
        if candidate.objectName() == LABELLED_ROW_OBJECT_NAME
        and candidate.layout().itemAt(0).widget().text() == "Visible"
    )
    checkbox = row.findChild(QCheckBox)
    assert checkbox.isChecked() is True

    shown = render_scene(viewer.controller, viewer.scene.id)

    checkbox.setChecked(False)  # the click
    assert visual.appearance.visible is False
    hidden = render_scene(viewer.controller, viewer.scene.id)
    assert not np.array_equal(shown, hidden), "the frame did not change"

    checkbox.setChecked(True)
    assert np.array_equal(shown, render_scene(viewer.controller, viewer.scene.id))

    # model -> widget: a write from anywhere else moves the checkbox.
    viewer.controller.update_appearance_field(visual.id, "visible", False)
    assert checkbox.isChecked() is False


# ---------------------------------------------------------------------------
# The remaining defect: the temporal accumulation history
# ---------------------------------------------------------------------------
#
# These are the only tests here that can see it.  They assert against
# ``CanvasView``'s own ``TemporalAccumulationPass`` rather than against pixels,
# because the pass lives on the canvas's renderer and the offscreen helper
# every other test uses does not have one.


def _canvas_view(controller, scene):
    return controller._render_manager._canvases[controller.get_canvas_ids(scene.id)[0]]


def _count_history_resets(monkeypatch, canvas_view) -> list[int]:
    """Record every time the accumulation history is discarded."""
    resets: list[int] = []
    original = canvas_view._accum_pass.reset
    monkeypatch.setattr(
        canvas_view._accum_pass,
        "reset",
        lambda *a, **k: (resets.append(1), original(*a, **k))[1],
    )
    return resets


@pytest.mark.asyncio
async def test_moving_the_camera_discards_the_accumulation_history(
    monkeypatch, controller, reslice
):
    """The mechanism exists and fires -- so the next test is not vacuous.

    This is also the reason dragging the canvas makes a stale visual vanish:
    it is the one path that resets the history.
    """
    scene, _visual = _mesh_scene(controller)
    await reslice(controller, scene.id)
    canvas_view = _canvas_view(controller, scene)
    assert canvas_view._accum_pass.enabled, "3D scenes accumulate"

    canvas_view._draw_frame()  # settle, as above
    resets = _count_history_resets(monkeypatch, canvas_view)
    canvas_view.camera.local.position = (99.0, 99.0, 99.0)
    canvas_view._draw_frame()

    assert resets, "a camera move must discard the history"


@pytest.mark.asyncio
async def test_hiding_a_visual_discards_the_accumulation_history(
    monkeypatch, controller, reslice
):
    """Hiding changes what should be drawn, so the history is stale.

    Without this the hidden visual lingered as a ghost: the history was
    invalidated only by a camera move, so the frames after the change were
    an average with a picture that no longer applied.  Dragging the canvas
    cleared it, which is what made the bug look like a redraw problem.

    The same argument applies to every other content change -- colormap,
    clim, opacity, a data commit, a transform -- which is why the
    invalidation hangs off ``request_draw`` rather than off ``visible``.
    """
    scene, visual = _mesh_scene(controller)
    await reslice(controller, scene.id)
    canvas_view = _canvas_view(controller, scene)
    # Settle first: ``reslice`` fits the camera, and that fit legitimately
    # resets the history on the next frame.  Without this the test would pass
    # on the camera's reset and prove nothing about the content change.
    canvas_view._draw_frame()

    resets = _count_history_resets(monkeypatch, canvas_view)
    controller.update_appearance_field(visual.id, "visible", False)
    canvas_view._draw_frame()

    assert resets, "a content change must discard the accumulation history"
