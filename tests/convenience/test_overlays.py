"""The overlay convenience API: ``Viewer`` methods, widgets and the dock."""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import OverlayControls, Viewer
from cellier.convenience.layout._controls_dock import OVERLAY_SELECTOR_TITLE
from cellier.convenience.layout._shared import (
    OVERLAY_PLACEHOLDER,
    overlay_control_specs,
    overlay_targets,
)
from cellier.convenience.layout._walk import render_dock
from cellier.data import ImageMemoryStore
from cellier.gui._overlay_fields import OVERLAY_CONTROLS, OVERLAY_FIELD_WIDGETS
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    CenteredAxes2D,
    SceneBoundingBox,
    SceneBoundingBoxAppearance,
)


def _viewer(gui: str = "offscreen") -> Viewer:
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui=gui)
    viewer.add_image(ImageMemoryStore(data=np.zeros((4, 6, 8), dtype=np.float32)))
    return viewer


@pytest.fixture(params=["qt", "jupyter"])
def toolkit(request):
    """``(gui, host class)`` for one toolkit."""
    if request.param == "qt":
        pytest.importorskip("qtpy")
        request.getfixturevalue("qtbot")
        from cellier.convenience._hosts import QtLayoutHost

        return "qt", QtLayoutHost
    pytest.importorskip("anywidget")
    from cellier.convenience._hosts import JupyterHost

    return "anywidget", JupyterHost


# ---------------------------------------------------------------------------
# Viewer API
# ---------------------------------------------------------------------------


def test_add_scene_overlay_attaches_to_the_scene():
    viewer = _viewer()

    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))

    assert viewer.scene.overlays == [box]
    assert viewer.controller.get_overlay(box.id) is box


def test_a_canvas_overlay_waits_for_the_canvas():
    viewer = _viewer()
    axes = viewer.add_canvas_overlay(CenteredAxes2D(name="axes"))
    assert viewer.overlays == (axes,)
    with pytest.raises(KeyError):
        viewer.controller.get_overlay(axes.id)

    viewer.add_canvas()

    (canvas_id,) = viewer.canvases
    assert viewer.scene.canvases[canvas_id].overlays == [axes]
    assert viewer.controller.get_overlay(axes.id) is axes
    assert viewer._pending_canvas_overlays == []


def test_a_canvas_overlay_added_after_the_canvas_attaches_at_once():
    viewer = _viewer()
    viewer.add_canvas()

    axes = viewer.add_canvas_overlay(CenteredAxes2D(name="axes"))

    assert viewer.controller.get_overlay(axes.id) is axes


def test_overlays_lists_scene_then_canvas_then_pending():
    viewer = _viewer()
    viewer.add_canvas()
    attached = viewer.add_canvas_overlay(CenteredAxes2D(name="attached"))
    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))

    assert viewer.overlays == (box, attached)


@pytest.mark.parametrize(
    "method, overlay",
    [
        ("add_scene_overlay", CenteredAxes2D(name="axes")),
        ("add_canvas_overlay", SceneBoundingBox(name="box")),
    ],
)
def test_the_wrong_category_is_rejected(method, overlay):
    viewer = _viewer()
    with pytest.raises(TypeError, match="takes a"):
        getattr(viewer, method)(overlay)


def test_remove_overlay_takes_a_model_or_an_id():
    viewer = _viewer()
    viewer.add_canvas()
    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    axes = viewer.add_canvas_overlay(CenteredAxes2D(name="axes"))

    viewer.remove_overlay(box)
    viewer.remove_overlay(axes.id)

    assert viewer.overlays == ()
    with pytest.raises(KeyError):
        viewer.remove_overlay(box)


def test_remove_overlay_drops_a_pending_one():
    viewer = _viewer()
    axes = viewer.add_canvas_overlay(CenteredAxes2D(name="axes"))

    viewer.remove_overlay(axes)
    viewer.add_canvas()

    assert viewer.overlays == ()


def test_adding_and_removing_announces_a_controls_change():
    viewer = _viewer()
    calls: list = []
    viewer._controls_changed.connect(lambda: calls.append(1))

    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    viewer.remove_overlay(box)

    assert len(calls) == 2


def test_an_overlay_on_another_viewer_is_not_removed():
    viewer, other = _viewer(), _viewer()
    box = other.add_scene_overlay(SceneBoundingBox(name="box"))

    with pytest.raises(KeyError):
        viewer.remove_overlay(box)


# ---------------------------------------------------------------------------
# Specs (pure)
# ---------------------------------------------------------------------------


def test_every_overlay_control_has_a_widget_entry():
    for fields in OVERLAY_CONTROLS.values():
        for field in fields:
            assert field in OVERLAY_FIELD_WIDGETS


def test_overlay_targets_are_labelled_uniquely():
    viewer = _viewer()
    first = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    second = viewer.add_scene_overlay(SceneBoundingBox(name="box"))

    targets = overlay_targets(viewer)

    assert [t.key for t in targets] == [first.id, second.id]
    assert [t.label for t in targets] == ["box", "box (2)"]
    assert targets[0].visual_ids == [first.id]


def test_overlay_control_specs_read_the_model():
    box = SceneBoundingBox(
        name="box", appearance=SceneBoundingBoxAppearance(thickness=3.0)
    )
    specs = {spec.kind: spec for spec in overlay_control_specs(box)}

    assert list(specs) == list(OVERLAY_CONTROLS["scene_bounding_box"])
    assert specs["appearance.thickness"].values == {"initial_value": 3.0}
    assert specs["visible"].title == "Visible"

    corner = {
        spec.kind: spec for spec in overlay_control_specs(CenteredAxes2D(name="a"))
    }["appearance.corner"]
    assert "top_left" in corner.values["choices"]


def test_an_unknown_presentation_is_rejected():
    with pytest.raises(ValueError, match="OverlayControls"):
        OverlayControls(presentation="tabs")


# ---------------------------------------------------------------------------
# Widgets and the dock, on both toolkits
# ---------------------------------------------------------------------------


def _render(viewer, host_cls, presentation="collapsible_sections"):
    closeables: list = []
    root = render_dock(
        OverlayControls(presentation=presentation), viewer, host_cls(), closeables
    )
    (dock,) = closeables
    return dock, root


def _widget(dock, field: str):
    return next(w for w in dock.widgets if w.field == field)


def test_the_dock_follows_the_viewer(toolkit):
    gui, host_cls = toolkit
    viewer = _viewer(gui)
    dock, _root = _render(viewer, host_cls)
    assert dock.targets == []
    assert dock._placeholder == OVERLAY_PLACEHOLDER

    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    axes = viewer.add_canvas_overlay(CenteredAxes2D(name="axes"))

    assert [t.label for t in dock.targets] == ["box", "axes"]
    assert len(dock.widgets) == len(OVERLAY_CONTROLS["scene_bounding_box"]) + len(
        OVERLAY_CONTROLS["centered_axes_2d"]
    )

    viewer.remove_overlay(box)
    assert [t.key for t in dock.targets] == [axes.id]
    dock.close()


def test_a_dock_edit_reaches_the_model_and_back(toolkit):
    gui, host_cls = toolkit
    viewer = _viewer(gui)
    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    dock, _root = _render(viewer, host_cls)

    thickness = _widget(dock, "appearance.thickness")
    thickness.value = 5.0
    assert box.appearance.thickness == 5.0

    box.appearance.thickness = 2.5
    assert thickness.value == pytest.approx(2.5)

    visible = _widget(dock, "visible")
    visible.value = False
    assert box.visible is False

    box.appearance = SceneBoundingBoxAppearance(color=(1.0, 0.0, 0.0, 1.0))
    color = _widget(dock, "appearance.color")
    assert tuple(color.value) == pytest.approx((1.0, 0.0, 0.0, 1.0))
    dock.close()


def test_the_selector_presentation_names_overlays(toolkit):
    gui, host_cls = toolkit
    viewer = _viewer(gui)
    viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    viewer.add_scene_overlay(SceneBoundingBox(name="other"))

    dock, _root = _render(viewer, host_cls, presentation="selector")

    assert dock.selector is not None
    assert dock._selector_title == OVERLAY_SELECTOR_TITLE
    dock.close()


def test_closing_the_dock_unsubscribes_its_widgets(toolkit):
    gui, host_cls = toolkit
    viewer = _viewer(gui)
    box = viewer.add_scene_overlay(SceneBoundingBox(name="box"))
    dock, _root = _render(viewer, host_cls)
    thickness = _widget(dock, "appearance.thickness")

    dock.close()
    box.appearance.thickness = 9.0

    assert thickness.value != pytest.approx(9.0)
