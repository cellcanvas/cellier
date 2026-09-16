"""A controls dock serves every configured visual and follows its viewer.

``plans/multi_visual_controls.md``: an ``AppearanceControls()`` dock used to build
controls for the *first* configured visual and nothing else, once, at render time.  It
now offers a selector over every configured visual and rebuilds as visuals are added and
removed.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from cellier.convenience import (
    AppearanceControls,
    OrthoViewer,
    Viewer,
)
from cellier.convenience.gui._controls_config import InMemoryImageControlsConfig
from cellier.convenience.layout import Layout, RenderControls, VStack
from cellier.convenience.layout._controls_dock import APPEARANCE_PLACEHOLDER
from cellier.convenience.layout._walk import render_dock
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance

_PANELS = ("xy", "xz", "yz", "vol")


def _store() -> ImageMemoryStore:
    return ImageMemoryStore(data=np.zeros((4, 8, 8), dtype=np.float32))


def _add_image(viewer, name: str, *, controls: bool = True):
    return viewer.add_image(
        _store(),
        appearance=InMemoryImageAppearance(),
        name=name,
        controls=InMemoryImageControlsConfig(appearance=["clim"]) if controls else None,
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )


def _changes(viewer) -> list:
    calls: list = []
    viewer._controls_changed.connect(lambda: calls.append(None))
    return calls


def _subscriptions_of(viewer, widget) -> list:
    bus = viewer.controller._outgoing_events
    return [
        sub for subs in bus._subs.values() for sub in subs if sub.owner_id == widget._id
    ]


# ---------------------------------------------------------------------------
# The registry: recorded on add, pruned on removal
# ---------------------------------------------------------------------------


def test_recording_controls_emits_once():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    calls = _changes(viewer)

    visual = _add_image(viewer, "a")

    assert len(calls) == 1
    assert list(viewer._controls_configs) == [visual.id]


def test_an_add_without_controls_emits_nothing():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    calls = _changes(viewer)

    _add_image(viewer, "a", controls=False)

    assert calls == []


def test_removing_a_configured_visual_prunes_and_emits():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    a = _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    calls = _changes(viewer)

    viewer.controller.remove_visual(a.id)

    assert len(calls) == 1
    assert list(viewer._controls_configs) == [b.id]
    assert a.id not in viewer._visual_groups


def test_removing_an_unconfigured_visual_emits_nothing():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    _add_image(viewer, "a")
    plain = _add_image(viewer, "plain", controls=False)
    calls = _changes(viewer)

    viewer.controller.remove_visual(plain.id)

    assert calls == []


def test_removing_a_scene_prunes_its_visuals():
    viewer = Viewer(spatial_axes("z", "y", "x"))
    _add_image(viewer, "a")
    _add_image(viewer, "b")

    viewer.controller.remove_scene(viewer.scene.id)

    assert viewer._controls_configs == {}
    assert viewer._visual_groups == {}


def test_ortho_removing_a_sibling_keeps_the_group():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = _add_image(ortho, "a")
    rep_id = visuals["xy"].id

    ortho.controller.remove_visual(visuals["yz"].id)

    assert list(ortho._controls_configs) == [rep_id]
    assert ortho._visual_groups[rep_id] == [
        visuals[key].id for key in ("xy", "xz", "vol")
    ]


def test_ortho_removing_the_representative_rekeys_in_place():
    """The config moves to the next sibling and keeps its place in the order."""
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    a = _add_image(ortho, "a")
    b = _add_image(ortho, "b")
    config = ortho._controls_configs[a["xy"].id]

    ortho.controller.remove_visual(a["xy"].id)

    assert list(ortho._controls_configs) == [a["xz"].id, b["xy"].id]
    assert ortho._controls_configs[a["xz"].id] is config
    assert ortho._visual_groups[a["xz"].id] == [
        a[key].id for key in ("xz", "yz", "vol")
    ]


def test_ortho_removing_every_panel_drops_the_entry():
    ortho = OrthoViewer(spatial_axes("z", "y", "x"))
    visuals = _add_image(ortho, "a")

    for key in _PANELS:
        ortho.controller.remove_visual(visuals[key].id)

    assert ortho._controls_configs == {}
    assert ortho._visual_groups == {}


def test_a_dropped_viewer_is_not_kept_alive_by_its_controller():
    """The removal subscription is weak: the controller may outlive the viewer."""
    viewer = Viewer(spatial_axes("z", "y", "x"))
    controller = viewer.controller
    ref = weakref.ref(viewer)

    del viewer
    gc.collect()

    assert ref() is None
    assert controller is not None


# ---------------------------------------------------------------------------
# The dock, on both toolkits
# ---------------------------------------------------------------------------


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


def _render(viewer, host_cls, spec=None):
    closeables: list = []
    root = render_dock(
        AppearanceControls() if spec is None else spec, viewer, host_cls(), closeables
    )
    (dock,) = closeables
    return dock, root


def _assert_shows(dock, root, gui: str) -> None:
    """The slot displays the selector (when there is one) and the controls."""
    shown = ([dock.selector] if dock.selector is not None else []) + dock.widgets
    if gui == "qt":
        assert all(root.isAncestorOf(item.widget) for item in shown)
    else:
        assert list(root.children) == [item.widget for item in shown]


def _assert_placeholder(root, gui: str, text: str) -> None:
    if gui == "qt":
        from qtpy.QtWidgets import QLabel

        assert text in [label.text() for label in root.findChildren(QLabel)]
    else:
        assert list(root.children) == []
        assert root.title == text


def test_two_configured_visuals_get_a_selector(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    a = _add_image(viewer, "a")
    _add_image(viewer, "b")

    dock, root = _render(viewer, host_cls)

    assert dock.selector.labels == ("a", "b")
    assert dock.selector.index == 0
    assert dock.selected.visual_ids == [a.id]
    assert dock.widgets
    _assert_shows(dock, root, gui)


def test_one_configured_visual_has_no_selector(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")

    dock, root = _render(viewer, host_cls)

    assert dock.selector is None
    assert dock.widgets
    _assert_shows(dock, root, gui)


def test_selecting_swaps_to_the_other_visuals_controls(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, root = _render(viewer, host_cls)
    a_widgets = dock.widgets

    dock.selector.select(1)

    assert dock.selected.visual_ids == [b.id]
    assert dock.selector.index == 1
    assert not any(widget in a_widgets for widget in dock.widgets)
    _assert_shows(dock, root, gui)

    # Every target's controls stay built and wired, so switching back shows
    # the same objects rather than a rebuild.
    assert all(_subscriptions_of(viewer, widget) for widget in a_widgets)
    dock.selector.select(0)
    assert dock.widgets == a_widgets
    _assert_shows(dock, root, gui)


def test_selecting_never_builds_widgets(toolkit):
    """On marimo a widget cannot be built outside a running cell.

    A selection arrives as a front-end message, outside any cell, so building
    there raised inside marimo and left the dock showing closed controls.
    """
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    _add_image(viewer, "b")
    dock, _root = _render(viewer, host_cls)
    builds: list = []
    original = dock._build
    dock._build = lambda target: (builds.append(target), original(target))[1]

    dock.selector.select(1)
    dock.selector.select(0)

    assert builds == []


def test_hidden_qt_controls_survive_the_old_column_being_deleted(qtbot):
    """Swapping deletes the old column; controls not shown must not go with it."""
    from qtpy.QtCore import QCoreApplication, QEvent

    from cellier.convenience._hosts import QtLayoutHost

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    _add_image(viewer, "a")
    _add_image(viewer, "b")
    dock, root = _render(viewer, QtLayoutHost)
    a_widgets = dock.widgets

    dock.selector.select(1)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    dock.selector.select(0)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    # A deleted C++ object raises RuntimeError on any call.
    assert all(root.isAncestorOf(widget.widget) for widget in a_widgets)


def test_the_selected_controls_drive_the_selected_visual(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    a = _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, _root = _render(viewer, host_cls)
    dock.selector.select(1)

    viewer.controller.update_single_appearance_field(b.id, "clim", (0.2, 0.8))

    # The selected control follows b, and a is untouched.
    assert a.single.clim == pytest.approx((0.0, 1.0))
    assert any(_subscriptions_of(viewer, widget) for widget in dock.widgets)


def test_a_visual_added_after_render_joins_without_taking_the_selection(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, root = _render(viewer, host_cls)
    dock.selector.select(1)
    widgets = dock.widgets

    _add_image(viewer, "c")

    assert dock.selector.labels == ("a", "b", "c")
    assert dock.selector.index == 1
    assert dock.selected.visual_ids == [b.id]
    assert dock.widgets == widgets  # not rebuilt
    _assert_shows(dock, root, gui)


def test_a_second_visual_added_after_render_brings_the_selector(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    dock, root = _render(viewer, host_cls)
    assert dock.selector is None

    _add_image(viewer, "b")

    assert dock.selector.labels == ("a", "b")
    _assert_shows(dock, root, gui)


def test_removing_the_selected_visual_falls_back_to_the_first(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    a = _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, root = _render(viewer, host_cls)
    dock.selector.select(1)
    old = dock.widgets

    viewer.controller.remove_visual(b.id)

    assert dock.selected.visual_ids == [a.id]
    assert dock.selector is None  # one target left
    assert all(_subscriptions_of(viewer, widget) == [] for widget in old)
    _assert_shows(dock, root, gui)


def test_removing_every_visual_shows_the_placeholder_and_an_add_fills_it(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    a = _add_image(viewer, "a")
    dock, root = _render(viewer, host_cls)

    viewer.controller.remove_visual(a.id)

    assert dock.targets == []
    assert dock.widgets == []
    _assert_placeholder(root, gui, APPEARANCE_PLACEHOLDER)

    c = _add_image(viewer, "c")

    assert dock.selected.visual_ids == [c.id]
    assert dock.selector is None
    _assert_shows(dock, root, gui)


def test_an_unconfigured_viewer_renders_a_placeholder_a_later_add_fills(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    dock, root = _render(viewer, host_cls)
    _assert_placeholder(root, gui, APPEARANCE_PLACEHOLDER)

    a = _add_image(viewer, "a")

    assert dock.selected.visual_ids == [a.id]
    _assert_shows(dock, root, gui)


def test_duplicate_names_are_told_apart(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "image")
    _add_image(viewer, "image")

    dock, _root = _render(viewer, host_cls)

    assert dock.selector.labels == ("image", "image (2)")


def test_closing_the_dock_stops_following_the_viewer(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    dock, _root = _render(viewer, host_cls)
    old = dock.widgets

    dock.close()
    _add_image(viewer, "b")

    assert len(dock.targets) == 1
    assert dock.widgets == []
    assert all(_subscriptions_of(viewer, widget) == [] for widget in old)


def test_the_ortho_dock_drives_every_panel_of_the_selected_add(toolkit):
    gui, host_cls = toolkit
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(ortho, "a")
    b = _add_image(ortho, "b")

    dock, _root = _render(ortho, host_cls)

    assert dock.selector.labels == ("a", "b")
    dock.selector.select(1)
    assert dock.selected.visual_ids == [b[key].id for key in _PANELS]


def test_the_ortho_dock_keeps_its_selection_when_the_representative_goes(toolkit):
    gui, host_cls = toolkit
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(ortho, "a")
    b = _add_image(ortho, "b")
    dock, _root = _render(ortho, host_cls)
    dock.selector.select(1)

    ortho.controller.remove_visual(b["xy"].id)

    assert dock.selector.index == 1
    assert dock.selected.visual_ids == [b[key].id for key in ("xz", "yz", "vol")]


# ---------------------------------------------------------------------------
# The selector widgets
# ---------------------------------------------------------------------------


def _selector(toolkit):
    gui, _host_cls = toolkit
    from cellier.convenience._backend import backend_for

    return backend_for(gui).target_selector(["a", "b"], 0, title="Visual")


def test_set_choices_does_not_emit(toolkit):
    selector = _selector(toolkit)
    emitted: list = []
    selector.selected.connect(emitted.append)

    selector.set_choices(["a", "b", "c"], 2)

    assert emitted == []
    assert selector.labels == ("a", "b", "c")
    assert selector.index == 2


def test_select_emits_the_index(toolkit):
    selector = _selector(toolkit)
    emitted: list = []
    selector.selected.connect(emitted.append)

    selector.select(1)

    assert emitted == [1]
    assert selector.index == 1


def test_the_slot_sends_children_as_composition_references():
    pytest.importorskip("anywidget")
    from uuid import uuid4

    from cellier.gui.anywidget import AnywidgetSlot
    from cellier.gui.anywidget.visuals import AnywidgetVisibleToggle

    child = AnywidgetVisibleToggle(uuid4(), initial_value=True)
    slot = AnywidgetSlot(children=[child])

    assert slot.get_state()["children"] == [f"anywidget:{child.model_id}"]


# ---------------------------------------------------------------------------
# Layout.single: controls sharing a dock stack instead of overwriting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"appearance": "right", "render": "right"},
            {"right_dock": [AppearanceControls, RenderControls]},
        ),
        (
            {"appearance": "top", "render": "top"},
            {"top_dock": [AppearanceControls, RenderControls]},
        ),
        (
            {"appearance": "left", "render": "right"},
            {"left_dock": [AppearanceControls], "right_dock": [RenderControls]},
        ),
    ],
)
def test_layout_single_stacks_controls_sharing_a_dock(kwargs, expected):
    layout = Layout.single("canvas", **kwargs)

    for name in ("left_dock", "right_dock", "top_dock", "bottom_dock"):
        spec = getattr(layout, name)
        types = expected.get(name)
        if types is None:
            assert spec is None
        elif len(types) == 1:
            assert type(spec) is types[0]
        else:
            assert isinstance(spec, VStack)
            assert [type(item) for item in spec.items] == types
