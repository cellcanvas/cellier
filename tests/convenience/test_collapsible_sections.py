"""``AppearanceControls(presentation="collapsible_sections")``.

Every configured visual is shown at once, each in its own collapsible section,
and the dock follows the viewer exactly as the selector presentation does
(``tests/convenience/test_multi_visual_controls.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.convenience import AppearanceControls, Viewer
from cellier.convenience.gui._controls_config import InMemoryImageControlsConfig
from cellier.convenience.layout._controls_dock import APPEARANCE_PLACEHOLDER
from cellier.convenience.layout._walk import render_dock
from cellier.data import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageAppearance, InMemoryImageSingleAppearance

_SECTIONS = "collapsible_sections"


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


def _add_image(viewer, name: str):
    return viewer.add_image(
        ImageMemoryStore(data=np.zeros((4, 8, 8), dtype=np.float32)),
        appearance=InMemoryImageAppearance(),
        name=name,
        controls=InMemoryImageControlsConfig(appearance=["clim"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )


def _render(viewer, host_cls):
    closeables: list = []
    root = render_dock(
        AppearanceControls(presentation=_SECTIONS), viewer, host_cls(), closeables
    )
    (dock,) = closeables
    return dock, root


def _subscriptions_of(viewer, widget) -> list:
    bus = viewer.controller._outgoing_events
    return [
        sub for subs in bus._subs.values() for sub in subs if sub.owner_id == widget._id
    ]


def _assert_shows_sections(dock, root, gui: str) -> None:
    """The slot displays exactly the sections, and each holds its controls."""
    if gui == "qt":
        assert all(root.isAncestorOf(section.widget) for section in dock.sections)
    else:
        assert list(root.children) == [section.widget for section in dock.sections]
    for target, section in zip(dock.targets, dock.sections):
        widgets = dock._built[target.key][1]
        assert widgets
        if gui == "qt":
            assert all(section.content.isAncestorOf(w.widget) for w in widgets)
        else:
            assert list(section.children) == [w.widget for w in widgets]


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def test_the_selector_is_still_the_default():
    assert AppearanceControls().presentation == "selector"


def test_an_unknown_presentation_is_refused_at_construction():
    with pytest.raises(ValueError, match="collapsible_sections"):
        AppearanceControls(presentation="tabs")


# ---------------------------------------------------------------------------
# The dock, on both toolkits
# ---------------------------------------------------------------------------


def test_every_configured_visual_gets_a_titled_section(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    _add_image(viewer, "b")

    dock, root = _render(viewer, host_cls)

    assert dock.selector is None
    assert [section.title for section in dock.sections] == ["a", "b"]
    _assert_shows_sections(dock, root, gui)


def test_only_the_first_section_starts_expanded(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    _add_image(viewer, "b")

    dock, _root = _render(viewer, host_cls)

    assert [section.expanded for section in dock.sections] == [True, False]


def test_every_sections_controls_are_built_and_wired(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    _add_image(viewer, "b")

    dock, _root = _render(viewer, host_cls)

    per_target = [dock._built[target.key][1] for target in dock.targets]
    assert all(per_target)
    assert dock.widgets == [w for widgets in per_target for w in widgets]
    assert all(_subscriptions_of(viewer, widget) for widget in dock.widgets)


def test_an_added_visual_gets_a_collapsed_section_and_the_rest_are_kept(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    dock, root = _render(viewer, host_cls)
    (first,) = dock.sections
    first.set_expanded(False)

    _add_image(viewer, "b")

    assert dock.sections[0] is first
    assert not first.expanded  # the user's choice survives the refresh
    assert [section.title for section in dock.sections] == ["a", "b"]
    assert not dock.sections[1].expanded
    _assert_shows_sections(dock, root, gui)


def test_removing_a_visual_drops_its_section_and_releases_its_controls(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    a = _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, root = _render(viewer, host_cls)
    kept = dock.sections[0]
    released = list(dock._built[dock.targets[1].key][1])

    viewer.controller.remove_visual(b.id)

    assert [target.key for target in dock.targets] == [a.id]
    assert dock.sections == [kept]
    assert all(_subscriptions_of(viewer, widget) == [] for widget in released)
    _assert_shows_sections(dock, root, gui)


def test_a_surviving_section_is_retitled_when_its_label_changes(toolkit):
    """Duplicate names are told apart by position, so a removal renames one."""
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    first = _add_image(viewer, "image")
    _add_image(viewer, "image")
    dock, _root = _render(viewer, host_cls)
    second = dock.sections[1]
    assert second.title == "image (2)"

    viewer.controller.remove_visual(first.id)

    assert dock.sections == [second]
    assert second.title == "image"


def test_an_empty_dock_shows_the_placeholder_and_a_later_add_opens(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    dock, root = _render(viewer, host_cls)

    if gui == "qt":
        from qtpy.QtWidgets import QLabel

        assert APPEARANCE_PLACEHOLDER in [
            label.text() for label in root.findChildren(QLabel)
        ]
    else:
        assert list(root.children) == []
        assert root.title == APPEARANCE_PLACEHOLDER

    _add_image(viewer, "a")

    assert [section.expanded for section in dock.sections] == [True]
    _assert_shows_sections(dock, root, gui)


def test_select_expands_the_targets_section(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, _root = _render(viewer, host_cls)

    dock.select(b.id)

    assert dock.selected.key == b.id
    assert dock.sections[1].expanded


def test_expanding_never_builds_widgets(toolkit):
    """As with the selector: nothing may be built outside a marimo cell."""
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    b = _add_image(viewer, "b")
    dock, _root = _render(viewer, host_cls)
    builds: list = []
    original = dock._build
    dock._build = lambda target: (builds.append(target), original(target))[1]

    dock.select(b.id)
    dock.sections[0].set_expanded(False)

    assert builds == []


def test_closing_the_dock_stops_following_the_viewer(toolkit):
    gui, host_cls = toolkit
    viewer = Viewer(spatial_axes("z", "y", "x"), gui=gui)
    _add_image(viewer, "a")
    dock, _root = _render(viewer, host_cls)
    old = dock.widgets

    dock.close()
    _add_image(viewer, "b")

    assert dock.sections == []
    assert dock.widgets == []
    assert all(_subscriptions_of(viewer, widget) == [] for widget in old)


# ---------------------------------------------------------------------------
# The anywidget section itself
# ---------------------------------------------------------------------------


def test_the_anywidget_section_sends_children_as_composition_references():
    """Composition references are what let a section mount inside a slot on marimo."""
    pytest.importorskip("anywidget")
    from uuid import uuid4

    from cellier.gui.anywidget._collapsible_section import (
        AnywidgetCollapsibleSection,
    )
    from cellier.gui.anywidget.visuals import AnywidgetVisibleToggle

    child = AnywidgetVisibleToggle(uuid4(), initial_value=True)
    section = AnywidgetCollapsibleSection("image", [child], expanded=True)

    state = section.get_state()
    assert state["children"] == [f"anywidget:{child.model_id}"]
    assert state["title"] == "image"
    assert state["expanded"] is True
