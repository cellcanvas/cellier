"""The ortho clipping widget, its dock node, and the flag / dock node check.

``plans/clipping_planes_linking_design.md`` section 6.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import (
    AppearanceControls,
    HStack,
    LabelsControlsConfig,
    Layout,
    OrthoClippingControls,
    OrthoViewer,
    RenderControls,
    Viewer,
    VStack,
)
from cellier.convenience._hosts import JupyterHost, QtLayoutHost
from cellier.convenience.gui._ortho_clipping import (
    MODES,
    PLACEHOLDER,
    OrthoClippingWidget,
)
from cellier.convenience.layout._shared import dock_node_names, missing_dock_node
from cellier.convenience.layout._walk import render_dock, render_layout
from cellier.data import LabelMemoryStore
from cellier.gui._clipping_planes import rows_from_planes
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane

PANELS = ("xy", "xz", "yz", "vol")
TOOLKITS = ["qt", "anywidget"]

#: Mode -> per dataset, (title after the dataset's name, panels, gizmo toggle).
EXPECTED = {
    "all": [("All views", ("xy", "xz", "yz", "vol"), True)],
    "2d": [("2D views", ("xy", "xz", "yz"), False), ("3D view", ("vol",), True)],
    None: [
        ("XY", ("xy",), False),
        ("XZ", ("xz",), False),
        ("YZ", ("yz",), False),
        ("3D view", ("vol",), True),
    ],
}


def _host(toolkit):
    return QtLayoutHost() if toolkit == "qt" else JupyterHost()


def _add(ortho, name="cells", *, flagged=True, appearance=True):
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in "zyx"),
    )
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = ClippingPlane.from_point_normal(system, (4, 4, 4.0), (0, 0, 1))
    return ortho.add_labels(
        store,
        name=name,
        clipping_planes=(plane,),
        controls=LabelsControlsConfig(appearance=appearance, clipping_controls=flagged),
    )


@pytest.fixture
def make_ortho(qtbot):
    """Ortho viewers with their canvases, closed after the test."""
    made = []

    def _make(mode="all") -> OrthoViewer:
        ortho = OrthoViewer(
            spatial_axes("z", "y", "x"), gui="offscreen", link_clipping_planes=mode
        )
        for scene in ortho.scenes.values():
            ortho.controller.add_canvas(scene.id)
        made.append(ortho)
        return ortho

    yield _make
    for ortho in made:
        ortho.controller.close()


@pytest.fixture
def make_widget(qtbot):
    """Ortho clipping widgets, closed after the test."""
    made = []

    def _make(ortho, toolkit) -> OrthoClippingWidget:
        widget = OrthoClippingWidget(ortho, _host(toolkit))
        if toolkit == "qt":
            qtbot.addWidget(widget.root)
        made.append(widget)
        return widget

    yield _make
    for widget in made:
        widget.close()


def _title(control) -> str:
    if hasattr(control, "comm"):
        return control.title
    return control.widget.title()


def _shown(widget, visuals_by_name) -> list[tuple]:
    """``(title, panels, has a gizmo toggle)`` of each control on show."""
    panel_of = {
        visual.id: key
        for visuals in visuals_by_name.values()
        for key, visual in visuals.items()
    }
    return [
        (
            _title(control),
            tuple(panel_of[i] for i in control.visual_ids),
            control.editor.has_gizmo,
        )
        for control in widget.widgets
    ]


def _in_step(widget, controller) -> bool:
    """Every control, shown or hidden, shows the planes of its first visual."""
    return all(
        rows_from_planes(
            controller.get_visual_model(control.visual_ids[0]).clipping_planes
        )
        == [dict(row) for row in control.editor.rows]
        for controls in widget.controls
        for control in controls.values()
    )


def _slot_children(widget, toolkit) -> list:
    """What the slot holds now: the selector first, then the controls."""
    if toolkit == "qt":
        return list(widget._slot._leaves)
    return list(widget._slot.slot.children)


# -- the widget -----------------------------------------------------------------


@pytest.mark.parametrize("toolkit", TOOLKITS)
@pytest.mark.parametrize("mode", MODES)
def test_one_control_per_link_group(make_ortho, make_widget, toolkit, mode):
    ortho = make_ortho(mode)
    visuals = {"cells": _add(ortho, "cells"), "nuclei": _add(ortho, "nuclei")}
    _add(ortho, "plain", flagged=False)  # no clipping controls asked for
    widget = make_widget(ortho, toolkit)

    assert widget.datasets == ["cells", "nuclei"]
    assert widget.selector.index == MODES.index(mode)
    assert widget.selector.labels == ("All views", "2D views", "Not linked")
    assert _shown(widget, visuals) == [
        (f"{name}: {title}", panels, gizmo)
        for name in ("cells", "nuclei")
        for title, panels, gizmo in EXPECTED[mode]
    ]
    # Six controls are built per dataset, whatever the mode shows.
    assert [sorted(controls) for controls in widget.controls] == [
        ["2d", "all", "vol", "xy", "xz", "yz"]
    ] * 2
    # The slot shows the selector and then the controls.
    children = _slot_children(widget, toolkit)
    assert len(children) == 1 + 2 * len(EXPECTED[mode])


@pytest.mark.parametrize("toolkit", TOOLKITS)
@pytest.mark.parametrize(
    ("old", "new"), [(a, b) for a in MODES for b in MODES if a != b]
)
@pytest.mark.parametrize("through", ["mode", "selector"])
def test_a_mode_change_swaps_what_is_shown_and_builds_nothing(
    make_ortho, make_widget, toolkit, old, new, through
):
    ortho = make_ortho(old)
    controller = ortho.controller
    visuals = {"cells": _add(ortho)}
    widget = make_widget(ortho, toolkit)
    built = [control for controls in widget.controls for control in controls.values()]
    # Make the panels differ as far as the old mode lets them.
    for x, key in zip((1.0, 2.0, 3.0, 5.0), PANELS):
        visual = visuals["cells"][key]
        first = visual.clipping_planes[0]
        plane = first.plane.model_copy(update={"offset": x})
        controller.set_clipping_planes(
            visual.id, (first.model_copy(update={"plane": plane}),)
        )

    if through == "mode":
        ortho.clipping_controller.mode = new
    else:
        widget.selector.select(MODES.index(new))  # as a user would
    assert ortho.clipping_controller.mode == new
    assert widget.selector.index == MODES.index(new)
    assert _shown(widget, visuals) == [
        (f"cells: {title}", panels, gizmo) for title, panels, gizmo in EXPECTED[new]
    ]
    now = [control for controls in widget.controls for control in controls.values()]
    assert all(a is b for a, b in zip(built, now))
    assert len(built) == len(now)
    # A control shows the state of every visual it writes to.
    for control in widget.widgets:
        rows = [dict(row) for row in control.editor.rows]
        for visual_id in control.visual_ids:
            planes = controller.get_visual_model(visual_id).clipping_planes
            assert rows_from_planes(planes) == rows


@pytest.mark.parametrize("toolkit", TOOLKITS)
@pytest.mark.parametrize("mode", MODES)
def test_an_edit_reaches_the_controls_link_group_only(
    make_ortho, make_widget, toolkit, mode
):
    ortho = make_ortho(mode)
    visuals = _add(ortho)
    other = _add(ortho, "other")
    widget = make_widget(ortho, toolkit)

    control = widget.widgets[0]
    control.editor.set_position(0, 7.0)
    reached = {
        key for key, v in visuals.items() if v.clipping_planes[0].plane.offset == 7.0
    }
    assert reached == set(EXPECTED[mode][0][1])
    assert all(v.clipping_planes[0].plane.offset == 4.0 for v in other.values())
    # Hidden controls heard it too: correct the moment they are shown.
    assert _in_step(widget, ortho.controller)
    assert control.editor.rows[0]["position"] == 7.0


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_the_gizmo_toggle_follows_the_canvas_across_modes(
    make_ortho, make_widget, toolkit
):
    ortho = make_ortho("all")
    controller = ortho.controller
    visuals = _add(ortho)
    widget = make_widget(ortho, toolkit)
    vol_canvas = ortho._panel_canvas("vol")
    plane = visuals["vol"].clipping_planes[0]

    (all_views,) = widget.widgets
    all_views.editor.set_gizmo(0, True)
    session = controller.get_clipping_plane_gizmo(vol_canvas)
    assert (session.visual_id, session.plane_id) == (visuals["vol"].id, plane.id)
    assert all_views.editor.gizmo_plane == str(plane.id)

    # The hidden "3D view" control heard it, so it is right when shown.
    ortho.clipping_controller.mode = "2d"
    flat, volume = widget.widgets
    assert not flat.editor.has_gizmo
    assert volume.editor.gizmo_plane == str(plane.id)
    assert not session.closed

    volume.editor.set_gizmo(0, False)
    assert controller.get_clipping_plane_gizmo(vol_canvas) is None
    ortho.clipping_controller.mode = "all"
    assert all_views.editor.gizmo_plane is None


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_the_widget_follows_datasets_added_and_removed(
    make_ortho, make_widget, toolkit
):
    ortho = make_ortho("2d")
    widget = make_widget(ortho, toolkit)
    # No flagged dataset: the selector and a placeholder.
    assert widget.datasets == []
    assert widget.widgets == []
    assert len(_slot_children(widget, toolkit)) == 1
    if toolkit == "anywidget":
        assert widget._slot.slot.title == PLACEHOLDER
    # The selector works with nothing under it.
    widget.selector.select(MODES.index(None))
    assert ortho.clipping_controller.mode is None
    ortho.clipping_controller.mode = "2d"

    first = _add(ortho, "cells")
    assert widget.datasets == ["cells"]
    if toolkit == "anywidget":
        assert widget._slot.slot.title == ""
    kept = widget.controls[0]
    second = _add(ortho, "cells")  # the same name again
    assert widget.datasets == ["cells", "cells (2)"]
    assert all(widget.controls[0][name] is kept[name] for name in kept)
    assert [_title(c) for c in widget.widgets] == [
        "cells: 2D views",
        "cells: 3D view",
        "cells (2): 2D views",
        "cells (2): 3D view",
    ]

    for visual in first.values():
        ortho.controller.remove_visual(visual.id)
    # The one that is left is now the only "cells": its controls are rebuilt
    # with that title.
    assert widget.datasets == ["cells"]
    assert [tuple(c.visual_ids) for c in widget.widgets] == [
        tuple(second[key].id for key in ("xy", "xz", "yz")),
        (second["vol"].id,),
    ]
    for visual in second.values():
        ortho.controller.remove_visual(visual.id)
    assert widget.datasets == []
    assert len(_slot_children(widget, toolkit)) == 1


#: Mode -> the shown controls' panels with yz gone; then with vol gone too.
_WITHOUT_YZ = {
    "all": [("xy", "xz", "vol")],
    "2d": [("xy", "xz"), ("vol",)],
    None: [("xy",), ("xz",), ("vol",)],
}
_WITHOUT_YZ_VOL = {
    "all": [("xy", "xz")],
    "2d": [("xy", "xz")],
    None: [("xy",), ("xz",)],
}
#: The same, as titles: a control of several panels that has lost some is
#: called by the panels it has left.
_TITLES_WITHOUT_YZ = {
    "all": ["cells: XY, XZ, 3D view"],
    "2d": ["cells: XY, XZ", "cells: 3D view"],
    None: ["cells: XY", "cells: XZ", "cells: 3D view"],
}
_TITLES_WITHOUT_YZ_VOL = {
    "all": ["cells: XY, XZ"],
    "2d": ["cells: XY, XZ"],
    None: ["cells: XY", "cells: XZ"],
}


@pytest.mark.parametrize("toolkit", TOOLKITS)
@pytest.mark.parametrize("mode", MODES)
def test_a_dataset_that_lost_panels_keeps_controls_for_the_rest(
    make_ortho, make_widget, toolkit, mode
):
    """L-D21: the dataset stays in the widget."""
    ortho = make_ortho(mode)
    controller = ortho.controller
    visuals = _add(ortho)
    widget = make_widget(ortho, toolkit)

    controller.remove_visual(visuals.pop("yz").id)
    assert widget.datasets == ["cells"]
    assert sorted(widget.controls[0]) == ["2d", "all", "vol", "xy", "xz"]
    shown = _shown(widget, {"cells": visuals})
    assert [panels for _title, panels, _gizmo in shown] == _WITHOUT_YZ[mode]
    assert [title for title, _panels, _gizmo in shown] == _TITLES_WITHOUT_YZ[mode]
    assert [gizmo for _title, _panels, gizmo in shown] == [
        "vol" in panels for panels in _WITHOUT_YZ[mode]
    ]
    assert _in_step(widget, controller)

    # An edit through the rebuilt control reaches the panels that are left.
    widget.widgets[0].editor.set_position(0, 7.0)
    assert {
        key for key, v in visuals.items() if v.clipping_planes[0].plane.offset == 7.0
    } == set(_WITHOUT_YZ[mode][0])

    # Without the 3D panel no control has a gizmo toggle.
    controller.remove_visual(visuals.pop("vol").id)
    assert sorted(widget.controls[0]) == ["2d", "all", "xy", "xz"]
    shown = _shown(widget, {"cells": visuals})
    assert [panels for _title, panels, _gizmo in shown] == _WITHOUT_YZ_VOL[mode]
    assert [title for title, _panels, _gizmo in shown] == _TITLES_WITHOUT_YZ_VOL[mode]
    assert not any(gizmo for _title, _panels, gizmo in shown)

    for visual in visuals.values():
        controller.remove_visual(visual.id)
    assert widget.datasets == []


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_a_closed_widget_releases_its_controls(make_ortho, toolkit, qtbot):
    ortho = make_ortho()
    controller = ortho.controller
    visuals = _add(ortho)
    bus = controller._outgoing_events
    before = sum(len(subs) for subs in bus._subs.values())

    widget = OrthoClippingWidget(ortho, _host(toolkit))
    if toolkit == "qt":
        qtbot.addWidget(widget.root)
    assert sum(len(subs) for subs in bus._subs.values()) > before
    widget.close()
    assert sum(len(subs) for subs in bus._subs.values()) == before
    # Nothing follows the viewer any more.
    ortho.clipping_controller.mode = None
    _add(ortho, "later")
    controller.remove_visual(visuals["yz"].id)
    assert widget.controls == []


def test_the_widget_needs_the_3d_canvas_built_first(qtbot):
    """No lazy binding: a control is not built before its gizmo's canvas."""
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        # Nothing to build yet, so nothing to bind.
        empty = OrthoClippingWidget(ortho, JupyterHost())
        empty.close()
        _add(ortho)
        with pytest.raises(ValueError, match="no canvas yet"):
            OrthoClippingWidget(ortho, JupyterHost())
    finally:
        ortho.controller.close()


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_a_build_that_fails_part_way_releases_what_it_built(
    make_ortho, make_widget, toolkit, monkeypatch
):
    """Controls built before the failure are not left wired to the bus."""
    ortho = make_ortho()
    bus = ortho.controller._outgoing_events
    widget = make_widget(ortho, toolkit)
    before = sum(len(subs) for subs in bus._subs.values())

    target = ortho._clipping_gizmo_target
    asked = []

    def fails_on_the_third(visual_ids):
        asked.append(visual_ids)
        if len(asked) == 3:
            raise RuntimeError("no target")
        return target(visual_ids)

    monkeypatch.setattr(ortho, "_clipping_gizmo_target", fails_on_the_third)
    # psygnal wraps what a callback of ``_controls_changed`` raises.
    with pytest.raises(Exception, match="no target"):
        _add(ortho)
    assert len(asked) == 3
    assert widget.controls == []
    after = sum(len(subs) for subs in bus._subs.values())
    # What the dataset itself subscribed (the linker, the registry) is all
    # that was added: no control of the failed build is left.
    monkeypatch.undo()
    other = make_ortho()
    other_bus = other.controller._outgoing_events
    other_before = sum(len(subs) for subs in other_bus._subs.values())
    _add(other)
    other_after = sum(len(subs) for subs in other_bus._subs.values())
    assert after - before == other_after - other_before


# -- the dock node ----------------------------------------------------------------


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_the_dock_node_renders_the_widget(make_ortho, toolkit, qtbot):
    ortho = make_ortho("2d")
    visuals = {"cells": _add(ortho)}
    closeables: list = []
    root = render_dock(OrthoClippingControls(), ortho, _host(toolkit), closeables)
    (widget,) = closeables
    try:
        if toolkit == "qt":
            qtbot.addWidget(root)
        assert isinstance(widget, OrthoClippingWidget)
        assert root is widget.root
        assert _shown(widget, visuals) == [
            (f"cells: {title}", panels, gizmo)
            for title, panels, gizmo in EXPECTED["2d"]
        ]
    finally:
        widget.close()


def test_the_dock_node_is_for_an_ortho_viewer(qtbot):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        with pytest.raises(TypeError, match="needs an OrthoViewer"):
            render_dock(OrthoClippingControls(), viewer, JupyterHost(), [])
    finally:
        viewer.controller.close()


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_an_ortho_appearance_dock_has_no_clipping_control(make_ortho, toolkit, qtbot):
    from cellier.gui._clipping_planes import ClippingPlanesEditor

    ortho = make_ortho()
    _add(ortho)
    closeables: list = []
    root = render_dock(AppearanceControls(), ortho, _host(toolkit), closeables)
    (dock,) = closeables
    try:
        if toolkit == "qt":
            qtbot.addWidget(root)
        assert len(dock.targets) == 2  # the 2D group and the 3D group
        for target in dock.targets:
            dock.select(target.key)
            assert dock.widgets
            assert not any(
                isinstance(getattr(control, "editor", None), ClippingPlanesEditor)
                for control in dock.widgets
            )
    finally:
        dock.close()


# -- flags and dock nodes must match (L-D20) --------------------------------------


class _Center:
    """A center leaf that composes to nothing much, on any host."""

    def compose(self, host):
        return host.stack([])


def test_dock_node_names_walks_the_stacks():
    layout = Layout(
        center=_Center(),
        left_dock=VStack([AppearanceControls(), HStack([OrthoClippingControls()])]),
        bottom_dock=RenderControls(),
    )
    assert dock_node_names(layout) == {
        "AppearanceControls",
        "OrthoClippingControls",
        "RenderControls",
    }
    assert dock_node_names(Layout(center=_Center())) == frozenset()


_BOTH = frozenset({"AppearanceControls", "OrthoClippingControls"})
_NONE: frozenset = frozenset()
_APPEARANCE = frozenset({"AppearanceControls"})
_CLIPPING = frozenset({"OrthoClippingControls"})
_VIEWER, _ORTHO = "AppearanceControls", "OrthoClippingControls"

#: (appearance, clipping_controls, dock nodes, the viewer's clipping node,
#: what the message names, or None when every flag has its node).
_FLAG_CASES = [
    # Nothing asked for: nothing needed.
    (False, False, _NONE, _VIEWER, None),
    (False, False, _NONE, _ORTHO, None),
    # Appearance needs its dock, on both viewers.
    (True, False, _NONE, _VIEWER, "AppearanceControls()"),
    (["opacity"], False, _NONE, _ORTHO, "AppearanceControls()"),
    (True, False, _APPEARANCE, _VIEWER, None),
    # On a Viewer the clipping control is in the appearance dock.
    (True, True, _APPEARANCE, _VIEWER, None),
    (False, True, _APPEARANCE, _VIEWER, "appearance=True"),
    (False, True, _NONE, _VIEWER, "appearance=True"),
    # On an OrthoViewer it has its own dock.
    (False, True, _NONE, _ORTHO, "OrthoClippingControls()"),
    (True, True, _APPEARANCE, _ORTHO, "OrthoClippingControls()"),
    (False, True, _CLIPPING, _ORTHO, None),
    (True, True, _BOTH, _ORTHO, None),
]


@pytest.mark.parametrize(
    ("appearance", "clipping", "nodes", "clipping_node", "missing"), _FLAG_CASES
)
def test_a_flag_needs_its_dock_node(
    appearance, clipping, nodes, clipping_node, missing
):
    config = LabelsControlsConfig(appearance=appearance, clipping_controls=clipping)
    problem = missing_dock_node(config, nodes, clipping_node=clipping_node)
    if missing is None:
        assert problem is None
    else:
        assert missing in problem


@pytest.mark.parametrize("toolkit", TOOLKITS)
def test_rendering_an_ortho_layout_without_the_clipping_dock_raises(
    make_ortho, toolkit, qtbot
):
    ortho = make_ortho()
    _add(ortho, "cells")
    host = _host(toolkit)
    with pytest.raises(ValueError, match=r"'cells'.*OrthoClippingControls\(\)"):
        render_layout(
            Layout(center=_Center(), left_dock=AppearanceControls()), ortho, host
        )
    with pytest.raises(ValueError, match=r"'cells'.*AppearanceControls\(\)"):
        render_layout(
            Layout(center=_Center(), left_dock=OrthoClippingControls()), ortho, host
        )
    # A refused render leaves the viewer unchecked, as before it.
    assert ortho._rendered_dock_nodes is None
    _add(ortho, "more")

    view = render_layout(
        Layout(
            center=_Center(),
            left_dock=AppearanceControls(),
            right_dock=OrthoClippingControls(),
        ),
        ortho,
        host,
    )
    try:
        if toolkit == "qt":
            qtbot.addWidget(view.root)
        assert ortho._rendered_dock_nodes == _BOTH
    finally:
        for closeable in view.closeables:
            closeable.close()


def test_a_render_that_raises_after_the_check_leaves_the_viewer_unchecked(qtbot):
    """The dock nodes are kept only once the whole layout is built."""
    # No canvases: the flags have their nodes, and the clipping dock raises.
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        _add(ortho, "cells")
        with pytest.raises(ValueError, match="no canvas yet"):
            render_layout(
                Layout(
                    center=_Center(),
                    left_dock=AppearanceControls(),
                    right_dock=OrthoClippingControls(),
                ),
                ortho,
                _host("anywidget"),
            )
        assert ortho._rendered_dock_nodes is None
        # So a visual added next is not checked against that layout.
        _add(ortho, "more")
    finally:
        ortho.controller.close()


def test_adding_a_flagged_visual_to_a_rendered_ortho_layout_without_its_dock_raises(
    make_ortho,
):
    ortho = make_ortho()
    view = render_layout(
        Layout(center=_Center(), left_dock=AppearanceControls()), ortho, JupyterHost()
    )
    try:
        n_visuals = sum(len(scene.visuals) for scene in ortho.scenes.values())
        with pytest.raises(ValueError, match=r"'cells'.*OrthoClippingControls\(\)"):
            _add(ortho, "cells")
        # Refused before anything was added.
        assert sum(len(s.visuals) for s in ortho.scenes.values()) == n_visuals
        assert ortho.clipping_controller.groups == []
        # Without the flag it is added, and the appearance dock follows.
        _add(ortho, "cells", flagged=False)
        (dock,) = view.closeables
        assert len(dock.targets) == 2
    finally:
        for closeable in view.closeables:
            closeable.close()


def test_a_dock_node_without_a_flag_shows_its_placeholder(make_ortho):
    """Not an error: the dock fills in when a flagged visual is added."""
    ortho = make_ortho()
    _add(ortho, "plain", flagged=False, appearance=False)
    view = render_layout(
        Layout(
            center=_Center(),
            left_dock=AppearanceControls(),
            right_dock=OrthoClippingControls(),
        ),
        ortho,
        JupyterHost(),
    )
    try:
        dock, widget = view.closeables
        assert dock.targets == []
        assert widget.datasets == []
        _add(ortho, "cells")
        assert len(dock.targets) == 2
        assert widget.datasets == ["cells"]
    finally:
        for closeable in view.closeables:
            closeable.close()


def _viewer_add(viewer, name, **flags):
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in "zyx"),
    )
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    return viewer.add_labels(store, name=name, controls=LabelsControlsConfig(**flags))


def test_a_viewer_flag_needs_the_appearance_dock(qtbot):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        viewer.add_canvas()
        # Not rendered: flags are recorded and nothing is checked.
        _viewer_add(viewer, "cells", appearance=True, clipping_controls=True)
        with pytest.raises(ValueError, match=r"'cells'.*AppearanceControls\(\)"):
            render_layout(Layout(center=_Center()), viewer, JupyterHost())
        with pytest.raises(ValueError, match=r"'cells'.*AppearanceControls\(\)"):
            render_layout(
                Layout(center=_Center(), left_dock=RenderControls()),
                viewer,
                JupyterHost(),
            )

        view = render_layout(
            Layout(center=_Center(), left_dock=AppearanceControls()),
            viewer,
            JupyterHost(),
        )
        try:
            # Rendered with the dock: flagged visuals can follow.
            _viewer_add(viewer, "more", appearance=True, clipping_controls=True)
            # Clipping controls with no appearance controls: nowhere to go.
            with pytest.raises(ValueError, match=r"'bare'.*appearance=True"):
                _viewer_add(viewer, "bare", clipping_controls=True)
            assert [v.name for v in viewer.scene.visuals] == ["cells", "more"]
        finally:
            for closeable in view.closeables:
                closeable.close()
    finally:
        viewer.controller.close()


def test_a_viewer_rendered_with_no_dock_refuses_a_flagged_add(qtbot):
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        viewer.add_canvas()
        _viewer_add(viewer, "plain")  # no flags
        render_layout(Layout(center=_Center()), viewer, JupyterHost())
        with pytest.raises(ValueError, match=r"'cells'.*AppearanceControls\(\)"):
            _viewer_add(viewer, "cells", appearance=True)
        _viewer_add(viewer, "quiet")  # no flags: fine
        assert [v.name for v in viewer.scene.visuals] == ["plain", "quiet"]
    finally:
        viewer.controller.close()
