"""``OrthoClippingController``: linked clipping planes across the ortho panels.

``plans/clipping_planes_linking_design.md`` sections 4 and 5.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer
from cellier.data import LabelMemoryStore
from cellier.events import ClippingPlanesUpdateEvent
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane

PANELS = ("xy", "xz", "yz", "vol")
MODES = ("all", "2d", None)
#: Mode -> panel -> the panels that carry its tuple (itself included).
LINKED = {
    "all": {key: set(PANELS) for key in PANELS},
    "2d": {**{key: {"xy", "xz", "yz"} for key in ("xy", "xz", "yz")}, "vol": {"vol"}},
    None: {key: {key} for key in PANELS},
}


def _store() -> LabelMemoryStore:
    system = DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in "zyx"),
    )
    return LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )


def _plane(store, x=4.0) -> ClippingPlane:
    system = store.data_coordinate_systems[0]
    return ClippingPlane.from_point_normal(system, (4, 4, x), (0, 0, 1))


def _moved(visual, x: float) -> tuple[ClippingPlane, ...]:
    """The visual's planes with the first moved to *x*, ids kept."""
    first, *rest = visual.clipping_planes
    plane = first.plane.model_copy(update={"offset": float(x)})
    return (first.model_copy(update={"plane": plane}), *rest)


def _at(visuals, x: float) -> set[str]:
    """The panels whose first plane sits at *x*."""
    return {
        key
        for key, visual in visuals.items()
        if visual.clipping_planes and visual.clipping_planes[0].plane.offset == x
    }


@pytest.fixture
def make_ortho(qtbot):
    """Build ortho viewers that are closed after the test."""
    made = []

    def _make(mode="all", **kwargs) -> OrthoViewer:
        ortho = OrthoViewer(
            spatial_axes("z", "y", "x"),
            gui="offscreen",
            link_clipping_planes=mode,
            **kwargs,
        )
        made.append(ortho)
        return ortho

    yield _make
    for ortho in made:
        ortho.controller.close()


def _add(ortho, name="cells"):
    store = _store()
    return ortho.add_labels(store, name=name, clipping_planes=(_plane(store),)), store


class _Events:
    """The clipping events of some visuals, by panel name."""

    def __init__(self, controller, visuals, prefix="") -> None:
        self.planes: list[tuple[str, object]] = []
        self.drags: list[tuple[str, str, str | None]] = []
        owner = uuid4()
        for key, visual in visuals.items():
            name = f"{prefix}{key}"
            controller.on_clipping_planes_changed(
                visual.id,
                lambda e, n=name: self.planes.append((n, e.source_id)),
                owner_id=owner,
            )
            controller.on_clipping_interaction(
                visual.id,
                lambda e, n=name: self.drags.append((n, e.phase, e.reason)),
                owner_id=owner,
            )

    def clear(self) -> None:
        self.planes.clear()
        self.drags.clear()


# -- the modes ------------------------------------------------------------------


def test_the_mode_must_be_one_of_three(make_ortho, qtbot):
    with pytest.raises(ValueError, match="Unknown clipping link mode"):
        make_ortho("3d")
    ortho = make_ortho()
    assert ortho.clipping_controller.mode == "all"
    with pytest.raises(ValueError, match="Unknown clipping link mode"):
        ortho.clipping_controller.mode = True
    assert ortho.clipping_controller.mode == "all"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("edited", PANELS)
def test_a_change_reaches_the_panels_the_mode_links(make_ortho, mode, edited):
    ortho = make_ortho(mode)
    visuals, _ = _add(ortho)
    # Every panel starts with the tuple given to add_*, whatever the mode.
    assert _at(visuals, 4.0) == set(PANELS)

    events = _Events(ortho.controller, visuals)
    source = uuid4()
    ortho.controller.set_clipping_planes(
        visuals[edited].id, _moved(visuals[edited], 7.0), source_id=source
    )
    assert _at(visuals, 7.0) == LINKED[mode][edited]
    # One event per visual changed, stamped with the originator's id.
    assert sorted(events.planes) == sorted(
        (key, source) for key in LINKED[mode][edited]
    )
    # Linked visuals carry the same planes, ids included.
    for key in LINKED[mode][edited]:
        assert visuals[key].clipping_planes == visuals[edited].clipping_planes


@pytest.mark.parametrize("mode", MODES)
def test_link_groups_are_what_the_mode_links(make_ortho, mode):
    ortho = make_ortho(mode)
    visuals, _ = _add(ortho)
    linker = ortho.clipping_controller
    (group,) = linker.groups
    assert group == {key: visuals[key].id for key in PANELS}
    ids = {visuals[key].id: key for key in PANELS}
    named = [[ids[i] for i in link] for link in linker.link_groups(group)]
    assert (
        named
        == {
            "all": [["xy", "xz", "yz", "vol"]],
            "2d": [["xy", "xz", "yz"], ["vol"]],
            None: [["xy"], ["xz"], ["yz"], ["vol"]],
        }[mode]
    )


def test_every_way_of_changing_a_visual_is_linked(make_ortho):
    ortho = make_ortho()
    controller = ortho.controller
    visuals, store = _add(ortho)
    events = _Events(controller, visuals)

    # Direct assignment: stamped with the controller's id on every panel.
    visuals["xz"].clipping_planes = _moved(visuals["xz"], 5.0)
    assert _at(visuals, 5.0) == set(PANELS)
    assert {source for _key, source in events.planes} == {controller._id}
    assert len(events.planes) == 4
    events.clear()

    # An update event from a widget: stamped with the widget's id.
    widget = uuid4()
    controller.incoming_events.emit(
        ClippingPlanesUpdateEvent(
            source_id=widget,
            visual_id=visuals["yz"].id,
            clipping_planes=_moved(visuals["yz"], 6.0),
        )
    )
    assert _at(visuals, 6.0) == set(PANELS)
    assert sorted(events.planes) == sorted((key, widget) for key in PANELS)
    events.clear()

    # One plane, as a gizmo moves it; adding and removing planes.
    plane = visuals["vol"].clipping_planes[0]
    controller.set_clipping_plane(
        visuals["vol"].id, plane.id, _plane(store, 2.0).plane, source_id=widget
    )
    assert _at(visuals, 2.0) == set(PANELS)
    visuals["xy"].clipping_planes = (*visuals["xy"].clipping_planes, _plane(store, 8.0))
    assert {len(v.clipping_planes) for v in visuals.values()} == {2}
    visuals["vol"].clipping_planes = ()
    assert {len(v.clipping_planes) for v in visuals.values()} == {0}
    events.clear()

    # The tuple a visual already has: nothing.
    controller.set_clipping_planes(visuals["xy"].id, ())
    assert events.planes == []


def test_a_refused_tuple_changes_no_linked_visual(make_ortho):
    ortho = make_ortho()
    visuals, _ = _add(ortho)
    events = _Events(ortho.controller, visuals)
    foreign = _plane(_store())  # a plane in another store's coordinates
    with pytest.raises(ValueError):
        ortho.controller.set_clipping_planes(visuals["vol"].id, (foreign,))
    assert _at(visuals, 4.0) == set(PANELS)
    assert events.planes == []


def test_datasets_are_not_linked_to_each_other(make_ortho):
    ortho = make_ortho()
    a, _ = _add(ortho, "a")
    b, _ = _add(ortho, "b")
    a["vol"].clipping_planes = _moved(a["vol"], 7.0)
    assert _at(a, 7.0) == set(PANELS)
    assert _at(b, 4.0) == set(PANELS)
    assert len(ortho.clipping_controller.groups) == 2


# -- changing the mode ------------------------------------------------------------


def _differing(ortho, visuals) -> None:
    """Give each panel its own position, as far as the mode lets it."""
    for x, key in zip((1.0, 2.0, 3.0, 5.0), PANELS):
        ortho.controller.set_clipping_planes(visuals[key].id, _moved(visuals[key], x))


#: (old, new) -> the position each panel holds after the switch, starting
#: from ``_differing``.  The 3D view wins; xy wins when only 2D is linked.
_AFTER_SWITCH = {
    ("all", "2d"): {"xy": 5.0, "xz": 5.0, "yz": 5.0, "vol": 5.0},
    ("all", None): {"xy": 5.0, "xz": 5.0, "yz": 5.0, "vol": 5.0},
    ("2d", None): {"xy": 3.0, "xz": 3.0, "yz": 3.0, "vol": 5.0},
    ("2d", "all"): {"xy": 5.0, "xz": 5.0, "yz": 5.0, "vol": 5.0},
    (None, "all"): {"xy": 5.0, "xz": 5.0, "yz": 5.0, "vol": 5.0},
    (None, "2d"): {"xy": 1.0, "xz": 1.0, "yz": 1.0, "vol": 5.0},
}
#: (old, new) -> the panels a switch writes to.
_WRITTEN = {
    ("all", "2d"): [],
    ("all", None): [],
    ("2d", None): [],
    ("2d", "all"): ["xy", "xz", "yz"],
    (None, "all"): ["xy", "xz", "yz"],
    (None, "2d"): ["xz", "yz"],
}


@pytest.mark.parametrize(("old", "new"), list(_AFTER_SWITCH))
def test_a_mode_change_writes_only_what_becomes_linked(make_ortho, old, new):
    ortho = make_ortho(old)
    linker = ortho.clipping_controller
    visuals, _ = _add(ortho)
    _differing(ortho, visuals)
    events = _Events(ortho.controller, visuals)
    modes = []
    linker.mode_changed.connect(modes.append)

    linker.mode = new
    assert linker.mode == new
    assert modes == [new]
    positions = {key: v.clipping_planes[0].plane.offset for key, v in visuals.items()}
    assert positions == _AFTER_SWITCH[old, new]
    # One event per visual written, stamped with the linker's id.
    assert sorted(events.planes) == sorted(
        (key, linker.id) for key in _WRITTEN[old, new]
    )

    # The same mode again: nothing.
    events.clear()
    linker.mode = new
    assert modes == [new]
    assert events.planes == []

    # And the new mode is in force.
    ortho.controller.set_clipping_planes(visuals["xz"].id, _moved(visuals["xz"], 9.0))
    assert _at(visuals, 9.0) == LINKED[new]["xz"]


def test_a_gizmo_keeps_its_plane_across_mode_changes(make_ortho):
    ortho = make_ortho("2d")
    controller = ortho.controller
    visuals, _ = _add(ortho)
    for scene in ortho.scenes.values():
        controller.add_canvas(scene.id)
    plane = visuals["vol"].clipping_planes[0]
    session = ortho.add_clipping_plane_gizmo(visuals, plane)
    controller.set_clipping_planes(visuals["xy"].id, _moved(visuals["xy"], 2.0))

    for mode in ("all", None, "2d", "all"):
        ortho.clipping_controller.mode = mode
        assert not session.closed
        assert session.plane_id == plane.id
    # The 3D view won: its plane, the gizmo's, is on every panel.
    assert _at(visuals, 4.0) == set(PANELS)


# -- drags ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_a_drag_is_reported_by_every_linked_visual(make_ortho, mode):
    """In an event loop: one start and one end per linked visual per drag."""
    ortho = make_ortho(mode)
    controller = ortho.controller
    visuals, _ = _add(ortho)
    events = _Events(controller, visuals)

    with controller.clipping_interaction(visuals["xz"].id):
        for x in (5.0, 6.0, 7.0):
            controller.set_clipping_planes(visuals["xz"].id, _moved(visuals["xz"], x))
    linked = LINKED[mode]["xz"]
    assert _at(visuals, 7.0) == linked
    assert sorted(events.drags) == sorted(
        [(key, "start", None) for key in linked]
        + [(key, "end", "release") for key in linked]
    )
    # The origin starts first, so the scopes are open for the first tick.
    assert events.drags[0] == ("xz", "start", None)
    assert all(
        controller.clipping_interaction_state(v.id) == "idle" for v in visuals.values()
    )
    assert ortho.clipping_controller._forwarded == {}


def test_without_an_event_loop_every_change_is_a_drag_on_every_linked_visual(
    make_ortho,
):
    ortho = make_ortho()
    controller = ortho.controller
    visuals, _ = _add(ortho)
    events = _Events(controller, visuals)
    with controller.clipping_interaction(visuals["vol"].id):
        for x in (5.0, 6.0):
            controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], x))
    # No stillness timer: every visual settles at once, after each change.
    for key in PANELS:
        drags = [(phase, reason) for name, phase, reason in events.drags if name == key]
        assert drags == [("start", None), ("end", "settle")] * 2
    assert ortho.clipping_controller._forwarded == {}


async def test_a_change_outside_a_scope_reports_no_drag(make_ortho):
    ortho = make_ortho()
    visuals, _ = _add(ortho)
    events = _Events(ortho.controller, visuals)
    ortho.controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 6.0))
    assert _at(visuals, 6.0) == set(PANELS)
    assert events.drags == []


async def test_a_mode_change_in_a_drag_closes_the_forwarded_scopes(make_ortho):
    ortho = make_ortho("all")
    controller = ortho.controller
    linker = ortho.clipping_controller
    visuals, _ = _add(ortho)
    events = _Events(controller, visuals)

    with controller.clipping_interaction(visuals["vol"].id):
        controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 5.0))
        assert set(linker._forwarded) == {visuals["vol"].id}
        linker.mode = "2d"
        assert linker._forwarded == {}
        ended = {key for key, phase, _reason in events.drags if phase == "end"}
        assert ended == {"xy", "xz", "yz"}
        controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 6.0))
    assert _at(visuals, 6.0) == {"vol"}
    assert _at(visuals, 5.0) == {"xy", "xz", "yz"}
    assert all(
        controller.clipping_interaction_state(v.id) == "idle" for v in visuals.values()
    )


# -- two datasets linked by user code (L-D17, L12) ------------------------------


def _link_xy(controller, a, b, *, forward_scope: bool):
    """User code: b's xy planes follow a's xy planes' positions."""
    user = uuid4()

    def copy(event) -> None:
        moved = tuple(
            old.model_copy(
                update={
                    "plane": old.plane.model_copy(update={"offset": new.plane.offset})
                }
            )
            for old, new in zip(b["xy"].clipping_planes, event.clipping_planes)
        )
        controller.set_clipping_planes(b["xy"].id, moved, source_id=user)

    controller.on_clipping_planes_changed(a["xy"].id, copy, owner_id=user)
    if forward_scope:

        def forward(event) -> None:
            if event.phase == "start":
                controller.begin_clipping_interaction(b["xy"].id, source_id=user)
            else:
                controller.end_clipping_interaction(b["xy"].id, source_id=user)

        controller.on_clipping_interaction(a["xy"].id, forward, owner_id=user)
    return user


#: Mode -> the panel of ``a`` edited: one the mode links to ``a``'s xy.
_ORIGIN = {"all": "vol", "2d": "xz", None: "xy"}


@pytest.mark.parametrize("mode", MODES)
def test_a_change_made_by_user_code_in_a_handler_is_linked_too(make_ortho, mode):
    """No linker-wide guard: the second dataset's panels are not left behind."""
    ortho = make_ortho(mode)
    controller = ortho.controller
    a, _ = _add(ortho, "a")
    b, _ = _add(ortho, "b")
    user = _link_xy(controller, a, b, forward_scope=False)
    events = _Events(controller, b)

    origin = a[_ORIGIN[mode]]
    controller.set_clipping_planes(origin.id, _moved(origin, 7.0))
    assert _at(a, 7.0) == LINKED[mode]["xy"]
    assert _at(b, 7.0) == LINKED[mode]["xy"]
    assert sorted(events.planes) == sorted((key, user) for key in LINKED[mode]["xy"])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("forward_scope", [False, True])
async def test_a_drag_reaches_the_second_dataset_only_if_the_user_forwards_it(
    make_ortho, mode, forward_scope
):
    ortho = make_ortho(mode)
    controller = ortho.controller
    a, _ = _add(ortho, "a")
    b, _ = _add(ortho, "b")
    _link_xy(controller, a, b, forward_scope=forward_scope)
    events = _Events(controller, {**{f"a/{k}": v for k, v in a.items()}})
    events_b = _Events(controller, b)

    origin = a[_ORIGIN[mode]]
    with controller.clipping_interaction(origin.id):
        for x in (5.0, 6.0, 7.0):
            controller.set_clipping_planes(origin.id, _moved(origin, x))
    linked = LINKED[mode]["xy"]
    assert _at(b, 7.0) == linked
    assert {name for name, _phase, _reason in events.drags} == {
        f"a/{key}" for key in linked
    }
    expected_b = (
        sorted(
            [(key, "start", None) for key in linked]
            + [(key, "end", "release") for key in linked]
        )
        if forward_scope
        else []
    )
    assert sorted(events_b.drags) == expected_b
    assert ortho.clipping_controller._forwarded == {}
    assert all(
        controller.clipping_interaction_state(v.id) == "idle"
        for v in (*a.values(), *b.values())
    )


# -- removal and close ------------------------------------------------------------


def test_a_removed_panel_leaves_the_rest_linked(make_ortho):
    ortho = make_ortho()
    controller = ortho.controller
    linker = ortho.clipping_controller
    visuals, _ = _add(ortho)
    _add(ortho, "other")

    removed = visuals.pop("yz")
    controller.remove_visual(removed.id)
    (group, other_group) = linker.groups
    assert group == {key: visuals[key].id for key in ("xy", "xz", "vol")}
    assert len(other_group) == 4
    assert linker.link_groups(group) == [[v.id for v in visuals.values()]]
    assert removed.id not in linker._scope_ids

    controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 7.0))
    assert _at(visuals, 7.0) == {"xy", "xz", "vol"}

    # The last visual of a dataset: the dataset is forgotten.
    for visual in visuals.values():
        controller.remove_visual(visual.id)
    assert linker.groups == [other_group]
    assert set(linker._scope_ids) == set(other_group.values())


async def test_removal_in_a_drag_leaves_no_scope_open(make_ortho):
    ortho = make_ortho()
    controller = ortho.controller
    linker = ortho.clipping_controller
    visuals, _ = _add(ortho)

    # A forward target goes: the drag carries on for the others.
    with controller.clipping_interaction(visuals["vol"].id):
        controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 5.0))
        controller.remove_visual(visuals.pop("yz").id)
        controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 6.0))
    assert _at(visuals, 6.0) == {"xy", "xz", "vol"}
    assert linker._forwarded == {}

    # The origin goes: the scopes it forwarded are closed.
    vol = visuals.pop("vol")
    controller.begin_clipping_interaction(vol.id, source_id=(source := uuid4()))
    controller.set_clipping_planes(vol.id, _moved(vol, 7.0))
    assert set(linker._forwarded) == {vol.id}
    controller.remove_visual(vol.id)
    controller.end_clipping_interaction(vol.id, source_id=source)
    assert linker._forwarded == {}
    assert all(
        controller.clipping_interaction_state(v.id) == "idle" for v in visuals.values()
    )
    # The two that are left are still linked.
    controller.set_clipping_planes(visuals["xy"].id, _moved(visuals["xy"], 8.0))
    assert _at(visuals, 8.0) == {"xy", "xz"}


def test_a_closed_linker_links_nothing(make_ortho):
    ortho = make_ortho()
    visuals, _ = _add(ortho)
    ortho.clipping_controller.close()
    assert ortho.clipping_controller.groups == []
    ortho.controller.set_clipping_planes(visuals["vol"].id, _moved(visuals["vol"], 7.0))
    assert _at(visuals, 7.0) == {"vol"}


# -- save and load (L-D18, L13) ---------------------------------------------------


def _saved(make_ortho, tmp_path, mode, *, differ: bool):
    """Save a viewer in *mode*; with *differ*, one panel moved alone."""
    ortho = make_ortho(mode)
    visuals, _ = _add(ortho)
    if differ:
        key = "vol" if mode == "2d" else "xy"
        ortho.controller.set_clipping_planes(visuals[key].id, _moved(visuals[key], 7.0))
    path = tmp_path / f"ortho-{mode}-{differ}.json"
    ortho.to_file(path)
    positions = {key: v.clipping_planes[0].plane.offset for key, v in visuals.items()}
    return path, positions


@pytest.mark.parametrize("saved", MODES)
@pytest.mark.parametrize("loaded", MODES)
def test_a_loaded_viewer_is_linked_as_the_mode_says(
    make_ortho, tmp_path, saved, loaded
):
    path, positions = _saved(make_ortho, tmp_path, saved, differ=False)
    ortho = OrthoViewer.from_file(path, link_clipping_planes=loaded)
    try:
        linker = ortho.clipping_controller
        assert linker.mode == loaded
        (group,) = linker.groups
        visuals = {
            key: ortho.controller.get_visual_model(visual_id)
            for key, visual_id in group.items()
        }
        assert set(visuals) == set(PANELS)
        assert ortho.image_group(visuals["xy"]) == [visuals[key].id for key in PANELS]
        # The load wrote nothing.
        assert {
            key: v.clipping_planes[0].plane.offset for key, v in visuals.items()
        } == positions
        for edited, x in (("vol", 2.0), ("xy", 3.0)):
            ortho.controller.set_clipping_planes(
                visuals[edited].id, _moved(visuals[edited], x)
            )
            assert _at(visuals, x) == LINKED[loaded][edited]
    finally:
        ortho.controller.close()


#: (saved, loaded) pairs where panels that differ in the file would be linked.
_REFUSED = [("2d", "all"), (None, "all"), (None, "2d")]


@pytest.mark.parametrize(
    ("saved", "loaded"),
    [(s, m) for s in ("2d", None) for m in MODES],
)
def test_a_file_whose_panels_disagree_under_the_mode_is_refused(
    make_ortho, tmp_path, saved, loaded
):
    path, positions = _saved(make_ortho, tmp_path, saved, differ=True)
    if (saved, loaded) in _REFUSED:
        with pytest.raises(ValueError, match=r"'cells'.*link_clipping_planes"):
            OrthoViewer.from_file(path, link_clipping_planes=loaded)
        return
    ortho = OrthoViewer.from_file(path, link_clipping_planes=loaded)
    try:
        (group,) = ortho.clipping_controller.groups
        assert {
            key: ortho.controller.get_visual_model(i).clipping_planes[0].plane.offset
            for key, i in group.items()
        } == positions
    finally:
        ortho.controller.close()


def test_loaded_datasets_of_one_name_are_paired_in_add_order(make_ortho, tmp_path):
    ortho = make_ortho()
    first, _ = _add(ortho, "cells")
    second, _ = _add(ortho, "cells")
    second["vol"].clipping_planes = _moved(second["vol"], 7.0)
    path = tmp_path / "twins.json"
    ortho.to_file(path)

    loaded = OrthoViewer.from_file(path)
    try:
        groups = loaded.clipping_controller.groups
        assert groups == [
            {key: first[key].id for key in PANELS},
            {key: second[key].id for key in PANELS},
        ]
    finally:
        loaded.controller.close()
