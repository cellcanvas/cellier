"""The clipping plane gizmo session (clipping plane gizmo design v2, 5.2 to 5.5).

Headless, on the rig of ``test_plane_gizmo.py``: a drag is synthetic pointer
input on an offscreen canvas, and what is checked is the model.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.clipping import handle_changes_plane, initial_anchor
from cellier.data import ImageMemoryStore
from cellier.data._axes import scale_and_translation_transform
from cellier.events import (
    ClippingPlanesChangedEvent,
    PlaneGizmoChangedEvent,
    PlaneGizmoUpdateEvent,
    PlaneInteractionEvent,
)
from cellier.visuals import ClippingPlane, InMemoryImageSingleAppearance
from tests.render.test_plane_gizmo import CAPTURES, Rig


class GizmoRig(Rig):
    """The plane gizmo rig, with a clipped visual and a log of the bus."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.store = self.controller.get_data_store(
            UUID(str(self.visual.data_store_id))
        )
        self.system = self.store.data_coordinate_system
        self.log: list = []
        owner = uuid4()
        self.controller.on_clipping_planes_changed(
            self.visual.id, self.log.append, owner_id=owner
        )
        self.controller.on_plane_interaction(
            self.visual.id, self.log.append, owner_id=owner
        )
        for view in self.views:
            self.controller.on_plane_gizmo_changed(
                view.canvas_id, self.log.append, owner_id=owner
            )

    def plane(self, x=12.0, normal=(0, 0, 1), **kwargs) -> ClippingPlane:
        """A plane through data ``(z, y, x) = (15.5, 15.5, x)``."""
        return ClippingPlane.from_point_normal(
            self.system, (15.5, 15.5, x), normal, **kwargs
        )

    def clip(self, *planes: ClippingPlane) -> None:
        self.visual.clipping_planes = planes
        self.log.clear()

    def session(self, item: ClippingPlane, view=None):
        view = view or self.view
        session = self.controller.add_clipping_plane_gizmo(
            self.visual.id, view.canvas_id, item.id
        )
        self.frame(view=view, n=2)
        return session

    def gizmo(self, session):
        view = self.controller.get_canvas_view(session.canvas_id)
        return view.get_plane_gizmo(session.id)

    def drag(self, session, group: str, index: int, dx: float, dy: float) -> None:
        """Grab a handle, move it in two steps with a frame each, release."""
        x, y = self.handle(self.gizmo(session), group, index)
        self.send("pointer_down", x, y)
        for step in (0.5, 1.0):
            self.send("pointer_move", x + dx * step, y + dy * step)
            self.frame()
        self.send("pointer_up", x + dx, y + dy, buttons=())
        self.frame()

    def model_plane(self, item: ClippingPlane):
        return self.controller.get_clipping_plane(self.visual.id, item.id).plane

    def of(self, kind) -> list:
        return [event for event in self.log if isinstance(event, kind)]


@pytest.fixture
def make_rig():
    rigs: list[GizmoRig] = []
    CAPTURES.clear()

    def _make(**kwargs) -> GizmoRig:
        rig = GizmoRig(**kwargs)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.controller.close()
    CAPTURES.clear()


# -- pure rules -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "axis", "changes"),
    [
        ("translate", 0, True),
        ("translate", 1, False),
        ("translate", 2, False),
        ("translate", (1, 2), False),
        ("translate", (0, 1), True),
        ("translate", (0, 2), True),
        ("rotate", 0, False),
        ("rotate", 1, True),
        ("rotate", 2, True),
        ("scale", 0, False),
    ],
)
def test_which_handles_change_the_plane(kind, axis, changes):
    assert handle_changes_plane(kind, axis) is changes


def test_the_anchor_is_the_plane_point_nearest_the_orbit_point():
    bounds = [[0, 0, 0], [30, 30, 30]]
    # The plane x == 20; the camera orbits about (5, 7, 9).
    anchor = initial_anchor((2.0, 0.0, 0.0), 40.0, (5.0, 7.0, 9.0), bounds)
    np.testing.assert_allclose(anchor, (20.0, 7.0, 9.0))
    # That point is outside the visual's box: the box centre is used.
    anchor = initial_anchor((1.0, 0.0, 0.0), 20.0, (5.0, 70.0, 9.0), bounds)
    np.testing.assert_allclose(anchor, (20.0, 15.0, 15.0))
    # Neither known: still a point on the plane.
    anchor = initial_anchor((1.0, 0.0, 0.0), 20.0, None, None)
    np.testing.assert_allclose(anchor, (20.0, 0.0, 0.0))


# -- creation -------------------------------------------------------------------


async def test_the_gizmo_is_placed_on_the_plane(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)

    point, normal = rig.gizmo(session).pose()
    # Rendered space is (x, y, z); the camera is fitted, so it orbits about
    # the volume's centre.
    np.testing.assert_allclose(normal, (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(point, (12.0, 15.5, 15.5), atol=1e-3)
    assert session.anchor == pytest.approx(point)
    assert rig.controller.get_plane_gizmo(rig.view.canvas_id) is session
    (event,) = rig.of(PlaneGizmoChangedEvent)
    assert (event.visual_id, event.plane_id) == (rig.visual.id, item.id)
    assert rig.of(ClippingPlanesChangedEvent) == []


async def test_a_disabled_plane_can_have_a_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0, enabled=False)
    rig.clip(item)
    session = rig.session(item)
    np.testing.assert_allclose(rig.gizmo(session).pose()[0][0], 12.0, atol=1e-3)
    rig.drag(session, "_translate1_children", 0, 30, 12)
    assert rig.model_plane(item) != item.plane
    assert not rig.controller.get_clipping_plane(rig.visual.id, item.id).enabled


async def test_what_is_refused(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane()
    rig.clip(item)
    controller, canvas_id = rig.controller, rig.view.canvas_id
    with pytest.raises(KeyError):
        controller.add_clipping_plane_gizmo(rig.visual.id, canvas_id, uuid4())
    with pytest.raises(KeyError):
        controller.add_clipping_plane_gizmo(rig.visual.id, uuid4(), item.id)

    other_scene = controller.add_scene(dim="3d", name="other")
    other = controller.add_image(
        data=ImageMemoryStore(data=np.zeros((4, 4, 4), np.float32)),
        scene_id=other_scene.id,
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="not in the scene"):
        controller.add_clipping_plane_gizmo(other.id, canvas_id, item.id)

    rig.scene.dims.selection.displayed_axes = (1, 2)
    with pytest.raises(ValueError, match="3D canvas"):
        controller.add_clipping_plane_gizmo(rig.visual.id, canvas_id, item.id)
    assert controller.get_plane_gizmo(canvas_id) is None
    assert rig.of(PlaneGizmoChangedEvent) == []
    assert rig.view._plane_gizmos == {}


# -- the gizmo -> the model -----------------------------------------------------


async def test_a_drag_along_the_normal_moves_the_plane_once_per_frame(make_rig):
    rig = make_rig()
    await rig.start()
    item, other = rig.plane(12.0), rig.plane(20.0, normal=(0, 0, -1))
    rig.clip(item, other)
    session = rig.session(item)
    rig.log.clear()

    rig.drag(session, "_translate1_children", 0, 40, 16)

    moved = rig.model_plane(item)
    point, normal = rig.gizmo(session).pose()
    # The model's plane is where the gizmo is.
    np.testing.assert_allclose(moved.normal, (0.0, 0.0, 1.0), atol=1e-6)
    assert moved.offset == pytest.approx(point[0], abs=1e-4)
    assert abs(moved.offset - 12.0) > 1.0
    np.testing.assert_allclose(normal, (1.0, 0.0, 0.0), atol=1e-6)
    # The id, the other plane and the order are kept.
    planes = rig.visual.clipping_planes
    assert [p.id for p in planes] == [item.id, other.id]
    assert planes[1] == other
    # Two frames and the release: at most three assignments, in one drag.
    changes = rig.of(ClippingPlanesChangedEvent)
    assert 1 <= len(changes) <= 3
    assert {event.source_id for event in changes} == {session.id}
    interactions = rig.of(PlaneInteractionEvent)
    assert [(e.phase, e.reason) for e in interactions] == [
        ("start", None),
        ("end", "release"),
    ]
    assert rig.controller.plane_interaction_state(rig.visual.id) == "idle"
    assert not session.dragging


async def test_a_tilt_turns_the_plane_about_the_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    anchor = np.array(session.anchor)

    rig.drag(session, "_rotate_children", 2, 25, -30)

    moved = rig.model_plane(item)
    normal_zyx = moved.normal / np.linalg.norm(moved.normal)
    assert not np.allclose(normal_zyx, (0, 0, 1), atol=1e-2)
    # The anchor stays on the plane: data (z, y, x) is rendered (x, y, z)
    # reversed here.
    distance = float(moved.normal @ anchor[::-1] - moved.offset)
    assert distance == pytest.approx(0.0, abs=1e-4)
    np.testing.assert_allclose(session.anchor, anchor, atol=1e-4)


@pytest.mark.parametrize(
    ("group", "kind", "axis"),
    [
        ("_translate1_children", "translate", 1),  # along an in-plane axis
        ("_translate2_children", "translate", (1, 2)),  # the two in-plane axes
        ("_rotate_children", "rotate", 0),  # about the normal
    ],
)
async def test_a_drag_that_cannot_change_the_plane_assigns_nothing(
    make_rig, group, kind, axis
):
    assert not handle_changes_plane(kind, axis)
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    gizmo = rig.gizmo(session)
    index = [_dim(e.dim) for e in getattr(gizmo.gizmo, group)].index(axis)
    before = gizmo.pose()
    anchor = session.anchor
    rig.log.clear()

    rig.drag(session, group, index, 30, 22)

    assert (gizmo.handle_kind, gizmo.handle_axis) == (kind, axis)
    assert rig.of(ClippingPlanesChangedEvent) == []
    assert rig.visual.clipping_planes == (item,)
    if kind == "translate":
        assert gizmo.pose() != before  # the gizmo itself moved
        assert session.anchor != anchor
        # The new anchor is still on the plane x == 12.
        assert session.anchor[0] == pytest.approx(12.0, abs=1e-4)
    else:
        assert gizmo.proxy.local.rotation.tolist() != [0.0, 0.0, 0.0, 1.0]
    # The scope opened and closed with nothing inside it.
    assert rig.of(PlaneInteractionEvent) == []
    assert not rig.controller._plane_interaction_driver.scope_open(rig.visual.id)


def _dim(dim):
    return int(dim) if np.ndim(dim) == 0 else tuple(int(v) for v in dim)


async def test_a_new_grab_ends_a_scope_whose_release_was_lost(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    x, y = rig.handle(rig.gizmo(session), "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.send("pointer_move", x + 20, y + 8)
    rig.frame()
    assert rig.controller.plane_interaction_state(rig.visual.id) == "active"
    # No release.  The next press drops the capture and ends the drag.
    x, y = rig.handle(rig.gizmo(session), "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    phases = [(e.phase, e.reason) for e in rig.of(PlaneInteractionEvent)]
    assert phases == [("start", None), ("end", "release")]
    rig.send("pointer_up", x, y, buttons=())
    assert not rig.controller._plane_interaction_driver.scope_open(rig.visual.id)


# -- the model -> the gizmo -----------------------------------------------------


async def test_a_plane_moved_elsewhere_moves_the_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    gizmo = rig.gizmo(session)

    # A slide: the gizmo goes to the nearest point of the new plane.
    rig.controller.set_clipping_plane(rig.visual.id, item.id, rig.plane(18.0).plane)
    point, normal = gizmo.pose()
    np.testing.assert_allclose(point, (18.0, 15.5, 15.5), atol=1e-3)
    np.testing.assert_allclose(normal, (1.0, 0.0, 0.0), atol=1e-6)

    # A flip keeps the place and turns the normal round.
    flipped = rig.plane(18.0, normal=(0, 0, -1)).plane
    rig.visual.clipping_planes = (
        rig.visual.clipping_planes[0].model_copy(update={"plane": flipped}),
    )
    point, normal = gizmo.pose()
    np.testing.assert_allclose(point, (18.0, 15.5, 15.5), atol=1e-3)
    np.testing.assert_allclose(normal, (-1.0, 0.0, 0.0), atol=1e-6)
    assert not session.closed


async def test_a_change_to_another_plane_leaves_the_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    item, other = rig.plane(12.0), rig.plane(20.0, normal=(0, 0, -1))
    rig.clip(item, other)
    session = rig.session(item)
    # Slide the gizmo within its plane, away from where a re-placement
    # would put it.
    rig.drag(session, "_translate1_children", 1, 20, 14)
    before = rig.gizmo(session).pose()
    rig.controller.set_clipping_plane(rig.visual.id, other.id, rig.plane(22.0).plane)
    rig.controller.set_clipping_planes(
        rig.visual.id,
        [p.model_copy(update={"enabled": False}) for p in rig.visual.clipping_planes],
    )
    assert rig.gizmo(session).pose() == before


async def test_a_transform_change_moves_the_gizmo_with_the_visual(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    rig.visual.transform = scale_and_translation_transform(
        rig.system,
        rig.scene.dims.world_coordinate_system,
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 5.0),
    )
    point, _normal = rig.gizmo(session).pose()
    assert point[0] == pytest.approx(17.0, abs=1e-3)
    assert rig.model_plane(item) == item.plane


# -- self-closing (G-P5) and one per canvas ------------------------------------


async def test_removing_the_plane_closes_the_session(make_rig):
    rig = make_rig()
    await rig.start()
    item, other = rig.plane(12.0), rig.plane(20.0, normal=(0, 0, -1))
    rig.clip(item, other)
    session = rig.session(item)
    rig.log.clear()
    rig.visual.clipping_planes = (other,)
    assert session.closed
    assert rig.view._plane_gizmos == {}
    assert rig.controller.get_plane_gizmo(rig.view.canvas_id) is None
    (event,) = rig.of(PlaneGizmoChangedEvent)
    assert (event.visual_id, event.plane_id) == (None, None)
    session.close()  # again: nothing
    assert len(rig.of(PlaneGizmoChangedEvent)) == 1


async def test_leaving_3d_closes_the_session(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    rig.scene.dims.selection.displayed_axes = (1, 2)
    assert session.closed
    assert rig.view._plane_gizmos == {}


async def test_removing_the_visual_or_the_canvas_closes_the_session(make_rig):
    rig = make_rig(n_canvases=2)
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    first = rig.session(item, view=rig.views[0])
    second = rig.session(item, view=rig.views[1])
    assert not first.closed  # one per canvas, not one per plane

    rig.controller.remove_canvas(rig.views[1].canvas_id)
    assert second.closed
    assert not first.closed
    rig.controller.remove_visual(rig.visual.id)
    assert first.closed
    assert rig.views[0]._plane_gizmos == {}


async def test_closing_mid_drag_ends_the_drag(make_rig):
    rig = make_rig()
    await rig.start()
    item = rig.plane(12.0)
    rig.clip(item)
    session = rig.session(item)
    x, y = rig.handle(rig.gizmo(session), "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.send("pointer_move", x + 20, y + 8)
    rig.frame()
    assert rig.controller.plane_interaction_state(rig.visual.id) == "active"
    session.close()
    assert rig.controller.plane_interaction_state(rig.visual.id) == "idle"
    assert not CAPTURES
    rig.send("pointer_up", x + 20, y + 8, buttons=())


async def test_a_canvas_has_one_gizmo_and_a_new_one_replaces_it(make_rig):
    rig = make_rig()
    await rig.start()
    item, other = rig.plane(12.0), rig.plane(20.0, normal=(0, 0, -1))
    rig.clip(item, other)
    first = rig.session(item)
    rig.log.clear()
    second = rig.session(other)
    assert first.closed
    assert not second.closed
    assert list(rig.view._plane_gizmos) == [second.id]
    # One event for the replacement: the new plane.
    (event,) = rig.of(PlaneGizmoChangedEvent)
    assert event.plane_id == other.id

    # A refused request leaves the gizmo there is.
    with pytest.raises(KeyError):
        rig.controller.add_clipping_plane_gizmo(
            rig.visual.id, rig.view.canvas_id, uuid4()
        )
    assert not second.closed
    assert rig.controller.get_plane_gizmo(rig.view.canvas_id) is second


async def test_the_request_event_opens_and_closes(make_rig):
    rig = make_rig()
    await rig.start()
    item, other = rig.plane(12.0), rig.plane(20.0, normal=(0, 0, -1))
    rig.clip(item, other)
    controller, canvas_id = rig.controller, rig.view.canvas_id
    widget = uuid4()

    def request(plane, enabled):
        controller._incoming_events.emit(
            PlaneGizmoUpdateEvent(
                source_id=widget,
                visual_id=rig.visual.id,
                plane_id=plane.id,
                canvas_id=canvas_id,
                kind="clipping",
                enabled=enabled,
            )
        )

    request(item, True)
    session = controller.get_plane_gizmo(canvas_id)
    assert session.plane_id == item.id
    assert rig.of(PlaneGizmoChangedEvent)[-1].source_id == widget

    # Switching off a plane that does not have the gizmo does nothing.
    request(other, False)
    assert controller.get_plane_gizmo(canvas_id) is session
    request(item, False)
    assert session.closed
    last = rig.of(PlaneGizmoChangedEvent)[-1]
    assert (last.plane_id, last.source_id) == (None, widget)


# -- a component on an axis the view does not show (5.5) ----------------------


async def test_a_plane_tilted_on_a_hidden_axis_has_no_gizmo(make_rig):
    rig = make_rig()
    controller = rig.controller
    scene = controller.add_scene(
        dim="3d",
        name="tzyx",
        coordinate_system=[
            ("t", "time"),
            ("z", "space"),
            ("y", "space"),
            ("x", "space"),
        ],
    )
    store = ImageMemoryStore(data=np.zeros((3, 8, 8, 8), np.float32))
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(color_map="gray", clim=(0.0, 1.0)),
    )
    controller.add_canvas(scene.id, canvas_size=(160, 120))
    canvas_id = controller.get_canvas_ids(scene.id)[-1]
    controller.reslice_all()
    system = store.data_coordinate_system
    flat = ClippingPlane.from_point_normal(system, (0, 4, 4, 4), (0, 0, 0, 1))
    tilted = ClippingPlane.from_point_normal(system, (1, 4, 4, 4), (1, 0, 0, 1))
    visual.clipping_planes = (flat, tilted)

    with pytest.raises(ValueError, match="does not show"):
        controller.add_clipping_plane_gizmo(visual.id, canvas_id, tilted.id)

    session = controller.add_clipping_plane_gizmo(visual.id, canvas_id, flat.id)
    point, normal = (
        controller.get_canvas_view(canvas_id).get_plane_gizmo(session.id).pose()
    )
    assert point[0] == pytest.approx(4.0, abs=1e-4)
    np.testing.assert_allclose(normal, (1.0, 0.0, 0.0), atol=1e-6)
    # An outside edit that tilts it in time ends the session.
    controller.set_clipping_plane(visual.id, flat.id, tilted.plane)
    assert session.closed
