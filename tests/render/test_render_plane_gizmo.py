"""The gizmo on a render plane (plane rendering design v3, 8.3; Phase 8).

Headless, on the rig of ``test_plane_gizmo.py``: a drag is synthetic pointer
input on an offscreen canvas, picked and handled by pygfx's own gizmo code,
and what is checked is the model.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import pylinalg as la
import pytest

from cellier.clipping import (
    ClippingPlaneGizmoController,
    RenderPlaneGizmoController,
    scale_handle_axes,
    scaled_extent,
)
from cellier.controller import CellierController
from cellier.data import ImageMemoryStore
from cellier.events import (
    PlaneGizmoChangedEvent,
    PlaneGizmoUpdateEvent,
    PlaneInteractionEvent,
    RenderPlanesChangedEvent,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    RenderPlane,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading
from tests.render.test_plane_gizmo import CAPTURES, SIZE, Rig, _ball

AXES = ("z", "y", "x")
#: World ``(z, y, x)`` of the middle of the (32, 32, 32) volume.
MIDDLE = (15.5, 15.5, 15.5)
SETTLE_S = 0.03


def _directions(n: int) -> np.ndarray:
    """*n* unit camera directions, the same every run."""
    vectors = np.random.default_rng(0).normal(size=(n, 3))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class PlaneRig(Rig):
    """One in-memory image in ``"plane"`` mode, and a log of the bus."""

    def __init__(self, *, n_canvases: int = 1, time_axis: bool = False) -> None:
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        if time_axis:
            self.scene = self.controller.add_scene(
                coordinate_system=[("t", "time"), *spatial_axes(*AXES)],
                dim="3d",
                name="scene",
            )
            data = np.stack([_ball(), _ball()])
        else:
            self.scene = self.controller.add_scene(dim="3d", name="scene")
            data = _ball()
        self.world = self.controller._model.scenes[
            self.scene.id
        ].dims.world_coordinate_system
        self.visual = self.controller.add_image(
            data=ImageMemoryStore(data=data),
            scene_id=self.scene.id,
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="plane"
            ),
        )
        self._finish(n_canvases)

    def _finish(self, n_canvases: int) -> None:
        self.views = []
        for _ in range(n_canvases):
            self.controller.add_canvas(self.scene.id, canvas_size=SIZE)
            canvas_id = self.controller.get_canvas_ids(self.scene.id)[-1]
            self.views.append(self.controller.get_canvas_view(canvas_id))
        self.view = self.views[0]
        self.events = []
        self.in_draw = []
        self.log: list = []
        owner = uuid4()
        self.controller.on_render_planes_changed(
            self.visual.id, self.log.append, owner_id=owner
        )
        self.controller.on_plane_interaction(
            self.visual.id, self.log.append, owner_id=owner
        )
        for view in self.views:
            self.controller.on_plane_gizmo_changed(
                view.canvas_id, self.log.append, owner_id=owner
            )

    # -- planes --------------------------------------------------------------

    def plane(self, normal=(1.0, 0.3, 0.2), up=(0.0, 1.0, 0.1), **kwargs):
        """An oblique plane through the middle, bounded on both axes."""
        kwargs.setdefault("extent_0", (-8.0, 10.0))
        kwargs.setdefault("extent_1", (-6.0, 7.0))
        return RenderPlane.from_point_normal(
            self.world, MIDDLE, normal, axes=AXES, up=up, **kwargs
        )

    def show(self, *planes: RenderPlane) -> None:
        self.controller.set_render_planes(self.visual.id, planes)
        self.log.clear()

    def model(self, plane: RenderPlane) -> RenderPlane:
        return self.controller.get_render_plane(self.visual.id, plane.id)

    def session(self, plane: RenderPlane, view=None) -> RenderPlaneGizmoController:
        view = view or self.view
        session = self.controller.add_render_plane_gizmo(
            self.visual.id, view.canvas_id, plane.id
        )
        self.frame(view=view, n=2)
        return session

    def gizmo(self, session):
        view = self.controller.get_canvas_view(session.canvas_id)
        return view.get_plane_gizmo(session.id)

    def of(self, kind) -> list:
        return [event for event in self.log if isinstance(event, kind)]

    # -- camera and drags ----------------------------------------------------

    def look(self, direction) -> None:
        """Look at the volume's middle from *direction* (rendered ``xyz``)."""
        camera = self.view.camera
        centre = np.array(MIDDLE[::-1])
        distance = float(np.linalg.norm(np.asarray(camera.world.position) - centre))
        camera.local.position = centre + np.asarray(direction) * distance
        camera.look_at(tuple(centre))
        self.frame(n=2)

    def arrow(self, session, group: str, index: int):
        """A handle's screen position and the unit direction out to it."""
        gizmo = self.gizmo(session)
        at = np.array(self.handle(gizmo, group, index))
        origin = np.array(self.screen(gizmo.gizmo.world.position))
        out = at - origin
        length = float(np.linalg.norm(out))
        return at, (out / length if length > 0 else out), length

    def drag(self, session, at, delta) -> tuple[str, object] | None:
        """Press at *at*, move by *delta* in two framed steps, release.

        Returns the handle pygfx grabbed, ``(kind, axis)``, or ``None`` when
        the press landed on no handle.
        """
        self.send("pointer_down", at[0], at[1])
        if not session.dragging:
            self.send("pointer_up", at[0], at[1], buttons=())
            return None
        gizmo = self.gizmo(session)
        grabbed = (gizmo.handle_kind, gizmo.handle_axis)
        for step in (0.5, 1.0):
            self.send("pointer_move", at[0] + delta[0] * step, at[1] + delta[1] * step)
            self.frame()
        self.send("pointer_up", at[0] + delta[0], at[1] + delta[1], buttons=())
        self.frame()
        return grabbed


@pytest.fixture
def make_rig():
    rigs: list = []
    CAPTURES.clear()

    def _make(cls=PlaneRig, **kwargs):
        rig = cls(**kwargs)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.controller.close()
    CAPTURES.clear()


def _frame_of(plane: RenderPlane) -> np.ndarray:
    """``[normal, axis_0, axis_1]`` rows of a plane, on its own axes."""
    axis_0 = np.asarray(plane.in_plane_axis_0)
    axis_1 = np.asarray(plane.in_plane_axis_1)
    return np.array([np.cross(axis_0, axis_1), axis_0, axis_1])


# -- pure rules -----------------------------------------------------------------


def test_a_scale_handle_needs_both_sides_bounded(make_rig):
    rig = make_rig()
    assert scale_handle_axes(rig.plane()) == (1, 2)
    assert scale_handle_axes(rig.plane(extent_0=(None, 4.0))) == (2,)
    assert scale_handle_axes(rig.plane(extent_1=(-3.0, None))) == (1,)
    assert (
        scale_handle_axes(rig.plane(extent_0=(None, None), extent_1=(None, None))) == ()
    )


def test_a_scale_factor_multiplies_both_sides_about_the_origin():
    assert scaled_extent((-2.0, 6.0), 1.5) == (-3.0, 9.0)
    assert scaled_extent((1.0, 4.0), 0.5) == (0.5, 2.0)
    # An unbounded side has no size; a vanishing factor would leave min == max.
    assert scaled_extent((None, 6.0), 2.0) == (None, 6.0)
    assert scaled_extent((-2.0, 6.0), 0.0) == (-2.0, 6.0)
    assert scaled_extent((-2.0, 6.0), float("nan")) == (-2.0, 6.0)


# -- the gizmo sits on the plane ------------------------------------------------


async def test_the_gizmo_carries_the_planes_whole_frame(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    assert isinstance(session, RenderPlaneGizmoController)
    assert session.kind == "render"
    gizmo = rig.gizmo(session)
    point, normal = gizmo.pose()
    axis_0, axis_1, scale = gizmo.frame()
    # Rendered (x, y, z) is the world's (z, y, x) reversed.
    np.testing.assert_allclose(point, plane.origin[::-1], atol=1e-5)
    np.testing.assert_allclose(axis_0, plane.in_plane_axis_0[::-1], atol=1e-6)
    np.testing.assert_allclose(axis_1, plane.in_plane_axis_1[::-1], atol=1e-6)
    np.testing.assert_allclose(normal, np.cross(axis_0, axis_1), atol=1e-6)
    assert scale == (1.0, 1.0)
    # Opening a gizmo changes no plane.
    assert rig.of(RenderPlanesChangedEvent) == []
    assert rig.model(plane) == plane


async def test_a_plane_changed_elsewhere_moves_the_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    moved = RenderPlane.from_point_normal(
        rig.world,
        (12.0, 18.0, 14.0),
        (0.2, 1.0, -0.4),
        axes=AXES,
        id=plane.id,
        extent_0=(-5.0, 5.0),
    )
    rig.controller.set_render_plane(rig.visual.id, plane.id, moved)
    gizmo = rig.gizmo(session)
    np.testing.assert_allclose(gizmo.pose()[0], moved.origin[::-1], atol=1e-5)
    axis_0, axis_1, _ = gizmo.frame()
    np.testing.assert_allclose(axis_0, moved.in_plane_axis_0[::-1], atol=1e-6)
    np.testing.assert_allclose(axis_1, moved.in_plane_axis_1[::-1], atol=1e-6)
    # extent_1 became unbounded: its scale handle went.
    assert gizmo.gizmo.scale_axes == {1}
    assert not session.closed


async def test_a_disabled_plane_can_have_a_gizmo(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane(enabled=False)
    rig.show(plane)
    session = rig.session(plane)
    assert not session.closed
    np.testing.assert_allclose(
        rig.gizmo(session).pose()[0], plane.origin[::-1], atol=1e-5
    )


# -- hidden scale handles -------------------------------------------------------


async def test_no_scale_handle_on_an_axis_with_an_unbounded_side(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane(extent_0=(-8.0, 10.0), extent_1=(None, 7.0))
    rig.show(plane)
    session = rig.session(plane)
    widget = rig.gizmo(session).gizmo
    rig.look((0.5, 0.6, 0.62))
    visible = [child.visible for child in widget._scale_children]
    # Axis 0 is the normal and never scales; axis 2 has an unbounded side.
    assert visible == [False, True, False]
    assert not widget._center_sphere.visible

    # Bounding the side in the model shows the handle; unbounding hides both.
    rig.controller.set_render_plane(
        rig.visual.id, plane.id, plane.model_copy(update={"extent_1": (-6.0, 7.0)})
    )
    rig.frame(n=2)
    assert [child.visible for child in widget._scale_children] == [False, True, True]
    rig.controller.set_render_plane(
        rig.visual.id,
        plane.id,
        plane.model_copy(update={"extent_0": (None, None), "extent_1": (None, None)}),
    )
    rig.frame(n=2)
    assert [child.visible for child in widget._scale_children] == [False] * 3


async def test_a_clipping_gizmo_still_has_no_scale_handles(make_rig):
    rig = make_rig()
    await rig.start()
    store = rig.controller.get_data_store(UUID(str(rig.visual.data_store_id)))
    item = ClippingPlane.from_point_normal(
        store.data_coordinate_system, MIDDLE, (0, 0, 1)
    )
    rig.visual.clipping_planes = (item,)
    session = rig.controller.add_clipping_plane_gizmo(
        rig.visual.id, rig.view.canvas_id, item.id
    )
    rig.look((0.5, 0.6, 0.62))
    widget = rig.gizmo(session).gizmo
    assert [child.visible for child in widget._scale_children] == [False] * 3


# -- C6: each handle changes exactly its model fields ----------------------------

#: ``(group, index)`` of every handle a render plane gizmo shows.
HANDLES = [
    ("_translate1_children", 0),
    ("_translate1_children", 1),
    ("_translate1_children", 2),
    ("_translate2_children", 0),
    ("_translate2_children", 1),
    ("_translate2_children", 2),
    ("_rotate_children", 0),
    ("_rotate_children", 1),
    ("_rotate_children", 2),
    ("_scale_children", 1),
    ("_scale_children", 2),
]


def _wrong_fields(before: RenderPlane, after: RenderPlane, kind, axis, along) -> list:
    """What a drag of handle ``(kind, axis)`` changed that it should not have.

    *along* says whether a scale drag went out along its handle's arrow.
    """
    wrong: list[str] = []
    frame = _frame_of(before)
    moved = np.asarray(after.origin) - np.asarray(before.origin)
    same_axes = (
        after.in_plane_axis_0 == before.in_plane_axis_0
        and after.in_plane_axis_1 == before.in_plane_axis_1
    )
    same_extents = (
        after.extent_0 == before.extent_0 and after.extent_1 == before.extent_1
    )
    if kind == "translate":
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        outside = [i for i in range(3) if i not in axes]
        if not same_axes:
            wrong.append("axes")
        if not same_extents:
            wrong.append("extents")
        if np.linalg.norm(moved) == 0.0:
            wrong.append("origin did not move")
        if np.abs(frame[outside] @ moved).max(initial=0.0) > 1e-4:
            wrong.append("origin left the handle's axes")
    elif kind == "rotate":
        if after.origin != before.origin:
            wrong.append("origin")
        if not same_extents:
            wrong.append("extents")
        if same_axes:
            wrong.append("axes did not turn")
        # The frame turned about the handle's own axis.
        if not np.allclose(_frame_of(after)[axis], frame[axis], atol=1e-5):
            wrong.append("the rotation axis moved")
    elif kind == "scale":
        if after.origin != before.origin:
            wrong.append("origin")
        if not same_axes:
            wrong.append("axes")
        mine, other = (
            ("extent_0", "extent_1") if axis == 1 else ("extent_1", "extent_0")
        )
        if getattr(after, other) != getattr(before, other):
            wrong.append("the other extent")
        low, high = getattr(before, mine)
        new_low, new_high = getattr(after, mine)
        factor = new_high / high
        if not np.isclose(new_low / low, factor):
            wrong.append("not symmetric about the origin")
        if along and not factor > 1.0:
            wrong.append("a drag along the arrow shrank the plane")
    else:
        wrong.append(f"unknown handle {kind}")
    return wrong


async def test_each_handle_changes_exactly_its_model_fields(make_rig):
    """C6: every handle, through pygfx's own picking and handlers, 40 views."""
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    drags: dict[str, int] = {}
    wrong: list = []
    for direction in _directions(40):
        rig.look(direction)
        for group, index in HANDLES:
            at, out, length = rig.arrow(session, group, index)
            if length < 12:
                continue  # seen end-on: no direction to drag along
            # Rings are dragged across the arrow, the rest along it.
            if group == "_rotate_children":
                delta = 30 * np.array([-out[1], out[0]])
            else:
                delta = 30 * out
            grabbed = rig.drag(session, at, delta)
            if grabbed is None:
                continue  # hidden at this angle, or behind another handle
            kind, axis = grabbed
            after = rig.model(plane)
            along = (group, index) == ("_scale_children", axis)
            for fault in _wrong_fields(plane, after, kind, axis, along):
                wrong.append((tuple(np.round(direction, 2)), kind, axis, fault))
            label = (
                f"{kind} normal"
                if axis == 0
                else f"{kind} in plane"
                if kind != "scale"
                else "scale"
            )
            drags[label] = drags.get(label, 0) + 1
            # Back to the start for the next handle.
            rig.controller.set_render_plane(rig.visual.id, plane.id, plane)
            rig.frame(n=2)
    assert wrong == []
    # Every kind of handle was exercised from many of the 40 views.
    assert set(drags) == {
        "translate normal",
        "translate in plane",
        "rotate normal",
        "rotate in plane",
        "scale",
    }
    assert min(drags.values()) >= 20, drags
    assert sum(drags.values()) >= 250, drags


async def test_a_scale_drag_along_the_arrow_grows_at_every_camera_angle(make_rig):
    """The handles are not turned to the camera, so pygfx's own drag is right.

    pygfx reverses a scale drag on a handle it has flipped, and reads the
    flip from a scale that does not hold it; with no flip there is nothing
    to get wrong.
    """
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    gizmo = rig.gizmo(session)
    widget, proxy = gizmo.gizmo, gizmo.proxy
    drags = shrank = away = 0
    for direction in _directions(40):
        rig.look(direction)
        assert (widget.gizmo_scale > 0).all()
        for dim in (1, 2):
            at, out, length = rig.arrow(session, "_scale_children", dim)
            if length < 15:
                continue
            handle = widget._scale_children[dim]
            offset = np.asarray(handle.world.position) - np.asarray(
                widget.world.position
            )
            away += int(offset @ np.asarray(direction) < 0)
            end = SimpleNamespace(x=at[0] + 60 * out[0], y=at[1] + 60 * out[1])
            widget._handle_start("scale", SimpleNamespace(x=at[0], y=at[1]), handle)
            widget._handle_scale_move(end)
            widget._ref = None
            shrank += int(proxy.local.scale[dim] < 1.0)
            proxy.local.scale = (1.0, 1.0, 1.0)
            drags += 1
    assert drags >= 60
    assert shrank == 0
    # The test would pass for the wrong reason if no handle pointed away.
    assert away >= 20


# -- extents by the scale handles -----------------------------------------------


async def test_a_scale_drag_multiplies_the_extent_and_folds_in_on_release(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    rig.look((0.5, 0.6, 0.62))
    gizmo = rig.gizmo(session)
    at, out, _ = rig.arrow(session, "_scale_children", 1)

    rig.send("pointer_down", at[0], at[1])
    assert (gizmo.handle_kind, gizmo.handle_axis) == ("scale", 1)
    rig.send("pointer_move", at[0] + 40 * out[0], at[1] + 40 * out[1])
    rig.frame()
    # Held: the gizmo is scaled, and the model's extent follows it.
    factor = gizmo.frame()[2][0]
    assert factor > 1.05
    held = rig.model(plane)
    np.testing.assert_allclose(held.extent_0, np.array(plane.extent_0) * factor)
    assert held.extent_1 == plane.extent_1
    assert held.origin == plane.origin
    # What a control hears during the drag is the growing extent.
    assert rig.of(RenderPlanesChangedEvent)[-1].render_planes == (held,)
    assert rig.of(RenderPlanesChangedEvent)[-1].source_id == session.id

    rig.send("pointer_up", at[0] + 40 * out[0], at[1] + 40 * out[1], buttons=())
    rig.frame()
    # Released: the factor stays in the extent and the gizmo's scale is 1.
    assert rig.model(plane) == held
    assert gizmo.frame()[2] == (1.0, 1.0)
    np.testing.assert_allclose(gizmo.proxy.local.scale, (1.0, 1.0, 1.0))

    # A second drag starts from the size the first one left.
    at, out, _ = rig.arrow(session, "_scale_children", 1)
    rig.send("pointer_down", at[0], at[1])
    rig.send("pointer_move", at[0] + 30 * out[0], at[1] + 30 * out[1])
    rig.frame()
    second = gizmo.frame()[2][0]
    np.testing.assert_allclose(
        rig.model(plane).extent_0, np.array(held.extent_0) * second
    )
    rig.send("pointer_up", at[0] + 30 * out[0], at[1] + 30 * out[1], buttons=())
    assert gizmo.frame()[2] == (1.0, 1.0)


# -- one interaction scope per drag ---------------------------------------------


async def test_a_drag_is_one_plane_interaction(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    session = rig.session(plane)
    rig.look((0.5, 0.6, 0.62))
    at, out, _ = rig.arrow(session, "_translate1_children", 0)
    assert rig.drag(session, at, 30 * out) == ("translate", 0)
    interactions = [(e.phase, e.reason) for e in rig.of(PlaneInteractionEvent)]
    assert interactions == [("start", None), ("end", "release")]
    assert rig.controller.plane_interaction_state(rig.visual.id) == "idle"
    assert all(e.source_id == session.id for e in rig.of(RenderPlanesChangedEvent))
    assert len(rig.of(RenderPlanesChangedEvent)) == 2  # one per framed move


class MultiscaleRig(PlaneRig):
    """A multiscale image in plane mode: a drag holds its plan."""

    def __init__(self, tmp_path, *, n_canvases: int = 1) -> None:
        self.controller = CellierController(gui="offscreen")
        self.controller.camera_reslice_enabled = False
        self.controller._render_manager.config.scheduler.dims_settle_s = SETTLE_S
        fx.write_pyramid(tmp_path, fx.ISO)
        store, _ = fx.open_pyramid(tmp_path, fx.ISO)
        self.scene = self.controller.add_scene(dim="3d", name="scene")
        self.world = self.controller._model.scenes[
            self.scene.id
        ].dims.world_coordinate_system
        self.visual = self.controller.add_image_multiscale(
            data=store,
            scene_id=self.scene.id,
            appearance=MultiscaleImageAppearance(force_level=1),
            single=MultiscaleImageSingleAppearance(
                color_map="viridis", clim=(0.0, 65535.0), render_mode="plane"
            ),
            render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        )
        self._finish(n_canvases)

    def plane(self, **kwargs):
        centre = tuple(s / 2 for s in fx.ISO.shape0)
        return RenderPlane.from_point_normal(
            self.world,
            centre,
            (1.0, 0.3, 0.2),
            axes=AXES,
            up=(0.0, 1.0, 0.1),
            extent_0=(-30.0, 30.0),
            extent_1=(-30.0, 30.0),
            **kwargs,
        )

    def look(self, direction) -> None:
        camera = self.view.camera
        centre = np.array([s / 2 for s in fx.ISO.shape0][::-1])
        distance = float(np.linalg.norm(np.asarray(camera.world.position) - centre))
        camera.local.position = centre + np.asarray(direction) * distance
        camera.look_at(tuple(centre))
        self.frame(n=2)


def _record_plans(monkeypatch) -> list[str]:
    from cellier.render.scene_manager import SceneManager

    plans: list[str] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for visual_id, cfg in visual_configs.items():
            targets = request.target_visual_ids
            if targets is None or visual_id in targets:
                plans.append(cfg.plan_mode.name)
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


async def test_a_gizmo_drag_holds_the_plan_and_loads_once_at_release(
    make_rig, tmp_path, monkeypatch
):
    """B1, through the real gizmo callbacks: counts only."""
    rig = make_rig(MultiscaleRig, tmp_path=tmp_path)
    plane = rig.plane()
    rig.controller.set_render_planes(rig.visual.id, (plane,))
    await rig.start()
    rig.log.clear()
    session = rig.session(plane)
    rig.look((0.5, 0.6, 0.62))
    await drain_loading(rig.controller)
    plans = _record_plans(monkeypatch)
    scheduler = rig.controller._render_manager.scheduler
    at, out, _ = rig.arrow(session, "_translate1_children", 0)

    rig.send("pointer_down", at[0], at[1])
    assert session.dragging
    for step in (0.25, 0.5, 0.75, 1.0):
        rig.send("pointer_move", at[0] + 40 * step * out[0], at[1] + 40 * step * out[1])
        rig.frame()
        assert plans == []  # nothing planned while the handle moves
        assert scheduler.core.idle()
    assert rig.controller.plane_interaction_state(rig.visual.id) == "active"
    assert rig.model(plane).origin != plane.origin
    rig.send("pointer_up", at[0] + 40 * out[0], at[1] + 40 * out[1], buttons=())
    # The target loads once, at the release.
    assert plans == ["FULL"]
    interactions = [(e.phase, e.reason) for e in rig.of(PlaneInteractionEvent)]
    assert interactions == [("start", None), ("end", "release")]
    await drain_loading(rig.controller)


async def test_a_pause_ends_the_motion_and_the_next_move_starts_another(
    make_rig, tmp_path, monkeypatch
):
    rig = make_rig(MultiscaleRig, tmp_path=tmp_path)
    plane = rig.plane()
    rig.controller.set_render_planes(rig.visual.id, (plane,))
    await rig.start()
    rig.log.clear()
    session = rig.session(plane)
    rig.look((0.5, 0.6, 0.62))
    await drain_loading(rig.controller)
    plans = _record_plans(monkeypatch)
    at, out, _ = rig.arrow(session, "_translate1_children", 0)

    rig.send("pointer_down", at[0], at[1])
    rig.send("pointer_move", at[0] + 15 * out[0], at[1] + 15 * out[1])
    rig.frame()
    assert plans == []
    # The handle is held still: the motion settles and the plane refines.
    await asyncio.sleep(SETTLE_S * 4)
    assert session.dragging
    assert plans == ["FULL"]
    rig.send("pointer_move", at[0] + 30 * out[0], at[1] + 30 * out[1])
    rig.frame()
    assert plans == ["FULL"]  # moving again: held again
    rig.send("pointer_up", at[0] + 30 * out[0], at[1] + 30 * out[1], buttons=())
    assert plans == ["FULL", "FULL"]
    interactions = [(e.phase, e.reason) for e in rig.of(PlaneInteractionEvent)]
    assert interactions == [
        ("start", None),
        ("end", "settle"),
        ("start", None),
        ("end", "release"),
    ]
    await drain_loading(rig.controller)


# -- closing --------------------------------------------------------------------


async def _open(make_rig, **kwargs):
    rig = make_rig(**kwargs)
    await rig.start()
    plane = rig.plane()
    other = rig.plane(normal=(0.0, 1.0, 0.0))
    rig.show(plane, other)
    return rig, plane, other, rig.session(plane)


def _closed_event(rig) -> PlaneGizmoChangedEvent:
    return rig.of(PlaneGizmoChangedEvent)[-1]


async def test_removing_the_plane_closes_the_session(make_rig):
    rig, _plane, other, session = await _open(make_rig)
    rig.controller.set_render_planes(rig.visual.id, (other,))
    assert session.closed
    assert rig.controller.get_plane_gizmo(rig.view.canvas_id) is None
    assert _closed_event(rig)[2:] == (None, None, None)
    with pytest.raises(KeyError):
        rig.view.get_plane_gizmo(session.id)


async def test_removing_another_plane_leaves_the_session(make_rig):
    rig, plane, _other, session = await _open(make_rig)
    rig.controller.set_render_planes(rig.visual.id, (plane,))
    assert not session.closed


async def test_leaving_plane_mode_closes_the_session(make_rig):
    rig, _plane, _other, session = await _open(make_rig)
    rig.visual.single.render_mode = "mip"
    assert session.closed
    assert _closed_event(rig).plane_id is None


async def test_leaving_3d_closes_the_session(make_rig):
    rig, _plane, _other, session = await _open(make_rig)
    rig.controller.set_displayed_axes(rig.scene.id, (1, 2))
    assert session.closed
    assert _closed_event(rig).plane_id is None


async def test_other_displayed_axes_close_the_session(make_rig):
    rig, _plane, _other, session = await _open(make_rig, time_axis=True)
    assert not session.closed
    # Still 3D, but t, y, x: the plane's z, y, x are not the displayed ones.
    with pytest.warns(UserWarning, match="not drawn"):
        rig.controller.set_displayed_axes(rig.scene.id, (0, 2, 3))
    assert session.closed
    assert rig.controller.get_plane_gizmo(rig.view.canvas_id) is None


async def test_removing_the_visual_closes_the_session(make_rig):
    rig, _plane, _other, session = await _open(make_rig)
    rig.controller.remove_visual(rig.visual.id)
    assert session.closed


async def test_closing_mid_drag_ends_the_interaction(make_rig):
    rig, plane, other, session = await _open(make_rig)
    rig.look((0.5, 0.6, 0.62))
    at, out, _ = rig.arrow(session, "_translate1_children", 0)
    rig.send("pointer_down", at[0], at[1])
    rig.send("pointer_move", at[0] + 20 * out[0], at[1] + 20 * out[1])
    rig.frame()
    assert rig.controller.plane_interaction_state(rig.visual.id) == "active"
    rig.controller.set_render_planes(rig.visual.id, (other,))
    assert session.closed
    assert not CAPTURES
    del plane


# -- refusals -------------------------------------------------------------------


async def test_a_gizmo_is_refused_outside_plane_mode(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    rig.visual.single.render_mode = "mip"
    canvas_id = rig.view.canvas_id
    assert "plane" in rig.controller.render_plane_gizmo_blocked(
        rig.visual.id, canvas_id, plane.id
    )
    with pytest.raises(ValueError, match="'plane' render mode"):
        rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, plane.id)
    assert rig.controller.get_plane_gizmo(canvas_id) is None
    rig.visual.single.render_mode = "plane"
    assert (
        rig.controller.render_plane_gizmo_blocked(rig.visual.id, canvas_id, plane.id)
        == ""
    )


async def test_a_gizmo_is_refused_in_2d_and_for_an_unknown_plane(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    canvas_id = rig.view.canvas_id
    with pytest.raises(KeyError):
        rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, uuid4())
    rig.controller.set_displayed_axes(rig.scene.id, (1, 2))
    assert "2D" in rig.controller.render_plane_gizmo_blocked(
        rig.visual.id, canvas_id, plane.id
    )
    with pytest.raises(ValueError, match="2D"):
        rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, plane.id)


async def test_a_plane_off_the_displayed_axes_is_refused(make_rig):
    rig = make_rig(time_axis=True)
    await rig.start()
    plane = RenderPlane.from_point_normal(
        rig.world, (0.0, 15.5, 15.5), (0, 0, 1), axes=("t", "y", "x")
    )
    with pytest.warns(UserWarning, match="not drawn"):
        rig.show(plane)
    canvas_id = rig.view.canvas_id
    reason = rig.controller.render_plane_gizmo_blocked(
        rig.visual.id, canvas_id, plane.id
    )
    assert "axes other than" in reason
    with pytest.raises(ValueError, match="three axes"):
        rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, plane.id)


# -- one gizmo per canvas, for both kinds ---------------------------------------


def _clip(rig) -> ClippingPlane:
    store = rig.controller.get_data_store(UUID(str(rig.visual.data_store_id)))
    item = ClippingPlane.from_point_normal(
        store.data_coordinate_system, MIDDLE, (0, 0, 1), enabled=False
    )
    rig.visual.clipping_planes = (item,)
    rig.log.clear()
    return item


async def test_a_clipping_and_a_render_gizmo_are_exclusive(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    item = _clip(rig)
    canvas_id = rig.view.canvas_id

    clipping = rig.controller.add_clipping_plane_gizmo(
        rig.visual.id, canvas_id, item.id
    )
    assert isinstance(clipping, ClippingPlaneGizmoController)
    assert rig.controller.get_plane_gizmo(canvas_id).kind == "clipping"
    render = rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, plane.id)
    assert clipping.closed
    assert not render.closed
    assert rig.controller.get_plane_gizmo(canvas_id) is render
    again = rig.controller.add_clipping_plane_gizmo(rig.visual.id, canvas_id, item.id)
    assert render.closed
    assert rig.controller.get_plane_gizmo(canvas_id) is again
    # One event per replacement, naming the kind and the plane: never "none".
    assert [(e.kind, e.plane_id) for e in rig.of(PlaneGizmoChangedEvent)] == [
        ("clipping", item.id),
        ("render", plane.id),
        ("clipping", item.id),
    ]
    assert len(rig.view._plane_gizmos) == 1
    rig.controller.remove_plane_gizmo(canvas_id)
    assert rig.of(PlaneGizmoChangedEvent)[-1][2:] == (None, None, None)
    assert not rig.view._plane_gizmos


async def test_a_refused_gizmo_leaves_the_one_the_canvas_had(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    item = _clip(rig)
    canvas_id = rig.view.canvas_id
    render = rig.controller.add_render_plane_gizmo(rig.visual.id, canvas_id, plane.id)
    rig.log.clear()
    with pytest.raises(KeyError):
        rig.controller.add_clipping_plane_gizmo(rig.visual.id, canvas_id, uuid4())
    assert not render.closed
    assert rig.controller.get_plane_gizmo(canvas_id) is render
    assert rig.of(PlaneGizmoChangedEvent) == []
    del item


async def test_the_request_event_names_the_kind(make_rig):
    rig = make_rig()
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    item = _clip(rig)
    canvas_id = rig.view.canvas_id
    widget = uuid4()

    def request(kind, plane_id, enabled=True):
        rig.controller.incoming_events.emit(
            PlaneGizmoUpdateEvent(
                source_id=widget,
                visual_id=rig.visual.id,
                plane_id=plane_id,
                canvas_id=canvas_id,
                kind=kind,
                enabled=enabled,
            )
        )

    request("render", plane.id)
    session = rig.controller.get_plane_gizmo(canvas_id)
    assert (session.kind, session.plane_id) == ("render", plane.id)
    event = rig.of(PlaneGizmoChangedEvent)[-1]
    assert (event.source_id, event.kind, event.plane_id) == (widget, "render", plane.id)

    # Switching a clipping toggle off does not close a render plane's gizmo.
    request("clipping", plane.id, enabled=False)
    assert rig.controller.get_plane_gizmo(canvas_id) is session
    request("clipping", item.id)
    assert rig.controller.get_plane_gizmo(canvas_id).kind == "clipping"
    assert session.closed
    request("clipping", item.id, enabled=False)
    assert rig.controller.get_plane_gizmo(canvas_id) is None
    with pytest.raises(Exception, match="Unknown plane gizmo kind"):
        request("slab", plane.id)


async def test_each_canvas_has_its_own_gizmo(make_rig):
    rig = make_rig(n_canvases=2)
    await rig.start()
    plane = rig.plane()
    rig.show(plane)
    first = rig.session(plane, view=rig.views[0])
    second = rig.session(plane, view=rig.views[1])
    assert not first.closed and not second.closed
    # A drag in one canvas moves the gizmo of the other.
    rig.look((0.5, 0.6, 0.62))
    at, out, _ = rig.arrow(first, "_translate1_children", 0)
    assert rig.drag(first, at, 30 * out) == ("translate", 0)
    moved = rig.model(plane)
    assert moved.origin != plane.origin
    np.testing.assert_allclose(
        rig.gizmo(second).pose()[0], moved.origin[::-1], atol=1e-5
    )
    assert la.vec_dist(rig.gizmo(first).pose()[0], rig.gizmo(second).pose()[0]) < 1e-5
