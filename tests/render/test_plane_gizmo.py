"""The render side of the plane gizmo (clipping plane gizmo design v2, 5.1).

Headless: offscreen canvases, synthetic input through ``canvas.submit_event``
as ``tests/render/test_camera_interaction.py`` does.  The gizmo sits inside
an ISO volume, so every drag also shows that a handle can be grabbed there.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pygfx as gfx
import pylinalg as la
import pytest

from cellier.controller import CellierController
from cellier.data import ImageMemoryStore
from cellier.events import PlaneGizmoMovedEvent
from cellier.render import RenderManagerConfig
from cellier.render._config import AmbientOcclusionConfig
from cellier.render._visual_lut import AO_EXCLUDED_BIT, get_shared_visual_lut
from cellier.visuals import InMemoryImageSingleAppearance
from tests.render.conftest import drain_loading

SIZE = (320, 240)
#: The volume is (z, y, x) = (32, 32, 32); rendered space is (x, y, z).
CENTRE = (15.5, 15.5, 15.5)
CAPTURES = gfx.objects.EventTarget.pointer_captures


def _ball(n: int = 32) -> np.ndarray:
    z, y, x = np.indices((n, n, n)) - (n - 1) / 2
    return ((z**2 + y**2 + x**2) < (0.4 * n) ** 2).astype(np.float32)


class Rig:
    """One controller, one 3D scene with an ISO ball, offscreen canvases."""

    def __init__(self, *, n_canvases: int = 1, ambient_occlusion: bool = False):
        self.controller = CellierController(
            gui="offscreen",
            render_config=RenderManagerConfig(
                ambient_occlusion=AmbientOcclusionConfig(enabled=ambient_occlusion)
            ),
        )
        self.controller.camera_reslice_enabled = False
        self.scene = self.controller.add_scene(dim="3d", name="scene")
        self.visual = self.controller.add_image(
            data=ImageMemoryStore(data=_ball()),
            scene_id=self.scene.id,
            single=InMemoryImageSingleAppearance(
                color_map="gray", clim=(0.0, 1.0), render_mode="iso", iso_threshold=0.5
            ),
        )
        self.views = []
        for _ in range(n_canvases):
            self.controller.add_canvas(self.scene.id, canvas_size=SIZE)
            canvas_id = self.controller.get_canvas_ids(self.scene.id)[-1]
            self.views.append(self.controller.get_canvas_view(canvas_id))
        self.view = self.views[0]
        self.events: list[PlaneGizmoMovedEvent] = []
        self.in_draw: list[bool] = []

    async def start(self) -> None:
        self.controller.fit_camera(self.scene.id)
        self.controller.reslice_all()
        await drain_loading(self.controller)
        for view in self.views:
            self.frame(view=view, n=2)

    def add_gizmo(self, view=None):
        view = view or self.view
        gizmo = view.add_plane_gizmo(uuid4(), CENTRE, (1.0, 0.0, 0.0), screen_size=90)

        def record(event: PlaneGizmoMovedEvent) -> None:
            self.events.append(event)
            self.in_draw.append(bool(view._drawing))

        self.controller._outgoing_events.subscribe(
            PlaneGizmoMovedEvent, record, entity_id=gizmo.gizmo_id
        )
        self.frame(view=view, n=2)
        return gizmo

    def frame(self, view=None, n: int = 1) -> np.ndarray:
        view = view or self.view
        out = None
        for _ in range(n):
            out = np.asarray(view.widget.draw())
        return out

    def send(self, kind: str, x, y, *, button=1, buttons=(1,), view=None) -> None:
        view = view or self.view
        view.widget.submit_event(
            {
                "event_type": kind,
                "x": float(x),
                "y": float(y),
                "button": button if kind != "pointer_move" else 0,
                "buttons": tuple(buttons),
                "modifiers": (),
                "ntouches": 0,
                "touches": {},
                "pointer_id": 0,
            }
        )
        view.widget._process_events()

    def screen(self, world_pos, view=None) -> tuple[float, float]:
        view = view or self.view
        width, height = view.widget.get_logical_size()
        ndc = la.vec_transform(world_pos, view.camera.camera_matrix)
        return (ndc[0] + 1) / 2 * width, (1 - ndc[1]) / 2 * height

    def handle(self, gizmo, group: str, index: int, view=None):
        element = getattr(gizmo.gizmo, group)[index]
        return self.screen(element.world.position, view=view)

    def phases(self) -> list[str]:
        return [event.phase for event in self.events]


@pytest.fixture
def make_rig():
    rigs: list[Rig] = []
    CAPTURES.clear()

    def _make(**kwargs) -> Rig:
        rig = Rig(**kwargs)
        rigs.append(rig)
        return rig

    yield _make
    for rig in rigs:
        rig.controller.close()
    CAPTURES.clear()


# -- a drag ---------------------------------------------------------------------


async def test_a_drag_is_a_start_one_move_per_frame_and_an_end(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    start_point, start_normal = gizmo.pose()

    rig.send("pointer_down", x, y)
    assert rig.phases() == ["start"]
    assert rig.events[0].handle_kind == "translate"
    assert rig.events[0].handle_axis == 0
    assert rig.events[0].point == start_point
    assert rig.events[0].canvas_id == rig.view.canvas_id
    assert gizmo.dragging

    # Several pointer moves in one frame: nothing is reported until the frame.
    for step in (8, 16, 24):
        rig.send("pointer_move", x + step, y + step * 0.5)
    assert rig.phases() == ["start"]
    rig.frame()
    assert rig.phases() == ["start", "move"]
    moved = rig.events[-1]
    assert moved.point == gizmo.pose()[0]
    assert moved.point != start_point
    # A one-axis translate handle moves the point along the normal only.
    np.testing.assert_allclose(moved.normal, start_normal, atol=1e-6)
    np.testing.assert_allclose(moved.point[1:], start_point[1:], atol=1e-4)

    # A frame with no move reports nothing.
    rig.frame()
    assert rig.phases() == ["start", "move"]

    # The release reports the pose still pending, then the end, at once.
    rig.send("pointer_move", x + 40, y + 20)
    rig.send("pointer_up", x + 40, y + 20, buttons=())
    assert rig.phases() == ["start", "move", "move", "end"]
    assert rig.events[-2].point == gizmo.pose()[0]
    assert rig.events[-1].point == gizmo.pose()[0]
    assert not gizmo.dragging
    assert not CAPTURES
    rig.frame()
    assert rig.phases() == ["start", "move", "move", "end"]


async def test_a_rotate_drag_turns_the_normal_about_the_point(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    point, normal = gizmo.pose()
    # Rotating about axis 1 or 2 tilts the plane; about axis 0 it only spins.
    x, y = rig.handle(gizmo, "_rotate_children", 2)
    rig.send("pointer_down", x, y)
    assert (rig.events[0].handle_kind, rig.events[0].handle_axis) == ("rotate", 2)
    rig.send("pointer_move", x + 25, y - 30)
    rig.send("pointer_up", x + 25, y - 30, buttons=())
    end = rig.events[-1]
    np.testing.assert_allclose(end.point, point, atol=1e-5)
    assert np.linalg.norm(end.normal) == pytest.approx(1.0)
    assert not np.allclose(end.normal, normal, atol=1e-3)


async def test_a_click_without_a_move_is_a_start_and_an_end(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    before = gizmo.pose()
    rig.send("pointer_down", x, y)
    rig.send("pointer_move", x + 1, y)  # inside pygfx's dead zone
    rig.send("pointer_up", x + 1, y, buttons=())
    assert rig.phases() == ["start", "end"]
    assert gizmo.pose() == before


async def test_a_drag_does_not_move_the_camera(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    camera = rig.view.capture_camera_state()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.send("pointer_move", x + 30, y + 10)
    rig.frame()
    rig.send("pointer_up", x + 30, y + 10, buttons=())
    rig.frame()
    assert rig.view.capture_camera_state() == camera
    # A drag off the gizmo still orbits.
    rig.send("pointer_down", 5, 5)
    assert len(rig.view._controller_3d._actions) > 0
    rig.send("pointer_up", 5, 5, buttons=())
    assert rig.phases() == ["start", "move", "end"]


# -- where the move is reported (G-P7, G-P8) ---------------------------------------


async def test_a_move_is_reported_ahead_of_the_draw_and_resets_accumulation(
    make_rig, monkeypatch
):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    resets = []
    reset = rig.view._accum_pass.reset
    monkeypatch.setattr(
        rig.view._accum_pass, "reset", lambda: (resets.append(1), reset())[1]
    )

    def frames_that_reset() -> int:
        resets.clear()
        rig.frame()
        first = len(resets)
        rig.frame()
        assert len(resets) == first  # and the next frame keeps accumulating
        return first

    rig.frame(n=2)
    assert frames_that_reset() == 0

    # An in-plane move changes no plane, and still must not ghost.
    x, y = rig.handle(gizmo, "_translate1_children", 1)
    rig.send("pointer_down", x, y)
    assert rig.view._accum_dirty
    assert frames_that_reset() == 1  # the grab: a handle is highlighted
    rig.send("pointer_move", x + 20, y + 10)
    assert frames_that_reset() == 1
    rig.send("pointer_up", x + 20, y + 10, buttons=())
    assert frames_that_reset() == 1  # the release: the highlight goes

    # Every report came from outside the draw callback: planning must not
    # run inside one.
    assert rig.phases() == ["start", "move", "end"]
    assert rig.in_draw == [False, False, False]


async def test_a_grab_ends_a_draw_hold(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    rig.view.hold_draws(30.0, lambda: True)
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    assert rig.view._hold_waiting is None
    rig.view.hold_draws(30.0, lambda: True)
    rig.send("pointer_move", x + 20, y)
    assert rig.view._hold_waiting is None
    rig.send("pointer_up", x + 20, y, buttons=())


# -- a lost release (G-P9) ---------------------------------------------------------


async def test_a_press_in_another_canvas_ends_a_drag_whose_release_was_lost(make_rig):
    rig = make_rig(n_canvases=2)
    await rig.start()
    first, second = rig.views
    gizmo = rig.add_gizmo(first)
    x, y = rig.handle(gizmo, "_translate1_children", 0, view=first)
    rig.send("pointer_down", x, y, view=first)
    rig.send("pointer_move", x + 20, y + 10, view=first)
    assert gizmo.dragging
    assert CAPTURES

    # The release never comes.  A left press in the other canvas.
    rig.send("pointer_down", 40, 40, view=second)
    assert rig.phases() == ["start", "move", "end"]
    assert not gizmo.dragging
    assert len(second._controller_3d._actions) > 0  # the same press orbits
    rig.send("pointer_up", 40, 40, buttons=(), view=second)
    assert not CAPTURES

    # A hover over a handle afterwards moves nothing.
    rig.frame(view=first)
    before = gizmo.pose()
    hx, hy = rig.handle(gizmo, "_translate1_children", 1, view=first)
    rig.send("pointer_move", hx, hy, buttons=(), view=first)
    rig.frame(view=first)
    assert gizmo.pose() == before
    assert rig.phases() == ["start", "move", "end"]


async def test_the_next_press_in_the_same_canvas_ends_a_lost_drag(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.send("pointer_down", 5, 5)  # off the gizmo
    assert rig.phases() == ["start", "end"]
    assert len(rig.view._controller_3d._actions) > 0
    rig.send("pointer_up", 5, 5, buttons=())
    assert not CAPTURES


async def test_a_right_press_during_a_live_drag_leaves_it(make_rig):
    rig = make_rig(n_canvases=2)
    await rig.start()
    first, second = rig.views
    gizmo = rig.add_gizmo(first)
    x, y = rig.handle(gizmo, "_translate1_children", 0, view=first)
    rig.send("pointer_down", x, y, view=first)
    rig.send("pointer_down", 40, 40, button=2, buttons=(1, 2), view=second)
    assert gizmo.dragging
    assert CAPTURES
    assert rig.phases() == ["start"]
    rig.send("pointer_up", x, y, buttons=(), view=first)
    assert rig.phases() == ["start", "end"]


# -- what a plane does not have (G-P6) ---------------------------------------------


async def test_the_centre_sphere_and_scale_handles_cannot_be_grabbed(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    inner = gizmo.gizmo
    assert inner._mode == "object"
    assert not inner._center_sphere.visible
    assert not any(element.visible for element in inner._scale_children)
    # No element is depth tested, or the volume would win every pick.
    assert not any(element.material.depth_test for element in inner.children)
    scale = gizmo.proxy.local.scale.copy()

    presses = [rig.screen(gizmo.proxy.world.position)]
    presses += [rig.handle(gizmo, "_scale_children", index) for index in range(3)]
    for x, y in presses:
        rig.send("pointer_down", x, y)
        rig.send("pointer_move", x + 15, y + 15)
        rig.send("pointer_up", x + 15, y + 15, buttons=())
        rig.view._controller_3d._actions.clear()
        rig.frame()
    assert rig.events == []
    np.testing.assert_array_equal(gizmo.proxy.local.scale, scale)
    assert inner._mode == "object"


def test_the_private_pygfx_names_the_gizmo_relies_on_exist():
    """Fails loudly on a pygfx upgrade that renames them (R3)."""
    inner = gfx.TransformGizmo(gfx.Group())
    for name in (
        "_ref",
        "_camera",
        "_viewport",
        "_mode",
        "_center_sphere",
        "_scale_children",
        "_translate1_children",
        "_translate2_children",
        "_rotate_children",
    ):
        assert hasattr(inner, name), name
    for name in ("_update_visibility", "_highlight", "update_gizmo", "process_event"):
        assert callable(getattr(gfx.TransformGizmo, name)), name
    assert isinstance(CAPTURES, dict)
    # A handle's axis: an index, or a pair for a two-axis translate handle.
    assert [element.dim for element in inner._translate1_children] == [0, 1, 2]
    assert [element.dim for element in inner._rotate_children] == [0, 1, 2]
    assert sorted(tuple(element.dim) for element in inner._translate2_children) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]


# -- the pass (G-P3, G-P12) --------------------------------------------------------


class _Quad:
    """A canvas overlay: one screen-space quad that writes no pick ids."""

    def __init__(self, ndc_xy) -> None:
        self.overlay_scene = gfx.Scene()
        self.overlay_camera = gfx.NDCCamera()
        quad = gfx.Mesh(
            gfx.plane_geometry(0.3, 0.3),
            gfx.MeshBasicMaterial(
                color=(0.0, 1.0, 1.0, 1.0), depth_test=False, depth_write=False
            ),
        )
        quad.local.position = (ndc_xy[0], ndc_xy[1], 0.0)
        self.overlay_scene.add(quad)

    def on_frame(self, width, height) -> None:
        pass


async def test_the_gizmo_is_drawn_under_the_overlays_and_stays_draggable(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    width, height = rig.view.widget.get_logical_size()

    def pixel() -> tuple[int, int, int]:
        # Adding an overlay does not reset the accumulation history.
        rig.view.invalidate_accumulation()
        frame = rig.frame(n=2)
        row = int(y * frame.shape[0] / height)
        column = int(x * frame.shape[1] / width)
        r, g, b = (int(v) for v in frame[row, column, :3])
        return r, g, b

    bare = pixel()
    assert bare[0] > 150  # the handle of axis 0, drawn over the volume
    assert bare[1] < 80

    rig.view.add_overlay(_Quad((x / width * 2 - 1, -(y / height * 2 - 1))))
    r, g, b = pixel()
    assert r < 60  # the overlay is on top
    assert g > 200
    assert b > 200

    rig.send("pointer_down", x, y)
    rig.send("pointer_up", x, y, buttons=())
    assert rig.phases() == ["start", "end"]


async def test_the_gizmo_is_not_drawn_in_2d_and_a_switch_ends_a_drag(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    rendered = []
    render = rig.view._renderer.render
    rig.view._renderer.render = lambda scene, *a, **k: (
        rendered.append(scene),
        render(scene, *a, **k),
    )[1]
    rig.frame()
    assert gizmo.scene in rendered

    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.view.switch_dim("2d")
    assert rig.phases() == ["start", "end"]
    assert not gizmo.dragging
    assert not CAPTURES
    rendered.clear()
    rig.frame()
    assert gizmo.scene not in rendered

    rig.view.switch_dim("3d")
    rendered.clear()
    rig.frame()
    assert gizmo.scene in rendered


async def test_a_hidden_gizmo_is_not_drawn_and_hiding_ends_a_drag(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    shown = rig.frame(n=2)
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    gizmo.set_visible(False)
    assert rig.phases() == ["start", "end"]
    hidden = rig.frame(n=2)
    assert np.abs(shown.astype(int) - hidden.astype(int)).max() > 100
    # Nothing to grab where the handle was.
    rig.send("pointer_down", x, y)
    rig.send("pointer_up", x, y, buttons=())
    assert rig.phases() == ["start", "end"]
    gizmo.set_visible(True)
    rig.view._controller_3d._actions.clear()
    rig.frame(n=2)
    rig.send("pointer_down", x, y)
    rig.send("pointer_up", x, y, buttons=())
    assert rig.phases() == ["start", "end", "start", "end"]


# -- adding and removing -----------------------------------------------------------


async def test_removing_a_gizmo_ends_its_drag_and_leaves_nothing_behind(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    view = rig.view
    with pytest.raises(ValueError, match="already has gizmo"):
        view.add_plane_gizmo(gizmo.gizmo_id, CENTRE, (1, 0, 0))
    assert view.get_plane_gizmo(gizmo.gizmo_id) is gizmo
    assert view.plane_gizmo_object_ids() == gizmo.object_ids()

    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.send("pointer_move", x + 20, y + 10)
    view.remove_plane_gizmo(gizmo.gizmo_id)
    assert rig.phases() == ["start", "move", "end"]
    assert not CAPTURES
    assert view.plane_gizmo_object_ids() == set()
    with pytest.raises(KeyError):
        view.get_plane_gizmo(gizmo.gizmo_id)
    view.remove_plane_gizmo(gizmo.gizmo_id)  # ignored

    # No pass, no report, and the handle is gone from the pick buffer.
    rig.frame(n=2)
    rig.send("pointer_down", x, y)
    rig.send("pointer_up", x, y, buttons=())
    rig.frame()
    assert rig.phases() == ["start", "move", "end"]


async def test_closing_the_canvas_removes_its_gizmos(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()
    x, y = rig.handle(gizmo, "_translate1_children", 0)
    rig.send("pointer_down", x, y)
    rig.view.close()
    assert not CAPTURES
    assert rig.view.plane_gizmo_object_ids() == set()
    rig.view.close()


# -- the pose ----------------------------------------------------------------------


async def test_set_pose_turns_the_proxy_by_the_smallest_rotation(make_rig):
    rig = make_rig()
    await rig.start()
    gizmo = rig.add_gizmo()

    def in_plane() -> np.ndarray:
        return la.vec_transform_quat((0.0, 1.0, 0.0), gizmo.proxy.local.rotation)

    gizmo.set_pose((1.0, 2.0, 3.0), (0.0, 0.0, 5.0))
    point, normal = gizmo.pose()
    assert point == (1.0, 2.0, 3.0)
    np.testing.assert_allclose(normal, (0.0, 0.0, 1.0), atol=1e-6)
    # Turned about y: the in-plane y axis did not move, so no spin was added.
    np.testing.assert_allclose(in_plane(), (0.0, 1.0, 0.0), atol=1e-6)

    # A flipped normal is a half turn about an in-plane axis.
    gizmo.set_pose((1.0, 2.0, 3.0), (0.0, 0.0, -1.0))
    np.testing.assert_allclose(gizmo.pose()[1], (0.0, 0.0, -1.0), atol=1e-6)
    np.testing.assert_allclose(in_plane(), (0.0, 1.0, 0.0), atol=1e-6)

    # The same pose again changes nothing.
    rotation = gizmo.proxy.local.rotation.copy()
    gizmo.set_pose((1.0, 2.0, 3.0), (0.0, 0.0, -1.0))
    np.testing.assert_allclose(gizmo.proxy.local.rotation, rotation, atol=1e-9)

    with pytest.raises(ValueError, match="all zero"):
        gizmo.set_pose(CENTRE, (0.0, 0.0, 0.0))


# -- ambient occlusion (G-P4) ------------------------------------------------------


def _erode(mask: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        inner = mask.copy()
        inner[1:, :] &= mask[:-1, :]
        inner[:-1, :] &= mask[1:, :]
        inner[:, 1:] &= mask[:, :-1]
        inner[:, :-1] &= mask[:, 1:]
        mask = inner
    return mask


def _differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(axis=-1) > 12


async def test_the_handles_get_no_ambient_occlusion(make_rig):
    rig = make_rig(ambient_occlusion=True)
    await rig.start()
    manager = rig.controller._render_manager
    lut = get_shared_visual_lut()

    def pair() -> tuple[np.ndarray, np.ndarray]:
        manager.ambient_occlusion_enabled = True
        with_ao = rig.frame(n=30)
        manager.ambient_occlusion_enabled = False
        without = rig.frame(n=30)
        manager.ambient_occlusion_enabled = True
        return with_ao, without

    hidden_ao, hidden_plain = pair()
    assert _differs(hidden_ao, hidden_plain).sum() > 100  # the pass does something
    assert not rig.view._ssao_pass._has_exclusions

    gizmo = rig.add_gizmo()
    shown_ao, shown_plain = pair()
    # The entries are in the table and the pass has its lookup on.
    rig.frame()
    assert rig.view._ssao_pass._has_exclusions
    for object_id in gizmo.object_ids():
        assert lut.get_entry(object_id) & AO_EXCLUDED_BIT
    # Inside the handles (two pixels in from their edge) nothing changes.
    inside = _erode(_differs(shown_plain, hidden_plain), 2)
    assert inside.sum() > 200
    assert _differs(shown_ao, shown_plain)[inside].sum() == 0

    rig.view.remove_plane_gizmo(gizmo.gizmo_id)
    rig.frame()
    assert not rig.view._ssao_pass._has_exclusions
    for object_id in gizmo.object_ids():
        assert not lut.get_entry(object_id) & AO_EXCLUDED_BIT
