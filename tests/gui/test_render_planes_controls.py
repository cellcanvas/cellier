"""The "Render planes" control on both toolkits (plane rendering design 9.2).

The control sends ``RenderPlanesUpdateEvent`` and follows
``RenderPlanesChangedEvent``.  It is the "Clipping planes" control with the
extents added, in world coordinates.  Also here: "Plane" in the render-mode
combos (9.1).
"""

from __future__ import annotations

import itertools
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import ImageMemoryStore, LabelMemoryStore
from cellier.events import (
    AppearanceUpdateEvent,
    ChannelAppearanceUpdateEvent,
    SingleAppearanceUpdateEvent,
)
from cellier.gui._render_planes import (
    RENDER_PLANES_TITLE,
    RenderPlaneGizmoTarget,
    box_cut,
    flipped,
    get_render_plane_gizmo_data,
    get_render_planes_data_from_visual,
    rows_from_planes,
    with_normal,
    with_position,
    with_side_bounded,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageChannelAppearance,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    RenderPlane,
)

_SERIALS = itertools.count(1)
_QTBOT: list = []
AXES = ("z", "y", "x")
#: The image is (z, y, x) = (10, 20, 40) voxels: world box 0 to 10 / 20 / 40
#: (a voxel's centre is at its index, so the box is -0.5 to 9.5 and so on).
SHAPE = (10, 20, 40)
BOX = [[-0.5, 9.5], [-0.5, 19.5], [-0.5, 39.5]]
CENTRE = (4.5, 9.5, 19.5)


@pytest.fixture(autouse=True)
def _own_qt_controls(qtbot):
    """Give every Qt control a test builds to ``qtbot``, which closes it."""
    _QTBOT.append(qtbot)
    yield
    _QTBOT.clear()


@pytest.fixture
def controller(qtbot):
    controller = CellierController(gui="offscreen")
    controller.camera_reslice_enabled = False
    yield controller
    controller.close()


def _add_image(controller, *, render_mode="plane", canvas=False, name="image"):
    """An in-memory image in a 3D scene; returns ``(scene, visual)``."""
    scene = controller.add_scene(dim="3d", name=f"scene-{name}")
    data = np.random.default_rng(0).random(SHAPE, dtype=np.float32)
    visual = controller.add_image(
        data=ImageMemoryStore(data=data),
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode=render_mode),
        name=name,
    )
    if canvas:
        controller.add_canvas(scene_id=scene.id)
    return scene, visual


def _world(controller, scene):
    return controller.get_scene(scene.id).dims.world_coordinate_system


def _plane(controller, scene, point=CENTRE, normal=(1, 0, 0), **kwargs) -> RenderPlane:
    return RenderPlane.from_point_normal(
        _world(controller, scene), point, normal, axes=AXES, **kwargs
    )


def _make(toolkit, controller, visual, *, visual_ids=None, canvas_id=None, **extra):
    data = get_render_planes_data_from_visual(controller, visual.id, **extra)
    if canvas_id is not None:
        data.update(get_render_plane_gizmo_data(controller, visual.id, canvas_id))
    ids = visual.id if visual_ids is None else visual_ids
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtRenderPlanesControls

        widget = QtRenderPlanesControls(ids, **data)
        _QTBOT[-1].addWidget(widget.widget)
    else:
        from cellier.gui.anywidget.visuals import AnywidgetRenderPlanesControls

        widget = AnywidgetRenderPlanesControls(ids, **data)
    controller.connect_widget(widget, subscription_specs=widget.subscription_specs())
    return widget


def _act(widget, action, index=None, value=None) -> None:
    """Do one thing as a user would."""
    if hasattr(widget, "comm"):  # anywidget: what the front end sends
        widget.edit = {
            "action": action,
            "index": index,
            "value": value,
            "serial": next(_SERIALS),
        }
        return
    if action == "add":
        widget._add.click()
        return
    row = widget.row(index)
    if action == "remove":
        row.remove.click()
    elif action == "flip":
        row.flip.click()
    elif action == "gizmo":
        assert row.gizmo.isChecked() != value
        row.gizmo.click()
    elif action == "enabled":
        row.enabled.setChecked(value)
    elif action == "facing":
        row.facing[tuple(value)].click()
    elif action == "position":
        row.position.setValue(value)
    elif action == "component":
        row.components[value[0]].setValue(value[1])
    elif action == "side":
        row.sides[(value[0], value[1])].setValue(value[2])
    elif action == "bounded":
        row.unbounded[(value[0], value[1])].setChecked(not value[2])


def _n_rows(widget) -> int:
    return len(widget.rows) if hasattr(widget, "comm") else len(widget._rows)


def _blocked(widget) -> str:
    return widget.blocked


def _can_add(widget) -> bool:
    return widget.can_add if hasattr(widget, "comm") else widget._add.isEnabled()


def _gizmo_states(widget) -> list[tuple[bool, str]]:
    if hasattr(widget, "comm"):
        return [(row["gizmo"], row["gizmo_blocked"]) for row in widget.rows]
    return [
        (row.gizmo.isChecked(), "" if row.gizmo.isEnabled() else row.gizmo.toolTip())
        for row in widget._rows
    ]


def _frame(plane: RenderPlane) -> np.ndarray:
    axis_0, axis_1 = np.array(plane.in_plane_axis_0), np.array(plane.in_plane_axis_1)
    return np.array([np.cross(axis_0, axis_1), axis_0, axis_1])


# -- the toolkit-free half ------------------------------------------------------


def _free_plane(**kwargs) -> RenderPlane:
    return RenderPlane(
        coordinate_system=uuid4(),
        axes=AXES,
        origin=(3.0, 4.0, 5.0),
        in_plane_axis_0=(0.0, 1.0, 0.0),
        in_plane_axis_1=(0.0, 0.0, 1.0),
        **kwargs,
    )


def test_a_row_shows_the_normal_the_position_and_the_extents():
    plane = _free_plane(extent_1=(-2.0, 6.0), enabled=False)
    (row,) = rows_from_planes([plane], {})
    assert row == {
        "id": str(plane.id),
        "enabled": False,
        "axes": ["z", "y", "x"],
        "normal": [1.0, 0.0, 0.0],
        "position": 3.0,
        "extent_0": [None, None],
        "extent_1": [-2.0, 6.0],
    }


def test_a_position_edit_keeps_the_origins_in_plane_coordinates():
    plane = with_normal(_free_plane(), (1.0, 1.0, 0.0))
    moved = with_position(plane, 9.0)
    normal = _frame(plane)[0]
    assert np.asarray(moved.origin) @ normal == pytest.approx(9.0)
    for axis in (plane.in_plane_axis_0, plane.in_plane_axis_1):
        assert np.asarray(moved.origin) @ axis == pytest.approx(
            np.asarray(plane.origin) @ axis
        )
    assert moved.in_plane_axis_0 == plane.in_plane_axis_0
    assert moved.id == plane.id


@pytest.mark.parametrize(
    "normal", [(0.0, 1.0, 0.0), (1.0, 2.0, -0.5), (-1.0, 0.0, 0.0), (0.3, -0.2, 0.9)]
)
def test_a_normal_edit_is_the_smallest_rotation_and_keeps_the_spin(normal):
    """The frame turns about the axis perpendicular to both normals."""
    # A frame spun 25 degrees about its normal, to see the spin kept.
    spin = np.radians(25.0)
    plane = _free_plane().model_copy(
        update={
            "in_plane_axis_0": (0.0, float(np.cos(spin)), float(np.sin(spin))),
            "in_plane_axis_1": (0.0, float(-np.sin(spin)), float(np.cos(spin))),
        }
    )
    turned = with_normal(plane, normal)
    before, after = _frame(plane), _frame(turned)
    target = np.asarray(normal) / np.linalg.norm(normal)
    np.testing.assert_allclose(after[0], target, atol=1e-12)
    np.testing.assert_allclose(after @ after.T, np.eye(3), atol=1e-12)
    assert turned.origin == plane.origin  # about the origin
    rotation = after.T @ before  # takes the old frame to the new
    # The rotation's angle is the angle between the normals: no extra spin.
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    assert angle == pytest.approx(np.arccos(np.clip(before[0] @ target, -1, 1)))


def test_a_normal_edit_refuses_zero():
    with pytest.raises(ValueError, match="all zero"):
        with_normal(_free_plane(), (0.0, 0.0, 0.0))


def test_a_flip_reverses_the_normal_and_keeps_the_rectangle():
    plane = _free_plane(extent_0=(-1.0, 2.0), extent_1=(-3.0, None))
    turned = flipped(plane)
    np.testing.assert_allclose(_frame(turned)[0], -_frame(plane)[0])
    assert turned.origin == plane.origin
    assert turned.extent_0 == plane.extent_0
    # Axis 1 is reversed and its extent mirrored: the same points.
    assert turned.extent_1 == (None, 3.0)
    assert flipped(turned) == plane


def test_the_box_cut_of_a_plane():
    # x = 5 plane of a box 0..10, 0..20, 0..40, origin off centre.
    plane = _free_plane().model_copy(update={"origin": (5.0, 4.0, 30.0)})
    box = [[0.0, 10.0], [0.0, 20.0], [0.0, 40.0]]
    assert box_cut(plane, box) == ((-4.0, 16.0), (-30.0, 10.0))
    # A plane that misses the box has no cut.
    assert box_cut(with_position(plane, 50.0), box) is None
    # An oblique plane: the polygon's span, checked against its corners.
    oblique = with_normal(plane, (1.0, 1.0, 0.0))
    (low_0, high_0), (low_1, high_1) = box_cut(oblique, box)
    assert low_0 < 0 < high_0 and low_1 < 0 < high_1
    for along_0 in (low_0, high_0):
        point = np.array(oblique.origin) + along_0 * np.array(oblique.in_plane_axis_0)
        # A corner of the cut is on the box's surface, not outside it.
        assert np.all(point >= np.array(box)[:, 0] - 40) and np.isfinite(point).all()


def test_bounding_a_side_fills_in_the_box_cut():
    box = [[0.0, 10.0], [0.0, 20.0], [0.0, 40.0]]
    plane = _free_plane().model_copy(update={"origin": (5.0, 4.0, 30.0)})
    bounded = with_side_bounded(plane, 0, 1, True, box)
    assert bounded.extent_0 == (None, 16.0)
    bounded = with_side_bounded(bounded, 1, 0, True, box)
    assert bounded.extent_1 == (-30.0, None)
    assert with_side_bounded(bounded, 0, 1, False, box).extent_0 == (None, None)
    # The cut would not leave min < max against the other side: just past it.
    tight = plane.model_copy(update={"extent_0": (None, -10.0)})
    low, high = with_side_bounded(tight, 0, 0, True, box).extent_0
    assert low < high == -10.0
    # A plane off the box still gets a finite, ordered pair.
    off = with_position(plane, 99.0)
    low, high = with_side_bounded(
        with_side_bounded(off, 0, 0, True, box), 0, 1, True, box
    ).extent_0
    assert low < high


def test_the_control_data_is_read_through_the_controller(controller):
    scene, visual = _add_image(controller)
    data = get_render_planes_data_from_visual(controller, visual.id)
    world = _world(controller, scene)
    assert data["coordinate_system"] == str(world.id)
    assert data["axis_names"] == ["z", "y", "x"]
    assert data["axis_ids"] == [str(axis.id) for axis in world.axes]
    assert data["planes"] == ()
    assert data["scene_id"] == str(scene.id)
    state = data["state_source"]()
    assert state == {"blocked": "", "displayed_axes": data["axis_ids"], "bounds": BOX}


# -- the control ----------------------------------------------------------------


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_it_adds_moves_toggles_and_removes_planes(controller, toolkit):
    _scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)

    _act(widget, "add")
    # Through the middle of the data, facing the first displayed axis,
    # unbounded: what controller.default_render_plane builds.
    (added,) = visual.render_planes
    expected = controller.default_render_plane(visual.id)
    assert added.model_copy(update={"id": expected.id}) == expected
    assert added.origin == CENTRE
    assert added.extent_0 == added.extent_1 == (None, None)
    plane_id = added.id

    _act(widget, "position", 0, 2.0)
    assert visual.render_planes[0].origin == (2.0, 9.5, 19.5)
    assert visual.render_planes[0].id == plane_id  # an edit keeps the id
    _act(widget, "enabled", 0, False)
    assert visual.render_planes[0].enabled is False
    assert _n_rows(widget) == 1  # a disabled plane stays in the list

    _act(widget, "add")
    assert len(visual.render_planes) == 2
    _act(widget, "facing", 1, [1, 1])
    np.testing.assert_allclose(
        _frame(visual.render_planes[1])[0], [0, 1, 0], atol=1e-12
    )
    _act(widget, "facing", 1, [2, -1])
    np.testing.assert_allclose(
        _frame(visual.render_planes[1])[0], [0, 0, -1], atol=1e-12
    )
    assert visual.render_planes[1].origin == CENTRE  # turned in place

    second_id = visual.render_planes[1].id
    _act(widget, "remove", 0)
    assert [plane.id for plane in visual.render_planes] == [second_id]
    assert _n_rows(widget) == 1
    assert widget.error == ""
    widget.close()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_position_and_normal_edits_keep_the_frames_spin(controller, toolkit):
    scene, visual = _add_image(controller)
    # A frame no constructor default gives: spun about the normal.
    plane = _plane(controller, scene, up=(0.0, 1.0, 0.4))
    visual.render_planes = (plane,)
    widget = _make(toolkit, controller, visual)

    _act(widget, "position", 0, 7.0)
    moved = visual.render_planes[0]
    assert moved.in_plane_axis_0 == plane.in_plane_axis_0
    assert moved.in_plane_axis_1 == plane.in_plane_axis_1
    assert moved.origin == (7.0, *plane.origin[1:])

    _act(widget, "component", 0, [1, 1.0])  # the normal (1, 0, 0) -> (1, 1, 0)
    turned = visual.render_planes[0]
    expected = with_normal(moved, (1.0, 1.0, 0.0))
    np.testing.assert_allclose(_frame(turned), _frame(expected), atol=1e-9)
    assert turned.origin == moved.origin
    assert widget.error == ""


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_flip_changes_the_model_and_nothing_drawn(controller, toolkit):
    scene, visual = _add_image(controller)
    plane = _plane(controller, scene, extent_1=(-3.0, 8.0))
    visual.render_planes = (plane,)
    widget = _make(toolkit, controller, visual)
    _act(widget, "flip", 0)
    assert visual.render_planes[0] == flipped(plane)
    assert widget.editor.rows[0]["normal"] == [-1.0, 0.0, 0.0]
    assert widget.editor.rows[0]["position"] == -CENTRE[0]


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_extents_and_their_unbounded_boxes(controller, toolkit):
    scene, visual = _add_image(controller)
    plane = _plane(controller, scene)  # x = 4.5; axis 0 is y, axis 1 is x
    visual.render_planes = (plane,)
    widget = _make(toolkit, controller, visual)

    # Unticking "unbounded" fills in where the data box cuts the plane.
    _act(widget, "bounded", 0, [0, 0, True])
    assert visual.render_planes[0].extent_0 == (-10.0, None)
    _act(widget, "bounded", 0, [0, 1, True])
    assert visual.render_planes[0].extent_0 == (-10.0, 10.0)
    _act(widget, "bounded", 0, [1, 1, True])
    assert visual.render_planes[0].extent_1 == (None, 20.0)

    _act(widget, "side", 0, [0, 1, 4.0])
    assert visual.render_planes[0].extent_0 == (-10.0, 4.0)
    assert widget.editor.rows[0]["extent_0"] == [-10.0, 4.0]

    # min >= max is refused like any bad edit: the last rows, and the error.
    _act(widget, "side", 0, [0, 0, 6.0])
    assert visual.render_planes[0].extent_0 == (-10.0, 4.0)
    assert "min < max" in widget.error
    assert widget.editor.rows[0]["extent_0"] == [-10.0, 4.0]
    if toolkit == "qt":
        assert widget.row(0).sides[(0, 0)].value() == -10.0

    # Ticking the box clears the side.
    _act(widget, "bounded", 0, [0, 0, False])
    assert visual.render_planes[0].extent_0 == (None, 4.0)
    assert widget.error == ""
    if toolkit == "qt":
        row = widget.row(0)
        assert row.unbounded[(0, 0)].isChecked()
        assert not row.sides[(0, 0)].isEnabled()
        assert row.sides[(0, 1)].isEnabled()


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_add_is_disabled_at_four_planes(controller, toolkit):
    _scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)
    for _ in range(3):
        _act(widget, "add")
        assert _can_add(widget)
    _act(widget, "add")
    assert len(visual.render_planes) == 4
    assert not _can_add(widget)
    if toolkit == "anywidget":
        _act(widget, "add")  # what a stale front end might still send
        assert len(visual.render_planes) == 4
        assert "at most 4" in widget.error
    _act(widget, "remove", 0)
    assert _can_add(widget)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_change_from_elsewhere_is_shown_and_sends_nothing(controller, toolkit):
    scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)
    sent: list = []
    widget.changed.connect(sent.append)
    visual.render_planes = (
        _plane(controller, scene, point=(7.0, 0.0, 0.0)),
        _plane(
            controller, scene, normal=(0, 0, 1), enabled=False, extent_0=(-1.0, 1.0)
        ),
    )
    assert _n_rows(widget) == 2
    assert widget.editor.rows[0]["position"] == 7.0
    assert widget.editor.rows[1]["enabled"] is False
    assert widget.editor.rows[1]["extent_0"] == [-1.0, 1.0]
    assert sent == []


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_refused_edit_shows_the_last_rows_and_the_error(controller, toolkit):
    _scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)
    _act(widget, "add")
    before = widget.editor.rows
    # Zeroing the only non-zero entry leaves no direction.
    _act(widget, "component", 0, [0, 0.0])
    assert "all zero" in widget.error
    assert widget.editor.rows == before
    assert len(visual.render_planes) == 1
    if toolkit == "qt":
        assert widget.row(0).components[0].value() == 1.0
    _act(widget, "position", 0, 3.0)  # the next good edit clears it
    assert widget.error == ""


def test_the_qt_row_names_the_planes_own_axes(controller):
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes(*AXES)], dim="3d", name="tzyx"
    )
    data = np.zeros((2, *SHAPE), dtype=np.float32)
    visual = controller.add_image(
        data=ImageMemoryStore(data=data),
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode="plane"),
    )
    widget = _make("qt", controller, visual)
    _act(widget, "add")
    # A new plane is on the three axes the view displays: z, y, x, not t.
    world = _world(controller, scene)
    assert visual.render_planes[0].axes == tuple(axis.id for axis in world.axes[1:])
    assert [button.text() for button in widget.row(0).facing.values()] == [
        "+z",
        "-z",
        "+y",
        "-y",
        "+x",
        "-x",
    ]
    assert widget.row(0).facing[(0, 1)].isChecked()


def test_a_moved_plane_keeps_its_row_widgets(controller):
    _scene, visual = _add_image(controller)
    widget = _make("qt", controller, visual)
    _act(widget, "add")
    row = widget.row(0)
    slider = row.slider
    _act(widget, "position", 0, 3.0)
    visual.render_planes = (
        visual.render_planes[0].model_copy(update={"origin": (6.0, 0.0, 0.0)}),
    )
    assert widget.row(0) is row
    assert widget.row(0).slider is slider
    assert row.position.value() == 6.0


def test_the_anywidget_sends_one_update_for_an_edit_delivered_twice(controller):
    _scene, visual = _add_image(controller)
    widget = _make("anywidget", controller, visual)
    _act(widget, "add")
    sent: list = []
    widget.changed.connect(sent.append)
    edit = {"action": "position", "index": 0, "value": 3.0, "serial": 99}
    widget.edit = edit
    widget.edit = dict(edit, serial=100)  # the same edit, delivered again
    assert len(sent) == 1


def test_the_anywidget_puts_back_what_a_host_overwrites(controller):
    """marimo sends the front end's whole state back with every edit."""
    _scene, visual = _add_image(controller)
    widget = _make("anywidget", controller, visual)
    _act(widget, "add")
    stale = {"rows": [], "error": "old", "blocked": "old", "can_add": False}
    shown = {name: getattr(widget, name) for name in stale}
    for name, value in stale.items():
        setattr(widget, name, value)
        assert getattr(widget, name) == shown[name]
    assert len(widget.rows) == 1


def test_the_anywidget_ignores_a_malformed_edit(controller):
    _scene, visual = _add_image(controller)
    widget = _make("anywidget", controller, visual)
    _act(widget, "add")
    before = visual.render_planes
    for action, index, value in [
        ("position", 5, 1.0),
        ("position", "0", 1.0),
        ("facing", 0, [3, 1]),
        ("component", 0, 1.0),
        ("side", 0, [0, 2, 1.0]),
        ("bounded", 0, [2, 0, True]),
        ("side", 0, [0, 1]),
        ("nonsense", 0, None),
    ]:
        _act(widget, action, index, value)
    assert visual.render_planes == before


def test_a_group_control_edits_every_visual(controller):
    scene, visual = _add_image(controller)
    data = np.zeros(SHAPE, dtype=np.float32)
    other = controller.add_image(
        data=ImageMemoryStore(data=data),
        scene_id=scene.id,
        single=InMemoryImageSingleAppearance(render_mode="plane"),
    )
    widget = _make("qt", controller, visual, visual_ids=[visual.id, other.id])
    _act(widget, "add")
    assert visual.render_planes == other.render_planes
    assert len(other.render_planes) == 1


def test_both_toolkits_carry_the_shared_title():
    from cellier.gui.anywidget.visuals import AnywidgetRenderPlanesControls
    from cellier.gui.qt.visuals import QtRenderPlanesControls

    assert RENDER_PLANES_TITLE == "Render planes"
    assert QtRenderPlanesControls.DEFAULT_TITLE == RENDER_PLANES_TITLE
    assert AnywidgetRenderPlanesControls.DEFAULT_TITLE == RENDER_PLANES_TITLE
    assert AnywidgetRenderPlanesControls.title.default_value == RENDER_PLANES_TITLE


def test_the_explicit_api_is_exported_from_cellier_gui():
    import cellier.gui as gui

    assert gui.RenderPlaneGizmoTarget is RenderPlaneGizmoTarget
    assert gui.get_render_planes_data_from_visual is get_render_planes_data_from_visual
    assert gui.get_render_plane_gizmo_data is get_render_plane_gizmo_data


# -- disabled, with the reason --------------------------------------------------


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_control_is_disabled_outside_plane_mode(controller, toolkit):
    scene, visual = _add_image(controller, render_mode="mip")
    visual.render_planes = (_plane(controller, scene),)
    widget = _make(toolkit, controller, visual)
    assert "render mode" in _blocked(widget)
    assert not _can_add(widget)
    assert _n_rows(widget) == 1  # the rows are shown, not editable
    if toolkit == "qt":
        assert not widget._body.isEnabled()
        assert widget._reason.text() == _blocked(widget)
    else:
        # A stale front end's edit is refused with the reason.
        _act(widget, "position", 0, 1.0)
        assert widget.error == _blocked(widget)
        assert visual.render_planes[0].origin == CENTRE

    # Entering plane mode enables it, with no other prompt.
    visual.single.render_mode = "plane"
    assert _blocked(widget) == ""
    assert _can_add(widget)
    if toolkit == "qt":
        assert widget._body.isEnabled()
        assert widget._reason.isHidden()
    _act(widget, "position", 0, 1.0)
    assert visual.render_planes[0].origin[0] == 1.0
    assert widget.error == ""


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_control_is_disabled_in_a_2d_view(controller, toolkit):
    scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)
    _act(widget, "add")
    controller.set_displayed_axes(scene.id, (1, 2))
    assert "2D" in _blocked(widget)
    assert not _can_add(widget)
    controller.set_displayed_axes(scene.id, (0, 1, 2))
    assert _blocked(widget) == ""
    assert _can_add(widget)


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_a_caller_can_give_its_own_reason(controller, toolkit):
    """What ``OrthoViewer``'s slice planes mode uses for read-only rows."""
    scene, visual = _add_image(controller, canvas=True)
    visual.render_planes = (_plane(controller, scene),)
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    reason = ["The planes follow the sliders."]
    widget = _make(
        toolkit, controller, visual, canvas_id=canvas_id, blocked=lambda: reason[0]
    )
    assert _blocked(widget) == reason[0]
    # The gizmo toggle is disabled with the same reason.
    assert _gizmo_states(widget) == [(False, reason[0])]
    reason[0] = ""
    widget.refresh()
    assert _blocked(widget) == ""
    assert _gizmo_states(widget) == [(False, "")]


# -- the gizmo toggle: a radio across both kinds of plane -----------------------


def _gizmo_scene(controller):
    scene, visual = _add_image(controller, canvas=True)
    visual.render_planes = (
        _plane(controller, scene),
        _plane(controller, scene, normal=(0, 1, 0)),
    )
    (canvas_id,) = controller.get_canvas_ids(scene.id)
    return scene, visual, canvas_id


async def test_the_gizmo_data_is_for_the_visual_and_canvas_named(controller):
    scene, visual, canvas_id = _gizmo_scene(controller)
    first, second = visual.render_planes
    data = get_render_plane_gizmo_data(controller, visual.id, canvas_id)
    assert data["gizmo_target"] == RenderPlaneGizmoTarget(
        visual_id=visual.id, canvas_id=canvas_id, scene_id=scene.id
    )
    assert data["gizmo_plane"] is None
    assert data["gizmo_blocked"](str(first.id)) == ""
    controller.add_render_plane_gizmo(visual.id, canvas_id, second.id)
    data = get_render_plane_gizmo_data(controller, visual.id, canvas_id)
    assert data["gizmo_plane"] == str(second.id)
    with pytest.raises(KeyError):
        get_render_plane_gizmo_data(controller, visual.id, uuid4())


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_a_control_given_no_gizmo_target_draws_no_toggle(controller, toolkit):
    _scene, visual = _add_image(controller)
    widget = _make(toolkit, controller, visual)
    _act(widget, "add")
    assert not widget.editor.has_gizmo
    if toolkit == "qt":
        assert widget.row(0).gizmo.isHidden()
    else:
        assert "gizmo" not in widget.rows[0]
        _act(widget, "gizmo", 0, True)  # ignored
    assert controller._plane_gizmos == {}


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_the_toggle_puts_the_gizmo_on_one_plane_at_a_time(controller, toolkit):
    _scene, visual, canvas_id = _gizmo_scene(controller)
    widget = _make(toolkit, controller, visual, canvas_id=canvas_id)
    first, second = visual.render_planes
    assert _gizmo_states(widget) == [(False, ""), (False, "")]

    _act(widget, "gizmo", 0, True)
    session = controller.get_plane_gizmo(canvas_id)
    assert (session.kind, session.plane_id) == ("render", first.id)
    assert _gizmo_states(widget) == [(True, ""), (False, "")]
    _act(widget, "gizmo", 1, True)
    assert controller.get_plane_gizmo(canvas_id).plane_id == second.id
    assert _gizmo_states(widget) == [(False, ""), (True, "")]
    _act(widget, "gizmo", 1, False)
    assert controller.get_plane_gizmo(canvas_id) is None
    assert _gizmo_states(widget) == [(False, ""), (False, "")]
    assert widget.error == ""


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_the_radio_runs_across_clipping_and_render_plane_controls(
    controller, toolkit
):
    """One gizmo per canvas: the two controls' toggles are one set (H4)."""
    from cellier.gui._clipping_planes import (
        get_clipping_plane_gizmo_data,
        get_clipping_planes_data_from_visual,
    )

    _scene, visual, canvas_id = _gizmo_scene(controller)
    store = controller.get_data_store(UUID(str(visual.data_store_id)))
    visual.clipping_planes = (
        ClippingPlane.from_point_normal(
            store.data_coordinate_system, (0, 0, 20), (0, 0, 1), enabled=False
        ),
    )
    render = _make(toolkit, controller, visual, canvas_id=canvas_id)
    data = {
        **get_clipping_planes_data_from_visual(visual, store),
        **get_clipping_plane_gizmo_data(controller, visual.id, canvas_id),
    }
    if toolkit == "qt":
        from cellier.gui.qt.visuals import QtClippingPlanesControls

        clipping = QtClippingPlanesControls(visual.id, **data)
        _QTBOT[-1].addWidget(clipping.widget)
    else:
        from cellier.gui.anywidget.visuals import AnywidgetClippingPlanesControls

        clipping = AnywidgetClippingPlanesControls(visual.id, **data)
    controller.connect_widget(
        clipping, subscription_specs=clipping.subscription_specs()
    )

    def press(widget, on):
        if toolkit == "qt":
            widget.row(0).gizmo.click()
        else:
            widget.edit = {
                "action": "gizmo",
                "index": 0,
                "value": on,
                "serial": next(_SERIALS),
            }

    press(render, True)
    assert _gizmo_states(render)[0] == (True, "")
    assert _gizmo_states(clipping) == [(False, "")]
    # The clipping row takes the canvas's gizmo: the render row lets go.
    press(clipping, True)
    assert controller.get_plane_gizmo(canvas_id).kind == "clipping"
    assert _gizmo_states(clipping) == [(True, "")]
    assert _gizmo_states(render) == [(False, ""), (False, "")]
    press(render, True)
    assert controller.get_plane_gizmo(canvas_id).kind == "render"
    assert _gizmo_states(clipping) == [(False, "")]
    assert _gizmo_states(render)[0] == (True, "")


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_the_toggle_follows_a_gizmo_opened_and_closed_elsewhere(
    controller, toolkit
):
    _scene, visual, canvas_id = _gizmo_scene(controller)
    widget = _make(toolkit, controller, visual, canvas_id=canvas_id)
    _first, second = visual.render_planes
    session = controller.add_render_plane_gizmo(visual.id, canvas_id, second.id)
    assert _gizmo_states(widget) == [(False, ""), (True, "")]
    late = _make(toolkit, controller, visual, canvas_id=canvas_id)
    assert _gizmo_states(late) == [(False, ""), (True, "")]
    _act(widget, "remove", 1)
    assert session.closed
    assert _gizmo_states(widget) == [(False, "")]
    assert _gizmo_states(late) == [(False, "")]


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
async def test_the_extent_fields_follow_a_gizmo_scale_drag(controller, toolkit):
    """What a scale handle writes during a drag reaches the fields (14.9)."""
    _scene, visual, canvas_id = _gizmo_scene(controller)
    plane = visual.render_planes[0].model_copy(update={"extent_0": (-4.0, 6.0)})
    controller.set_render_plane(visual.id, plane.id, plane)
    widget = _make(toolkit, controller, visual, canvas_id=canvas_id)
    session = controller.add_render_plane_gizmo(visual.id, canvas_id, plane.id)
    # The session's own write, as its scale handler makes it mid drag.
    with controller.plane_interaction(visual.id):
        session._assign({"extent_0": (-6.0, 9.0)})
        assert widget.editor.rows[0]["extent_0"] == [-6.0, 9.0]
        if toolkit == "qt":
            assert widget.row(0).sides[(0, 1)].value() == 9.0


# -- 9.1: "Plane" in the render-mode combos -------------------------------------


def test_the_image_rows_that_mean_nothing_in_plane_mode_are_hidden():
    from cellier.gui._image_controls import row_visible

    for field in ("iso_threshold", "attenuation"):
        assert not row_visible(field, ["plane"], 3)
    assert row_visible("render_mode", ["plane"], 3)
    assert not row_visible("render_mode", ["plane"], 2)  # 3D only, as today
    assert row_visible("iso_threshold", ["iso"], 3)


def _requests(controller, widget_id):
    """Send what a control sends for a render-mode choice."""

    def single(visual, mode):
        controller.incoming_events.emit(
            SingleAppearanceUpdateEvent(
                source_id=widget_id,
                visual_id=visual.id,
                field="render_mode",
                value=mode,
            )
        )

    def channel(visual, index, mode):
        controller.incoming_events.emit(
            ChannelAppearanceUpdateEvent(
                source_id=widget_id,
                visual_id=visual.id,
                channel_index=index,
                field="render_mode",
                value=mode,
            )
        )

    def labels(visual, mode):
        controller.incoming_events.emit(
            AppearanceUpdateEvent(
                source_id=widget_id,
                visual_id=visual.id,
                field="render_mode",
                value=mode,
            )
        )

    return single, channel, labels


def test_choosing_plane_on_a_visual_with_no_planes_adds_a_centred_one(controller):
    _scene, visual = _add_image(controller, render_mode="mip")
    single, _channel, _labels = _requests(controller, uuid4())
    single(visual, "plane")
    assert visual.single.render_mode == "plane"
    (plane,) = visual.render_planes
    assert plane.origin == CENTRE
    np.testing.assert_allclose(_frame(plane)[0], [1.0, 0.0, 0.0])
    assert plane.extent_0 == plane.extent_1 == (None, None)

    # Choosing it again, or with planes already there, adds nothing.
    single(visual, "mip")
    single(visual, "plane")
    assert visual.render_planes == (plane,)


def test_the_model_does_not_add_a_plane_by_itself(controller):
    """Only a control's request does: the mode set from code adds nothing."""
    _scene, visual = _add_image(controller, render_mode="mip")
    visual.single.render_mode = "plane"
    assert visual.render_planes == ()
    visual.single.render_mode = "mip"
    controller.update_single_appearance_field(visual.id, "render_mode", "plane")
    assert visual.render_planes == ()


def test_choosing_plane_in_a_2d_view_adds_nothing(controller):
    scene, visual = _add_image(controller, render_mode="mip")
    controller.set_displayed_axes(scene.id, (1, 2))
    single, _channel, _labels = _requests(controller, uuid4())
    single(visual, "plane")
    assert visual.single.render_mode == "plane"
    assert visual.render_planes == ()


def test_a_labels_combo_adds_a_plane_too(controller):
    scene = controller.add_scene(dim="3d", name="labels")
    data = np.zeros(SHAPE, dtype=np.int32)
    visual = controller.add_labels(
        data=LabelMemoryStore(data=data),
        scene_id=scene.id,
        appearance=InMemoryLabelsAppearance(),
    )
    _single, _channel, labels = _requests(controller, uuid4())
    labels(visual, "plane")
    assert visual.appearance.render_mode == "plane"
    assert len(visual.render_planes) == 1
    assert visual.render_planes[0].origin == CENTRE


def test_one_channels_combo_moves_every_channel_and_adds_one_plane(controller):
    """Composite: the combos move together (D-P53), and one plane is added."""
    scene = controller.add_scene(
        coordinate_system=[("c", "channel"), *spatial_axes(*AXES)],
        dim="3d",
        name="channels",
    )
    data = np.random.default_rng(0).random((3, *SHAPE), dtype=np.float32)
    visual = controller.add_image(
        data=ImageMemoryStore(data=data),
        scene_id=scene.id,
        channel_axis=0,
        composite=True,
        channels={
            index: InMemoryImageChannelAppearance(render_mode="mip")
            for index in range(3)
        },
    )
    _single, channel, _labels = _requests(controller, uuid4())
    channel(visual, 1, "plane")
    assert [visual.channels[k].render_mode for k in range(3)] == ["plane"] * 3
    assert len(visual.render_planes) == 1
    assert visual.render_planes[0].origin == CENTRE
    channel(visual, 0, "iso")
    assert [visual.channels[k].render_mode for k in range(3)] == ["iso"] * 3
    assert len(visual.render_planes) == 1  # leaving the mode keeps the planes


@pytest.mark.parametrize("toolkit", ["qt", "anywidget"])
def test_the_planes_control_shows_the_plane_a_combo_added(controller, toolkit):
    _scene, visual = _add_image(controller, render_mode="mip")
    widget = _make(toolkit, controller, visual)
    assert _blocked(widget) != ""
    single, _channel, _labels = _requests(controller, uuid4())
    single(visual, "plane")
    assert _blocked(widget) == ""
    assert _n_rows(widget) == 1
    assert widget.editor.rows[0]["position"] == CENTRE[0]
