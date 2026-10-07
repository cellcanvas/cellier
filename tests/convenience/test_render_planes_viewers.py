"""Render planes through the convenience viewers (plane rendering design 10).

``Viewer`` arguments, the gizmo method and the dock's control (10.1);
``OrthoViewer`` pass-through and ``OrthoPlaneController``'s ``"slices"``
mode (10.2).
"""

from __future__ import annotations

import warnings
from uuid import UUID, uuid4

import numpy as np
import pytest

from cellier.convenience import (
    InMemoryImageControlsConfig,
    LabelsControlsConfig,
    OrthoViewer,
    Viewer,
)
from cellier.convenience._ortho_planes import SLICES_REASON
from cellier.data import ImageMemoryStore, LabelMemoryStore, PointsMemoryStore
from cellier.events import PlaneInteractionEvent, RenderPlanesChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    ClippingPlane,
    InMemoryImageSingleAppearance,
    InMemoryLabelsAppearance,
    MultiscaleImageAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleImageSingleAppearance,
    RenderPlane,
)
from tests import _plane_fixtures as fx
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import drain_loading

AXES = ("z", "y", "x")
SHAPE = (8, 12, 16)
SETTLE_S = 0.03


def _image() -> ImageMemoryStore:
    data = np.random.default_rng(0).random(SHAPE, dtype=np.float32)
    return ImageMemoryStore(data=data)


def _plane_single() -> InMemoryImageSingleAppearance:
    return InMemoryImageSingleAppearance(render_mode="plane")


def _world(viewer):
    scene = viewer.scenes["vol"] if hasattr(viewer, "scenes") else viewer.scene
    return scene.dims.world_coordinate_system


def _plane(viewer, point=(4.0, 6.0, 8.0), normal=(1, 0, 0), **kwargs) -> RenderPlane:
    return RenderPlane.from_point_normal(
        _world(viewer), point, normal, axes=AXES, **kwargs
    )


@pytest.fixture
def make_viewer(qtbot):
    made = []

    def _make(cls=Viewer, **kwargs):
        if cls is Viewer:
            kwargs.setdefault("dim", "3d")
        viewer = cls(spatial_axes(*AXES), gui="offscreen", **kwargs)
        viewer.controller.camera_reslice_enabled = False
        made.append(viewer)
        return viewer

    yield _make
    for viewer in made:
        viewer.controller.close()


def _normals(planes) -> list[list[float]]:
    return [
        np.round(np.cross(p.in_plane_axis_0, p.in_plane_axis_1), 9).tolist()
        for p in planes
    ]


def _positions(planes) -> list[float]:
    """Where each of three slice planes sits along its own axis."""
    return [float(plane.origin[k]) for k, plane in enumerate(planes)]


# -- Viewer (10.1) --------------------------------------------------------------


def test_the_viewer_passes_render_planes_and_the_mode(make_viewer):
    viewer = make_viewer()
    plane = _plane(viewer)
    image = viewer.add_image(_image(), single=_plane_single(), render_planes=(plane,))
    assert image.render_planes == (plane,)
    assert image.plane_mode()
    labels = viewer.add_labels(
        LabelMemoryStore(data=np.ones(SHAPE, np.int32)),
        appearance=InMemoryLabelsAppearance(render_mode="plane"),
        render_planes=(plane,),
    )
    assert labels.render_planes == (plane,)
    assert labels.plane_mode()
    # A bad plane is refused as by the controller: another world system.
    other = make_viewer()
    with pytest.raises(ValueError, match="world coordinate system"):
        viewer.add_image(_image(), render_planes=(_plane(other),))


def test_the_viewer_passes_render_planes_to_the_multiscale_visuals(
    make_viewer, tmp_path
):
    viewer = make_viewer()
    plane = _plane(viewer)
    fx.write_pyramid(tmp_path / "image", fx.ISO)
    store, _ = fx.open_pyramid(tmp_path / "image", fx.ISO)
    image = viewer.add_image_multiscale(
        store,
        single=MultiscaleImageSingleAppearance(render_mode="plane"),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
        render_planes=(plane,),
    )
    assert image.render_planes == (plane,)
    assert image.plane_mode()


async def test_the_viewer_puts_a_gizmo_on_a_render_plane(make_viewer):
    viewer = make_viewer()
    plane, other = _plane(viewer), _plane(viewer, normal=(0, 1, 0))
    visual = viewer.add_image(
        _image(), single=_plane_single(), render_planes=(plane, other)
    )
    viewer.add_canvas()
    (canvas_id,) = viewer.canvases
    session = viewer.add_render_plane_gizmo(visual, plane)
    assert (session.kind, session.canvas_id, session.plane_id) == (
        "render",
        canvas_id,
        plane.id,
    )
    assert viewer.controller.get_plane_gizmo(canvas_id) is session
    # By id, on the other plane: the canvas's one gizmo moves there.
    second = viewer.add_render_plane_gizmo(visual.id, other.id)
    assert session.closed
    # The canvas's one gizmo serves both kinds: a clipping gizmo takes it.
    store = viewer.controller.get_data_store(UUID(str(visual.data_store_id)))
    item = ClippingPlane.from_point_normal(
        store.data_coordinate_system, (4, 6, 8), (0, 0, 1), enabled=False
    )
    visual.clipping_planes = (item,)
    clipping = viewer.add_clipping_plane_gizmo(visual, item)
    assert second.closed
    viewer.remove_plane_gizmo()
    assert clipping.closed
    assert viewer.controller.get_plane_gizmo(canvas_id) is None


def _dock_controls(viewer, backend="anywidget") -> list:
    """Every control of every dock target, as the layout walk builds them."""
    from cellier.convenience._backend import backend_for
    from cellier.convenience.layout._shared import appearance_targets
    from cellier.convenience.layout._walk import build_appearance_widgets

    built = []
    for target in appearance_targets(viewer):
        built += build_appearance_widgets(
            target.visual,
            target.config,
            viewer.controller,
            target.visual_ids,
            backend=backend_for(backend),
            clipping_gizmo_target=getattr(viewer, "_clipping_gizmo_target", None),
            render_planes_target=getattr(viewer, "_render_planes_target", None),
        )
    return built


def _plane_controls(viewer, backend="anywidget") -> list:
    return [
        widget
        for widget in _dock_controls(viewer, backend)
        if type(widget).__name__.endswith("RenderPlanesControls")
    ]


@pytest.mark.parametrize("backend", ["anywidget", "qt"])
async def test_the_dock_builds_the_render_planes_control(make_viewer, backend, qtbot):
    viewer = make_viewer()
    visual = viewer.add_image(
        _image(),
        single=_plane_single(),
        controls=InMemoryImageControlsConfig(
            appearance=True, render_plane_controls=True
        ),
    )
    viewer.add_canvas()
    (control,) = _plane_controls(viewer, backend)
    if backend == "qt":
        qtbot.addWidget(control.widget)
    assert control.visual_ids == (visual.id,)
    assert control.editor.has_gizmo  # the viewer's one canvas
    assert control.editor.blocked == ""
    # It is wired: an edit reaches the model, and the model's change the rows.
    control.editor.add()
    assert len(visual.render_planes) == 1
    viewer.controller.set_render_planes(visual.id, ())
    assert control.editor.rows == []


def test_the_control_is_opt_in_and_only_for_visuals_with_planes(make_viewer):
    viewer = make_viewer()
    viewer.add_image(_image(), controls=InMemoryImageControlsConfig(appearance=True))
    viewer.add_labels(
        LabelMemoryStore(data=np.ones(SHAPE, np.int32)),
        name="with",
        controls=LabelsControlsConfig(appearance=True, render_plane_controls=True),
    )
    viewer.add_points(PointsMemoryStore(positions=np.zeros((3, 3), np.float32)))
    controls = _plane_controls(viewer)
    assert len(controls) == 1
    assert not controls[0].editor.has_gizmo  # no canvas: no toggle
    # Not in plane mode: built, disabled, with the reason.
    assert "render mode" in controls[0].blocked


def test_the_flag_needs_the_appearance_dock(make_viewer):
    from cellier.convenience.layout._shared import missing_dock_node

    config = InMemoryImageControlsConfig(render_plane_controls=True)
    problem = missing_dock_node(
        config, frozenset({"AppearanceControls"}), clipping_node="AppearanceControls"
    )
    assert "render_plane_controls=True and no appearance controls" in problem
    config = InMemoryImageControlsConfig(appearance=True, render_plane_controls=True)
    for node in ("AppearanceControls", "OrthoClippingControls"):
        assert (
            missing_dock_node(
                config, frozenset({"AppearanceControls"}), clipping_node=node
            )
            is None
        )


# -- OrthoViewer: passing through (10.2) ----------------------------------------


def test_render_planes_go_to_the_3d_panel_only(make_viewer):
    ortho = make_viewer(OrthoViewer)
    plane = _plane(ortho)
    visuals = ortho.add_image(_image(), single=_plane_single(), render_planes=(plane,))
    assert visuals["vol"].render_planes == (plane,)
    assert visuals["vol"].plane_mode()
    for key in ("xy", "xz", "yz"):
        assert visuals[key].render_planes == ()
    # The mode is on every panel's model and acts in the 3D panel alone: a
    # 2D view draws its slice whatever the mode (D-P19).
    controller = ortho.controller
    assert controller._draws_planes(visuals["vol"])
    assert not any(controller._draws_planes(visuals[key]) for key in ("xy", "xz", "yz"))
    labels = ortho.add_labels(
        LabelMemoryStore(data=np.ones(SHAPE, np.int32)),
        appearance=InMemoryLabelsAppearance(render_mode="plane"),
        render_planes=(plane,),
    )
    assert labels["vol"].render_planes == (plane,)
    assert labels["xy"].render_planes == ()


async def test_the_ortho_gizmo_is_on_the_3d_panels_visual(make_viewer):
    ortho = make_viewer(OrthoViewer)
    plane = _plane(ortho)
    visuals = ortho.add_image(_image(), single=_plane_single(), render_planes=(plane,))
    controller = ortho.controller
    with pytest.raises(ValueError, match="no canvas yet"):
        ortho.add_render_plane_gizmo(visuals, plane)
    for scene in ortho.scenes.values():
        controller.add_canvas(scene.id)
    vol_canvas = controller.get_canvas_ids(ortho.scenes["vol"].id)[0]
    # Any panel's visual names the group; the session is on the 3D one.
    session = ortho.add_render_plane_gizmo(visuals["xy"], plane)
    assert (session.visual_id, session.canvas_id) == (visuals["vol"].id, vol_canvas)
    ortho.remove_plane_gizmo()
    assert session.closed


async def test_the_ortho_dock_shows_the_control_for_the_3d_view_only(make_viewer):
    ortho = make_viewer(OrthoViewer)
    visuals = ortho.add_image(
        _image(),
        single=_plane_single(),
        controls=InMemoryImageControlsConfig(
            appearance=True, render_plane_controls=True
        ),
    )
    for scene in ortho.scenes.values():
        ortho.controller.add_canvas(scene.id)
    group_2d = [visuals[key].id for key in ("xy", "xz", "yz")]
    assert ortho._render_planes_target(group_2d) is None
    # One control, for the 3D view's group, editing the 3D panel's visual.
    (control,) = _plane_controls(ortho)
    assert control.visual_ids == (visuals["vol"].id,)
    assert control.editor.has_gizmo
    control.editor.add()
    assert len(visuals["vol"].render_planes) == 1
    assert visuals["xy"].render_planes == ()


# -- OrthoViewer: the "slices" mode (10.2) --------------------------------------


def _slices(make_viewer, *, mode=True, **kwargs):
    ortho = make_viewer(OrthoViewer, **kwargs)
    visuals = ortho.add_image(_image(), single=_plane_single())
    ortho.dims_controller.set_slice_positions({0: 3.0, 1: 5.0, 2: 7.0})
    if mode:
        ortho.plane_controller.mode = "slices"
    return ortho, visuals


def test_the_mode_is_off_by_default_and_checked(make_viewer):
    ortho = make_viewer(OrthoViewer)
    visuals = ortho.add_image(_image(), single=_plane_single())
    assert ortho.plane_controller.mode is None
    assert visuals["vol"].render_planes == ()  # the mode writes nothing unasked
    with pytest.raises(ValueError, match="Unknown plane mode"):
        ortho.plane_controller.mode = "all"
    assert ortho.plane_controller.blocked_reason() == ""


def test_turning_the_mode_on_writes_three_planes_at_the_slice_positions(make_viewer):
    ortho, visuals = _slices(make_viewer, mode=False)
    own = _plane(ortho, normal=(1, 1, 0))
    ortho.controller.set_render_planes(visuals["vol"].id, (own,))
    modes: list = []
    ortho.plane_controller.mode_changed.connect(modes.append)

    ortho.plane_controller.mode = "slices"
    planes = visuals["vol"].render_planes
    # One plane per 2D panel, facing the axis that panel slices: xy slices
    # z, xz slices y, yz slices x.
    assert _normals(planes) == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert _positions(planes) == [3.0, 5.0, 7.0]
    assert tuple(p.id for p in planes) == ortho.plane_controller.plane_ids
    assert all(p.extent_0 == p.extent_1 == (None, None) and p.enabled for p in planes)
    assert own not in planes  # the mode replaced the visual's own plane
    assert modes == ["slices"]
    for key in ("xy", "xz", "yz"):
        assert visuals[key].render_planes == ()
    ortho.plane_controller.mode = "slices"  # the current mode: nothing
    assert modes == ["slices"]


def test_every_image_and_labels_vol_visual_gets_the_planes(make_viewer):
    """D-P62: whatever its render mode; other visual types are left alone."""
    ortho, visuals = _slices(make_viewer)
    volume = ortho.add_image(_image(), name="volume")  # in a volume mode
    labels = ortho.add_labels(LabelMemoryStore(data=np.ones(SHAPE, np.int32)))
    points = ortho.add_points(PointsMemoryStore(positions=np.zeros((3, 3), np.float32)))
    expected = visuals["vol"].render_planes
    assert len(expected) == 3
    assert volume["vol"].render_planes == expected
    assert labels["vol"].render_planes == expected
    assert not hasattr(points["vol"], "render_planes")
    assert set(ortho.plane_controller.visual_ids) == {
        visuals["vol"].id,
        volume["vol"].id,
        labels["vol"].id,
    }
    # A volume-mode visual carries them and draws them once in plane mode.
    assert not volume["vol"].plane_mode()
    ortho.controller.set_image_render_mode(volume["vol"].id, "plane")
    assert volume["vol"].render_planes == expected


@pytest.mark.parametrize(("panel", "axis"), [("xy", 0), ("xz", 1), ("yz", 2)])
def test_a_slider_moves_its_plane(make_viewer, panel, axis):
    ortho, visuals = _slices(make_viewer)
    log: list = []
    ortho.controller.on_render_planes_changed(
        visuals["vol"].id, log.append, owner_id=uuid4()
    )
    before = visuals["vol"].render_planes
    # The 2D panel's own slider, as its widget moves it.
    ortho.controller.update_slice_indices(ortho.scenes[panel].id, {axis: 6.0})
    after = visuals["vol"].render_planes
    expected = [3.0, 5.0, 7.0]
    expected[axis] = 6.0
    assert _positions(after) == expected
    assert _normals(after) == _normals(before)
    assert [p.id for p in after] == [p.id for p in before]  # moved, not replaced
    # One change of the tuple for one slider move, stamped as the mode's.
    assert len(log) == 1
    assert log[0].source_id == ortho.plane_controller.id


def test_a_plane_moved_by_hand_moves_its_slider_and_the_other_visuals(make_viewer):
    """D-P61: a change made on a visual is carried over, not thrown away."""
    ortho, visuals = _slices(make_viewer)
    other = ortho.add_labels(LabelMemoryStore(data=np.ones(SHAPE, np.int32)))
    controller = ortho.controller
    vol = visuals["vol"]
    xz_plane = vol.render_planes[1]

    moved = xz_plane.model_copy(update={"origin": (3.0, 9.0, 7.0)})
    controller.set_render_plane(vol.id, xz_plane.id, moved)
    # The xz panel's slider (world y), on every panel: the axes are linked.
    for scene in ortho.scenes.values():
        assert scene.dims.selection.slice_indices[1] == 9.0
    assert _positions(vol.render_planes) == [3.0, 9.0, 7.0]
    assert other["vol"].render_planes == vol.render_planes

    # The same through a direct assignment, on the other visual.
    planes = list(other["vol"].render_planes)
    planes[0] = planes[0].model_copy(update={"origin": (1.0, 9.0, 7.0)})
    other["vol"].render_planes = tuple(planes)
    assert ortho.scenes["xy"].dims.selection.slice_indices[0] == 1.0
    assert _positions(vol.render_planes) == [1.0, 9.0, 7.0]
    assert vol.render_planes == other["vol"].render_planes


def test_extents_and_enabled_set_by_hand_are_kept_and_copied(make_viewer):
    ortho, visuals = _slices(make_viewer)
    other = ortho.add_labels(LabelMemoryStore(data=np.ones(SHAPE, np.int32)))
    vol = visuals["vol"]
    first = vol.render_planes[0]
    edited = first.model_copy(update={"extent_0": (-2.0, 3.0), "enabled": False})
    ortho.controller.set_render_plane(vol.id, first.id, edited)
    assert vol.render_planes[0] == edited
    assert other["vol"].render_planes == vol.render_planes
    # A slider move afterwards keeps them: only the position changes.
    ortho.dims_controller.set_slice_position(0, 4.0)
    kept = vol.render_planes[0]
    assert kept.extent_0 == (-2.0, 3.0) and kept.enabled is False
    assert kept.origin[0] == 4.0
    assert other["vol"].render_planes == vol.render_planes
    # A visual added later takes the planes the others carry.
    late = ortho.add_image(_image(), name="late")
    assert late["vol"].render_planes == vol.render_planes


def test_a_tuple_that_is_not_the_three_slice_planes_is_put_back(make_viewer):
    ortho, visuals = _slices(make_viewer)
    controller = ortho.controller
    vol = visuals["vol"]
    slices = vol.render_planes
    tilted = RenderPlane.from_point_normal(
        _world(ortho), slices[0].origin, (1, 0.5, 0), axes=AXES, id=slices[0].id
    )
    foreign = _plane(ortho)
    cases = {
        "tilted": (tilted, *slices[1:]),
        "removed": slices[:2],
        "added": (*slices, foreign),
        "replaced": (foreign, *slices[1:]),
        "cleared": (),
    }
    with pytest.warns(UserWarning, match="three slice planes") as caught:
        for planes in cases.values():
            controller.set_render_planes(vol.id, planes)
            assert vol.render_planes == slices
    assert len(caught) == 1  # said once
    # No slider moved.
    assert ortho.plane_controller.positions() == (3.0, 5.0, 7.0)


def test_turning_the_mode_off_leaves_the_planes_free_to_edit(make_viewer):
    ortho, visuals = _slices(make_viewer)
    vol = visuals["vol"]
    slices = vol.render_planes
    ortho.plane_controller.mode = None
    assert vol.render_planes == slices  # left as they are
    assert ortho.plane_controller.blocked_reason() == ""
    # Free: a tilt stays, a slider no longer moves a plane.
    tilted = RenderPlane.from_point_normal(
        _world(ortho), slices[0].origin, (1, 0.5, 0), axes=AXES, id=slices[0].id
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ortho.controller.set_render_planes(vol.id, (tilted,))
    assert vol.render_planes == (tilted,)
    ortho.dims_controller.set_slice_position(0, 6.0)
    assert vol.render_planes == (tilted,)


async def test_no_gizmo_on_the_slice_planes(make_viewer):
    ortho, visuals = _slices(make_viewer)
    for scene in ortho.scenes.values():
        ortho.controller.add_canvas(scene.id)
    plane = visuals["vol"].render_planes[0]
    with pytest.raises(ValueError, match="have no gizmo"):
        ortho.add_render_plane_gizmo(visuals, plane)
    ortho.plane_controller.mode = None
    session = ortho.add_render_plane_gizmo(visuals, plane)
    assert not session.closed
    # Turning the mode on gives the visual planes of the same ids in the
    # same places: the session stays, and the control says why it is off.
    ortho.plane_controller.mode = "slices"
    assert ortho.plane_controller.blocked_reason() == SLICES_REASON


@pytest.mark.parametrize("backend", ["anywidget", "qt"])
async def test_the_control_shows_the_slice_planes_read_only(
    make_viewer, backend, qtbot
):
    ortho = make_viewer(OrthoViewer)
    visuals = ortho.add_image(
        _image(),
        single=_plane_single(),
        controls=InMemoryImageControlsConfig(
            appearance=True, render_plane_controls=True
        ),
    )
    for scene in ortho.scenes.values():
        ortho.controller.add_canvas(scene.id)
    (control,) = _plane_controls(ortho, backend)
    if backend == "qt":
        qtbot.addWidget(control.widget)
    assert control.editor.blocked == ""

    ortho.plane_controller.mode = "slices"
    # The mode change reached the control: three rows, disabled, with why.
    assert control.editor.blocked == SLICES_REASON
    assert len(control.editor.rows) == 3
    described = control.editor.describe()
    assert all(row["gizmo_blocked"] == SLICES_REASON for row in described)
    assert not control.editor.can_add
    if backend == "qt":
        assert not control._body.isEnabled()
        assert control._reason.text() == SLICES_REASON
        assert not control._add.isEnabled()
    else:
        assert control.blocked == SLICES_REASON
        assert control.can_add is False
    # An edit is refused with the reason; the rows follow the sliders.
    control.editor.set_position(0, 1.0)
    assert visuals["vol"].render_planes[0].origin[0] == 0.0
    ortho.dims_controller.set_slice_position(0, 5.0)
    assert control.editor.rows[0]["position"] == 5.0

    ortho.plane_controller.mode = None
    assert control.editor.blocked == ""
    control.editor.set_position(0, 1.0)
    assert visuals["vol"].render_planes[0].origin[0] == 1.0


def test_unlinked_axes_move_the_slicing_panel_and_the_3d_panel(make_viewer):
    ortho = make_viewer(OrthoViewer, link_axes=False)
    visuals = ortho.add_image(_image(), single=_plane_single())
    controller = ortho.controller
    for (key, axis), value in zip(
        (("xy", 0), ("xz", 1), ("yz", 2)), (3.0, 5.0, 7.0), strict=True
    ):
        controller.update_slice_indices(ortho.scenes[key].id, {axis: value})
    ortho.plane_controller.mode = "slices"
    vol = visuals["vol"]
    # Each plane is at its own panel's position.
    assert _positions(vol.render_planes) == [3.0, 5.0, 7.0]
    first = vol.render_planes[0]
    controller.set_render_plane(
        vol.id, first.id, first.model_copy(update={"origin": (6.0, 5.0, 7.0)})
    )
    assert ortho.scenes["xy"].dims.selection.slice_indices[0] == 6.0
    assert ortho.scenes["vol"].dims.selection.slice_indices[0] == 6.0
    assert ortho.scenes["xz"].dims.selection.slice_indices[0] == 0.0  # not linked


def test_a_removed_visual_is_forgotten(make_viewer):
    ortho, visuals = _slices(make_viewer)
    other = ortho.add_labels(LabelMemoryStore(data=np.ones(SHAPE, np.int32)))
    ortho.controller.remove_visual(other["vol"].id)
    assert ortho.plane_controller.visual_ids == (visuals["vol"].id,)
    ortho.dims_controller.set_slice_position(0, 6.0)  # nothing raises
    assert visuals["vol"].render_planes[0].origin[0] == 6.0


def test_the_clipping_link_is_as_before(make_viewer):
    ortho, visuals = _slices(make_viewer)
    store = ortho.controller.get_data_store(UUID(str(visuals["vol"].data_store_id)))
    item = ClippingPlane.from_point_normal(
        store.data_coordinate_system, (4, 6, 8), (0, 0, 1)
    )
    visuals["xz"].clipping_planes = (item,)
    assert {v.clipping_planes for v in visuals.values()} == {(item,)}
    # And the slice planes took no notice.
    assert _positions(visuals["vol"].render_planes) == [3.0, 5.0, 7.0]


# -- a 2D scrub holds the 3D visual's plan (B9) ---------------------------------


def _record_plans(monkeypatch, visual_id) -> list[str]:
    from cellier.render.scene_manager import SceneManager

    plans: list[str] = []
    original = SceneManager.plan_chunked

    def spy(self, request, visual_configs):
        for planned, cfg in visual_configs.items():
            targets = request.target_visual_ids
            if planned == visual_id and (targets is None or planned in targets):
                plans.append(cfg.plan_mode.name)
        return original(self, request, visual_configs)

    monkeypatch.setattr(SceneManager, "plan_chunked", spy)
    return plans


async def test_a_2d_scrub_holds_the_vol_plan_and_loads_once(
    make_viewer, tmp_path, monkeypatch
):
    ortho = make_viewer(OrthoViewer)
    controller = ortho.controller
    controller._render_manager.config.scheduler.dims_settle_s = SETTLE_S
    fx.write_pyramid(tmp_path, fx.ISO)
    store, _ = fx.open_pyramid(tmp_path, fx.ISO)
    visuals = ortho.add_image_multiscale(
        store,
        appearance=MultiscaleImageAppearance(force_level=1),
        single=MultiscaleImageSingleAppearance(
            color_map="viridis", clim=(0.0, 65535.0), render_mode="plane"
        ),
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS, block_size=16),
    )
    vol = visuals["vol"]
    for scene in ortho.scenes.values():
        controller.add_canvas(scene.id, canvas_size=(200, 125))
    ortho.dims_controller.set_slice_positions({0: 40.0, 1: 40.0, 2: 40.0})
    ortho.plane_controller.mode = "slices"
    for scene in ortho.scenes.values():
        controller.fit_camera(scene.id)
    controller.reslice_all()
    await drain_loading(controller)

    drags: list = []
    changes: list = []
    owner = uuid4()
    controller.on_plane_interaction(vol.id, drags.append, owner_id=owner)
    controller.on_render_planes_changed(vol.id, changes.append, owner_id=owner)
    plans = _record_plans(monkeypatch, vol.id)
    xy = ortho.scenes["xy"]
    widget = uuid4()

    # The xy panel's z slider, scrubbed as its widget scrubs it.
    controller.begin_dims_interaction(xy.id, source_id=widget)
    for z in (44.0, 48.0, 52.0, 56.0):
        controller.update_slice_indices(
            xy.id, {0: z}, source_id=widget, interactive=True
        )
        # The plane follows every tick, and the 3D visual plans nothing.
        assert vol.render_planes[0].origin[0] == z
        assert plans == []
    assert controller.plane_interaction_state(vol.id) == "active"
    assert len(changes) == 4
    controller.end_dims_interaction(xy.id, source_id=widget)
    # Released: the target is planned once.
    assert plans == ["FULL"]
    assert [(e.phase, e.reason) for e in drags] == [
        ("start", None),
        ("end", "release"),
    ]
    assert all(isinstance(e, PlaneInteractionEvent) for e in drags)
    assert all(isinstance(e, RenderPlanesChangedEvent) for e in changes)
    await drain_loading(controller)

    # With the mode off a scrub is no plane interaction.
    ortho.plane_controller.mode = None
    drags.clear()
    controller.begin_dims_interaction(xy.id, source_id=widget)
    controller.update_slice_indices(
        xy.id, {0: 30.0}, source_id=widget, interactive=True
    )
    controller.end_dims_interaction(xy.id, source_id=widget)
    assert drags == []
    assert vol.render_planes[0].origin[0] == 56.0
    await drain_loading(controller)
