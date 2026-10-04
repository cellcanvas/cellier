"""Clipping planes through the convenience viewers (design 3.3, D20, D30)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.convenience import OrthoViewer, Viewer
from cellier.data import LabelMemoryStore, PointsMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.transform import Axis, DataCoordinateSystem
from cellier.visuals import ClippingPlane


def _system() -> DataCoordinateSystem:
    return DataCoordinateSystem(
        name="data",
        datastore_id=uuid4(),
        axes=tuple(Axis(name=n, axis_type="space", sampling="discrete") for n in "zyx"),
    )


def _plane(system, x=4.0) -> ClippingPlane:
    return ClippingPlane.from_point_normal(system, (4, 4, x), (0, 0, 1))


def test_store_then_planes_then_visual(qtbot):
    """D30: the plane is built from the store's system before the visual."""
    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = _plane(store.data_coordinate_systems[0])
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_labels(store, clipping_planes=(plane,))
        assert visual.clipping_planes == (plane,)
        visual.clipping_planes = (plane, _plane(system, 6.0))
        assert len(visual.clipping_planes) == 2
    finally:
        viewer.controller.close()


def test_a_store_with_no_system_gets_planes_after_the_visual(qtbot):
    store = PointsMemoryStore(positions=np.zeros((3, 3), dtype=np.float32))
    assert store.data_coordinate_systems == []
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_points(store)
        visual.clipping_planes = (_plane(store.data_coordinate_systems[0]),)
        assert len(visual.clipping_planes) == 1
    finally:
        viewer.controller.close()


def test_the_ortho_panels_share_one_tuple(qtbot):
    """D20: one store, one system, so the planes are linked across views."""
    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = _plane(system)
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        visuals = ortho.add_labels(store, clipping_planes=(plane,))
        assert {v.clipping_planes for v in visuals.values()} == {(plane,)}

        moved = (_plane(system, 6.0),)
        visuals["xz"].clipping_planes = moved
        assert {v.clipping_planes for v in visuals.values()} == {moved}
        visuals["vol"].clipping_planes = ()
        assert {v.clipping_planes for v in visuals.values()} == {()}

        # Refused on one panel: no panel changes.
        with pytest.raises(Exception, match="data coordinate system"):
            visuals["xy"].clipping_planes = (_plane(_system()),)
        assert {v.clipping_planes for v in visuals.values()} == {()}
    finally:
        ortho.controller.close()


def test_planes_survive_save_and_load(qtbot, tmp_path):
    system = _system()
    store = PointsMemoryStore(
        positions=np.arange(12, dtype=np.float32).reshape(4, 3),
        data_coordinate_systems=[system],
    )
    planes = (_plane(system), _plane(system, 6.0).model_copy(update={"enabled": False}))
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_points(store, clipping_planes=planes)
        visual_id = visual.id
        viewer.to_file(tmp_path / "viewer.json")
    finally:
        viewer.controller.close()

    restored = Viewer.from_file(tmp_path / "viewer.json")
    try:
        controller = restored.controller
        visual = controller.get_visual_model(visual_id)
        assert visual.clipping_planes == planes
        # Still the store's system, and already on the render visual.
        store = controller.get_data_store(system.datastore_id)
        assert visual.clipping_planes[0].plane.coordinate_system == (
            store.data_coordinate_system.id
        )
        scene_manager = controller._render_manager._scenes[restored.scene.id]
        assert scene_manager.get_visual(visual_id).clipping_planes == planes
    finally:
        restored.controller.close()


# -- the gizmo (clipping plane gizmo design v2, 7) ------------------------------


def test_the_viewer_puts_a_gizmo_on_a_plane(qtbot):
    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane, other = _plane(system), _plane(system, 6.0)
    viewer = Viewer(spatial_axes("z", "y", "x"), dim="3d", gui="offscreen")
    try:
        visual = viewer.add_labels(store, clipping_planes=(plane, other))
        viewer.add_canvas()
        (canvas_id,) = viewer.canvases
        session = viewer.add_clipping_plane_gizmo(visual, plane)
        assert (session.canvas_id, session.plane_id) == (canvas_id, plane.id)
        assert viewer.controller.get_clipping_plane_gizmo(canvas_id) is session

        # By id, on the other plane: the canvas's one gizmo moves there.
        second = viewer.add_clipping_plane_gizmo(visual.id, other.id)
        assert session.closed
        viewer.remove_clipping_plane_gizmo()
        assert second.closed
        assert viewer.controller.get_clipping_plane_gizmo(canvas_id) is None
        viewer.remove_clipping_plane_gizmo()  # none: nothing
    finally:
        viewer.controller.close()


def test_the_ortho_gizmo_is_on_the_3d_panel_and_the_2d_panels_follow(qtbot):
    from cellier.gui._clipping_planes import gizmo_seed

    system = _system()
    store = LabelMemoryStore(
        data=np.ones((8, 9, 10), np.int32), data_coordinate_systems=[system]
    )
    plane = _plane(system)
    ortho = OrthoViewer(spatial_axes("z", "y", "x"), gui="offscreen")
    try:
        visuals = ortho.add_labels(store, clipping_planes=(plane,))
        controller = ortho.controller
        for scene in ortho.scenes.values():
            controller.add_canvas(scene.id)
        vol_canvas = controller.get_canvas_ids(ortho.scenes["vol"].id)[0]

        # Any panel's visual names the group; the session is on the 3D one.
        session = ortho.add_clipping_plane_gizmo(visuals["xy"], plane)
        assert session.visual_id == visuals["vol"].id
        assert session.canvas_id == vol_canvas

        # A move made as the gizmo makes it reaches every panel.
        moved = _plane(system, 6.0).plane
        controller.set_clipping_plane(
            visuals["vol"].id, plane.id, moved, source_id=session.id
        )
        assert {v.clipping_planes[0].plane for v in visuals.values()} == {moved}
        assert {v.clipping_planes[0].id for v in visuals.values()} == {plane.id}

        # A control of the panel group draws its toggle for the 3D panel.
        seed = gizmo_seed(controller, [v.id for v in visuals.values()])
        assert seed["gizmo"]["visual_id"] == str(visuals["vol"].id)
        assert seed["gizmo"]["canvas_id"] == str(vol_canvas)
        assert seed["gizmo_plane"] == str(plane.id)

        ortho.remove_clipping_plane_gizmo()
        assert session.closed
    finally:
        ortho.controller.close()
