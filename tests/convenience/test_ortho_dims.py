"""``OrthoDimsController``: one world point across the four panels (design 3.6)."""

from __future__ import annotations

import numpy as np
from cmap import Colormap

from cellier.convenience import OrthoViewer
from cellier.convenience._ortho_dims import OrthoDimsController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import DimsChangedEvent, DimsUpdateEvent, SliderAxesChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals import InMemoryImageAppearance, InMemoryImageSingleAppearance

#: ``(t, z, y, x)``: spatial axes 1-3, one extra axis.
_WORLD = [("t", "time"), *spatial_axes("z", "y", "x")]


def _positions(viewer: OrthoViewer) -> list[dict[int, float]]:
    return [
        dict(scene.dims.selection.slice_indices) for scene in viewer.scenes.values()
    ]


def _record(viewer: OrthoViewer, event_type=DimsChangedEvent) -> list:
    events: list = []
    for scene in viewer.scenes.values():
        viewer.controller._outgoing_events.subscribe(
            event_type, events.append, entity_id=scene.id
        )
    return events


def _add_image(viewer: OrthoViewer) -> None:
    store = ImageMemoryStore(data=np.zeros((8, 8, 8), dtype=np.float32))
    viewer.add_image(
        store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map=Colormap("gray")),
    )


def test_viewer_exposes_its_dims_controller():
    viewer = OrthoViewer(_WORLD)
    assert isinstance(viewer.dims_controller, OrthoDimsController)
    assert viewer.dims_controller.scene_ids == tuple(
        viewer.scenes[key].id for key in ("xy", "xz", "yz", "vol")
    )


def test_set_slice_position_updates_every_panel_with_one_source_id():
    viewer = OrthoViewer(_WORLD)
    events = _record(viewer)

    viewer.dims_controller.set_slice_position(2, 4.5)

    assert all(position[2] == 4.5 for position in _positions(viewer))
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_set_slider_override_updates_every_panel():
    viewer = OrthoViewer(_WORLD)
    _add_image(viewer)
    events = _record(viewer, SliderAxesChangedEvent)

    viewer.dims_controller.set_slider_override(0, True)

    for scene in viewer.scenes.values():
        assert scene.dims.slider_overrides == {0: True}
        assert scene.slider_axes == (0, 1, 2, 3)
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_moving_the_xy_z_slider_moves_z_everywhere():
    """The XY panel's z slider is the stored z of XZ, YZ and the volume."""
    viewer = OrthoViewer(_WORLD)
    xy = viewer.scenes["xy"]
    widget_id = object()  # stands in for the widget's own id

    viewer.controller.incoming_events.emit(
        DimsUpdateEvent(
            source_id=widget_id,
            scene_id=xy.id,
            slice_indices={1: 6.0},
            displayed_axes=None,
        )
    )

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    assert positions[0][1] == 6.0


def test_a_direct_model_edit_is_mirrored_without_looping():
    viewer = OrthoViewer(_WORLD)
    xz = viewer.scenes["xz"]
    events = _record(viewer)

    xz.dims.selection.slice_indices = {**xz.dims.selection.slice_indices, 0: 5.0}

    assert all(position[0] == 5.0 for position in _positions(viewer))
    # One change on the edited panel, one mirrored change on each other.
    assert len(events) == 4


def test_thickness_is_mirrored():
    viewer = OrthoViewer(_WORLD)
    viewer.controller.update_thickness(viewer.scenes["yz"].id, {0: 1.5})
    for scene in viewer.scenes.values():
        assert scene.dims.selection.thickness == {0: 1.5}


def test_displayed_axes_are_never_mirrored():
    viewer = OrthoViewer(_WORLD)
    before = {
        key: scene.dims.selection.displayed_axes for key, scene in viewer.scenes.items()
    }

    viewer.controller.set_displayed_axes(viewer.scenes["xy"].id, (1, 3))

    assert viewer.scenes["xy"].dims.selection.displayed_axes == (1, 3)
    for key in ("xz", "yz", "vol"):
        assert viewer.scenes[key].dims.selection.displayed_axes == before[key]


def test_construction_copies_the_first_panel_when_panels_disagree():
    viewer = OrthoViewer(_WORLD, link_axes=False)
    controller = viewer.controller
    controller.update_slice_indices(viewer.scenes["xy"].id, {0: 2.0, 2: 3.0})
    controller.update_slice_indices(viewer.scenes["yz"].id, {0: 9.0})
    controller.set_slider_override(viewer.scenes["vol"].id, 0, False)

    OrthoDimsController(
        controller, [viewer.scenes[k] for k in ("xy", "xz", "yz", "vol")]
    )

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    assert positions[0][0] == 2.0
    assert all(scene.dims.slider_overrides == {} for scene in viewer.scenes.values())


def test_unlinked_panels_are_not_mirrored():
    viewer = OrthoViewer(_WORLD, link_axes=False)
    assert viewer.dims_controller is None
    assert not viewer.axis_sync_enabled

    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})

    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0


def test_re_enabling_sync_brings_the_panels_back_into_agreement():
    viewer = OrthoViewer(_WORLD)
    viewer.axis_sync_enabled = False
    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})
    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0

    viewer.axis_sync_enabled = True

    assert all(position[0] == 4.0 for position in _positions(viewer))


def test_close_stops_mirroring():
    viewer = OrthoViewer(_WORLD)
    viewer.dims_controller.close()
    viewer.controller.update_slice_indices(viewer.scenes["xy"].id, {0: 4.0})
    assert viewer.scenes["xz"].dims.selection.slice_indices[0] == 0.0


def test_center_slices_sets_all_four_panels_once(image_store):
    viewer = OrthoViewer(spatial_axes("z", "y", "x"))
    viewer.controller.add_data_store(image_store)
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    events = _record(viewer)

    viewer.center_slices()

    positions = _positions(viewer)
    assert all(position == positions[0] for position in positions)
    # One write per panel, all from the dims controller: no mirrored echoes.
    assert len(events) == 4
    assert {event.source_id for event in events} == {viewer.dims_controller.id}


def test_save_load_keeps_the_panels_in_agreement(tmp_path):
    viewer = OrthoViewer(_WORLD)
    _add_image(viewer)
    viewer.dims_controller.set_slice_positions({0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0})
    viewer.dims_controller.set_slider_override(0, True)
    path = tmp_path / "ortho.json"
    viewer.to_file(path)

    loaded = OrthoViewer.from_file(path)

    positions = _positions(loaded)
    assert all(position == positions[0] for position in positions)
    assert positions[0] == {0: 1.0, 1: 3.0, 2: 4.0, 3: 5.0}
    slider_axes = {scene.slider_axes for scene in loaded.scenes.values()}
    assert slider_axes == {(0, 1, 2, 3)}
    assert loaded.axis_sync_enabled
