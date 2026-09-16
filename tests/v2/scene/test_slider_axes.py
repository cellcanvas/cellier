"""``Scene.slider_axes`` and ``DimsManager.slider_overrides`` (design 3.5)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from cmap import Colormap

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.data.points._points_memory_store import PointsMemoryStore
from cellier.events import SliderAxesChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.viewer_model import ViewerModel
from cellier.visuals import (
    ImageVisual,
    InMemoryImageAppearance,
    InMemoryImageChannelAppearance,
    PointsMarkerAppearance,
)

#: The lightsheet world of the design's section 3.5 example.
_TCZYX = [("t", "time"), ("c", "channel"), *spatial_axes("z", "y", "x")]


def _appearance() -> InMemoryImageAppearance:
    return InMemoryImageAppearance()


def _points(controller, scene):
    """A (z, y, x) points visual, broadcast over the leading world axes."""
    store = PointsMemoryStore(positions=np.zeros((2, 3), dtype=np.float32))
    return controller.add_points(
        store, scene.id, appearance=PointsMarkerAppearance(size=1.0)
    )


def _lightsheet(*, composite: bool):
    """World ``(t, c, z, y, x)`` with a ``tczyx`` image and broadcast points."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    store = ImageMemoryStore(data=np.zeros((2, 3, 4, 5, 6), dtype=np.float32))
    if composite:
        image = controller.add_image(
            store,
            scene.id,
            channel_axis=1,
            composite=True,
            channels={
                0: InMemoryImageChannelAppearance(color_map=Colormap("green")),
                1: InMemoryImageChannelAppearance(color_map=Colormap("magenta")),
            },
        )
    else:
        image = controller.add_image(store, scene.id, appearance=_appearance())
    _points(controller, scene)
    return controller, scene, image


# ---------------------------------------------------------------------------
# The design's table, row by row
# ---------------------------------------------------------------------------


def test_composite_image_does_not_derive_its_channel_axis():
    _controller, scene, _image = _lightsheet(composite=True)
    assert scene.slider_axes == (0, 2, 3, 4)


def test_single_image_derives_every_axis():
    _controller, scene, _image = _lightsheet(composite=False)
    assert scene.slider_axes == (0, 1, 2, 3, 4)


def test_an_override_hides_a_derived_axis():
    controller, scene, _image = _lightsheet(composite=False)
    controller.set_slider_override(scene.id, 0, False)
    assert scene.dims.slider_overrides == {0: False}
    assert scene.slider_axes == (1, 2, 3, 4)


def test_overrides_hide_one_axis_and_force_another():
    controller, scene, _image = _lightsheet(composite=True)
    controller.set_slider_override(scene.id, 0, False)
    controller.set_slider_override(scene.id, 1, True)
    assert scene.dims.slider_overrides == {0: False, 1: True}
    assert scene.slider_axes == (1, 2, 3, 4)


def test_clearing_an_override_returns_the_axis_to_automatic():
    controller, scene, _image = _lightsheet(composite=False)
    controller.set_slider_override(scene.id, 0, False)
    controller.set_slider_override(scene.id, 0, None)
    assert scene.dims.slider_overrides == {}
    assert scene.slider_axes == (0, 1, 2, 3, 4)


# ---------------------------------------------------------------------------
# What counts
# ---------------------------------------------------------------------------


def test_a_scene_without_visuals_has_no_slider_axes():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    assert scene.slider_axes == ()


def test_broadcast_axes_feed_nothing():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    _points(controller, scene)
    assert scene.slider_axes == (2, 3, 4)


def test_hidden_visuals_count():
    controller, scene, image = _lightsheet(composite=False)
    controller.set_visual_visible(image.id, False)
    assert scene.slider_axes == (0, 1, 2, 3, 4)


def test_a_visual_without_a_transform_contributes_nothing():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    store = ImageMemoryStore(data=np.zeros((2, 3, 4, 5, 6), dtype=np.float32))
    scene.visuals.append(
        ImageVisual(name="loose", data_store_id=str(store.id), appearance=_appearance())
    )
    assert scene.slider_axes == ()


def test_a_transform_without_axis_correspondence_raises():
    """D37: that case is not designed yet, so the error is not swallowed."""
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")

    def _no_correspondence():
        raise ValueError("a shear")

    scene.visuals.append(
        SimpleNamespace(
            transform=SimpleNamespace(axis_correspondence=_no_correspondence)
        )
    )
    with pytest.raises(ValueError, match="a shear"):
        _ = scene.slider_axes


# ---------------------------------------------------------------------------
# SliderAxesChangedEvent
# ---------------------------------------------------------------------------


def _record(controller, scene):
    events: list[SliderAxesChangedEvent] = []
    controller._outgoing_events.subscribe(
        SliderAxesChangedEvent, events.append, entity_id=scene.id
    )
    return events


def test_adding_and_removing_a_visual_emits_on_change_only():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    events = _record(controller, scene)

    first = _points(controller, scene)
    assert [event.slider_axes for event in events] == [(2, 3, 4)]

    second = _points(controller, scene)  # same axes: no event
    assert len(events) == 1

    controller.remove_visual(second.id)
    assert len(events) == 1
    controller.remove_visual(first.id)
    assert [event.slider_axes for event in events] == [(2, 3, 4), ()]


def test_a_direct_composite_assignment_emits():
    """``visual.composite = ...`` moves the derived set without a controller call."""
    controller, scene, image = _lightsheet(composite=True)
    events = _record(controller, scene)
    image.composite = False
    assert [event.slider_axes for event in events] == [(0, 1, 2, 3, 4)]
    image.composite = True
    assert [event.slider_axes for event in events] == [(0, 1, 2, 3, 4), (0, 2, 3, 4)]


def test_an_empty_composite_still_composites_its_axis():
    """D16: no drawn channel, but ``c`` is still not a derived slider axis."""
    controller, scene, image = _lightsheet(composite=True)
    controller.remove_channel(image.id, 0)
    controller.remove_channel(image.id, 1)
    assert image.channels == {}
    assert scene.slider_axes == (0, 2, 3, 4)


def test_an_override_emits_only_when_the_set_changes():
    controller, scene, _image = _lightsheet(composite=False)
    events = _record(controller, scene)

    controller.set_slider_override(scene.id, 2, True)  # already derived
    assert events == []
    controller.set_slider_override(scene.id, 0, False)
    assert [event.slider_axes for event in events] == [(1, 2, 3, 4)]


def test_a_direct_override_assignment_emits():
    controller, scene, _image = _lightsheet(composite=False)
    events = _record(controller, scene)
    scene.dims.slider_overrides = {1: False}
    assert [event.slider_axes for event in events] == [(0, 2, 3, 4)]


def test_an_override_change_does_not_reslice():
    from cellier.events import DimsChangedEvent

    controller, scene, _image = _lightsheet(composite=False)
    dims_events: list = []
    controller._outgoing_events.subscribe(
        DimsChangedEvent, dims_events.append, entity_id=scene.id
    )
    controller.set_slider_override(scene.id, 0, False)
    assert dims_events == []


def test_a_direct_transform_assignment_emits():
    """``visual.transform = ...`` moves the derived set without a controller call."""
    world = [("w", "space"), *spatial_axes("z", "y", "x")]
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=world, dim="2d")
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    image = controller.add_image(store, scene.id, appearance=_appearance())
    assert scene.slider_axes == (1, 2, 3)
    events = _record(controller, scene)

    data = store.data_coordinate_system
    world_system = scene.dims.world_coordinate_system
    image.transform = AffineTransform.from_axis_map(
        data,
        world_system,
        axis_map={
            data.axes[0].id: world_system.axes[0].id,
            data.axes[1].id: world_system.axes[2].id,
            data.axes[2].id: world_system.axes[3].id,
        },
        broadcast_output_axes=(world_system.axes[1].id,),
    )

    assert scene.slider_axes == (0, 2, 3)
    assert [event.slider_axes for event in events] == [(0, 2, 3)]


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_scene_creation_seeds_every_axis():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    assert set(scene.dims.selection.slice_indices) == {0, 1, 2, 3, 4}


def test_a_slider_edit_keeps_the_other_positions():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    controller.update_slice_indices(scene.id, {0: 3.0, 2: 1.5})
    controller.update_slice_indices(scene.id, {1: 2.0})
    assert scene.dims.selection.slice_indices == {
        0: 3.0,
        1: 2.0,
        2: 1.5,
        3: 0.0,
        4: 0.0,
    }


def test_update_slice_indices_with_an_unknown_axis_raises():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    before = dict(scene.dims.selection.slice_indices)
    with pytest.raises(ValueError, match="outside the scene's world"):
        controller.update_slice_indices(scene.id, {0: 1.0, 7: 1.0})
    assert scene.dims.selection.slice_indices == before


def test_set_slider_override_with_an_unknown_axis_raises():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    with pytest.raises(ValueError, match="outside the scene's world"):
        controller.set_slider_override(scene.id, 9, True)


def test_moving_a_displayed_axis_position_does_not_reslice():
    from cellier.events import DimsChangedEvent

    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_TCZYX, dim="3d")
    events: list = []
    controller._outgoing_events.subscribe(
        DimsChangedEvent, events.append, entity_id=scene.id
    )
    resliced: list = []
    controller.reslice_scene = lambda scene_id, **_: resliced.append(scene_id)

    controller.update_slice_indices(scene.id, {3: 4.0})  # y is displayed
    controller.update_slice_indices(scene.id, {0: 1.0})  # t is sliced

    assert [event.region_changed for event in events] == [False, True]
    assert resliced == [scene.id]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_save_load_keeps_overrides_and_positions_and_derives_slider_axes():
    controller, scene, _image = _lightsheet(composite=True)
    controller.set_slider_override(scene.id, 0, False)
    controller.set_slider_override(scene.id, 1, True)
    controller.update_slice_indices(scene.id, {0: 1.0, 1: 1.0, 2: 2.0})
    before = scene.slider_axes

    dumped = controller.to_model().model_dump(mode="json")
    dims = dumped["scenes"][str(scene.id)]["dims"]
    assert dims["slider_overrides"] == {"0": False, "1": True}
    assert set(dims["selection"]["slice_indices"]) == {"0", "1", "2", "3", "4"}
    assert "slider_axes" not in str(dumped)

    loaded = ViewerModel.model_validate(dumped)
    assert loaded.scenes[scene.id].slider_axes == before == (1, 2, 3, 4)
