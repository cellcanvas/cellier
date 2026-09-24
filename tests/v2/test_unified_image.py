"""The unified in-memory image visual (unified image design 3.1 - 3.4, 3.8).

One visual draws an image single-channel or composited.  These tests drive it
through the controller and plan against the real per-canvas selection, so the
slicing rule, the slot pool and the model bridges are all exercised together.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.controller import CellierController, _visual_render_config
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events import ImageCompositeChangedEvent, SliderAxesChangedEvent
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    ImageVisual,
    InMemoryImageAppearance,
    InMemoryImageChannelAppearance,
    effective_transparency_mode,
)
from tests._planning import planned_requests_3d

#: World ``(c, y, x)``; a 2D scene displays ``(y, x)`` and slices ``c``.
_CYX = [("c", "channel"), *spatial_axes("y", "x")]


def _channels(*indices, **overrides):
    return {i: InMemoryImageChannelAppearance(**overrides) for i in indices}


def _setup(*, composite=False, channels=None, channel_axis=0, dim="2d", shape=None):
    controller = CellierController(gui="offscreen")
    world = _CYX if dim == "2d" else [("c", "channel"), *spatial_axes("z", "y", "x")]
    scene = controller.add_scene(coordinate_system=world, dim=dim)
    controller.add_canvas(scene.id)
    if shape is None:
        shape = (3, 8, 8) if dim == "2d" else (3, 4, 8, 8)
    store = ImageMemoryStore(data=np.ones(shape, dtype=np.float32))
    visual = controller.add_image(
        store,
        scene.id,
        channel_axis=channel_axis,
        composite=composite,
        channels=channels if channels is not None else _channels(0, 1),
    )
    return controller, scene, visual


def _gfx(controller, scene, visual):
    return controller._render_manager._scenes[scene.id].get_visual(visual.id)


def _plan(controller, scene, visual):
    """Plan the visual against the canvas's selection, in the scene's mode."""
    gfx = _gfx(controller, scene, visual)
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    selection = controller._selections_for_scene(scene.id)[canvas_id]
    dims_state = scene.dims.to_state()
    if len(dims_state.selection.displayed_axes) == 3:
        return planned_requests_3d(
            gfx,
            np.zeros(3),
            None,
            1.0,
            100.0,
            dims_state=dims_state,
            selection=selection,
        )
    return gfx.build_slice_request_2d(
        np.zeros(3), 100.0, 10.0, None, None, dims_state, selection=selection
    )


def _channel_indices(requests):
    return [request.axis_selections[0] for request in requests]


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_composite_without_a_channel_axis_raises():
    with pytest.raises(ValidationError, match="requires a channel_axis"):
        ImageVisual(name="i", data_store_id="s", composite=True)


def test_an_empty_composite_is_valid():
    visual = ImageVisual(name="i", data_store_id="s", channel_axis=0, composite=True)
    assert visual.channels == {}


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_a_fifth_channel_raises_in_2d_and_3d(dim):
    with pytest.raises(ValueError, match="max_channels"):
        _setup(dim=dim, shape=None, channels=_channels(0, 1, 2, 3, 4))

    controller, _scene, visual = _setup(dim=dim, channels=_channels(0, 1, 2, 3))
    with pytest.raises(ValueError, match="max_channels"):
        controller.add_channel(visual.id, 4, InMemoryImageChannelAppearance())
    assert set(visual.channels) == {0, 1, 2, 3}


def test_channel_axis_is_frozen():
    _controller, _scene, visual = _setup()
    with pytest.raises(ValidationError):
        visual.channel_axis = 1


def test_the_channel_axis_must_map_to_a_world_axis():
    """A transform that drops the channel axis is refused.

    Stores and worlds are checked for rank before the image check runs, so
    the check is driven directly with a transform that reaches only y and x.
    """
    from types import SimpleNamespace

    controller, scene, _visual = _setup()
    dropped = SimpleNamespace(axis_correspondence=lambda: {1: 1, 2: 2})
    stub = SimpleNamespace(channel_axis=0, composite=False, transform=dropped)
    with pytest.raises(ValueError, match="maps to no world axis"):
        controller._check_image_axes(scene.id, stub)


def test_compositing_a_displayed_axis_is_refused_on_add():
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(coordinate_system=_CYX, dim="2d")
    store = ImageMemoryStore(data=np.ones((3, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="displays"):
        controller.add_image(store, scene.id, channel_axis=1, composite=True)
    assert scene.visuals == []


def test_set_image_composite_refuses_a_displayed_axis_and_leaves_the_model():
    controller, _scene, visual = _setup(channel_axis=1)
    with pytest.raises(ValueError, match="displays"):
        controller.set_image_composite(visual.id, True)
    assert visual.composite is False


def test_displaying_a_composited_axis_is_refused_before_anything_changes():
    controller, scene, _visual = _setup(composite=True)
    with pytest.raises(ValueError, match="composites it"):
        controller.update_displayed_axes(scene.id, (0, 2))
    assert scene.dims.selection.displayed_axes == (1, 2)


# ---------------------------------------------------------------------------
# Slicing and fetching
# ---------------------------------------------------------------------------


async def test_single_mode_follows_the_slider_even_to_an_index_not_in_channels():
    controller, scene, visual = _setup(channels=_channels(0))
    controller.update_slice_indices(scene.id, {0: 2.0})
    assert _channel_indices(_plan(controller, scene, visual)) == [2]


async def test_single_mode_outside_the_data_draws_nothing():
    controller, scene, visual = _setup()
    controller.update_slice_indices(scene.id, {0: 7.0})
    assert _plan(controller, scene, visual) == []
    assert _gfx(controller, scene, visual)._slice_empty is True


async def test_composite_draws_the_visible_channels_whatever_the_slider():
    channels = _channels(0, 1, 2)
    channels[1].visible = False
    controller, scene, visual = _setup(composite=True, channels=channels)
    controller.update_slice_indices(scene.id, {0: 99.0})

    requests = _plan(controller, scene, visual)

    # Nothing for the hidden channel; one plane per drawn channel.
    assert _channel_indices(requests) == [0, 2]


async def test_composite_fetches_nothing_for_an_index_outside_the_axis():
    controller, scene, visual = _setup(composite=True, channels=_channels(0, 3))
    assert _channel_indices(_plan(controller, scene, visual)) == [0]


def test_appearance_visible_masters_composite():
    _controller, _scene, visual = _setup(composite=True)
    assert _visual_render_config(visual).slicing_enabled is True
    visual.appearance.visible = False
    assert _visual_render_config(visual).slicing_enabled is False


async def test_an_empty_composite_plans_nothing_and_still_composites_its_axis():
    controller, scene, visual = _setup(composite=True, channels={})

    assert _plan(controller, scene, visual) == []
    assert _visual_render_config(visual).slicing_enabled is False
    assert 0 not in scene.slider_axes
    with pytest.raises(ValueError, match="composites it"):
        controller.update_displayed_axes(scene.id, (0, 2))


async def test_adding_or_showing_a_channel_reslices_and_draws_it():
    controller, scene, visual = _setup(composite=True, channels={})
    resliced: list = []
    original = controller.reslice_visual
    controller.reslice_visual = lambda vid: (resliced.append(vid), original(vid))

    controller.add_channel(visual.id, 1, InMemoryImageChannelAppearance(visible=False))
    assert resliced == [visual.id]
    assert _plan(controller, scene, visual) == []

    visual.channels[1].visible = True
    assert resliced == [visual.id, visual.id]
    assert _channel_indices(_plan(controller, scene, visual)) == [1]


async def test_removing_the_last_channel_leaves_an_empty_composite():
    controller, scene, visual = _setup(composite=True, channels=_channels(0))
    controller.remove_channel(visual.id, 0)
    assert visual.channels == {}
    assert _plan(controller, scene, visual) == []


async def test_a_mode_switch_does_not_refetch_a_slot_that_holds_the_slice():
    controller, scene, visual = _setup(composite=True)
    gfx = _gfx(controller, scene, visual)
    requests = _plan(controller, scene, visual)
    gfx.on_data_ready_2d(
        [(request, np.zeros((8, 8), dtype=np.float32)) for request in requests]
    )
    slot_for_one = gfx._slot_for_key[1]

    controller.update_slice_indices(scene.id, {0: 1.0})
    visual.composite = False

    assert _plan(controller, scene, visual) == []
    assert gfx._drawn == {1: slot_for_one}


async def test_a_direct_composite_assignment_behaves_like_set_image_composite():
    controller, _scene, visual = _setup()
    composite_events: list = []
    slider_events: list = []
    controller._outgoing_events.subscribe(
        ImageCompositeChangedEvent, composite_events.append
    )
    controller._outgoing_events.subscribe(SliderAxesChangedEvent, slider_events.append)
    resliced: list = []
    original = controller.reslice_visual
    controller.reslice_visual = lambda vid: (resliced.append(vid), original(vid))

    visual.composite = True
    assert [event.composite for event in composite_events] == [True]
    assert [event.slider_axes for event in slider_events] == [(1, 2)]
    assert resliced == [visual.id]

    controller.set_image_composite(visual.id, False)
    assert [event.composite for event in composite_events] == [True, False]
    assert [event.slider_axes for event in slider_events] == [(1, 2), (0, 1, 2)]
    assert resliced == [visual.id, visual.id]  # once each, not twice


# ---------------------------------------------------------------------------
# The slot pool and the render layer
# ---------------------------------------------------------------------------


def test_an_image_without_a_channel_axis_has_one_slot():
    controller, scene, visual = _setup(channel_axis=None, channels={})
    assert len(_gfx(controller, scene, visual).slots) == 1


def test_an_image_with_a_channel_axis_has_max_channels_slots():
    controller, scene, visual = _setup()
    assert len(_gfx(controller, scene, visual).slots) == visual.max_channels


async def test_render_order_and_depth_reach_every_drawn_slot():
    controller, scene, visual = _setup(composite=True)
    controller.update_appearance_field(visual.id, "render_order", 5)
    controller.update_appearance_field(visual.id, "depth_compare", "<=")
    gfx = _gfx(controller, scene, visual)
    _plan(controller, scene, visual)

    drawn = [gfx.slots[index] for index in gfx._drawn.values()]
    assert len(drawn) == 2
    for slot in drawn:
        assert slot.node_2d.render_order == 5
        assert slot.node_2d.material.depth_compare == "<="
        # Two channel planes at one depth: depth testing off.
        assert slot.node_2d.material.depth_test is False

    visual.channels[1].visible = False
    _plan(controller, scene, visual)
    (only,) = [gfx.slots[index] for index in gfx._drawn.values()]
    assert only.node_2d.material.depth_test is True


async def test_overlapping_3d_channels_stop_writing_depth():
    """The first volume's hit depth would otherwise clip the next into speckle."""
    controller, scene, visual = _setup(composite=True, dim="3d")
    gfx = _gfx(controller, scene, visual)
    _plan(controller, scene, visual)

    drawn = [gfx.slots[index] for index in gfx._drawn.values()]
    assert len(drawn) == 2
    for slot in drawn:
        assert slot.node_3d.material.depth_write is False
        assert slot.node_3d.material.depth_test is True

    visual.channels[1].visible = False
    _plan(controller, scene, visual)
    (only,) = [gfx.slots[index] for index in gfx._drawn.values()]
    assert only.node_3d.material.depth_write is True


def test_one_bounding_box_per_mode():
    import pygfx as gfx_module

    controller, scene, visual = _setup()
    group = _gfx(controller, scene, visual).node_2d
    lines = [child for child in group.children if isinstance(child, gfx_module.Line)]
    assert len(lines) == 1


async def test_transparency_follows_the_mode_until_set():
    controller, scene, visual = _setup()
    gfx = _gfx(controller, scene, visual)
    assert effective_transparency_mode(visual) == "blend"

    visual.composite = True
    _plan(controller, scene, visual)
    assert effective_transparency_mode(visual) == "add"
    for index in gfx._drawn.values():
        assert gfx.slots[index].node_2d.material.alpha_mode == "add"

    controller.update_appearance_field(visual.id, "transparency_mode", "multiply")
    for index in gfx._drawn.values():
        assert gfx.slots[index].node_2d.material.alpha_mode == "multiply"
    visual.composite = False
    assert effective_transparency_mode(visual) == "multiply"


def test_remove_visual_cancels_its_slices_and_closes_it():
    controller, scene, visual = _setup(composite=True)
    gfx = _gfx(controller, scene, visual)
    cancelled: list = []
    coordinator = controller._render_manager._slice_coordinator
    original = coordinator.cancel_visual
    coordinator.cancel_visual = lambda s, c, v: (cancelled.append(v), original(s, c, v))

    controller.remove_visual(visual.id)

    assert cancelled == [visual.id]
    assert gfx.slots == ()
    assert gfx.node_2d.children == ()


def test_a_shared_appearance_defaults_everything():
    appearance = InMemoryImageAppearance()
    assert appearance.transparency_mode is None
    assert appearance.interpolation == "nearest"
