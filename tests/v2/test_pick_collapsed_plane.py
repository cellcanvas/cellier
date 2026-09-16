"""A pick names the plane that is on screen.

The render layer decodes only the axes it drew.  Every other component of
``ImagePickInfo.data_coordinate`` is a plane the visual collapsed, and the
visual reports it from the plan it last drew rather than the controller
re-deriving it from the dims state.  The two differ while a reslice is in
flight, and they used to differ permanently: the dims state carries a world
position whose rounding rule (half-up, centre-at-integer) is not the one the
pick payload documents (``floor``, ``[i, i + 1)``).
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.events._events import (
    ImagePickEvent,
    ImagePickInfo,
    ViewRay,
    _CanvasRawPointerEvent,
)
from cellier.render.render_manager import _ImageDisplayedDataCoord
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import AffineTransform, Axis
from cellier.visuals import InMemoryImageSingleAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests._v2 import data_system

CHANNEL_SHAPE = (4, 8, 10, 12)  # (c, z, y, x)


def _channel_viewer():
    """Demo 3's shape: a czyx image whose c axis is an ordinary sliced axis."""
    controller = CellierController(gui="offscreen")
    cs = world_coordinate_system(
        (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x")),
        name="world",
    )
    scene = controller.add_scene(coordinate_system=cs, dim="3d")
    controller.add_canvas(scene.id)

    array = np.zeros(CHANNEL_SHAPE, dtype=np.float32)
    # One marked voxel per channel, so a wrong channel reads zero.
    for channel in range(CHANNEL_SHAPE[0]):
        array[channel, 3, 4, 5] = channel + 1.0
    store = ImageMemoryStore(
        data=array,
        name="img",
        data_coordinate_systems=[
            data_system(("c", "z", "y", "x"), sampling="discrete")
        ],
    )
    data = store.data_coordinate_systems[0]
    world = scene.dims.world_coordinate_system
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axis_by_name(name).id: world.axis_by_name(name).id
            for name in ("c", "z", "y", "x")
        },
    )
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=InMemoryImageAppearance(),
        transform=transform,
        single=InMemoryImageSingleAppearance(color_map="gray"),
    )
    return controller, scene, visual, array


def _press(controller, scene, visual, displayed, *, collapsed):
    """One 3-D press whose pick decoded *displayed*, reporting *collapsed*."""
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())
    controller._on_raw_pointer_event(
        _CanvasRawPointerEvent(
            canvas_id=canvas_id,
            scene_id=scene.id,
            action="press",
            camera_type="3d",
            position_2d=None,
            ray=ViewRay(origin=np.zeros(3), direction=np.array([0.0, 0.0, 1.0])),
            hit_visual_id=visual.id,
            button=1,
            modifiers=(),
            buttons=(1,),
            gesture_id=None,
            pick_details=_ImageDisplayedDataCoord(
                displayed_data_coord=tuple(displayed),
                collapsed_data_indices=collapsed,
            ),
        )
    )
    assert len(received) == 1
    return received[0].pick_info


async def test_the_visual_names_the_plane_and_the_dims_state_does_not():
    """The reported channel is the one drawn, at a slider position that rounds up.

    ``c = 1.79`` renders channel 2 -- the assembler rounds half-up -- while
    flooring the raw pulled-back position would say channel 1.  Indexing the
    store with the wrong channel is silent: it returns the value stored there,
    which for a per-channel marker is zero.
    """
    controller, scene, visual, array = _channel_viewer()
    scene.dims.selection.displayed_axes = (1, 2, 3)
    scene.dims.selection.slice_indices = {0: 1.79}

    # pygfx (x, y, z) order reverses onto ascending displayed axes (z, y, x).
    details = _press(controller, scene, visual, (5.5, 4.5, 3.5), collapsed=((0, 2),))
    assert isinstance(details, ImagePickInfo)
    index = tuple(int(np.floor(value)) for value in details.data_coordinate)
    assert index == (2, 3, 4, 5)
    assert array[index] == 3.0  # channel 2's marker, not channel 1's zero
    # The value is read at that voxel; no channel_axis reports it as {0: ...}.
    assert details.channel_values == {0: 3.0}


async def test_the_plan_wins_over_a_slider_that_has_moved_since():
    """A pick is a question about the screen, not about the dims state.

    The slider says channel 0 and the visual last drew channel 3; until the
    reslice lands, channel 3 is what is on screen and so what a click hits.
    """
    controller, scene, visual, _array = _channel_viewer()
    scene.dims.selection.displayed_axes = (1, 2, 3)
    scene.dims.selection.slice_indices = {0: 0.0}

    details = _press(controller, scene, visual, (5.5, 4.5, 3.5), collapsed=((0, 3),))
    assert int(np.floor(details.data_coordinate[0])) == 3


async def test_floor_is_the_only_rule_a_consumer_needs():
    """Displayed and collapsed components share one convention.

    A collapsed axis has no sub-voxel position -- one plane was drawn -- so its
    component is that plane's centre.  Mixing conventions inside one tuple is
    what made ``floor`` right for half of it and wrong for the rest.
    """
    controller, scene, visual, _array = _channel_viewer()
    scene.dims.selection.displayed_axes = (1, 2, 3)
    scene.dims.selection.slice_indices = {0: 1.79}

    details = _press(controller, scene, visual, (5.25, 4.75, 3.5), collapsed=((0, 2),))
    assert tuple(details.data_coordinate) == (2.5, 3.5, 4.75, 5.25)


async def test_without_a_plan_the_dims_state_is_rounded_the_same_way():
    """The fallback lands on the plane the assembler would have fetched.

    A visual that reported nothing -- never resliced, or built headlessly --
    still has to answer.  Deriving the plane from the dims state is only
    correct if it rounds half-up like the assembler; flooring the raw position
    would name the neighbouring plane for any slider past a voxel's midpoint.
    """
    controller, scene, visual, _array = _channel_viewer()
    scene.dims.selection.displayed_axes = (1, 2, 3)
    scene.dims.selection.slice_indices = {0: 1.79}

    details = _press(controller, scene, visual, (5.5, 4.5, 3.5), collapsed=None)
    assert int(np.floor(details.data_coordinate[0])) == 2


async def test_a_live_reslice_reports_the_channel_it_fetched():
    """End to end: no synthesised pick payload, the real planning path.

    ``pick_collapsed_indices`` reads the plan the visual built from the
    controller's region, so this pins the whole chain rather than the
    controller's half of it.
    """
    controller, scene, visual, _array = _channel_viewer()
    scene.dims.selection.displayed_axes = (1, 2, 3)
    controller.update_slice_indices(scene.id, {0: 1.79})

    # Plan the way the slicer does, from the controller's own region, so the
    # visual records the same plan a real frame would have given it.
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    gfx_visual = controller._render_manager._scenes[scene.id].get_visual(visual.id)
    gfx_visual.build_slice_request(
        camera_pos_world=np.zeros(3),
        frustum_corners_world=None,
        fov_y_rad=1.0,
        screen_height_px=600.0,
        dims_state=scene.dims.to_state(),
        selection=controller._selections_for_scene(scene.id)[canvas_id],
    )
    assert gfx_visual.pick_collapsed_indices() == {0: 2}
