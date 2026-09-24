"""Asynchronous pick values for multiscale images and labels (design 3.7).

A multiscale visual's value is read at level 0 through the slicer, one read
per pick, per canvas: a newer pointer event cancels an in-flight ``move``
read, ``press`` and ``release`` reads always complete, and ``remove_visual``
cancels every read the visual has in flight.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.events import ImagePickEvent, LabelsPickEvent, ViewRay
from cellier.events._events import _CanvasRawPointerEvent
from cellier.render.render_manager import (
    _ImageDisplayedDataCoord,
    _LabelsDisplayedDataCoord,
)
from cellier.scene import spatial_axes
from cellier.visuals import (
    MultiscaleImageChannelAppearance,
    MultiscaleImageRenderConfig,
    MultiscaleLabelRenderConfig,
    MultiscaleLabelsAppearance,
)
from tests._gpu_budget import SMALL_BUDGETS
from tests.render.conftest import _write_multiscale_zarr

_CZYX = [("c", "channel"), *spatial_axes("z", "y", "x")]


def _pyramid(root, channels_first: bool, fill, dtype="float32"):
    from cellier.data.image._zarr_multiscale_store import MultiscaleZarrDataStore

    lead = (2,) if channels_first else ()
    _write_multiscale_zarr(
        root,
        levels=[("s0", (*lead, 8, 8, 8)), ("s1", (*lead, 4, 4, 4))],
        fill=fill,
        dtype=dtype,
    )
    ones = (1.0,) * len(lead)
    zeros = (0.0,) * len(lead)
    return MultiscaleZarrDataStore.from_scale_and_translation(
        zarr_path=str(root),
        scale_names=["s0", "s1"],
        level_scales=[(*ones, 1.0, 1.0, 1.0), (*ones, 2.0, 2.0, 2.0)],
        level_translations=[(*zeros, 0.0, 0.0, 0.0), (*zeros, 0.5, 0.5, 0.5)],
    )


@pytest.fixture
def czyx_store(tmp_path):
    """Channel ``c`` holds ``10 * (c + 1) + x`` everywhere."""

    def _fill(arr):
        x = np.arange(arr.shape[-1], dtype=arr.dtype)
        for channel in range(arr.shape[0]):
            arr[channel] = 10.0 * (channel + 1) + x

    return _pyramid(tmp_path, True, _fill)


@pytest.fixture
def labels_store(tmp_path):
    def _fill(arr):
        arr[...] = 7

    return _pyramid(tmp_path, False, _fill, dtype="int32")


def _raw(canvas_id, scene_id, visual_id, details, *, action, camera, gesture_id=None):
    return _CanvasRawPointerEvent(
        canvas_id=canvas_id,
        scene_id=scene_id,
        action=action,
        camera_type=camera,
        position_2d=np.array([1.0, 2.0]) if camera == "2d" else None,
        ray=(
            ViewRay(origin=np.zeros(3), direction=np.array([0.0, 0.0, 1.0]))
            if camera == "3d"
            else None
        ),
        hit_visual_id=visual_id,
        button=1,
        modifiers=(),
        buttons=(1,) if gesture_id is not None else (),
        gesture_id=gesture_id,
        pick_details=details,
    )


def _composite_image(controller, store, dim):
    scene = controller.add_scene(coordinate_system=_CZYX, dim=dim, name="scene")
    visual = controller.add_image_multiscale(
        store,
        scene.id,
        channel_axis=0,
        composite=True,
        channels={i: MultiscaleImageChannelAppearance() for i in range(2)},
        render_config=MultiscaleImageRenderConfig(**SMALL_BUDGETS),
    )
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, ImagePickEvent, received.append, owner_id=uuid4())
    return scene, visual, canvas_id, received


def _image_pick(x: float, *, dim: str, winner: int = 1):
    if dim == "2d":
        displayed = (x, 2.5)
    else:
        displayed = (x, 2.5, 3.5)
    return _ImageDisplayedDataCoord(
        displayed_data_coord=displayed,
        collapsed_data_indices=None,
        channel_index=winner,
        drawn_channels=(0, 1),
    )


async def test_a_2d_composite_value_arrives_after_the_handler(
    controller, drive_reslice, czyx_store
):
    scene, visual, canvas_id, received = _composite_image(controller, czyx_store, "2d")

    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            _image_pick(3.5, dim="2d"),
            action="press",
            camera="2d",
            gesture_id=uuid4(),
        )
    )
    assert received == []  # read asynchronously

    await drive_reslice(controller)

    (event,) = received
    assert event.pick_info.channel_values == {0: 13.0, 1: 23.0}
    assert int(np.floor(event.pick_info.data_coordinate[0])) == 1


async def test_a_3d_composite_reads_only_the_winner(
    controller, drive_reslice, czyx_store
):
    scene, visual, canvas_id, received = _composite_image(controller, czyx_store, "3d")

    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            _image_pick(5.5, dim="3d", winner=0),
            action="press",
            camera="3d",
            gesture_id=uuid4(),
        )
    )
    await drive_reslice(controller)

    (event,) = received
    assert event.pick_info.channel_values == {0: 15.0}


async def test_a_newer_pick_cancels_an_in_flight_move_read(
    controller, drive_reslice, czyx_store
):
    scene, visual, canvas_id, received = _composite_image(controller, czyx_store, "3d")

    for x in (1.5, 2.5, 3.5):
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                visual.id,
                _image_pick(x, dim="3d"),
                action="move",
                camera="3d",
            )
        )
    await drive_reslice(controller)

    (event,) = received
    assert event.pick_info.channel_values == {1: 23.0}


async def test_press_and_release_complete_when_moves_follow(
    controller, drive_reslice, czyx_store
):
    """Match by gesture id and action: the press arrives, the drag moves do not."""
    scene, visual, canvas_id, received = _composite_image(controller, czyx_store, "3d")
    gesture = uuid4()

    for action, gesture_id, x in [
        ("press", gesture, 1.5),
        ("move", gesture, 2.5),
        ("move", gesture, 3.5),
        ("release", gesture, 4.5),
        ("move", None, 5.5),
    ]:
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                visual.id,
                _image_pick(x, dim="3d"),
                action=action,
                camera="3d",
                gesture_id=gesture_id,
            )
        )
    await drive_reslice(controller)

    arrived = {
        (e.action, e.gesture_id): e.pick_info.channel_values[1] for e in received
    }
    assert arrived == {
        ("press", gesture): 21.0,
        ("release", gesture): 24.0,
        ("move", None): 25.0,
    }
    assert controller._pick_reads_by_visual == {}
    assert controller._move_pick_reads == {}


async def test_multiscale_labels_value_arrives_asynchronously(
    controller, drive_reslice, labels_store
):
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), dim="3d", name="scene"
    )
    visual = controller.add_labels_multiscale(
        labels_store,
        scene.id,
        appearance=MultiscaleLabelsAppearance(),
        render_config=MultiscaleLabelRenderConfig(**SMALL_BUDGETS),
    )
    canvas_id = uuid4()
    received: list = []
    controller.on_pick(canvas_id, LabelsPickEvent, received.append, owner_id=uuid4())

    controller._on_raw_pointer_event(
        _raw(
            canvas_id,
            scene.id,
            visual.id,
            _LabelsDisplayedDataCoord(displayed_data_coord=(1.5, 2.5, 3.5)),
            action="release",
            camera="3d",
            gesture_id=uuid4(),
        )
    )
    assert received == []
    await drive_reslice(controller)

    (event,) = received
    assert event.pick_info.value == 7
    assert event.pick_info.data_coordinate == (3.5, 2.5, 1.5)


async def test_remove_visual_cancels_its_reads_including_press_and_release(
    controller, drive_reslice, czyx_store
):
    scene, visual, canvas_id, received = _composite_image(controller, czyx_store, "3d")
    gesture = uuid4()
    for action in ("press", "release"):
        controller._on_raw_pointer_event(
            _raw(
                canvas_id,
                scene.id,
                visual.id,
                _image_pick(1.5, dim="3d"),
                action=action,
                camera="3d",
                gesture_id=gesture,
            )
        )
    assert len(controller._pick_reads_by_visual[visual.id]) == 2

    controller.remove_visual(visual.id)
    await drive_reslice(controller)

    assert received == []
    assert controller._pick_reads_by_visual == {}
