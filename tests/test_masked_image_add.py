"""Adding a MaskedImage: the refusals (masked image design v2, 4.3 and 5.2).

Every refusal happens before a render object is built, and its message says
what to change.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    ImageVisual,
    MaskedImageRenderConfig,
    MaskedImageVisual,
)

SMALL = {"gpu_budget_bytes": 8 * 1024**2, "value_gpu_budget_bytes": 8 * 1024**2}


@pytest.fixture
def controller():
    controller = CellierController()
    yield controller
    controller.close()


def _scene(controller, axes, displayed=("z", "y", "x")):
    return controller.add_scene(coordinate_system=axes, displayed_axes=displayed).id


def _store(shape, name):
    rng = np.random.default_rng(0)
    return ImageMemoryStore(data=rng.random(shape).astype(np.float32), name=name)


def test_fewer_than_three_spatial_axes_is_refused(controller):
    scene_id = _scene(controller, spatial_axes("y", "x"), displayed=("y", "x"))
    with pytest.raises(ValueError, match=r"2 spatial axes.*needs three"):
        controller.add_masked_image(
            _store((8, 8), "mask"),
            _store((8, 8), "value"),
            scene_id,
            render_config=MaskedImageRenderConfig(**SMALL),
        )


def test_a_value_without_the_masks_time_axis_is_refused(controller):
    axes = [("t", "time"), *spatial_axes("z", "y", "x")]
    scene_id = _scene(controller, axes)
    mask = _store((2, 8, 8, 8), "mask")
    value = _store((8, 8, 8), "value")
    # With no value_to_mask the two must share a grid.
    with pytest.raises(ValueError, match="share a grid"):
        controller.add_masked_image(
            mask, value, scene_id, render_config=MaskedImageRenderConfig(**SMALL)
        )
    # Placed explicitly, the value still has no t: no broadcast.
    matrix = np.zeros((5, 4))
    matrix[1:4, 0:3] = np.eye(3)
    matrix[4, 3] = 1.0
    with pytest.raises(ValueError, match="not broadcast"):
        controller.add_masked_image(
            mask,
            value,
            scene_id,
            value_to_mask=matrix,
            render_config=MaskedImageRenderConfig(**SMALL),
        )


def test_value_to_mask_coupling_a_slider_to_space_is_refused(controller):
    axes = [("t", "time"), *spatial_axes("z", "y", "x")]
    scene_id = _scene(controller, axes)
    matrix = np.eye(5)
    matrix[1, 0] = 0.5  # the value's t moves the mask's z
    with pytest.raises(ValueError, match="slider axis may only map"):
        controller.add_masked_image(
            _store((2, 8, 8, 8), "mask"),
            _store((2, 8, 8, 8), "value"),
            scene_id,
            value_to_mask=matrix,
            render_config=MaskedImageRenderConfig(**SMALL),
        )


def test_value_to_mask_must_name_the_two_systems(controller):
    from cellier.transform import AffineTransform

    scene_id = _scene(controller, spatial_axes("z", "y", "x"))
    mask = _store((8, 8, 8), "mask")
    value = _store((8, 8, 8), "value")
    controller.add_data_store(mask)
    controller.add_data_store(value)
    controller._ensure_data_coordinate_systems(scene_id, mask)
    controller._ensure_data_coordinate_systems(scene_id, value)
    backwards = AffineTransform.from_matrix(
        np.eye(4), mask.data_coordinate_systems[0], value.data_coordinate_systems[0]
    )
    with pytest.raises(ValueError, match="must map the value store's level-0"):
        controller.add_masked_image(
            mask,
            value,
            scene_id,
            value_to_mask=backwards,
            render_config=MaskedImageRenderConfig(**SMALL),
        )


@pytest.mark.parametrize(
    ("side", "field"),
    [("mask", "gpu_budget_bytes"), ("value", "value_gpu_budget_bytes")],
)
def test_an_in_memory_side_that_does_not_fit_is_refused(controller, side, field):
    """256 cubed needs 512 bricks of 32; 128 MiB holds 511 (P0-16)."""
    scene_id = _scene(controller, spatial_axes("z", "y", "x"))
    small = ImageMemoryStore(data=np.zeros((32, 32, 32), np.float32), name="small")
    big = ImageMemoryStore(data=np.zeros((256, 256, 256), np.float32), name="big")
    mask, value = (big, small) if side == "mask" else (small, big)
    config = MaskedImageRenderConfig(
        gpu_budget_bytes=128 * 1024**2, value_gpu_budget_bytes=128 * 1024**2
    )
    with pytest.raises(ValueError, match=f"in-memory {side} image") as info:
        controller.add_masked_image(mask, value, scene_id, render_config=config)
    message = str(info.value)
    assert "needs 512 bricks" in message
    assert "holds 511" in message
    assert f"raise {field}, or use a multiscale store" in message
    assert not any(
        isinstance(v, MaskedImageVisual)
        for v in controller._model.scenes[scene_id].visuals
    )


def test_remove_data_store_refuses_while_the_value_is_read(controller):
    from cellier._masked_image import visual_store_ids

    scene_id = _scene(controller, spatial_axes("z", "y", "x"))
    mask = _store((8, 8, 8), "mask")
    value = _store((8, 8, 8), "value")
    visual = MaskedImageVisual(
        name="m",
        data_store_id=str(mask.id),
        value_store_id=str(value.id),
        level_transforms=[],
        value_level_transforms=[],
    )
    assert visual_store_ids(visual) == (mask.id, value.id)
    plain = ImageVisual(name="i", data_store_id=str(mask.id))
    assert visual_store_ids(plain) == (mask.id,)
    controller.add_data_store(mask)
    controller.add_data_store(value)
    # Put the model in the scene as add_visual would, without a render object.
    controller._model.scenes[scene_id].visuals.append(visual)
    with pytest.raises(ValueError, match="still referenced"):
        controller.remove_data_store(value.id)
    controller._model.scenes[scene_id].visuals.remove(visual)
    controller.remove_data_store(value.id)
