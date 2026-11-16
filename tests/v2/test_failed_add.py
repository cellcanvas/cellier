"""A refused add leaves the controller as it was.

An add registers its store and the store's coordinate systems before it checks
the visual and builds its render objects. When any of those refuses the
visual, the stores and systems the add registered are unregistered again, and
no visual is left in the scene. Stores registered before the add stay.

The refusal at render-build time (a brick grid past the device's 3D texture
limit) is in ``tests/render/lut_indirection/test_brick_grid_limit.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from cellier.visuals import ImageVisual, MaskedImageRenderConfig

#: World ``(c, y, x)``; the scene displays ``(y, x)``, so compositing ``y``
#: is refused.
_CYX = [("c", "channel"), *spatial_axes("y", "x")]


@pytest.fixture
def controller():
    controller = CellierController(gui="offscreen")
    yield controller
    controller.close()


def _store(shape=(3, 8, 8), name="image") -> ImageMemoryStore:
    return ImageMemoryStore(data=np.ones(shape, dtype=np.float32), name=name)


def _assert_unregistered(controller, store, scene) -> None:
    assert store.id not in controller._model.data.stores
    assert store.id not in controller._store_psygnal_handlers
    assert len(store.data_changed) == 0
    for system in store.data_coordinate_systems:
        assert system.id not in controller._coordinate_systems
    assert scene.visuals == []
    assert controller._visual_to_scene == {}


def test_a_refused_add_image_leaves_no_store_or_systems(controller):
    scene = controller.add_scene(coordinate_system=_CYX, displayed_axes=("y", "x"))
    store = _store()
    with pytest.raises(ValueError, match="displays"):
        controller.add_image(store, scene.id, channel_axis=1, composite=True)
    # The store was given coordinate systems on the way in, and keeps them.
    assert store.data_coordinate_systems
    _assert_unregistered(controller, store, scene)

    # The same store can be added again.
    visual = controller.add_image(store, scene.id, channel_axis=0, composite=True)
    assert scene.visuals == [visual]
    assert store.id in controller._model.data.stores
    assert store.data_coordinate_systems[0].id in controller._coordinate_systems


def test_a_refused_model_leaves_no_half_registered_visual(controller):
    """The image axes are checked before the visual is placed in the scene."""
    scene = controller.add_scene(coordinate_system=_CYX, displayed_axes=("y", "x"))
    store = _store()
    model = ImageVisual(
        name="bad", data_store_id=str(store.id), channel_axis=1, composite=True
    )
    with pytest.raises(ValueError, match="displays"):
        controller.add_visual(scene.id, model, data_store=store)
    _assert_unregistered(controller, store, scene)
    assert not any(
        sub.owner_id == model.id
        for subs in controller._outgoing_events._subs.values()
        for sub in subs
    )
    assert model.id not in controller._visual_psygnal_handlers


def test_a_refused_add_keeps_a_store_registered_before_it(controller):
    scene = controller.add_scene(coordinate_system=_CYX, displayed_axes=("y", "x"))
    store = _store()
    good = controller.add_image(store, scene.id)
    with pytest.raises(ValueError, match="displays"):
        controller.add_image(store, scene.id, channel_axis=1, composite=True)
    assert scene.visuals == [good]
    assert store.id in controller._model.data.stores
    assert store.id in controller._store_psygnal_handlers
    assert store.data_coordinate_systems[0].id in controller._coordinate_systems

    other = _store(name="other")
    controller.add_data_store(other)
    with pytest.raises(ValueError, match="displays"):
        controller.add_image(other, scene.id, channel_axis=1, composite=True)
    assert other.id in controller._model.data.stores
    assert other.data_coordinate_systems[0].id in controller._coordinate_systems


def test_a_refused_masked_image_leaves_neither_store(controller):
    scene = controller.add_scene(
        coordinate_system=[("t", "time"), *spatial_axes("z", "y", "x")],
        displayed_axes=("z", "y", "x"),
    )
    mask = _store((2, 8, 8, 8), "mask")
    value = _store((8, 8, 8), "value")
    small = MaskedImageRenderConfig(
        gpu_budget_bytes=8 * 1024**2, value_gpu_budget_bytes=8 * 1024**2
    )
    with pytest.raises(ValueError, match="share a grid"):
        controller.add_masked_image(mask, value, scene.id, render_config=small)
    _assert_unregistered(controller, mask, scene)
    _assert_unregistered(controller, value, scene)
