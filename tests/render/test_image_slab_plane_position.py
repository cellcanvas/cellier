"""An image on a slab draws the plane at the slice position, not near it.

The image slicing rule narrows a sliced axis's slab to one plane.  It used to
put that plane at the midpoint of the slab's bounds, ``((p - h) + (p + h)) /
2``, which can land an ulp away from ``p``; the zero-thickness plane there
then excluded the selection's own slice position and building the selection
raised (full affine hand check H4, 2026-10-19; not a transform effect -- the
identity fails the same way).  The plane now comes from the translation
(``RegionSelection.with_planes_at_slice``).
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data import LabelMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes
from tests._planning import planned_requests_2d
from tests.render._affine import multiscale_store

#: The H4 slider value: its slab midpoint at half thickness 1 is one ulp high.
ROUNDING_POSITION = 31.3796992481203

_SHAPE = (40, 8, 8)


def _add(controller, scene, family: str, tmp_path):
    data = np.zeros(_SHAPE, dtype=np.float32)
    data[31] = 1.0
    if family == "image":
        store = ImageMemoryStore(data=data, name="image")
        return controller.add_image(data=store, scene_id=scene.id)
    if family == "labels":
        store = LabelMemoryStore(data=data.astype(np.int32), name="labels")
        return controller.add_labels(data=store, scene_id=scene.id)
    store = multiscale_store(tmp_path, "multiscale", data)
    return controller.add_image_multiscale(data=store, scene_id=scene.id)


@pytest.mark.parametrize("position", [ROUNDING_POSITION, 31.0])
@pytest.mark.parametrize("family", ["image", "labels", "multiscale"])
async def test_a_slab_slice_draws_the_plane_at_the_slice_position(
    family, position, tmp_path
):
    p = ROUNDING_POSITION
    assert ((p - 1.0) + (p + 1.0)) / 2.0 != p  # the premise

    controller = CellierController(gui="offscreen")
    try:
        scene = controller.add_scene(
            coordinate_system=spatial_axes("z", "y", "x"), displayed_axes=("y", "x")
        )
        controller.add_canvas(scene.id)
        visual = _add(controller, scene, family, tmp_path)
        controller.update_thickness(scene.id, {0: 1.0})
        # Reslices the canvas, which raised here before the fix.
        controller.update_slice_indices(scene.id, {0: position})

        canvas_id = controller.get_canvas_ids(scene.id)[0]
        gfx = controller._render_manager._scenes[scene.id].get_visual(visual.id)
        requests = planned_requests_2d(
            gfx,
            camera_pos_world=np.zeros(3),
            viewport_width_px=100.0,
            world_width=10.0,
            view_min_world=None,
            view_max_world=None,
            dims_state=scene.dims.to_state(),
            selection=controller._selections_for_scene(scene.id)[canvas_id],
        )
        finest = [r for r in requests if r.scale_index == 0]
        assert finest
        assert {r.axis_selections[0] for r in finest} == {31}
    finally:
        controller.close()
