"""In-memory image and labels report the footprint of the data they draw.

Both visuals start on one- and two-voxel placeholder textures, and pygfx
derives a node's bounding box from its texture, so each node's box is pinned
to the data's footprint.  Two bugs, both found by the full affine hand checks
(2026-10-19) and neither a transform effect:

- **Labels framed their placeholder.**  The first visit to 3D fits the
  camera at once (``_switch_canvas_cameras``), before the 3D slice loads,
  and nothing refits afterwards; the labels nodes were not pinned, so a
  labels volume started in 2D and toggled to 3D was framed on an empty
  corner (H7).
- **The 2D box ignored the view.**  The image pinned its 2D box once, from
  the store's last two axes, but a ``(z, x)`` view's slice is ``Z x X``; and
  both visuals sized their outline wireframe only on the first upload, so a
  switch from ``(y, x)`` to ``(z, x)`` kept the old box (a hidden line still
  counts toward its group's box).

The boxes now follow the view's retained axes and each upload.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.convenience.capture import _wait_for_slicer
from cellier.data import LabelMemoryStore
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.transform import AffineTransform, Axis
from tests._v2 import data_system

_SHAPE = (6, 10, 14)  # (z, y, x)
_FAMILIES = ("image", "labels")


def _rotated_anisotropic() -> np.ndarray:
    """H7's transform: 30 degrees in ``(y, x)`` with ``y`` voxels 2x tall."""
    theta = np.deg2rad(30.0)
    linear = np.eye(3)
    linear[1:, 1:] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    linear = linear @ np.diag([1.0, 2.0, 1.0])
    centre = (np.asarray(_SHAPE, dtype=float) - 1.0) / 2.0
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = centre - linear @ centre
    return matrix


_MATRICES = {"identity": np.eye(4), "rotated": _rotated_anisotropic()}


def _viewer(family: str, displayed_axes, matrix=None):
    controller = CellierController(gui="offscreen")
    scene = controller.add_scene(
        coordinate_system=spatial_axes("z", "y", "x"), displayed_axes=displayed_axes
    )
    controller.add_canvas(scene.id, canvas_size=(200, 200))
    data = np.zeros(_SHAPE, dtype=np.float32)
    data[1:-1, 2:-2, 2:-2] = 1.0
    if family == "image":
        store = ImageMemoryStore(data=data, name="image")
    else:
        store = LabelMemoryStore(data=data.astype(np.int32), name="labels")
    controller._ensure_data_coordinate_systems(scene.id, store)
    transform = AffineTransform.from_matrix(
        np.eye(4) if matrix is None else matrix,
        store.data_coordinate_systems[0],
        scene.dims.world_coordinate_system,
        name="data_to_world",
    )
    add = controller.add_image if family == "image" else controller.add_labels
    add(data=store, scene_id=scene.id, transform=transform)
    return controller, scene


def _scene_box(controller, scene) -> np.ndarray:
    canvas_id = controller.get_canvas_ids(scene.id)[0]
    view = controller._render_manager._canvases[canvas_id]
    return view._get_scene_fn(scene.id).get_world_bounding_box()


def _world_box_xyz(matrix: np.ndarray) -> np.ndarray:
    """The data footprint ``[-0.5, n - 0.5]`` mapped to world, in pygfx order."""
    corners = np.array(
        [
            [z, y, x, 1.0]
            for z in (-0.5, _SHAPE[0] - 0.5)
            for y in (-0.5, _SHAPE[1] - 0.5)
            for x in (-0.5, _SHAPE[2] - 0.5)
        ]
    )
    world = (corners @ matrix.T)[:, 2::-1]
    return np.stack([world.min(axis=0), world.max(axis=0)])


#: A ``(z, x)`` view's box: rendered x is data x, rendered y is data z.
_ZX_BOX = [[-0.5, -0.5, 0.0], [_SHAPE[2] - 0.5, _SHAPE[0] - 0.5, 0.0]]


@pytest.mark.parametrize("kind", sorted(_MATRICES))
@pytest.mark.parametrize("family", _FAMILIES)
async def test_the_first_3d_fit_frames_the_whole_volume(family, kind):
    matrix = _MATRICES[kind]
    controller, scene = _viewer(family, ("y", "x"), matrix)
    try:
        controller.fit_camera(scene.id)
        controller.reslice_all()
        await _wait_for_slicer(controller)

        # The first visit to 3D fits here, before the 3D slice has loaded.
        controller.update_displayed_axes(scene.id, (0, 1, 2))
        np.testing.assert_allclose(
            _scene_box(controller, scene), _world_box_xyz(matrix), atol=1e-4
        )
        canvas_id = controller.get_canvas_ids(scene.id)[0]
        view = controller._render_manager._canvases[canvas_id]
        first_fit = np.array(view.camera.world.position)

        # Refitting once the data has landed changes nothing.
        await _wait_for_slicer(controller)
        controller.fit_camera(scene.id)
        np.testing.assert_allclose(view.camera.world.position, first_fit, atol=1e-4)
    finally:
        controller.close()


@pytest.mark.parametrize("family", _FAMILIES)
async def test_a_zx_view_box_is_z_by_x(family):
    """Not the store's last two axes, before the data lands or after."""
    controller, scene = _viewer(family, ("z", "x"))
    try:
        np.testing.assert_allclose(_scene_box(controller, scene), _ZX_BOX)
        controller.reslice_all()
        await _wait_for_slicer(controller)
        np.testing.assert_allclose(_scene_box(controller, scene), _ZX_BOX)
    finally:
        controller.close()


@pytest.mark.parametrize("family", _FAMILIES)
async def test_the_box_follows_a_switch_of_displayed_axes(family):
    """The outline wireframe was sized once, on the first upload."""
    controller, scene = _viewer(family, ("y", "x"))
    try:
        controller.reslice_all()
        await _wait_for_slicer(controller)
        controller.update_displayed_axes(scene.id, (0, 2))
        np.testing.assert_allclose(_scene_box(controller, scene), _ZX_BOX)
        await _wait_for_slicer(controller)
        np.testing.assert_allclose(_scene_box(controller, scene), _ZX_BOX)
    finally:
        controller.close()


async def test_a_composite_image_box_leaves_out_the_channel_axis():
    controller = CellierController(gui="offscreen")
    try:
        world = world_coordinate_system(
            (Axis(name="c", axis_type="channel"), *spatial_axes("z", "y", "x")),
            name="world",
        )
        scene = controller.add_scene(coordinate_system=world, displayed_axes=("z", "x"))
        controller.add_canvas(scene.id, canvas_size=(200, 200))
        store = ImageMemoryStore(
            data=np.ones((3, *_SHAPE), dtype=np.float32),
            name="image",
            data_coordinate_systems=[
                data_system(("c", "z", "y", "x"), sampling="discrete")
            ],
        )
        controller.add_image(store, scene.id, channel_axis=0, composite=True)
        controller.reslice_all()
        await _wait_for_slicer(controller)
        np.testing.assert_allclose(_scene_box(controller, scene), _ZX_BOX)
    finally:
        controller.close()
