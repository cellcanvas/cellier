"""``BaseVisual.transform`` is a v2 ``data -> world`` transform (Phase 3).

D18 forbids a coordinate-system-less identity, so the model field has no
default: a visual built before it reaches a scene does not yet know which
world it is going into.  ``CellierController.add_visual`` decides, and these
tests pin what it decides.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.controller import CellierController
from cellier.data._axes import (
    default_data_to_world,
    scale_and_translation_transform,
)
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.render.visuals._image import _norm_size_from_transform
from cellier.scene.dims import spatial_axes
from cellier.transform import AffineTransform
from cellier.visuals._image_memory import ImageVisual, InMemoryImageAppearance
from tests._displayed import trailing_axes
from tests._v2 import bound, scale_and_translation, systems


def _appearance():
    return InMemoryImageAppearance(color_map="gray", clim=(0.0, 1.0))


def _controller_and_scene(axes=None):
    controller = CellierController(gui="offscreen")
    axes = axes or spatial_axes("z", "y", "x")
    scene = controller.add_scene(
        coordinate_system=axes, displayed_axes=trailing_axes(axes, "3d")
    )
    return controller, scene


# ---------------------------------------------------------------------------
# The default
# ---------------------------------------------------------------------------


def test_an_unplaced_visual_has_no_transform():
    visual = ImageVisual(
        name="i",
        data_store_id=str(ImageMemoryStore(data=np.zeros((2, 2, 2))).id),
        appearance=_appearance(),
    )
    assert visual.transform is None


def test_add_visual_builds_the_identity_between_the_two_systems():
    controller, scene = _controller_and_scene()
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    visual = controller.add_image(
        data=store, scene_id=scene.id, appearance=_appearance()
    )
    transform = visual.transform
    assert isinstance(transform, AffineTransform)
    assert transform.input_coordinate_system == store.data_coordinate_system.id
    assert transform.output_coordinate_system == scene.dims.world_coordinate_system.id
    np.testing.assert_array_equal(transform.matrix, np.eye(4))


def test_a_sub_rank_dataset_is_broadcast_over_the_leading_world_axes():
    """Design 3.8 / D25: "exists at every T" is not the same claim as "sits at
    T = 0", and a zero row alone cannot tell them apart."""
    data, _ = systems(3, ("z", "y", "x"))
    _, world = systems(4, ("t", "z", "y", "x"))
    transform = default_data_to_world(data, world)
    assert transform.matrix.shape == (5, 4)
    assert transform.broadcast_axes == frozenset({world.axes[0].id})


def test_a_wider_dataset_projects_its_leading_axes_away():
    """A multichannel store's channel axis is composited by the visual and
    never reaches the world.  ``from_axis_map`` cannot express a dropped input
    axis, so the matrix is built directly."""
    data, _ = systems(4, ("c", "z", "y", "x"))
    _, world = systems(3, ("z", "y", "x"))
    transform = default_data_to_world(data, world)
    assert transform.matrix.shape == (4, 5)
    # The channel column is zero: nothing in the world moves with it.
    np.testing.assert_array_equal(transform.linear[:, 0], np.zeros(3))


# ---------------------------------------------------------------------------
# Naming the endpoints of a bare scale and offset
# ---------------------------------------------------------------------------
#
# ``transform_from_v1`` did this for a v1 matrix and went with v1 in Phase 8.
# What is left is the case that was never a migration affordance: a store that
# states its geometry as raw numbers -- a geff file's per-axis scale and
# offset (D23) -- and cannot name the scene's world itself.


def test_a_scale_and_offset_gets_its_endpoints_named():
    data, world = systems(3, ("z", "y", "x"))
    transform = scale_and_translation_transform(
        data, world, (2.0, 3.0, 4.0), (1.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(np.diag(transform.matrix), [2.0, 3.0, 4.0, 1.0])
    np.testing.assert_allclose(transform.translation, [1.0, 0.0, 0.0])
    assert transform.input_coordinate_system == data.id
    assert transform.output_coordinate_system == world.id


def test_a_scale_of_the_wrong_rank_is_refused_with_a_reason():
    data, world = systems(3, ("z", "y", "x"))
    with pytest.raises(ValueError, match="from_axis_map"):
        scale_and_translation_transform(data, world, (1.0, 1.0, 1.0, 1.0))


def test_a_transform_built_against_other_systems_is_refused():
    """A v2 transform names its endpoints, so a foreign one describes a
    different space -- and composing it would silently join two worlds."""
    controller, scene = _controller_and_scene()
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    foreign = scale_and_translation((2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="names its endpoints"):
        controller.add_image(
            data=store,
            scene_id=scene.id,
            appearance=_appearance(),
            transform=foreign,
        )


def test_a_transform_built_against_the_right_systems_is_kept():
    controller, scene = _controller_and_scene()
    store = ImageMemoryStore(data=np.zeros((4, 5, 6), dtype=np.float32))
    transform = bound(controller, scene.id, store, (2.0, 2.0, 2.0))
    visual = controller.add_image(
        data=store,
        scene_id=scene.id,
        appearance=_appearance(),
        transform=transform,
    )
    assert visual.transform is transform


# ---------------------------------------------------------------------------
# The shader helpers, redefined for a non-square transform (P10 / F0.6).
# The rotation guard and its three tests went with full affine design v2,
# Phase 4: the brick shaders take any affine (K3, K5).
# ---------------------------------------------------------------------------


def test_the_norm_size_matches_what_the_column_norms_gave():
    """The numbers are unchanged: on a diagonal transform the single entry per
    displayed axis is the column norm the old ``select_axes`` path computed."""
    transform = scale_and_translation((2.0, 0.5, 0.5), labels=("z", "y", "x"))
    dataset_size_xyz = np.array([256.0, 256.0, 64.0])
    norm = _norm_size_from_transform(transform, (0, 1, 2), dataset_size_xyz)
    # physical extent (x, y, z) = (128, 128, 128) -> all equal, longest is 1.
    np.testing.assert_allclose(norm, [1.0, 1.0, 1.0])


def test_the_norm_size_is_well_defined_on_a_non_square_transform():
    data, _ = systems(3, ("z", "y", "x"))
    _, world = systems(4, ("t", "z", "y", "x"))
    transform = default_data_to_world(data, world)
    norm = _norm_size_from_transform(transform, (1, 2, 3), np.array([64.0, 64.0, 64.0]))
    np.testing.assert_allclose(norm, [1.0, 1.0, 1.0])
