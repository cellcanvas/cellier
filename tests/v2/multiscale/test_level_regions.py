"""The region is pulled back level by level, through each level's own transform.

Design 3.11 A.  What this replaces is ``_build_world_to_level_transforms`` and
the precomputed ``inv_level_k @ inv_visual`` list it held -- and with it a
``2 ** (level - 1)`` assumption applied to every axis, which is the class of
bug this repo has already paid for once on a pyramid whose ``z`` was not
downsampled.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellier.render._spaces import build_render_spaces
from cellier.render.visuals._slicing import (
    axis_selections_from_box,
    round_world_to_voxel,
)
from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    DataCoordinateSystem,
    RegionSelection,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)

# Design 3.11's setup: three levels, ``z`` NOT downsampled.
_LEVEL_SHAPES = [(10, 64, 256, 256), (10, 64, 128, 128), (10, 64, 64, 64)]
_LABELS = ("t", "z", "y", "x")
_TYPES = ("time", "space", "space", "space")


def _axes():
    return tuple(
        Axis(name=name, axis_type=kind, unit=None if kind == "time" else "micrometer")
        for name, kind in zip(_LABELS, _TYPES)
    )


@pytest.fixture
def pyramid():
    """An anisotropic three-level pyramid, with its systems."""
    from uuid import uuid4

    store_id = uuid4()
    levels = [
        DataCoordinateSystem(name=f"level{k}", axes=_axes(), datastore_id=store_id)
        for k in range(3)
    ]
    level_transforms = [
        AffineTransform.from_axis_map(
            levels[k],
            levels[0],
            axis_map={levels[k].axes[i].id: levels[0].axes[i].id for i in range(4)},
            scale={
                levels[k].axes[0].id: 1.0,
                levels[k].axes[1].id: 1.0,  # z is not downsampled
                levels[k].axes[2].id: float(2**k),
                levels[k].axes[3].id: float(2**k),
            },
            translation={
                levels[k].axes[2].id: (2**k - 1) / 2,
                levels[k].axes[3].id: (2**k - 1) / 2,
            },
        )
        for k in range(3)
    ]
    world = WorldCoordinateSystem(name="world", axes=_axes())
    # 1.0 um z spacing, 0.1 um in plane, and 3.7's awkward time axis
    # (``T = 0.5 t + 0.25``) so the rounding step has something to do.
    data_to_world = AffineTransform.from_axis_map(
        levels[0],
        world,
        axis_map={levels[0].axes[i].id: world.axes[i].id for i in range(4)},
        scale={
            levels[0].axes[0].id: 0.5,
            levels[0].axes[1].id: 1.0,
            levels[0].axes[2].id: 0.1,
            levels[0].axes[3].id: 0.1,
        },
        translation={levels[0].axes[0].id: 0.25},
    )
    canvas_id = uuid4()
    rendered = RenderedCoordinateSystem.from_world(
        world, [world.axes[a].id for a in (1, 2, 3)], canvas_id
    )
    rendered_to_world = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={
            rendered.axes[i].id: world.axes[a].id for i, a in enumerate((1, 2, 3))
        },
        constant_output_axes={world.axes[0].id: 1.0},
    )
    visual = VisualCoordinateSystem.from_data(
        levels[0], [levels[0].axes[a].id for a in (1, 2, 3)], uuid4()
    )
    spaces = build_render_spaces(
        levels[0],
        visual,
        world,
        rendered,
        rendered_to_world,
        data_to_world,
        (1, 2, 3),
        data_levels=levels,
        level_transforms=level_transforms,
    )
    region = ConvexRegion.from_axis_slabs(world, {world.axes[0].id: (1.0, 0.0)})
    selection = RegionSelection(transform=rendered_to_world, region=region)
    return spaces, data_to_world, selection


def _level_boxes(pyramid):
    spaces, data_to_world, selection = pyramid
    level0 = data_to_world.imap_region(selection.region, spaces.world)
    return [
        transform.imap_region(level0, spaces.data).simplify().bounding_box()
        for transform in spaces.level_transforms
    ]


def test_every_level_agrees_because_t_is_not_downsampled(pyramid):
    """Design 3.11 A's numbers, exactly: ``t`` bbox ``[1.5, 1.5]`` at every
    level, voxel 2 at every level.  The per-level answer falls out of the
    per-level transform, not from a power-of-two rule."""
    for level, box in enumerate(_level_boxes(pyramid)):
        assert box.min_coordinate[0] == pytest.approx(1.5)
        assert box.max_coordinate[0] == pytest.approx(1.5)
        assert round_world_to_voxel(box.min_coordinate[0], _LEVEL_SHAPES[level][0]) == 2


def test_a_downsampled_axis_would_not_have_agreed(pyramid):
    """The contrast that makes the previous test mean something: an axis the
    pyramid *does* downsample gives a different voxel index per level, and a
    single ``2 ** (level - 1)`` factor applied to every axis is exactly what
    gets that wrong on an anisotropic pyramid."""
    spaces, data_to_world, _ = pyramid
    world = spaces.world
    # Bound y instead, which halves per level.
    region = ConvexRegion.from_axis_slabs(world, {world.axes[2].id: (12.8, 0.0)})
    level0 = data_to_world.imap_region(region, world)
    positions = [
        transform.imap_region(level0, spaces.data)
        .simplify()
        .bounding_box()
        .min_coordinate[2]
        for transform in spaces.level_transforms
    ]
    assert positions[0] == pytest.approx(128.0)
    assert positions[1] == pytest.approx(63.75)
    assert positions[2] == pytest.approx(31.625)


def test_the_displayed_axes_come_from_the_brick_window(pyramid):
    """Design 3.11 D: the displayed axes and the collapsed axes come from
    different places, and always did.  The window is LOD's and culling's; the
    collapsed axis is the region's."""
    box = _level_boxes(pyramid)[1]
    windows = {1: (-1, 33), 2: (95, 129), 3: (127, 161)}
    selections = axis_selections_from_box(box, _LEVEL_SHAPES[1], windows)
    assert selections == (2, (-1, 33), (95, 129), (127, 161))


def test_out_of_range_window_coordinates_are_passed_through(pyramid):
    """Negative and past-the-end coordinates are expected: the store clamps
    and zero-pads, as it did before this migration."""
    box = _level_boxes(pyramid)[2]
    selections = axis_selections_from_box(
        box, _LEVEL_SHAPES[2], {1: (-1, 33), 2: (31, 65), 3: (63, 97)}
    )
    assert selections == (2, (-1, 33), (31, 65), (63, 97))


def test_the_pull_back_needs_no_inverse(pyramid):
    """``imap_region`` is ``A^T`` on the normals and ``d - n . t`` on the
    offsets, so it works on a transform whose ``inverse()`` is ``None`` -- which
    is what a broadcast ``data -> world`` has."""
    spaces, data_to_world, selection = pyramid
    assert data_to_world.imap_region(selection.region, spaces.world) is not None
    for transform in spaces.level_transforms:
        assert transform.inverse() is not None  # these happen to be invertible
    # The operation itself never asks.
    level0 = data_to_world.imap_region(selection.region, spaces.world)
    assert not level0.is_empty()


# ---------------------------------------------------------------------------
# Site C: the camera composition
# ---------------------------------------------------------------------------


def test_the_camera_composition_matches_the_three_step_dance(pyramid):
    """The camera's level-k position, checked against what it replaces.

    What it replaced: ``select_axes`` + ``[[2, 1, 0]]`` + ``imap_coordinates``
    + ``[[2, 1, 0]]``.  Now: one named reversal and one ``map_coordinates``.
    The composed transform also returns the collapsed ``t`` -- the ``t`` the
    camera is looking at -- which is harmless and dropped.

    The comparison is against the square submatrix and an explicit inverse --
    the same three steps, written out.
    """
    from cellier.render._spaces import (
        cellier_to_pygfx_order,
        pygfx_to_cellier_order,
    )
    from cellier.render.visuals._image import _displayed_submatrix

    spaces, data_to_world, _ = pyramid
    camera_pygfx = np.array([[12.0, 9.0, 40.0]])

    chain = spaces.rendered_to_world.then(
        data_to_world.inverse(), spaces.world, spaces.data
    )
    level0 = chain.map_coordinates(pygfx_to_cellier_order(camera_pygfx))
    new = cellier_to_pygfx_order(level0[:, list(spaces.retained_axes)]).flatten()

    sub_3d = _displayed_submatrix(data_to_world, (1, 2, 3))
    camera_homogeneous = np.append(camera_pygfx[0, [2, 1, 0]], 1.0)
    old = (np.linalg.inv(sub_3d) @ camera_homogeneous)[:3][[2, 1, 0]]

    np.testing.assert_allclose(new, old, rtol=1e-6)
    # Design 3.11 C's printed values.
    np.testing.assert_allclose(level0.flatten(), [1.5, 40.0, 90.0, 120.0])
    np.testing.assert_allclose(new, [120.0, 90.0, 40.0])


def test_the_reversal_is_named_and_is_its_own_inverse():
    """The flip stays at the pygfx boundary, in named helpers, so moving
    it later is a change to two functions rather than a hunt for
    ``[[2, 1, 0]]``."""
    from cellier.render._spaces import (
        cellier_to_pygfx_order,
        pygfx_to_cellier_order,
    )

    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(pygfx_to_cellier_order(points), points[:, ::-1])
    np.testing.assert_array_equal(
        cellier_to_pygfx_order(pygfx_to_cellier_order(points)), points
    )
    # It works on a 2-vector too, which is the 2D canvas case.
    flat = np.array([[7.0, 8.0]])
    np.testing.assert_array_equal(pygfx_to_cellier_order(flat), [[8.0, 7.0]])
