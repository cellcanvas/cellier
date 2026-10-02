"""Geometry visuals slice against the pulled-back region (design 3.12, 3.13).

The only phase that is deliberately **not** behaviour-preserving.  Today the
stores compare a **world** slice position directly against **data**
coordinates, and the thickness is hardcoded to ``0.5`` in units nobody
states -- so a points visual with a non-identity transform renders in the
right place and slices against the wrong numbers.

Two changes land together and both change what a user sees:

* points, lines, meshes and graphs now slice against the transform;
* thickness is a **world** half-thickness, so on a scaled axis the number of
  elements shown changes.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from cellier.render._spaces import (
    axis_scales,
    data_slice_positions,
    geometry_data_region,
)
from cellier.transform import (
    AffineTransform,
    Axis,
    ConvexRegion,
    DataCoordinateSystem,
    RegionSelection,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)


def _systems(names, types=None):
    types = types or ["space"] * len(names)
    axes = tuple(Axis(name=n, axis_type=t) for n, t in zip(names, types))
    return axes


def _selection(world, displayed, slabs):
    """A selection a canvas showing *displayed* would emit for *slabs*."""
    canvas_id = uuid4()
    rendered = RenderedCoordinateSystem.from_world(
        world, [world.axes[a].id for a in displayed], canvas_id
    )
    embedding = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={
            rendered.axes[i].id: world.axes[a].id for i, a in enumerate(displayed)
        },
        constant_output_axes={
            world.axes[a].id: float(slabs.get(a, (0.0, 0.0))[0])
            for a in range(world.ndim)
            if a not in displayed
        },
    )
    region = ConvexRegion.from_axis_slabs(
        world,
        {world.axes[a].id: value for a, value in slabs.items()},
    )
    return RegionSelection(transform=embedding, region=region)


# ---------------------------------------------------------------------------
# Design 3.12: zyx data in a ZYX world
# ---------------------------------------------------------------------------


@pytest.fixture
def points_3_12():
    """Six points and a 2 um z spacing, exactly as design 3.12 sets them up."""
    data = DataCoordinateSystem(name="d", axes=_systems("zyx"), datastore_id=uuid4())
    world = WorldCoordinateSystem(name="w", axes=_systems("ZYX"))
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={data.axes[i].id: world.axes[i].id for i in range(3)},
        scale={
            data.axes[0].id: 2.0,
            data.axes[1].id: 0.5,
            data.axes[2].id: 0.5,
        },
        translation={data.axes[0].id: 10.0},
    )
    positions = np.array(
        [
            [1.6, 10, 20],
            [2.0, 12, 22],
            [2.4, 14, 24],
            [2.6, 16, 26],
            [13.0, 18, 28],
            [14.0, 30, 30],
        ]
    )
    return data, world, transform, positions


def test_the_points_world_positions_are_what_the_design_says(points_3_12):
    _data, _world, transform, positions = points_3_12
    world_z = transform.map_coordinates(positions)[:, 0]
    np.testing.assert_allclose(world_z, [13.2, 14.0, 14.8, 15.2, 36.0, 38.0])


def test_the_region_selects_the_points_that_are_on_the_slice(points_3_12):
    """Design 3.12's answer: p0, p1 and p2 lie at world Z of 13.2, 14.0 and
    14.8, all within ``14 +- 1``."""
    _data, world, transform, positions = points_3_12
    selection = _selection(world, (1, 2), {0: (14.0, 1.0)})
    region = transform.imap_region(selection.region, world)
    np.testing.assert_allclose(region.normals, [[2.0, 0, 0], [-2.0, 0, 0]])
    np.testing.assert_allclose(region.offsets, [5.0, -3.0])
    assert np.flatnonzero(region.contains(positions)).tolist() == [0, 1, 2]


def test_what_the_old_comparison_did_instead(points_3_12):
    """The latent bug, stated.  The store compared the **world** value 14
    against **data** ``z`` with a data-unit half-thickness, which selects the
    one point 24 um away from the slice plane and hides the three on it."""
    _data, _world, _transform, positions = points_3_12
    lo, hi = 14.0 - 0.5, 14.0 + 0.5
    old = (positions[:, 0] >= lo) & (positions[:, 0] <= hi)
    assert np.flatnonzero(old).tolist() == [5]


@pytest.mark.parametrize(
    ("half_thickness", "expected"),
    [(0.5, [1]), (1.0, [0, 1, 2]), (2.0, [0, 1, 2, 3])],
)
def test_thickness_is_now_a_world_quantity(points_3_12, half_thickness, expected):
    """Design 3.12's sweep.  Today's ``0.5`` is in data units, so on this
    transform its world equivalent is 1.0 um -- which is why D4's world
    default of 0.5 changes what is shown, deliberately."""
    _data, world, transform, positions = points_3_12
    selection = _selection(world, (1, 2), {0: (14.0, half_thickness)})
    region = transform.imap_region(selection.region, world)
    assert np.flatnonzero(region.contains(positions)).tolist() == expected


def test_a_three_dimensional_view_has_no_slabs_and_keeps_everything(points_3_12):
    """Today that is "``slice_indices`` is empty so the mask loop does not
    run".  Now it is an unbounded region and an all-True ``contains``: the
    same outcome, one rule instead of two."""
    _data, world, transform, positions = points_3_12
    region = ConvexRegion.unbounded(world)
    assert not region.half_spaces
    pulled = transform.imap_region(region, world)
    assert pulled.contains(positions).all()


def test_a_plane_selects_only_what_lies_on_it(points_3_12):
    """The scene's thickness is the one knob: an axis nobody gave a thickness
    is a plane, and a continuous axis keeps pure containment.  No family adds
    a floor."""
    data, world, transform, positions = points_3_12
    # World Z = 14 is data z = 2, where p1 sits exactly.
    on_a_point = _selection(world, (1, 2), {0: (14.0, 0.0)})
    region = geometry_data_region(on_a_point, transform, world, data)
    assert np.flatnonzero(region.contains(positions)).tolist() == [1]
    between = _selection(world, (1, 2), {0: (15.0, 0.0)})
    region = geometry_data_region(between, transform, world, data)
    assert not region.contains(positions).any()


def test_a_thickness_the_user_asked_for_is_the_band(points_3_12):
    data, world, transform, positions = points_3_12
    # World Z in [14, 16] is data z in [2, 3]: p1, p2 and p3.
    selection = _selection(world, (1, 2), {0: (15.0, 1.0)})
    region = geometry_data_region(selection, transform, world, data)
    assert np.flatnonzero(region.contains(positions)).tolist() == [1, 2, 3]


def test_oblique_needs_no_further_work(points_3_12):
    """Design 3.12: points are the one family where oblique selection is free
    once the region is the transport -- no chunk grid to intersect and no
    rounding rule to generalise."""
    _data, world, transform, positions = points_3_12
    normal = np.array([1.0, 0.6, 0.0])
    normal = normal / np.linalg.norm(normal)
    oblique = ConvexRegion.from_plane_slab(
        world, normal=normal, offset=16.0, half_thickness=1.0
    )
    region = transform.imap_region(oblique, world)
    assert np.flatnonzero(region.contains(positions)).tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Design 3.13: tzyx data in a TCZYX world -- the broadcast case
# ---------------------------------------------------------------------------


@pytest.fixture
def points_3_13():
    """Tracked points with a time axis but no channel, in a world with both."""
    data = DataCoordinateSystem(
        name="d",
        axes=_systems("tzyx", ["time", "space", "space", "space"]),
        datastore_id=uuid4(),
    )
    world = WorldCoordinateSystem(
        name="w",
        axes=_systems("TCZYX", ["time", "channel", "space", "space", "space"]),
    )
    transform = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={
            data.axes[0].id: world.axes[0].id,
            data.axes[1].id: world.axes[2].id,
            data.axes[2].id: world.axes[3].id,
            data.axes[3].id: world.axes[4].id,
        },
        scale={
            data.axes[0].id: 0.5,
            data.axes[1].id: 2.0,
            data.axes[2].id: 0.5,
            data.axes[3].id: 0.5,
        },
        translation={data.axes[0].id: 0.25, data.axes[1].id: 10.0},
        broadcast_output_axes=[world.axes[1].id],
    )
    positions = np.array(
        [
            [1.0, 5, 10, 20],
            [1.5, 6, 12, 22],
            [2.0, 7, 14, 24],
            [3.0, 8, 16, 26],
            [4.0, 9, 18, 28],
            [0.0, 4, 8, 18],
        ]
    )
    return data, world, transform, positions


def test_the_transform_has_an_all_zero_broadcast_row(points_3_13):
    _data, world, transform, _positions = points_3_13
    assert transform.matrix.shape == (6, 5)
    np.testing.assert_array_equal(transform.matrix[1], np.zeros(5))
    assert transform.broadcast_axes == frozenset({world.axes[1].id})


def test_without_d8_the_points_would_disappear(points_3_13):
    """The failure D8 exists to prevent, stated as a fact about the region.

    The forward map places every point at world ``C = 0``, which is the
    matrix's answer and not the truth -- the truth is "these points exist at
    every C", which no matrix can express (D25).  Pulled back without the
    drop rule, ``C = 2`` becomes ``0 <= -1.5`` and the region is empty: the
    points visual vanishes the moment the channel slider leaves zero, while
    the image next to it renders normally.
    """
    _data, world, transform, positions = points_3_13
    region = ConvexRegion.from_axis_slabs(
        world,
        {world.axes[0].id: (1.0, 0.5), world.axes[1].id: (2.0, 0.5)},
    )
    # The internal form is what the drop rule is layered on; passing no
    # broadcast indices reproduces the pre-D8 behaviour exactly.
    naive = transform._imap_region(region, ())
    assert naive.is_empty()
    assert not naive.contains(positions).any()


def test_with_d8_the_channel_constraint_is_dropped(points_3_13):
    """Design 3.13's answer: p0, p1 and p2 are at world T of 0.75, 1.00 and
    1.25, all inside ``1.0 +- 0.5``, and their channel is irrelevant because
    they have none."""
    _data, world, transform, positions = points_3_13
    region = ConvexRegion.from_axis_slabs(
        world,
        {world.axes[0].id: (1.0, 0.5), world.axes[1].id: (2.0, 0.5)},
    )
    pulled = transform.imap_region(region, world).simplify()
    assert not pulled.is_empty()
    np.testing.assert_allclose(pulled.normals, [[0.5, 0, 0, 0], [-0.5, 0, 0, 0]])
    np.testing.assert_allclose(pulled.offsets, [1.25, -0.25])
    assert np.flatnonzero(pulled.contains(positions)).tolist() == [0, 1, 2]


def test_there_is_no_rounding_anywhere_on_this_path(points_3_13):
    """The image path takes the box and snaps the collapsed axes to voxel
    indices; points are continuous, so the constraints apply to the
    coordinates directly.  The two families share the region and diverge only
    in what they do with it."""
    _data, world, transform, positions = points_3_13
    region = ConvexRegion.from_axis_slabs(world, {world.axes[0].id: (1.0, 0.5)})
    pulled = transform.imap_region(region, world)
    # t in [0.5, 2.5] -- a point at t = 2.0 is in, one at t = 3.0 is out, and
    # nothing was snapped to an integer on the way.
    assert pulled.contains(positions[2])
    assert not pulled.contains(positions[3])


# ---------------------------------------------------------------------------
# The graph's per-family policy
# ---------------------------------------------------------------------------


def test_axis_scales_convert_a_world_thickness_into_data_units(points_3_12):
    """The graph keeps its asymmetric window and its fade; what changes is
    the space the numbers are in."""
    data, _world, transform, _positions = points_3_12
    assert axis_scales(transform) == {0: 2.0, 1: 0.5, 2: 0.5}
    assert data.ndim == 3


def test_the_slice_position_moves_into_data_coordinates(points_3_12):
    _data, world, transform, _positions = points_3_12
    selection = _selection(world, (1, 2), {0: (14.0, 0.0)})
    positions = data_slice_positions(selection.region, transform, world)
    # world Z = 14 on a ``2 z + 10`` axis is data z = 2.
    assert positions == {0: pytest.approx(2.0)}
