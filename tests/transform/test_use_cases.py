"""End-to-end acceptance: every geometric operation in every use case.

All eight geometric operations work in all four use cases, with three cells
needing special handling.  Every cell of that table is asserted here.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.transform import (
    AffineTransform,
    Axis,
    AxisAlignedBoundingBox,
    ConvexRegion,
    DataCoordinateSystem,
    DegenerateNormalError,
    Plane,
    RegionSelection,
    RenderedCoordinateSystem,
    WorldCoordinateSystem,
)
from tests.transform._use_cases import MODEL_USE_CASES, channel_broadcast, scale_only

CASE_IDS = list(MODEL_USE_CASES)


def space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


# --- the table, row by row --------------------------------------------


@pytest.mark.parametrize("name", CASE_IDS)
def test_points_both_ways(name):
    _, _, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(0)
    points = rng.normal(size=(5, transform.input_ndim))
    world = transform.map_coordinates(points)
    assert world.shape == (5, transform.output_ndim)
    assert np.allclose(transform.imap_coordinates(world), points)


@pytest.mark.parametrize("name", CASE_IDS)
def test_directions_both_ways(name):
    _, _, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(5, transform.input_ndim))
    forward = transform.map_direction(vectors)
    assert np.allclose(transform.imap_direction(forward), vectors)


@pytest.mark.parametrize("name", CASE_IDS)
def test_normals_both_ways(name):
    _, _, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(2)
    normals = rng.normal(size=(5, transform.input_ndim))
    forward = transform.map_normal(normals)
    assert np.allclose(transform.imap_normal(forward), normals)


@pytest.mark.parametrize("name", CASE_IDS)
def test_planes_both_ways(name):
    data, world, transform = MODEL_USE_CASES[name]()
    rng = np.random.default_rng(3)
    normal = rng.normal(size=transform.input_ndim)
    plane = Plane(coordinate_system=data.id, normal=normal, offset=1.3)
    mapped = transform.map_plane(plane)
    assert mapped.coordinate_system == world.id
    back = transform.imap_plane(mapped)
    scale = np.linalg.norm(back.normal) / np.linalg.norm(plane.normal)
    assert np.allclose(back.normal, plane.normal * scale)
    assert back.offset == pytest.approx(plane.offset * scale)


@pytest.mark.parametrize("name", CASE_IDS)
def test_bounding_boxes_both_ways(name):
    data, world, transform = MODEL_USE_CASES[name]()
    ndim = transform.input_ndim
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.zeros(ndim),
        max_coordinate=np.arange(1.0, ndim + 1.0),
    )
    mapped = transform.map_bounding_box(box, world)
    assert mapped.coordinate_system == world.id
    back = transform.imap_bounding_box(mapped)
    # the round trip is a superset, never smaller
    assert np.all(back.min_coordinate <= box.min_coordinate + 1e-9)
    assert np.all(back.max_coordinate >= box.max_coordinate - 1e-9)


# --- the three cells the table marks as needing handling -------------


def test_channel_broadcast_reverse_normal_parallel_to_c_degenerates():
    """Section 4.2, 'normal w->d': degenerate if n is parallel to C."""
    _, _, transform = channel_broadcast()
    with pytest.raises(DegenerateNormalError):
        transform.imap_normal(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))


def test_channel_broadcast_reverse_plane_parallel_to_c_degenerates():
    """Section 4.2, 'plane w->d'."""
    _, world, transform = channel_broadcast()
    plane = Plane(
        coordinate_system=world.id, normal=[0.0, 1.0, 0.0, 0.0, 0.0], offset=3.0
    )
    with pytest.raises(DegenerateNormalError):
        transform.imap_plane(plane)


def test_channel_broadcast_forward_bounding_box_needs_broadcast_axes():
    """Section 4.2, 'bbox d->w': needs broadcast_axes."""
    data, world, transform = channel_broadcast()
    box = AxisAlignedBoundingBox(
        coordinate_system=data.id,
        min_coordinate=np.zeros(4),
        max_coordinate=np.ones(4),
    )
    with_declaration = transform.map_bounding_box(box, world)
    assert with_declaration.min_coordinate[1] == -np.inf
    assert with_declaration.max_coordinate[1] == np.inf

    undeclared = AffineTransform.from_matrix(transform.matrix, data, world)
    without = undeclared.map_bounding_box(box, world)
    assert without.min_coordinate[1] == without.max_coordinate[1] == 0.0


def test_channel_broadcast_forward_normal_has_a_zero_channel_component():
    """The world hyperplane contains the C direction, for free."""
    _, _, transform = channel_broadcast()
    mapped = transform.map_normal(np.array([0.3, -1.2, 0.7, 2.0]))
    assert mapped[1] == pytest.approx(0.0, abs=1e-12)


def test_channel_broadcast_reverse_plane_tilted_in_c_loses_the_tilt():
    """Not injective, and pinned as intended rather than fixed into an error."""
    _, world, transform = channel_broadcast()
    plane = Plane(
        coordinate_system=world.id, normal=[0.0, 1.0, 0.0, 0.0, 1.0], offset=0.0
    )
    pulled = transform.imap_plane(plane)
    remapped = transform.map_plane(pulled)
    assert remapped.normal[1] == pytest.approx(0.0)
    assert not np.allclose(remapped.normal, plane.normal)


# --- the rendered coordinate system, end to end ----------------------


def tczyx():
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space("Z"),
            space("Y"),
            space("X"),
        )
    )


def embedding(world, displayed, slices):
    rendered = RenderedCoordinateSystem.from_world(world, displayed, uuid4())
    transform = AffineTransform.from_axis_map(
        rendered,
        world,
        axis_map={name: name for name in displayed},
        constant_output_axes=slices,
        name="rendered to world",
    )
    return rendered, transform


def test_a_3d_render_of_a_tczyx_world():
    world = tczyx()
    rendered, transform = embedding(world, ("Z", "Y", "X"), {"T": 7.0, "C": 2.0})

    assert rendered.ndim == 3
    assert transform.matrix.shape == (6, 4)
    inverse = transform.inverse()
    assert inverse is not None
    assert inverse.input_coordinate_system == world.id
    assert inverse.output_coordinate_system == rendered.id

    point = np.array([3.0, 4.0, 5.0])
    assert np.allclose(transform.map_coordinates(point), [7.0, 2.0, 3.0, 4.0, 5.0])
    world_point = np.array([7.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(transform.imap_coordinates(world_point), point)


def test_a_2d_render_of_a_tczyx_world():
    world = tczyx()
    rendered, transform = embedding(world, ("Z", "Y"), {"T": 7.0, "C": 2.0, "X": 1.0})
    assert rendered.ndim == 2
    assert transform.matrix.shape == (6, 3)
    point = np.array([3.0, 4.0])
    assert np.allclose(transform.map_coordinates(point), [7.0, 2.0, 3.0, 4.0, 1.0])
    assert np.allclose(
        transform.imap_coordinates(transform.map_coordinates(point)), point
    )


def test_the_projection_drops_the_sliced_axes():
    world = tczyx()
    _, transform = embedding(world, ("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    inverse = transform.inverse()
    assert np.allclose(inverse.linear[:, :2], 0.0)


def test_a_world_point_off_the_slice_still_maps_onto_it():
    world = tczyx()
    _, transform = embedding(world, ("Z", "Y", "X"), {"T": 7.0, "C": 2.0})
    on_slice = np.array([7.0, 2.0, 3.0, 4.0, 5.0])
    off_slice = np.array([99.0, -4.0, 3.0, 4.0, 5.0])
    assert np.allclose(
        transform.imap_coordinates(on_slice), transform.imap_coordinates(off_slice)
    )


# --- composing data -> world with the inverse of rendered -> world ---


def test_data_to_rendered_composes_and_behaves():
    """The chain reads data -> world <- rendered; compose through the inverse."""
    data, world, data_to_world = scale_only()
    rendered, rendered_to_world = embedding(world, ("Z", "Y", "X"), {})
    world_to_rendered = rendered_to_world.inverse()
    assert world_to_rendered is not None

    data_to_rendered = data_to_world.then(world_to_rendered, world, rendered)
    assert data_to_rendered.input_coordinate_system == data.id
    assert data_to_rendered.output_coordinate_system == rendered.id
    assert data_to_rendered.matrix.shape == (4, 4)

    point = np.array([1.0, 2.0, 3.0])
    assert np.allclose(
        data_to_rendered.map_coordinates(point),
        world_to_rendered.map_coordinates(data_to_world.map_coordinates(point)),
    )
    assert np.allclose(
        data_to_rendered.imap_coordinates(data_to_rendered.map_coordinates(point)),
        point,
    )


def test_a_world_selection_pulls_back_into_data_space():
    """The operation the slicer runs, end to end."""
    data, world, data_to_world = channel_broadcast()
    selection = RegionSelection(
        transform=embedding(world, ("Z", "Y", "X"), {"T": 7.0, "C": 0.0})[1],
        region=ConvexRegion.from_axis_slabs(
            world, {"T": (7.0, 0.5), "C": (0.0, 0.5), "Z": (12.0, 1.0)}
        ),
    )
    in_data = data_to_world.imap_region(selection.region, world).simplify()
    assert in_data.coordinate_system == data.id
    assert in_data.ndim == 4
    assert not in_data.is_empty()

    # the two constraints on the broadcast axis were dropped by the drop rule
    # before A^T was applied, so they never reach the pulled-back region
    assert len(selection.region.half_spaces) == 6
    assert len(in_data.half_spaces) == 4

    box = in_data.bounding_box()
    assert box.min_coordinate[0] == pytest.approx(6.5)
    assert box.max_coordinate[0] == pytest.approx(7.5)
    assert box.min_coordinate[1] == pytest.approx(2.0)
    assert box.max_coordinate[1] == pytest.approx(6.0)
    assert box.min_coordinate[2] == -np.inf


def test_bounding_the_displayed_axes_culls_to_the_viewport():
    """A displayed axis may be bounded: "only fetch what is on screen".

    A region may bound **any** axis, displayed ones included.  Bounding
    the displayed axes is how a selection expresses a viewport crop, and
    it is the same argument as for not collapsing a region to a box
    early -- on a different axis and with a much bigger number.
    """
    volume = 1000.0  # the datastore is 1000^3 voxels at 0.1 um/voxel
    world = tczyx()
    data = DataCoordinateSystem(
        name="volume",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            space("z"),
            space("y"),
            space("x"),
        ),
    )
    data_to_world = AffineTransform.from_axis_map(
        data,
        world,
        axis_map={"t": "T", "z": "Z", "y": "Y", "x": "X"},
        scale={"z": 0.1, "y": 0.1, "x": 0.1},
        broadcast_output_axes=["C"],
    )

    def voxels(region):
        box = data_to_world.imap_region(region, world).simplify().bounding_box()
        lower = np.where(np.isneginf(box.min_coordinate), 0.0, box.min_coordinate)
        upper = np.where(np.isposinf(box.max_coordinate), volume, box.max_coordinate)
        return float(np.prod((upper - lower)[1:]))

    slice_only = ConvexRegion.from_axis_slabs(world, {"T": (7.0, 0.5), "C": (0.0, 0.5)})
    # the camera shows a 20x20 um window through a 4 um slab
    with_viewport = ConvexRegion.from_axis_slabs(
        world,
        {
            "T": (7.0, 0.5),
            "C": (0.0, 0.5),
            "Z": (50.0, 2.0),
            "Y": (50.0, 10.0),
            "X": (50.0, 10.0),
        },
    )

    assert voxels(slice_only) == 1_000_000_000
    assert voxels(with_viewport) == 40 * 200 * 200
    assert voxels(slice_only) / voxels(with_viewport) == pytest.approx(625.0)

    # and the selection that carries it validates
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4())
    assert RegionSelection(
        transform=AffineTransform.from_axis_map(
            rendered,
            world,
            axis_map={"Z": "Z", "Y": "Y", "X": "X"},
            constant_output_axes={"T": 7.0, "C": 0.0},
        ),
        region=with_viewport,
    )
