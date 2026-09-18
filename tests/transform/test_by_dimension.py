"""ByDimensionTransform (implementation plan, Phase 4).

The container that lets one axis be irregular while the rest stay affine.
The running scene is the design's worked example: a TZYX labels volume into
a TCZYX world, broadcasting over C.
"""

from uuid import uuid4

import numpy as np
import pytest

from cellier.transform import (
    AffineTransform,
    Axis,
    AxisAlignedBoundingBox,
    AxisCoordinates,
    ByDimensionTransform,
    CoordinateSystem,
    DataCoordinateSystem,
    NonAffineTransformError,
    NonUniformAxisTransform,
    WorldCoordinateSystem,
)

TIMES = tuple(float(t) for t in [0, 1, 2, 3, 4, 4.5, 5.0, 5.5, 6.0, 7, 8, 10, 12])


def _space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


@pytest.fixture
def scene():
    """TZYX labels -> TCZYX world, the design's worked example."""
    labels = DataCoordinateSystem(
        name="labels",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="frame"),
            _space("z"),
            _space("y"),
            _space("x"),
        ),
    )
    world = WorldCoordinateSystem(
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            Axis(name="c", axis_type="channel"),
            _space("z"),
            _space("y"),
            _space("x"),
        )
    )
    return labels, world


@pytest.fixture
def leaf(scene):
    labels, world = scene
    return NonUniformAxisTransform(
        coordinates=AxisCoordinates(values=TIMES),
        input_coordinate_system=CoordinateSystem(name="t", axes=(labels.axes[0],)).id,
        output_coordinate_system=CoordinateSystem(name="t", axes=(world.axes[0],)).id,
    )


@pytest.fixture
def transform(scene, leaf):
    labels, world = scene
    return ByDimensionTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        broadcast_output_axes=("c",),
        axis_transforms={"t": leaf},
    )


# -- construction ------------------------------------------------------------


def test_the_blocks_partition_both_axis_sets(transform):
    assert transform.input_ndim == 4
    assert transform.output_ndim == 5
    input_axes = sorted(a for b in transform.blocks for a in b.input_axes)
    output_axes = sorted(a for b in transform.blocks for a in b.output_axes)
    assert input_axes == [0, 1, 2, 3]
    assert output_axes == [0, 1, 2, 3, 4]


def test_it_validates_against_the_systems_it_names(transform, scene):
    labels, world = scene
    transform.validate_against(labels, world)


def test_the_broadcast_axis_rides_in_the_affine_block(transform):
    """A (z,y,x) -> (c,z,y,x) block: non-square, which is normal here."""
    affine_block = next(b for b in transform.blocks if len(b.input_axes) == 3)
    assert affine_block.input_axes == (1, 2, 3)
    assert affine_block.output_axes == (1, 2, 3, 4)


def test_a_scale_beside_a_per_axis_transform_raises(scene, leaf):
    labels, world = scene
    with pytest.raises(ValueError, match="two answers to one question"):
        ByDimensionTransform.from_axis_map(
            labels,
            world,
            axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
            scale={"t": 2.0, "z": 2.0},
            broadcast_output_axes=("c",),
            axis_transforms={"t": leaf},
        )


def test_even_a_no_op_scale_raises(scene, leaf):
    """1.0 is arithmetically nothing, but the shape of the call is wrong."""
    labels, world = scene
    with pytest.raises(ValueError, match="two answers to one question"):
        ByDimensionTransform.from_axis_map(
            labels,
            world,
            axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
            scale={"t": 1.0},
            broadcast_output_axes=("c",),
            axis_transforms={"t": leaf},
        )


def test_a_translation_beside_a_per_axis_transform_raises(scene, leaf):
    labels, world = scene
    with pytest.raises(ValueError, match="two answers to one question"):
        ByDimensionTransform.from_axis_map(
            labels,
            world,
            axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
            translation={"t": 3.0},
            broadcast_output_axes=("c",),
            axis_transforms={"t": leaf},
        )


# -- routing: both directions, because upstream fails silently one way -------


def test_mapping_forward_through_a_non_square_transform(transform):
    """The expanding direction, where ByDimension.apply raises IndexError."""
    mapped = transform.map_coordinates(np.array([6.4, 3.0, 4.0, 5.0]))
    assert mapped == pytest.approx([5.2, 0.0, 6.0, 4.0, 5.0])


def test_mapping_back_through_a_non_square_transform(transform):
    """The contracting direction, where ByDimension.apply is SILENTLY wrong.

    It would return a 5-column array with an uninitialised trailing value.
    This is the direction the whole pull-back path runs on, which is why
    both directions are pinned rather than just the forward one.
    """
    mapped = transform.imap_coordinates(np.array([5.2, 0.0, 6.0, 4.0, 5.0]))
    assert mapped.shape == (4,)
    assert mapped == pytest.approx([6.4, 3.0, 4.0, 5.0])


def test_the_round_trip_closes(transform):
    point = np.array([[6.4, 3.0, 4.0, 5.0], [2.0, 1.0, 1.0, 1.0]])
    mapped = transform.map_coordinates(point)
    assert transform.imap_coordinates(mapped) == pytest.approx(point)


def test_an_integer_input_is_not_truncated(transform):
    """empty_like would inherit int64 and truncate 4.5 to 4."""
    mapped = transform.map_coordinates(np.array([[5, 3, 4, 5]]))
    assert mapped.dtype == np.float64
    assert mapped[0, 0] == pytest.approx(4.5)


def test_a_float32_input_is_not_narrowed(transform):
    point = np.array([[6.4, 3.0, 4.0, 5.0]], dtype=np.float32)
    mapped = transform.map_coordinates(point)
    assert mapped.dtype == np.float64


def test_rank_is_preserved(transform):
    assert transform.map_coordinates(np.zeros(4)).shape == (5,)
    assert transform.map_coordinates(np.zeros((3, 4))).shape == (3, 5)


# -- bounding boxes ----------------------------------------------------------


def test_an_axis_aligned_slab_pulls_back_exactly(transform, scene):
    """The design's hand-computed numbers: [4.7, 5.7] s -> [5.4, 7.4] frames."""
    _, world = scene
    box = AxisAlignedBoundingBox(
        coordinate_system=world.id,
        min_coordinate=np.array([4.7, -np.inf, -np.inf, -np.inf, -np.inf]),
        max_coordinate=np.array([5.7, np.inf, np.inf, np.inf, np.inf]),
    )
    pulled = transform.imap_bounding_box(box)
    assert pulled.min_coordinate[0] == pytest.approx(5.4)
    assert pulled.max_coordinate[0] == pytest.approx(7.4)


def test_a_box_in_the_wrong_system_is_caught(transform, scene):
    labels, _ = scene
    box = AxisAlignedBoundingBox(
        coordinate_system=labels.id,
        min_coordinate=np.zeros(5),
        max_coordinate=np.ones(5),
    )
    with pytest.raises(ValueError, match="coordinate system"):
        transform.imap_bounding_box(box)


# -- affine-ness -------------------------------------------------------------


def test_a_non_affine_block_makes_the_whole_thing_non_affine(transform):
    assert transform.to_affine() is None


def test_an_all_affine_container_equals_the_plain_affine(scene):
    """Block-diagonal assembly must reproduce the one-matrix answer."""
    labels, world = scene
    common = {
        "axis_map": {"t": "t", "z": "z", "y": "y", "x": "x"},
        "scale": {"t": 0.5, "z": 2.0, "y": 1.0, "x": 1.0},
        "translation": {"z": 3.0},
        "broadcast_output_axes": ("c",),
    }
    by_dimension = ByDimensionTransform.from_axis_map(labels, world, **common)
    plain = AffineTransform.from_axis_map(labels, world, **common)

    assembled = by_dimension.to_affine()
    assert assembled is not None
    assert assembled.matrix == pytest.approx(plain.matrix)

    point = np.array([6.0, 3.0, 4.0, 5.0])
    assert by_dimension.map_coordinates(point) == pytest.approx(
        plain.map_coordinates(point)
    )


# -- restriction -------------------------------------------------------------


def test_restricting_the_non_uniform_axis_leaves_an_affine(transform, scene):
    """The check that could have sunk the design: the GPU still gets a matrix."""
    labels, _ = scene
    restricted = transform.restrict({0: 6.4}, labels)
    affine = restricted.to_affine()
    assert affine is not None
    assert affine.matrix.shape == (6, 4)


def test_the_restricted_transform_agrees_with_the_full_one(transform, scene):
    labels, _ = scene
    restricted = transform.restrict({0: 6.4}, labels)
    full = transform.map_coordinates(np.array([6.4, 3.0, 4.0, 5.0]))
    assert restricted.map_coordinates(np.array([3.0, 4.0, 5.0])) == pytest.approx(full)


def test_a_fully_fixed_block_is_evaluated_not_restricted(transform, scene):
    """Evaluating is what makes a sliced non-uniform axis free at the boundary."""
    labels, _ = scene
    restricted = transform.restrict({0: 6.4}, labels)
    # The T output row is now a constant equal to the leaf evaluated there.
    assert restricted.translation[0] == pytest.approx(5.2)


def test_restricting_nothing_returns_the_same_transform(transform):
    assert transform.restrict({}) is transform


def test_a_free_non_affine_block_refuses_and_names_the_axis(transform, scene):
    """Displaying a non-uniform axis has no 4x4 matrix -- the honest outcome."""
    labels, _ = scene
    with pytest.raises(NonAffineTransformError, match="stop displaying axis 't'"):
        transform.restrict({1: 3.0}, labels)


def test_restrict_does_not_round(transform, scene):
    """6.4 stays 6.4; rounding already happened one layer up."""
    labels, _ = scene
    restricted = transform.restrict({0: 6.4}, labels)
    assert restricted.translation[0] == pytest.approx(5.2)
    assert restricted.translation[0] != pytest.approx(5.0)


# -- the Jacobian refusals ---------------------------------------------------


def test_a_direction_refuses_and_names_the_axis(transform):
    with pytest.raises(NonAffineTransformError, match=r"\['t'\]"):
        transform.map_direction(np.array([1.0, 0.0, 0.0, 0.0]))


def test_an_all_affine_container_answers_directions(scene):
    labels, world = scene
    by_dimension = ByDimensionTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0},
        broadcast_output_axes=("c",),
    )
    mapped = by_dimension.map_direction(np.array([1.0, 1.0, 0.0, 0.0]))
    assert mapped == pytest.approx([1.0, 0.0, 2.0, 0.0, 0.0])


# -- inversion and equality --------------------------------------------------


def test_inverse_is_none_when_a_block_cannot_invert(transform):
    """The leaf's inverse is not itself a table, so it reports None."""
    assert transform.inverse() is None


def test_two_identical_containers_are_equal(scene, leaf):
    labels, world = scene
    common = {
        "axis_map": {"t": "t", "z": "z", "y": "y", "x": "x"},
        "scale": {"z": 2.0},
        "broadcast_output_axes": ("c",),
        "axis_transforms": {"t": leaf},
    }
    first = ByDimensionTransform.from_axis_map(labels, world, **common)
    second = ByDimensionTransform.from_axis_map(labels, world, **common)
    assert first == second
    assert hash(first) == hash(second)


def test_equality_is_total_against_a_foreign_type(transform):
    assert (transform == "not a transform") is False


# -- axis_correspondence (Phase 5) -------------------------------------------


def test_axis_correspondence_is_structural(transform):
    """Answered from the block declarations, with no matrix involved.

    This is the point of moving it onto the transform: the container has no
    matrix at all (a leaf is present), yet it can still say which world axis
    each data axis becomes.
    """
    assert transform.to_affine() is None
    assert transform.axis_correspondence() == {0: 0, 1: 2, 2: 3, 3: 4}


def test_the_leaf_answers_for_itself(leaf):
    assert leaf.axis_correspondence() == {0: 0}


def test_it_agrees_with_the_affine_answer_when_all_blocks_are_affine(scene):
    """Behaviour-preserving: same answer as reading the matrix."""
    labels, world = scene
    common = {
        "axis_map": {"t": "t", "z": "z", "y": "y", "x": "x"},
        "scale": {"z": 2.0},
        "broadcast_output_axes": ("c",),
    }
    by_dimension = ByDimensionTransform.from_axis_map(labels, world, **common)
    plain = AffineTransform.from_axis_map(labels, world, **common)
    assert by_dimension.axis_correspondence() == plain.axis_correspondence()


def test_the_render_layer_wrapper_delegates(transform):
    """_spaces.axis_correspondence is now a thin pass-through."""
    from cellier.render._spaces import axis_correspondence

    assert axis_correspondence(transform) == transform.axis_correspondence()


# -- input_domain -------------------------------------------------------------


def test_a_table_backed_axis_reports_its_index_span(transform, leaf):
    """The span of samples that exist, on the store's own edge convention.

    A caller that rounds a position to a whole sample needs this, or
    rounding leaves the domain at the top edge and maps to nothing.
    """
    assert leaf.input_domain() == {0: (-0.5, 12.5)}
    assert transform.input_domain() == {0: (-0.5, 12.5)}


def test_an_affine_block_contributes_no_domain(scene):
    """A matrix maps every real coordinate; bounds come from the store."""
    labels, world = scene
    affine_only = ByDimensionTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0},
        broadcast_output_axes=("c",),
    )
    assert affine_only.input_domain() == {}


def test_the_domain_is_keyed_by_container_axis(scene, leaf):
    """A block's local axis 0 is translated to the container's own index."""
    labels, world = scene
    permuted = ByDimensionTransform.from_axis_map(
        labels,
        world,
        axis_map={"t": "t", "z": "z", "y": "y", "x": "x"},
        scale={"z": 2.0, "y": 1.0, "x": 1.0},
        broadcast_output_axes=("c",),
        axis_transforms={"t": leaf},
    )
    # 't' is input axis 0 of the container and axis 0 of its own block, so
    # this also pins that the translation happens at all.
    assert set(permuted.input_domain()) == {0}


def test_rounding_the_top_of_the_domain_stays_on_a_real_sample(transform):
    """The demo-5 crash, at the transform level.

    Index 12.5 is the outer edge of frame 12's cell, not a midpoint between
    frame 12 and a frame 13.  Mapping 13 forward has no answer.
    """
    _low, high = transform.input_domain()[0]
    assert np.isfinite(transform.map_coordinates(np.array([high, 0.0, 0.0, 0.0]))[0])
    assert np.isnan(
        transform.map_coordinates(np.array([float(np.floor(high)) + 1, 0.0, 0.0, 0.0]))[
            0
        ]
    )


# -- broadcast axes and world bounds -------------------------------------------


def test_broadcast_output_axes_come_from_the_blocks(transform, scene):
    """Answered structurally, with no matrix: the time block has none."""
    _labels, world = scene
    assert transform.to_affine() is None
    assert transform.broadcast_output_axes() == frozenset({world.axes[1].id})


def test_a_leaf_and_a_plain_affine_report_no_broadcast(leaf, scene):
    labels, _world = scene
    assert leaf.broadcast_output_axes() == frozenset()
    square = AffineTransform.from_axis_map(
        labels,
        CoordinateSystem(name="out", axes=labels.axes),
        axis_map={axis.id: axis.id for axis in labels.axes},
    )
    assert square.broadcast_output_axes() == frozenset()


def test_world_bounds_through_a_lookup_axis_and_a_broadcast(transform, scene):
    """The light-sheet case: t through a table, c broadcast, zyx scaled."""
    from cellier.scene._bounds import visual_world_bounds

    _labels, world = scene
    extents = ((-0.5, len(TIMES) - 0.5), (-0.5, 3.5), (-0.5, 5.5), (-0.5, 7.5))

    low, high = visual_world_bounds(transform, extents, world)

    # t: the table's outer edges, half a gap beyond the first and last frame.
    assert low[0] == pytest.approx(TIMES[0] - (TIMES[1] - TIMES[0]) / 2)
    assert high[0] == pytest.approx(TIMES[-1] + (TIMES[-1] - TIMES[-2]) / 2)
    # c: broadcast, so no extent.
    assert np.isnan(low[1]) and np.isnan(high[1])
    np.testing.assert_allclose(low[2:], (-1.0, -0.5, -0.5))
    np.testing.assert_allclose(high[2:], (7.0, 5.5, 7.5))
