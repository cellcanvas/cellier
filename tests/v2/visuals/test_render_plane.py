"""``RenderPlane``: constructors, validation, identity (plane rendering 4.1)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from cellier.scene.dims import spatial_axes, world_coordinate_system
from cellier.visuals import MAX_RENDER_PLANES, RenderPlane
from cellier.visuals._render_plane import validate_render_planes


@pytest.fixture
def world():
    return world_coordinate_system(spatial_axes("z", "y", "x"), name="world")


@pytest.fixture
def world_tzyx():
    return world_coordinate_system(
        [("t", "time"), *spatial_axes("z", "y", "x")], name="world"
    )


def _frame(plane: RenderPlane) -> np.ndarray:
    return np.column_stack([plane.in_plane_axis_0, plane.in_plane_axis_1, plane.normal])


# -- constructors ---------------------------------------------------------------


def test_from_point_normal_builds_a_right_handed_unit_frame(world):
    plane = RenderPlane.from_point_normal(world, (1, 2, 3), (0, 3, 4))
    frame = _frame(plane)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(plane.normal, (0, 0.6, 0.8), atol=1e-12)
    assert plane.origin == (1.0, 2.0, 3.0)
    assert plane.coordinate_system == world.id
    assert plane.axes == tuple(axis.id for axis in world.axes)
    assert plane.extent_0 == (None, None)
    assert plane.extent_1 == (None, None)
    assert plane.enabled is True


def test_with_no_up_axis_0_is_the_axis_least_aligned_with_the_normal(world):
    plane = RenderPlane.from_point_normal(world, (0, 0, 0), (0.1, 0.2, 1.0))
    # z is least aligned with the normal: axis 0 is z projected into the plane.
    expected = np.array([1.0, 0.0, 0.0])
    normal = np.array(plane.normal)
    expected = expected - (expected @ normal) * normal
    np.testing.assert_allclose(
        plane.in_plane_axis_0, expected / np.linalg.norm(expected), atol=1e-12
    )


def test_up_is_projected_into_the_plane(world):
    plane = RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1), up=(0, 1, 5))
    np.testing.assert_allclose(plane.in_plane_axis_0, (0, 1, 0), atol=1e-12)
    np.testing.assert_allclose(plane.in_plane_axis_1, (-1, 0, 0), atol=1e-12)
    with pytest.raises(ValueError, match="parallel"):
        RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1), up=(0, 0, 2))


def test_the_two_constructors_agree(world):
    by_normal = RenderPlane.from_point_normal(
        world, (4, 5, 6), (1, -2, 0.5), up=(0, 1, 0)
    )
    by_rotation = RenderPlane.from_rotation(
        world, (4, 5, 6), _frame(by_normal), id=by_normal.id
    )
    assert by_rotation == by_normal


def test_from_rotation_takes_a_quaternion(world):
    # A quarter turn about the third axis of the plane's axes.
    half = np.sqrt(0.5)
    plane = RenderPlane.from_rotation(world, (0, 0, 0), (0, 0, half, half))
    np.testing.assert_allclose(plane.in_plane_axis_0, (0, 1, 0), atol=1e-12)
    np.testing.assert_allclose(plane.in_plane_axis_1, (-1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(plane.normal, (0, 0, 1), atol=1e-12)


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, -1.0]),  # a reflection
        np.diag([2.0, 1.0, 1.0]),  # not orthonormal
        np.eye(2),
        (1.0, 0.0, 0.0),
    ],
)
def test_from_rotation_refuses_what_is_not_a_rotation(world, rotation):
    with pytest.raises(ValueError, match="rotation"):
        RenderPlane.from_rotation(world, (0, 0, 0), rotation)


def test_axes_name_three_world_axes_and_are_stored_as_ids(world_tzyx):
    plane = RenderPlane.from_point_normal(
        world_tzyx, (1, 2, 3), (0, 0, 1), axes=("z", "y", "x")
    )
    assert plane.axes == tuple(world_tzyx.axis_by_name(n).id for n in "zyx")
    reordered = RenderPlane.from_point_normal(
        world_tzyx, (3, 2, 1), (1, 0, 0), axes=("x", "y", "z")
    )
    assert reordered.axes == tuple(world_tzyx.axis_by_name(n).id for n in "xyz")
    with pytest.raises(ValueError, match="pass axes="):
        RenderPlane.from_point_normal(world_tzyx, (1, 2, 3), (0, 0, 1))
    with pytest.raises(ValueError, match="three axes"):
        RenderPlane.from_point_normal(world_tzyx, (1, 2, 3), (0, 0, 1), axes=("z", "y"))
    with pytest.raises(KeyError):
        RenderPlane.from_point_normal(
            world_tzyx, (1, 2, 3), (0, 0, 1), axes=("z", "y", "w")
        )


# -- validation -----------------------------------------------------------------


def _raw(world, **fields) -> dict:
    return {
        "coordinate_system": world.id,
        "axes": tuple(axis.id for axis in world.axes),
        "origin": (0.0, 0.0, 0.0),
        "in_plane_axis_0": (1.0, 0.0, 0.0),
        "in_plane_axis_1": (0.0, 1.0, 0.0),
        **fields,
    }


def test_the_in_plane_axes_are_stored_normalised(world):
    plane = RenderPlane(
        **_raw(world, in_plane_axis_0=(3, 0, 0), in_plane_axis_1=(0, 0, 0.5))
    )
    assert plane.in_plane_axis_0 == (1.0, 0.0, 0.0)
    assert plane.in_plane_axis_1 == (0.0, 0.0, 1.0)
    assert plane.normal == (0.0, -1.0, 0.0)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"in_plane_axis_0": (0, 0, 0)}, "non-zero"),
        ({"in_plane_axis_1": (float("nan"), 0, 1)}, "finite"),
        ({"in_plane_axis_1": (float("inf"), 0, 1)}, "finite"),
        ({"in_plane_axis_1": (1.0, 1e-3, 0.0)}, "orthogonal"),
        ({"in_plane_axis_1": (1.0, 0.0, 0.0)}, "orthogonal"),
        ({"origin": (0.0, float("inf"), 0.0)}, "finite"),
        ({"extent_0": (2.0, 2.0)}, "min < max"),
        ({"extent_1": (3.0, -3.0)}, "min < max"),
        ({"extent_0": (float("-inf"), 1.0)}, "finite number or None"),
    ],
)
def test_a_bad_plane_is_refused(world, fields, match):
    with pytest.raises(ValidationError, match=match):
        RenderPlane(**_raw(world, **fields))


def test_the_axes_must_be_three_distinct(world):
    z, y, _x = (axis.id for axis in world.axes)
    with pytest.raises(ValidationError, match="distinct"):
        RenderPlane(**_raw(world, axes=(z, y, y)))
    with pytest.raises(ValidationError):
        RenderPlane(**_raw(world, axes=(z, y)))


@pytest.mark.parametrize(
    "extent", [(None, None), (-2.0, None), (None, 5.0), (-2.0, 5.0), (1.0, 2.0)]
)
def test_each_extent_side_is_bounded_or_not_on_its_own(world, extent):
    plane = RenderPlane(**_raw(world, extent_0=extent, extent_1=(None, 0.5)))
    assert plane.extent_0 == extent
    assert plane.extent_1 == (None, 0.5)


# -- identity -------------------------------------------------------------------


def test_the_plane_is_frozen(world):
    plane = RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1))
    with pytest.raises(ValidationError):
        plane.origin = (1.0, 1.0, 1.0)


def test_equality_includes_the_id(world):
    one = RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1))
    other = RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1))
    assert one.id != other.id
    assert one != other
    assert other.model_copy(update={"id": one.id}) == one
    moved = one.model_copy(update={"origin": (0.0, 0.0, 9.0)})
    assert moved.id == one.id
    assert moved != one
    assert hash(one) == hash(other.model_copy(update={"id": one.id}))


def test_json_round_trip(world):
    plane = RenderPlane.from_point_normal(
        world,
        (1.5, -2, 3),
        (0.3, 1, -0.2),
        extent_0=(-4.0, None),
        extent_1=(-1.0, 2.5),
        enabled=False,
    )
    again = RenderPlane.model_validate_json(plane.model_dump_json())
    assert again == plane
    assert again.axes == plane.axes  # ids, not their text


def test_axis_names_are_kept_as_names(world):
    plane = RenderPlane(**_raw(world, axes=("z", "y", "x")))
    assert plane.axes == ("z", "y", "x")
    assert RenderPlane.model_validate_json(plane.model_dump_json()) == plane


def test_a_given_id_is_kept(world):
    plane_id = uuid4()
    plane = RenderPlane.from_point_normal(world, (0, 0, 0), (1, 0, 0), id=plane_id)
    assert plane.id == plane_id


def test_a_tuple_is_validated_and_the_cap_is_four(world):
    plane = RenderPlane.from_point_normal(world, (0, 0, 0), (0, 0, 1))
    assert validate_render_planes([plane, plane.model_dump()]) == (plane, plane)
    with pytest.raises(ValidationError):
        validate_render_planes([plane, "not a plane"])
    assert MAX_RENDER_PLANES == 4


# -- on a visual ------------------------------------------------------------------


def test_a_visual_with_render_planes_round_trips_through_json(world):
    from cellier.visuals import ImageVisual, InMemoryImageSingleAppearance

    plane = RenderPlane.from_point_normal(world, (1, 2, 3), (0, 0, 1))
    visual = ImageVisual(
        name="image",
        data_store_id=str(uuid4()),
        single=InMemoryImageSingleAppearance(render_mode="plane"),
        render_planes=(plane,),
    )
    again = ImageVisual.model_validate_json(visual.model_dump_json())
    assert again.render_planes == (plane,)
    assert again.single.render_mode == "plane"
    assert again.plane_mode() is True


@pytest.mark.parametrize(
    "appearance_cls",
    [
        "InMemoryImageSingleAppearance",
        "InMemoryImageChannelAppearance",
        "MultiscaleImageSingleAppearance",
        "MultiscaleImageChannelAppearance",
        "InMemoryLabelsAppearance",
        "MultiscaleLabelsAppearance",
    ],
)
def test_plane_is_a_render_mode_of_every_image_and_labels_appearance(appearance_cls):
    import cellier.visuals as visuals

    appearance = getattr(visuals, appearance_cls)(render_mode="plane")
    assert appearance.render_mode == "plane"
