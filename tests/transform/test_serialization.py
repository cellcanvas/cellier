"""The np.ndarray field contract and round-trips."""

import json
from uuid import uuid4

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from cellier.transform import (
    AffineTransform,
    Axis,
    AxisAlignedBoundingBox,
    ConvexRegion,
    CoordinateSystem,
    CoordinateSystemType,
    DataCoordinateSystem,
    HalfSpace,
    Plane,
    RegionSelection,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)
from tests.transform._use_cases import channel_broadcast, scale_only


def space(name):
    return Axis(name=name, axis_type="space", unit="micrometer")


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


def every_model():
    """One instance of every model in the package."""
    data, world, transform = scale_only()
    rendered = RenderedCoordinateSystem.from_world(world, ("Z", "Y"), uuid4())
    world5 = tczyx()
    rendered5 = RenderedCoordinateSystem.from_world(world5, ("Z", "Y", "X"), uuid4())
    embedding = AffineTransform.from_axis_map(
        rendered5,
        world5,
        axis_map={"Z": "Z", "Y": "Y", "X": "X"},
        constant_output_axes={"T": 7.0, "C": 2.0},
    )
    return {
        "Axis": space("z"),
        "CoordinateSystem": CoordinateSystem(name="c", axes=(space("z"),)),
        "DataCoordinateSystem": data,
        "VisualCoordinateSystem": VisualCoordinateSystem.from_data(
            data, ("y", "x"), uuid4()
        ),
        "WorldCoordinateSystem": world,
        "RenderedCoordinateSystem": rendered,
        "AxisAlignedBoundingBox": AxisAlignedBoundingBox(
            coordinate_system=world.id,
            min_coordinate=[0.0, 1.0, 2.0],
            max_coordinate=[3.0, 4.0, 5.0],
        ),
        "Plane": Plane(coordinate_system=world.id, normal=[0.0, 0.0, 1.0], offset=2.0),
        "HalfSpace": HalfSpace(normal=[1.0, 0.0, 0.0], offset=5.0),
        "ConvexRegion": ConvexRegion.from_axis_slabs(world, {"Z": (1.0, 0.5)}),
        "AffineTransform": transform,
        "RegionSelection": RegionSelection(
            transform=embedding,
            region=ConvexRegion.from_axis_slabs(
                world5, {"T": (7.0, 0.5), "C": (2.0, 0.5)}
            ),
        ),
    }


@pytest.mark.parametrize("name", list(every_model()))
def test_every_model_round_trips_through_model_dump(name):
    """Without the coercing validator this raises, so it is a real guard."""
    instance = every_model()[name]
    back = type(instance).model_validate(instance.model_dump())
    assert back == instance


@pytest.mark.parametrize("name", list(every_model()))
def test_every_model_round_trips_through_json(name):
    instance = every_model()[name]
    back = type(instance).model_validate_json(instance.model_dump_json())
    assert back == instance


# --- infinity --------------------------------------------------------


def infinite_box():
    return AxisAlignedBoundingBox(
        coordinate_system=uuid4(),
        min_coordinate=[-np.inf, 1.0, -np.inf],
        max_coordinate=[np.inf, 2.0, np.inf],
    )


def test_infinite_bounds_survive_model_dump():
    box = infinite_box()
    back = AxisAlignedBoundingBox.model_validate(box.model_dump())
    assert back == box
    assert back.min_coordinate[0] == -np.inf


def test_infinite_bounds_survive_json_as_sentinel_strings():
    """model_dump_json would otherwise write null and read back nan."""
    box = infinite_box()
    dumped = box.model_dump_json()
    assert '"Infinity"' in dumped
    assert '"-Infinity"' in dumped

    back = AxisAlignedBoundingBox.model_validate_json(dumped)
    assert back.min_coordinate[0] == -np.inf
    assert back.max_coordinate[0] == np.inf
    assert not np.any(np.isnan(back.min_coordinate))
    assert not np.any(np.isnan(back.max_coordinate))


def test_the_dumped_json_is_strict_json_valid():
    """json.dumps(..., allow_nan=False) accepts it; a bare inf would not."""
    parsed = json.loads(infinite_box().model_dump_json())
    assert json.dumps(parsed, allow_nan=False)
    with pytest.raises(ValueError, match="not JSON compliant"):
        json.dumps([float("inf")], allow_nan=False)


def test_sentinel_strings_are_accepted_on_the_way_in():
    box = AxisAlignedBoundingBox(
        coordinate_system=uuid4(),
        min_coordinate=["-Infinity", 1.0],
        max_coordinate=["Infinity", 2.0],
    )
    assert box.min_coordinate[0] == -np.inf
    assert box.max_coordinate[0] == np.inf


# --- rejection rules -------------------------------------------------


def test_non_finite_values_are_rejected_everywhere_but_box_bounds():
    coordinate_system = uuid4()
    data, world, _ = scale_only()
    for factory in (
        lambda v: Plane(
            coordinate_system=coordinate_system, normal=[v, 0.0], offset=1.0
        ),
        lambda v: Plane(
            coordinate_system=coordinate_system, normal=[1.0, 0.0], offset=v
        ),
        lambda v: HalfSpace(normal=[v, 0.0], offset=1.0),
        lambda v: HalfSpace(normal=[1.0, 0.0], offset=v),
    ):
        for value in (np.inf, -np.inf, np.nan):
            with pytest.raises(ValidationError, match="finite"):
                factory(value)

    matrix = np.eye(4)
    for value in (np.inf, np.nan):
        matrix[0, 0] = value
        with pytest.raises(ValidationError, match="finite"):
            AffineTransform.from_matrix(matrix, data, world)


def test_nan_is_rejected_in_box_bounds_but_infinity_is_not():
    with pytest.raises(ValidationError, match="nan"):
        AxisAlignedBoundingBox(
            coordinate_system=uuid4(),
            min_coordinate=[np.nan, 0.0],
            max_coordinate=[1.0, 1.0],
        )
    assert infinite_box()


def test_a_bare_string_is_not_read_as_a_character_sequence():
    with pytest.raises(ValidationError):
        HalfSpace(normal="10", offset=1.0)


# --- equality and hashing --------------------------------------------


@pytest.mark.parametrize("name", list(every_model()))
def test_every_model_compares_and_hashes_without_raising(name):
    """A model with the default __eq__ on a bare array raises ValueError."""
    instance = every_model()[name]
    twin = type(instance).model_validate(instance.model_dump())
    assert instance == twin
    assert (instance == "not a model") is False
    assert (instance != "not a model") is True


@pytest.mark.parametrize(
    "name", ["AxisAlignedBoundingBox", "Plane", "HalfSpace", "AffineTransform"]
)
def test_array_holding_models_hash(name):
    instance = every_model()[name]
    twin = type(instance).model_validate(instance.model_dump())
    assert hash(instance) == hash(twin)
    assert len({instance, twin}) == 1


# --- the wire format ------------------------------------------------


def test_the_dumped_affine_is_a_nested_list_of_floats():
    _, _, transform = scale_only()
    dumped = transform.model_dump()["transform"]
    assert isinstance(dumped, list)
    assert isinstance(dumped[0], list)
    assert all(isinstance(value, float) for row in dumped for value in row)


def test_axis_dumps_type_under_the_rfc5_spelling():
    dumped = space("z").model_dump(by_alias=True)
    # `sampling` is a cellier extension, not an RFC-5 field: it says whether
    # this axis's coordinates are sample indices, which RFC-5 has no slot for.
    assert set(dumped) == {"name", "type", "unit", "sampling", "id"}
    assert dumped["type"] == "space"


def test_broadcast_axes_survive_json():
    _, world, transform = channel_broadcast()
    back = AffineTransform.model_validate_json(transform.model_dump_json())
    assert back.broadcast_axes == frozenset({world.axis_by_name("C").id})


def test_the_coordinate_system_union_round_trips_a_heterogeneous_list():
    class Holder(BaseModel):
        systems: tuple[CoordinateSystemType, ...]

    world = tczyx()
    holder = Holder(
        systems=(
            CoordinateSystem(name="plain", axes=(space("z"),)),
            DataCoordinateSystem(
                name="store", datastore_id=uuid4(), axes=(space("z"),)
            ),
            world,
            RenderedCoordinateSystem.from_world(world, ("Z", "Y"), uuid4()),
        )
    )
    assert Holder.model_validate_json(holder.model_dump_json()) == holder


def test_a_region_selection_containing_both_round_trips():
    instance = every_model()["RegionSelection"]
    back = RegionSelection.model_validate(instance.model_dump())
    assert back.transform == instance.transform
    assert back.region == instance.region
