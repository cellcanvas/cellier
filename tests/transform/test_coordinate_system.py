"""Tests for cellier.transform._coordinate_system."""

from functools import partial
from uuid import uuid4

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from cellier.transform import (
    Axis,
    CoordinateSystem,
    CoordinateSystemType,
    DataCoordinateSystem,
    RenderedCoordinateSystem,
    VisualCoordinateSystem,
    WorldCoordinateSystem,
)

space = partial(Axis, axis_type="space", unit="micrometer")


def zyx() -> CoordinateSystem:
    return CoordinateSystem(
        name="data", axes=(space(name="z"), space(name="y"), space(name="x"))
    )


def tzyx_data() -> DataCoordinateSystem:
    return DataCoordinateSystem(
        name="timelapse",
        datastore_id=uuid4(),
        axes=(
            Axis(name="t", axis_type="time", unit="second"),
            space(name="z"),
            space(name="y"),
            space(name="x"),
        ),
    )


def tczyx_world() -> WorldCoordinateSystem:
    return WorldCoordinateSystem(
        axes=(
            Axis(name="T", axis_type="time", unit="second"),
            Axis(name="C", axis_type="channel"),
            space(name="Z"),
            space(name="Y"),
            space(name="X"),
        )
    )


# --- CoordinateSystem ------------------------------------------------


def test_ndim_equals_len_axes():
    assert zyx().ndim == 3


def test_empty_axes_rejected():
    with pytest.raises(ValidationError):
        CoordinateSystem(name="empty", axes=())


def test_duplicate_axis_ids_rejected():
    axis = space(name="z")
    with pytest.raises(ValidationError):
        CoordinateSystem(name="dupe", axes=(axis, axis))


def test_duplicate_axis_names_accepted():
    system = CoordinateSystem(name="dupe", axes=(space(name="z"), space(name="z")))
    assert system.axis_names() == ("z", "z")


def test_axis_by_name_unique_match():
    system = zyx()
    assert system.axis_by_name("y").name == "y"


def test_axis_by_name_raises_on_ambiguity():
    system = CoordinateSystem(name="dupe", axes=(space(name="z"), space(name="z")))
    with pytest.raises(ValueError, match="ambiguous"):
        system.axis_by_name("z")


def test_axis_by_name_raises_on_miss():
    with pytest.raises(KeyError):
        zyx().axis_by_name("q")


def test_index_of_round_trips_against_axes():
    system = zyx()
    for index, axis in enumerate(system.axes):
        assert system.index_of(axis.id) == index


def test_index_of_raises_for_a_foreign_axis():
    with pytest.raises(KeyError):
        zyx().index_of(uuid4())


def test_resolve_accepts_a_name_or_an_id():
    system = zyx()
    assert system.resolve("y") == 1
    assert system.resolve(system.axes[1].id) == 1
    assert system.resolve_axis("y") is system.axes[1]


def test_resolve_by_id_works_where_a_name_is_ambiguous():
    """An id is never ambiguous, even when the names collide."""
    first, second = space(name="z"), space(name="z")
    system = CoordinateSystem(name="dupe", axes=(first, second))
    with pytest.raises(ValueError, match="ambiguous"):
        system.resolve("z")
    assert system.resolve(second.id) == 1


def test_coordinate_system_is_frozen():
    system = zyx()
    with pytest.raises(ValidationError):
        system.name = "other"


# --- subclasses ------------------------------------------------------


def test_data_coordinate_system_requires_datastore_id():
    with pytest.raises(ValidationError):
        DataCoordinateSystem(name="cells", axes=(space(name="z"),))
    system = DataCoordinateSystem(
        name="cells", datastore_id=uuid4(), axes=(space(name="z"),)
    )
    assert system.coordinate_system_type == "data"


def test_world_coordinate_system_name_defaults_to_world():
    assert WorldCoordinateSystem(axes=(space(name="Z"),)).name == "world"


def test_discriminated_union_round_trips_over_a_heterogeneous_list():
    class Holder(BaseModel):
        systems: tuple[CoordinateSystemType, ...]

    world = tczyx_world()
    holder = Holder(
        systems=(
            zyx(),
            DataCoordinateSystem(
                name="cells", datastore_id=uuid4(), axes=(space(name="z"),)
            ),
            VisualCoordinateSystem(visual_id=uuid4(), axes=(space(name="y"),)),
            world,
            RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), uuid4()),
        )
    )
    back = Holder.model_validate(holder.model_dump())
    assert [type(s) for s in back.systems] == [
        CoordinateSystem,
        DataCoordinateSystem,
        VisualCoordinateSystem,
        WorldCoordinateSystem,
        RenderedCoordinateSystem,
    ]
    assert back == holder


def test_union_adapter_selects_by_discriminator():
    adapter = TypeAdapter(CoordinateSystemType)
    world = tczyx_world()
    parsed = adapter.validate_python(world.model_dump())
    assert isinstance(parsed, WorldCoordinateSystem)


# --- RenderedCoordinateSystem ----------------------------------------


def test_rendered_rank_other_than_2_or_3_is_rejected():
    canvas_id = uuid4()
    with pytest.raises(ValidationError):
        RenderedCoordinateSystem(axes=(space(name="Z"),), canvas_id=canvas_id)
    with pytest.raises(ValidationError):
        RenderedCoordinateSystem.from_world(
            tczyx_world(), ("T", "C", "Z", "Y"), canvas_id
        )
    assert (
        RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Y"), canvas_id).ndim
        == 2
    )


def test_from_world_inherits_metadata_and_makes_fresh_ids():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(world, ("T", "Z", "Y"), uuid4())

    assert rendered.axis_names() == ("T", "Z", "Y")
    assert [a.axis_type for a in rendered.axes] == ["time", "space", "space"]
    assert [a.unit for a in rendered.axes] == ["second", "micrometer", "micrometer"]

    world_ids = {a.id for a in world.axes}
    assert not world_ids & {a.id for a in rendered.axes}


def test_displayed_axes_order_is_the_rendered_axis_order():
    world = tczyx_world()
    canvas_id = uuid4()
    xyz = RenderedCoordinateSystem.from_world(world, ("X", "Y", "Z"), canvas_id)
    zyx_ = RenderedCoordinateSystem.from_world(world, ("Z", "Y", "X"), canvas_id)
    assert xyz.axis_names() == ("X", "Y", "Z")
    assert zyx_.axis_names() == ("Z", "Y", "X")
    assert xyz.axis_names() != zyx_.axis_names()


def test_from_world_accepts_axis_ids():
    world = tczyx_world()
    rendered = RenderedCoordinateSystem.from_world(
        world, (world.axes[2].id, world.axes[3].id), uuid4()
    )
    assert rendered.axis_names() == ("Z", "Y")


def test_from_world_rejects_a_repeated_world_axis():
    with pytest.raises(ValueError, match="distinct"):
        RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Z"), uuid4())


def test_rendered_has_no_world_backref():
    """The link is carried by the transform, not by a second field."""
    rendered = RenderedCoordinateSystem.from_world(tczyx_world(), ("Z", "Y"), uuid4())
    assert "world_coordinate_system" not in type(rendered).model_fields


# --- VisualCoordinateSystem ------------------------------------------


def test_from_data_inherits_metadata_and_makes_fresh_ids():
    """`from_data` mirrors `from_world` exactly."""
    data = tzyx_data()
    visual_id = uuid4()
    visual = VisualCoordinateSystem.from_data(data, ("z", "y", "x"), visual_id)

    assert visual.coordinate_system_type == "visual"
    assert visual.name == "visual"
    assert visual.visual_id == visual_id
    assert visual.axis_names() == ("z", "y", "x")
    assert [a.axis_type for a in visual.axes] == ["space", "space", "space"]
    assert [a.unit for a in visual.axes] == ["micrometer"] * 3

    data_ids = {a.id for a in data.axes}
    assert not data_ids & {a.id for a in visual.axes}


def test_from_data_retained_axes_order_is_the_array_axis_order():
    """The retained axes are the order the returned array carries."""
    data = tzyx_data()
    visual = VisualCoordinateSystem.from_data(data, ("x", "z"), uuid4())
    assert visual.axis_names() == ("x", "z")


def test_from_data_accepts_axis_ids():
    data = tzyx_data()
    visual = VisualCoordinateSystem.from_data(
        data, (data.axes[1].id, data.axes[2].id), uuid4()
    )
    assert visual.axis_names() == ("z", "y")


def test_from_data_rejects_a_repeated_data_axis():
    with pytest.raises(ValueError, match="distinct"):
        VisualCoordinateSystem.from_data(tzyx_data(), ("z", "z"), uuid4())


def test_visual_system_has_no_rank_restriction_unlike_rendered():
    """A rendered system is 2D or 3D by R1; a visual system is what the
    request retained, which may be 1, 4 or more axes.
    """
    data = tzyx_data()
    visual_id = uuid4()
    all_axes = ("t", "z", "y", "x")
    assert VisualCoordinateSystem.from_data(data, ("x",), visual_id).ndim == 1
    assert VisualCoordinateSystem.from_data(data, all_axes, visual_id).ndim == 4


def test_visual_system_carries_a_visual_backref_not_a_datastore_one():
    """This space belongs to a visual, so `datastore_id` would lie."""
    visual = VisualCoordinateSystem.from_data(tzyx_data(), ("y", "x"), uuid4())
    assert "visual_id" in type(visual).model_fields
    assert "datastore_id" not in type(visual).model_fields


def test_visual_system_requires_a_visual_id():
    with pytest.raises(ValidationError):
        VisualCoordinateSystem(axes=(space(name="y"),))


def test_visual_system_round_trips_directly_and_through_the_union():
    visual = VisualCoordinateSystem.from_data(tzyx_data(), ("z", "y", "x"), uuid4())

    assert VisualCoordinateSystem.model_validate(visual.model_dump()) == visual

    adapter = TypeAdapter(CoordinateSystemType)
    parsed = adapter.validate_python(visual.model_dump())
    assert isinstance(parsed, VisualCoordinateSystem)
    assert parsed == visual


def test_two_visuals_over_one_store_get_distinct_systems():
    """The upload space is not intrinsic to a store, which is why it is
    not a `DataCoordinateSystem`.
    """
    data = tzyx_data()
    one = VisualCoordinateSystem.from_data(data, ("z", "y", "x"), uuid4())
    two = VisualCoordinateSystem.from_data(data, ("y", "x"), uuid4())
    assert one.id != two.id
    assert one.visual_id != two.visual_id
    assert one.axis_names() != two.axis_names()
