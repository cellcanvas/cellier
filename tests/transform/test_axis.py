"""Tests for cellier.transform._axis."""

from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from cellier.transform import Axis, AxisType


def test_axis_type_outside_the_rfc5_set_is_rejected():
    with pytest.raises(ValidationError):
        Axis(name="z", axis_type="spatial")


@pytest.mark.parametrize(
    "axis_type",
    ["array", "space", "time", "channel", "coordinate", "displacement"],
)
def test_every_rfc5_axis_type_is_accepted(axis_type):
    assert Axis(name="z", axis_type=axis_type).axis_type == axis_type


def test_axis_type_is_required():
    """No default, because a wrongly typed axis fails far from its cause."""
    with pytest.raises(ValidationError):
        Axis(name="z")


def test_both_spellings_parse_and_by_alias_emits_type():
    from_python = Axis(name="z", axis_type="space")
    from_wire = Axis.model_validate({"name": "z", "type": "space"})
    assert from_python.axis_type == from_wire.axis_type == "space"

    dumped = from_python.model_dump(by_alias=True)
    assert "type" in dumped
    assert "axis_type" not in dumped
    assert dumped["type"] == "space"


def test_round_trips_through_both_dump_flavours():
    axis = Axis(name="z", axis_type="space", unit="micrometer")
    assert Axis.model_validate(axis.model_dump()) == axis
    assert Axis.model_validate(axis.model_dump(by_alias=True)) == axis


def test_axis_is_frozen():
    axis = Axis(name="z", axis_type="space")
    with pytest.raises(ValidationError):
        axis.name = "y"


def test_two_axes_with_the_same_name_have_different_ids():
    first = Axis(name="z", axis_type="space")
    second = Axis(name="z", axis_type="space")
    assert first.name == second.name
    assert first.id != second.id


def test_unit_is_free_form_and_unvalidated():
    """RFC-5 says UDUNITS-2; nothing here parses it."""
    nonsense = "furlongs per " + "fortnight"
    assert Axis(name="z", axis_type="space", unit=nonsense).unit == nonsense
    assert Axis(name="z", axis_type="space").unit is None


def test_explicit_id_is_preserved():
    axis_id = uuid4()
    assert Axis(name="z", axis_type="space", id=axis_id).id == axis_id


def test_axis_type_alias_is_the_closed_literal():
    assert set(TypeAdapter(AxisType).json_schema()["enum"]) == {
        "array",
        "space",
        "time",
        "channel",
        "coordinate",
        "displacement",
    }
