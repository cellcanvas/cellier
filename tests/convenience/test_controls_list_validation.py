"""Validation of the ``controls=`` list of the ``add_*`` methods.

``plans/ordered_controls_list_design.md`` section 5.  ``controls=`` is a list
of control models or ``None``.  ``validate_controls`` is called by every
``add_*`` before anything is added, so a wrong list raises there and not as
a dock with a control missing.
"""

from __future__ import annotations

import pytest

from cellier.convenience.gui._controls_config import (
    ALLOWED_CONTROLS,
    FIELD_KINDS,
    AmbientOcclusionControl,
    BoundingBoxControl,
    ClippingPlanesControl,
    Control,
    DatasetInfoControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingConfigControl,
    LoadingIndicator,
    MeshLodControl,
    MeshSectionControl,
    OutlineControl,
    PickingControl,
    RenderPlanesControl,
    TrailControl,
    validate_controls,
)

KINDS = list(ALLOWED_CONTROLS)
IMAGE_KINDS = ["image", "multiscale_image"]
FIELD_CONTROL_KINDS = [kind for kind in KINDS if kind not in IMAGE_KINDS]
MULTISCALE_VOLUME_KINDS = ["multiscale_image", "multiscale_labels"]

ALL_CONTROLS = [
    ImageControl,
    FieldControls,
    LevelOfDetailControl,
    BoundingBoxControl,
    ClippingPlanesControl,
    RenderPlanesControl,
    OutlineControl,
    AmbientOcclusionControl,
    PickingControl,
    LoadingIndicator,
    LoadingConfigControl,
    MeshSectionControl,
    MeshLodControl,
    TrailControl,
    DatasetInfoControl,
]


def _make(control_cls: type[Control], kind: str) -> Control:
    """An instance of *control_cls* with a field that is valid for *kind*."""
    if control_cls in (ImageControl, FieldControls):
        return control_cls(fields=["visible"])
    return control_cls()


def _field_item(kind: str, fields: list[str]) -> Control:
    """The field control of *kind*, naming *fields*."""
    cls = ImageControl if kind in IMAGE_KINDS else FieldControls
    return cls(fields=fields)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def test_there_are_nine_visual_kinds_and_fifteen_models():
    assert KINDS == [
        "image",
        "multiscale_image",
        "labels",
        "multiscale_labels",
        "mesh",
        "multiscale_mesh",
        "points",
        "lines",
        "graph",
    ]
    assert list(FIELD_KINDS) == KINDS
    assert {cls for allowed in ALLOWED_CONTROLS.values() for cls in allowed} == set(
        ALL_CONTROLS
    )
    assert all(issubclass(cls, Control) for cls in ALL_CONTROLS)


@pytest.mark.parametrize(
    ("control_cls", "kinds"),
    [
        (ImageControl, IMAGE_KINDS),
        (FieldControls, FIELD_CONTROL_KINDS),
        (LevelOfDetailControl, MULTISCALE_VOLUME_KINDS),
        (BoundingBoxControl, KINDS),
        (ClippingPlanesControl, KINDS),
        (
            RenderPlanesControl,
            ["image", "multiscale_image", "labels", "multiscale_labels"],
        ),
        (OutlineControl, KINDS),
        (AmbientOcclusionControl, KINDS),
        (PickingControl, KINDS),
        (
            LoadingIndicator,
            ["multiscale_image", "multiscale_labels", "mesh", "multiscale_mesh"],
        ),
        (LoadingConfigControl, MULTISCALE_VOLUME_KINDS),
        (MeshSectionControl, ["mesh", "multiscale_mesh"]),
        (MeshLodControl, ["multiscale_mesh"]),
        (TrailControl, ["graph"]),
        (DatasetInfoControl, KINDS),
    ],
)
def test_each_model_is_allowed_on_the_kinds_of_the_design_table(control_cls, kinds):
    """Design section 4: accepted on its kinds, refused on every other."""
    for kind in KINDS:
        item = _make(control_cls, kind)
        if kind in kinds:
            assert validate_controls([item], kind, "v") == [item]
        else:
            with pytest.raises(ValueError, match=control_cls.__name__):
                validate_controls([item], kind, "v")


def test_no_level_of_detail_field_is_left_in_the_field_table():
    assert all("level_of_detail" not in fields for fields in FIELD_KINDS.values())


def test_attenuation_is_a_multiscale_image_field_only():
    assert "attenuation" in FIELD_KINDS["multiscale_image"]
    assert "attenuation" not in FIELD_KINDS["image"]


# ---------------------------------------------------------------------------
# No controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("controls", [None, [], ()])
def test_none_and_an_empty_list_are_no_controls(kind, controls):
    assert validate_controls(controls, kind, "v") is None


# ---------------------------------------------------------------------------
# The accepted form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_valid_list_comes_back_as_a_new_list_in_the_same_order(kind):
    controls = [
        DatasetInfoControl(),
        _field_item(kind, ["opacity", "visible"]),
        BoundingBoxControl(),
    ]

    validated = validate_controls(controls, kind, "v")

    assert validated == controls
    assert validated is not controls
    assert all(a is b for a, b in zip(validated, controls))


@pytest.mark.parametrize("kind", KINDS)
def test_every_allowed_model_together_is_accepted(kind):
    controls = [_make(cls, kind) for cls in ALLOWED_CONTROLS[kind]]
    assert validate_controls(controls, kind, "v") == controls


@pytest.mark.parametrize("kind", KINDS)
def test_every_field_name_of_a_kind_is_accepted(kind):
    item = _field_item(kind, list(FIELD_KINDS[kind]))
    assert validate_controls([item], kind, "v") == [item]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", [True, False])
def test_true_and_false_are_refused_with_their_own_message(kind, value):
    """Design D5: explicit only; there is no default list."""
    with pytest.raises(TypeError, match="list the controls to show") as excinfo:
        validate_controls(value, kind, "my visual")
    assert "None" in str(excinfo.value)
    assert "my visual" in str(excinfo.value)


@pytest.mark.parametrize("kind", KINDS)
def test_an_object_that_is_not_a_list_is_refused_naming_the_list_form(kind):
    """What an old config object, or any other value, gets."""

    class OldConfig:
        appearance = True

    with pytest.raises(TypeError, match="a list of control models") as excinfo:
        validate_controls(OldConfig(), kind, "v")
    assert "OldConfig" in str(excinfo.value)


def test_a_single_control_outside_a_list_is_refused():
    with pytest.raises(TypeError, match=r"controls=\[BoundingBoxControl"):
        validate_controls(BoundingBoxControl(), "image", "v")


def test_a_control_class_in_place_of_an_instance_is_refused():
    with pytest.raises(TypeError, match=r"write BoundingBoxControl\(\)"):
        validate_controls([BoundingBoxControl], "image", "v")


@pytest.mark.parametrize("item", ["visible", 3, None, {"visible": True}])
def test_an_item_that_is_not_a_control_is_refused(item):
    with pytest.raises(TypeError, match="a list of control models"):
        validate_controls([BoundingBoxControl(), item], "points", "v")


# ---------------------------------------------------------------------------
# Items a kind does not have
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", IMAGE_KINDS)
def test_field_controls_on_an_image_names_the_image_control(kind):
    with pytest.raises(ValueError, match=r"ImageControl\(fields="):
        validate_controls([FieldControls(["visible"])], kind, "v")


@pytest.mark.parametrize("kind", FIELD_CONTROL_KINDS)
def test_an_image_control_on_another_visual_names_field_controls(kind):
    with pytest.raises(ValueError, match=r"FieldControls\("):
        validate_controls([ImageControl(fields=["visible"])], kind, "v")


def test_a_mesh_lod_control_on_an_in_memory_mesh_is_refused():
    """Design F1: it built nothing and said nothing before."""
    with pytest.raises(ValueError, match="add_multiscale_mesh"):
        validate_controls([MeshLodControl()], "mesh", "v")
    assert validate_controls([MeshLodControl()], "multiscale_mesh", "v")


def test_the_loading_indicator_stays_allowed_on_an_in_memory_mesh():
    """Design F4."""
    assert validate_controls([LoadingIndicator()], "mesh", "v")


def test_the_refusal_names_the_method_the_visual_and_what_is_available():
    with pytest.raises(ValueError) as excinfo:
        validate_controls([TrailControl()], "points", "my points")

    message = str(excinfo.value)
    assert "TrailControl" in message
    assert "add_points" in message
    assert "my points" in message
    assert "FieldControls" in message


# ---------------------------------------------------------------------------
# The same type twice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_the_same_type_twice_is_refused(kind):
    """Design D7."""
    with pytest.raises(ValueError, match="BoundingBoxControl more than once"):
        validate_controls(
            [BoundingBoxControl(), DatasetInfoControl(), BoundingBoxControl()],
            kind,
            "v",
        )


@pytest.mark.parametrize("kind", KINDS)
def test_two_field_controls_are_refused(kind):
    with pytest.raises(ValueError, match="more than once"):
        validate_controls(
            [_field_item(kind, ["visible"]), _field_item(kind, ["opacity"])],
            kind,
            "v",
        )


# ---------------------------------------------------------------------------
# Field names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_a_misspelled_field_is_refused_with_the_nearest_match(kind):
    # Built, not written out: the typos hook corrects misspelled literals.
    misspelled = "visi" + "ble" + "s"
    with pytest.raises(ValueError, match="Did you mean 'visible'"):
        validate_controls([_field_item(kind, [misspelled])], kind, "v")


def test_the_motivating_misspelling_suggests_the_colormap_field():
    with pytest.raises(ValueError, match="Did you mean 'color_map'"):
        validate_controls([ImageControl(fields=["colo" + "ur" + "_map"])], "image", "v")


@pytest.mark.parametrize("kind", KINDS)
def test_a_name_with_no_near_match_raises_without_a_suggestion(kind):
    with pytest.raises(ValueError) as excinfo:
        validate_controls([_field_item(kind, ["zzzzzzzz"])], kind, "v")

    message = str(excinfo.value)
    assert "Did you mean" not in message
    assert "zzzzzzzz" in message
    for field in FIELD_KINDS[kind]:
        assert field in message


@pytest.mark.parametrize("kind", KINDS)
def test_the_first_bad_name_is_reported_among_valid_ones(kind):
    with pytest.raises(ValueError, match="nonsense"):
        validate_controls(
            [_field_item(kind, ["visible", "nonsense", "opacity"])], kind, "v"
        )


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("image", "attenuation"),
        ("image", "salt"),
        ("labels", "color_map"),
        ("labels", "iso_threshold"),
        ("mesh", "size"),
        ("points", "thickness"),
        ("lines", "size"),
        ("graph", "color"),
        ("points", "node_size"),
    ],
)
def test_a_field_of_another_kind_is_refused(kind, field):
    """A real name on the wrong visual: the vocabulary is per kind."""
    with pytest.raises(ValueError, match=field):
        validate_controls([_field_item(kind, [field])], kind, "v")


@pytest.mark.parametrize("kind", MULTISCALE_VOLUME_KINDS)
def test_level_of_detail_as_a_field_names_the_control_to_add(kind):
    with pytest.raises(ValueError, match=r"add LevelOfDetailControl\(\)"):
        validate_controls([_field_item(kind, ["level_of_detail"])], kind, "v")


@pytest.mark.parametrize(
    "kind", [kind for kind in KINDS if kind not in MULTISCALE_VOLUME_KINDS]
)
def test_level_of_detail_as_a_field_on_a_kind_without_it(kind):
    with pytest.raises(ValueError, match="LevelOfDetailControl") as excinfo:
        validate_controls([_field_item(kind, ["level_of_detail"])], kind, "v")
    assert "multiscale" in str(excinfo.value)


@pytest.mark.parametrize("kind", MULTISCALE_VOLUME_KINDS)
@pytest.mark.parametrize("old", ["lod_bias", "settled_lod_bias"])
def test_the_removed_bias_keys_name_the_control_too(kind, old):
    with pytest.raises(ValueError, match="LevelOfDetailControl"):
        validate_controls([_field_item(kind, [old])], kind, "v")


# ---------------------------------------------------------------------------
# Checks that need no visual kind: the models' own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("control_cls", [ImageControl, FieldControls])
def test_fields_are_required(control_cls):
    with pytest.raises(TypeError):
        control_cls()


@pytest.mark.parametrize("control_cls", [ImageControl, FieldControls])
def test_an_empty_field_list_is_refused(control_cls):
    with pytest.raises(ValueError, match="at least one field"):
        control_cls(fields=[])


@pytest.mark.parametrize("control_cls", [ImageControl, FieldControls])
@pytest.mark.parametrize("fields", [True, "visible", None, {"visible"}])
def test_fields_that_are_not_a_list_are_refused(control_cls, fields):
    """``True`` has no meaning: there is no "every field" value (design D5)."""
    with pytest.raises(TypeError, match="must be a list of field names"):
        control_cls(fields=fields)


@pytest.mark.parametrize("control_cls", [ImageControl, FieldControls])
def test_a_field_named_twice_is_refused(control_cls):
    with pytest.raises(ValueError, match="'visible' more than once"):
        control_cls(fields=["visible", "opacity", "visible"])


def test_field_controls_takes_its_list_positionally_and_keeps_its_order():
    assert FieldControls(["size", "color", "visible"]).fields == [
        "size",
        "color",
        "visible",
    ]


def test_image_decimals_default_to_two():
    assert ImageControl(fields=["clim"]).decimals == 2


@pytest.mark.parametrize("value", [-1, -5])
def test_negative_image_decimals_are_refused(value):
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        ImageControl(fields=["clim"], decimals=value)


@pytest.mark.parametrize("value", [1.5, "2", True])
def test_non_integer_image_decimals_are_refused(value):
    with pytest.raises(TypeError, match="decimals must be an int"):
        ImageControl(fields=["clim"], decimals=value)


def test_trail_axes_default_to_none():
    assert TrailControl().axes is None
    assert TrailControl(axes=["t", 3]).axes == ["t", 3]


@pytest.mark.parametrize("axes", [True, "t", [], [1.5], [True], 3])
def test_trail_axes_of_the_wrong_shape_are_refused(axes):
    with pytest.raises(TypeError, match=r"TrailControl\.axes"):
        TrailControl(axes=axes)


def test_dataset_info_defaults_to_asking_the_store():
    assert DatasetInfoControl().info is None


def test_the_base_class_has_no_fields():
    import dataclasses

    assert dataclasses.fields(Control) == ()
