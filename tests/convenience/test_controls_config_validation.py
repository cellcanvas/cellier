"""Field-name validation on the controls configs.

Stage 3 of ``plans/convenience_cleanup.md`` (section 9).  Two closed
vocabularies were unvalidated and failed the same silent way: an unknown
``appearance`` name matched no predicate so no control appeared, and an
unknown channel field name fell through every branch of the
channel widget's ``if/elif`` chain.  Neither errored, and the appearance case
was worse than nothing -- a dock still appeared, holding only the bounding
box, so a user concluded the control was unsupported.

Each config class now enforces its own contract in ``__post_init__``.  The
valid set is derived from ``APPEARANCE_CONTROLS``, which is also what
``appearance_specs`` dispatches on, so validation and rendering cannot
disagree (design section 6.5.2 decision 6, and section 10.1's rule that a name
is valid iff a widget exists for it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from cellier.convenience.gui._controls_config import (
    BaseControlsConfig,
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)

_IN_MEMORY_FIELDS = ["color_map", "clim", "render_mode", "iso_threshold"]
_MULTISCALE_ONLY = ["attenuation", "lod_bias"]


# ---------------------------------------------------------------------------
# Valid sets are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", _IN_MEMORY_FIELDS)
def test_in_memory_accepts_its_own_vocabulary(field):
    assert InMemoryImageControlsConfig(appearance=[field]).appearance == [field]


@pytest.mark.parametrize("field", _IN_MEMORY_FIELDS + _MULTISCALE_ONLY)
def test_multiscale_accepts_the_widened_vocabulary(field):
    """The subclass widens the parent's set, matching the model inheritance."""
    assert MultiscaleImageControlsConfig(appearance=[field]).appearance == [field]


def test_a_whole_valid_list_is_accepted():
    fields = _IN_MEMORY_FIELDS + _MULTISCALE_ONLY
    assert MultiscaleImageControlsConfig(appearance=fields).appearance == fields


# ---------------------------------------------------------------------------
# The per-subclass narrowing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", _MULTISCALE_ONLY)
def test_multiscale_only_fields_are_rejected_on_the_in_memory_config(field):
    """The likelier mistake (section 9.1): a real name, wrong config.

    ``lod_bias`` and ``attenuation`` exist on ``MultiscaleImageAppearance``
    and not on ``InMemoryImageAppearance``.  A closed ``Enum`` would not catch
    this -- the member exists either way -- which is why the vocabulary is
    per config class and not one global set.
    """
    with pytest.raises(ValueError, match=field):
        InMemoryImageControlsConfig(appearance=[field])


def test_the_error_names_the_config_class_and_its_valid_set():
    with pytest.raises(ValueError) as excinfo:
        InMemoryImageControlsConfig(appearance=["lod_bias"])

    message = str(excinfo.value)
    assert "InMemoryImageControlsConfig" in message
    for field in _IN_MEMORY_FIELDS:
        assert field in message


# ---------------------------------------------------------------------------
# Typos and their suggestions
# ---------------------------------------------------------------------------


def test_a_typo_is_rejected_with_the_nearest_match():
    """The motivating case: ``colour_map`` -> ``color_map``."""
    with pytest.raises(ValueError, match="Did you mean 'color_map'"):
        InMemoryImageControlsConfig(appearance=["colour_map"])


@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("clims", "clim"),
        ("render_modes", "render_mode"),
        ("lodbias", "lod_bias"),
    ],
)
def test_near_misses_suggest_the_right_field(typo, expected):
    with pytest.raises(ValueError, match=f"Did you mean '{expected}'"):
        MultiscaleImageControlsConfig(appearance=[typo])


def test_a_name_with_no_near_match_still_raises_without_a_suggestion():
    with pytest.raises(ValueError) as excinfo:
        InMemoryImageControlsConfig(appearance=["zzzzzzzz"])

    assert "Did you mean" not in str(excinfo.value)
    assert "zzzzzzzz" in str(excinfo.value)


def test_the_first_bad_name_is_reported_even_among_valid_ones():
    with pytest.raises(ValueError, match="nonsense"):
        InMemoryImageControlsConfig(appearance=["color_map", "nonsense", "clim"])


# ---------------------------------------------------------------------------
# The bool values are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [False, True])
def test_the_hide_show_switch_is_not_a_field_list(value):
    """``appearance=True``/``False`` bypasses validation entirely.

    ``True`` is still a dead value here; stage 5 gives it the "use the default
    list" meaning (section 11.2).
    """
    assert InMemoryImageControlsConfig(appearance=value).appearance is value


def test_an_empty_list_is_accepted():
    assert InMemoryImageControlsConfig(appearance=[]).appearance == []


# ---------------------------------------------------------------------------
# The base class
# ---------------------------------------------------------------------------


def test_the_base_class_drives_only_the_universal_fields():
    """``visible`` and ``opacity`` are on ``BaseAppearance``, so on everything.

    Stage 5 moved them onto the base map (section 6.5.2 decision 7), so every
    config class inherits them and no field is valid on one config and not
    another for no reason in the model.  ``BaseControlsConfig`` itself is
    still never instantiated directly -- it is the shared parent.
    """
    assert BaseControlsConfig(appearance=["visible", "opacity"]).appearance == [
        "visible",
        "opacity",
    ]
    with pytest.raises(ValueError, match="not a valid appearance field"):
        BaseControlsConfig(appearance=["color_map"])


def test_a_config_class_driving_nothing_says_so():
    """The empty-set message, still reachable for a config with no controls."""
    from dataclasses import dataclass as _dataclass

    @_dataclass
    class _NoControls(BaseControlsConfig):
        APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {}

    with pytest.raises(ValueError, match="accepts no appearance fields"):
        _NoControls(appearance=["visible"])


def test_a_new_config_class_gets_validation_from_its_control_map_alone():
    """One declaration, not a valid set kept in sync with a widget table.

    This is what makes section 10.1's rule structural: a subclass that adds a
    control adds a valid name by the same line, and cannot add one without.
    """

    @dataclass
    class _MeshConfig(BaseControlsConfig):
        APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {
            "wireframe": "toggle",
            "shininess": "float_spin",
        }

    assert _MeshConfig(appearance=["wireframe", "shininess"]).appearance == [
        "wireframe",
        "shininess",
    ]
    with pytest.raises(ValueError, match="Did you mean 'wireframe'"):
        _MeshConfig(appearance=["wirefame"])


# ---------------------------------------------------------------------------
# The residual: valid for the class, absent from the model
# ---------------------------------------------------------------------------


def test_a_field_valid_for_the_config_but_absent_from_the_model_is_reported():
    """Validation cannot catch this; only the renderer sees both halves.

    A ``MultiscaleImageControlsConfig`` on an in-memory visual: every name is
    valid for the config class, and the model still has no ``lod_bias``.
    ``appearance_specs`` reports it and the renderers warn (section 9.5
    step 4).
    """
    from cellier.convenience.layout._shared import appearance_specs
    from cellier.visuals import ImageVisual

    visual = ImageVisual(name="v0", data_store_id="store")

    specs, skipped = appearance_specs(
        visual, MultiscaleImageControlsConfig(appearance=["clim", "lod_bias"])
    )

    assert [spec.kind for spec in specs] == ["image", "aabb"]
    assert skipped == ["lod_bias"]


# ---------------------------------------------------------------------------
# decimals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_class", [InMemoryImageControlsConfig, MultiscaleImageControlsConfig]
)
def test_image_decimals_default_to_two(config_class):
    assert config_class().decimals == 2


@pytest.mark.parametrize("value", [-1, -5])
def test_negative_image_decimals_are_rejected(value):
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        MultiscaleImageControlsConfig(decimals=value)


@pytest.mark.parametrize("value", [1.5, "2", True])
def test_non_integer_image_decimals_are_rejected(value):
    with pytest.raises(TypeError, match="decimals must be an int"):
        InMemoryImageControlsConfig(decimals=value)
