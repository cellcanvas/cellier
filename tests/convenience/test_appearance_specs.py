"""Unit tests for the toolkit-neutral appearance decision layer.

Stage 1 of ``plans/convenience_cleanup.md`` (section 7.3) moved the "which
controls, in what order, with what values" question out of the two renderers
and into ``layout._shared.appearance_specs``, which is **pure** -- no
controller, no widgets, no toolkit import.  That is the coverage win the
refactor was for: the field-predicate matrix and the clim-range inference were
previously reachable only through a Qt or an anywidget fixture, and each was
tested against only one of the two copies (section 4.2).

Nothing here constructs a widget or a viewer.
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
from cellier.convenience.layout._shared import (
    appearance_specs,
    appearance_targets,
    next_selection,
    unique_labels,
)
from cellier.visuals._base_visual import AABBParams


class _FakeVisual:
    """The two attributes ``appearance_specs`` reads, and an id."""

    def __init__(self, appearance, aabb=None, visual_id="v0", name="image"):
        self.appearance = appearance
        self.aabb = AABBParams() if aabb is None else aabb
        self.id = visual_id
        self.name = name


def _in_memory(**single_kwargs):
    """A real in-memory image model; *single_kwargs* go to its ``single``."""
    from cellier.visuals import ImageVisual, InMemoryImageSingleAppearance

    return ImageVisual(
        name="image",
        data_store_id="store",
        single=InMemoryImageSingleAppearance(color_map="grays", **single_kwargs),
    )


def _multiscale(**single_kwargs):
    """A real multiscale image model; *single_kwargs* go to its ``single``."""
    from cellier.visuals import MultiscaleImageSingleAppearance, MultiscaleImageVisual
    from tests._v2 import level_transforms

    return MultiscaleImageVisual(
        name="image",
        data_store_id="store",
        level_transforms=level_transforms([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]),
        single=MultiscaleImageSingleAppearance(color_map="viridis", **single_kwargs),
    )


def kinds(result) -> list[str]:
    """The spec kinds, in order."""
    return [spec.kind for spec in result.specs]


# ── the field predicate matrix ───────────────────────────────────────────────


def test_every_multiscale_field_maps_to_a_control():
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(
            appearance=[
                "color_map",
                "clim",
                "render_mode",
                "iso_threshold",
                "attenuation",
                "lod_bias",
            ]
        ),
    )
    assert kinds(result) == ["image", "lod_bias", "aabb", "loading"]
    assert result.specs[0].values["fields"] == [
        "color_map",
        "clim",
        "render_mode",
        "iso_threshold",
        "attenuation",
    ]
    assert result.skipped == []


def test_every_image_field_collapses_into_one_control():
    """The unified image control draws them all (unified image design 3.10)."""
    for fields in (
        ["render_mode"],
        ["iso_threshold"],
        ["render_mode", "iso_threshold", "attenuation"],
    ):
        result = appearance_specs(
            _multiscale(), MultiscaleImageControlsConfig(appearance=fields)
        )
        assert kinds(result) == ["image", "aabb", "loading"], fields


def test_order_is_the_config_maps_order_not_the_callers():
    """Group order follows ``APPEARANCE_CONTROLS``, not the ``appearance`` list.

    The ``appearance`` docstring says "in display order"; it never has been.
    Pinned here as well as in the Qt acceptance test so a change shows up
    without a toolkit fixture.
    """
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=["clim", "color_map"])
    )
    assert kinds(result) == ["image", "aabb"]


# A field the config class does not know -- a typo, or a real name that does
# not apply to this config -- no longer reaches this function: stage 3 rejects
# it in ``__post_init__``.  See ``test_controls_config_validation.py``.


def test_a_field_missing_from_the_visuals_model_is_skipped():
    """A multiscale config on an in-memory visual: the model is narrower.

    The config class knows ``lod_bias``; the model does not carry it.  Both
    halves of the predicate have to hold.
    """
    result = appearance_specs(
        _in_memory(), MultiscaleImageControlsConfig(appearance=["clim", "lod_bias"])
    )
    assert kinds(result) == ["image", "aabb"]
    assert result.skipped == ["lod_bias"]


@pytest.mark.parametrize("appearance", [False, []])
def test_no_field_list_yields_no_specs(appearance):
    """``False`` and ``[]`` produce nothing -- not even the AABB."""
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=appearance)
    )
    assert result.specs == []


def test_appearance_true_resolves_to_the_config_classes_default_list():
    """Stage 5 gave the dead ``True`` value the obvious meaning."""
    result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=True)
    )
    assert kinds(result) == ["image", "aabb"]


def test_a_default_list_does_not_report_fields_the_model_lacks():
    """``True`` means "everything this config can drive", and models narrow it.

    A flat mesh has no ``shininess``.  Warning about that on every default
    panel would be noise, so only fields the caller *named* are reported as
    skipped.
    """
    from cellier.convenience.gui._controls_config import MeshControlsConfig
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    visual = _FakeVisual(MeshFlatAppearance())

    assert appearance_specs(visual, MeshControlsConfig(appearance=True)).skipped == []
    assert appearance_specs(
        visual, MeshControlsConfig(appearance=["shininess"])
    ).skipped == ["shininess"]


def test_a_visual_with_no_appearance_yields_no_specs():
    class _NoAppearance:
        id = "v0"
        aabb = AABBParams()

    result = appearance_specs(
        _NoAppearance(), InMemoryImageControlsConfig(appearance=["color_map"])
    )
    assert result.specs == []


# ── the values carried on each spec ──────────────────────────────────────────


def test_clim_range_is_inferred_from_the_current_clim():
    """Widened to include 0 and 1, matching what both builders used to do."""
    result = appearance_specs(
        _in_memory(clim=(-5.0, 200.0)),
        InMemoryImageControlsConfig(appearance=["clim"]),
    )
    (image_spec, _aabb) = result.specs
    assert image_spec.values["clim_range"] == [-5.0, 200.0]
    assert image_spec.values["single"]["clim"] == [-5.0, 200.0]


def test_a_clim_inside_the_unit_interval_is_widened_to_it():
    result = appearance_specs(
        _in_memory(clim=(0.25, 0.75)),
        InMemoryImageControlsConfig(appearance=["clim"]),
    )
    assert result.specs[0].values["clim_range"] == [0.0, 1.0]


def test_a_configured_clim_range_wins_over_the_inferred_one():
    result = appearance_specs(
        _in_memory(clim=(0.0, 1.0)),
        InMemoryImageControlsConfig(appearance=["clim"], clim_range=(0.0, 4095.0)),
    )
    assert result.specs[0].values["clim_range"] == [0.0, 4095.0]


def test_the_image_spec_carries_the_configured_decimals():
    default = appearance_specs(
        _in_memory(clim=(0.0, 1.0)), InMemoryImageControlsConfig(appearance=["clim"])
    )
    configured = appearance_specs(
        _in_memory(clim=(0.0, 1.0)),
        InMemoryImageControlsConfig(appearance=["clim"], decimals=0),
    )
    assert default.specs[0].values["decimals"] == 2
    assert configured.specs[0].values["decimals"] == 0


def test_the_image_spec_widens_clim_range_over_every_channel():
    """Composite channels count too, so every contrast slider fits its limits."""
    from cellier.visuals import InMemoryImageChannelAppearance

    visual = _in_memory()
    visual.channels = {0: InMemoryImageChannelAppearance(clim=(0.0, 4096.0))}
    result = appearance_specs(
        visual, InMemoryImageControlsConfig(appearance=["render_mode"])
    )
    assert result.specs[0].values["clim_range"] == [0.0, 4096.0]


def test_colormap_names_come_from_the_config():
    result = appearance_specs(
        _in_memory(),
        InMemoryImageControlsConfig(
            appearance=["color_map"], colormap_names=["magma", "grays"]
        ),
    )
    assert result.specs[0].values["colormap_names"] == ["magma", "grays"]
    # Normalised through ``colormap_to_str``, which is what the widgets take.
    assert result.specs[0].values["single"]["color_map"] == "colorbrewer:greys"


def test_the_aabb_spec_is_seeded_from_the_visual_not_from_defaults():
    visual = _in_memory()
    visual.aabb = AABBParams(enabled=True, line_width=7.5, color="#ff00ff")

    result = appearance_specs(
        visual, InMemoryImageControlsConfig(appearance=["color_map"])
    )
    aabb_spec = result.specs[-1]
    assert aabb_spec.kind == "aabb"
    assert aabb_spec.values == {
        "initial_enabled": True,
        "initial_line_width": 7.5,
        "initial_color": "#ff00ff",
    }


def test_dataset_info_is_appended_last_and_only_when_non_empty():
    config = MultiscaleImageControlsConfig(
        appearance=["color_map"], dataset_info=[("Scale levels", "4")]
    )
    result = appearance_specs(_multiscale(), config)
    assert kinds(result) == ["image", "aabb", "loading", "dataset_info"]
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}

    config.dataset_info = ()
    assert kinds(appearance_specs(_multiscale(), config)) == [
        "image",
        "aabb",
        "loading",
    ]


def test_dataset_info_rows_are_coerced_to_strings():
    """A value read off a store is rarely already a string.

    Both front ends display what they are given verbatim, so the neutral layer
    is where an int shape or a numpy dtype becomes text -- once, rather than
    once per toolkit.
    """
    config = MultiscaleImageControlsConfig(
        appearance=["color_map"], dataset_info=[("Scale levels", 4)]
    )
    result = appearance_specs(_multiscale(), config)
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}


def test_titles_are_shared_by_both_front_ends():
    """One title per kind, defined once -- the Qt group box reads these."""
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(
            appearance=["color_map", "clim", "render_mode", "lod_bias"]
        ),
    )
    assert [spec.title for spec in result.specs] == [
        "Image",
        "LOD bias",
        "Bounding box",
        "Data fetch status",
    ]


# ── the per-config-class dispatch (design section 6.5.2 decision 6) ──────────


def test_the_same_field_name_can_mean_different_controls_per_config():
    """``render_mode`` is not one control; the config class decides.

    The image models spell it ``mip``/``iso``/``minip`` and the labels models
    ``iso_categorical``/``flat_categorical``.  A global field-name table would
    hand a labels visual the volume-render widget; dispatching on the config
    makes that impossible rather than merely unlikely.
    """

    @dataclass
    class _FakeLabelsConfig(BaseControlsConfig):
        APPEARANCE_CONTROLS: ClassVar[dict[str, str]] = {"render_mode": "labels_render"}

    from cellier.visuals._label_memory import InMemoryLabelsAppearance

    labels_visual = _FakeVisual(InMemoryLabelsAppearance())
    image_result = appearance_specs(
        _in_memory(), InMemoryImageControlsConfig(appearance=["render_mode"])
    )
    labels_result = appearance_specs(
        labels_visual, _FakeLabelsConfig(appearance=["render_mode"])
    )

    assert kinds(image_result)[0] == "image"
    assert kinds(labels_result)[0] == "labels_render"
    # A single-field kind carries the field's value, plus the Literal's own
    # options where it has them -- which is how one labels config serves both
    # the in-memory and the multiscale render-mode vocabularies.
    assert labels_result.specs[0].values == {
        "initial_value": "iso_categorical",
        "choices": ("iso_categorical", "flat_categorical"),
    }


# ── appearance_targets ─────────────────────────────────────


class _FakeController:
    def __init__(self, visuals):
        self._visuals = {visual.id: visual for visual in visuals}

    def get_visual_model(self, visual_id):
        return self._visuals[visual_id]


class _FakeViewer:
    def __init__(self, visuals, configs, groups=None):
        self.controller = _FakeController(visuals)
        self._controls_configs = configs
        self._visual_groups = groups or {}


def test_targets_are_every_configured_visual_in_registration_order():
    first = _FakeVisual(None, visual_id="a", name="first")
    second = _FakeVisual(None, visual_id="b", name="second")
    config_a = InMemoryImageControlsConfig(appearance=["clim"])
    config_b = InMemoryImageControlsConfig(appearance=["color_map"])
    # Registered b before a: the order the user added them, not id or scene order.
    viewer = _FakeViewer([first, second], {"b": config_b, "a": config_a})

    targets = appearance_targets(viewer)

    assert [t.key for t in targets] == ["b", "a"]
    assert [t.visual for t in targets] == [second, first]
    assert [t.config for t in targets] == [config_b, config_a]
    assert [t.label for t in targets] == ["second", "first"]
    assert [t.visual_ids for t in targets] == [["b"], ["a"]]


def test_a_config_asking_for_no_panel_is_not_a_target():
    visuals = [_FakeVisual(None, visual_id=key) for key in ("a", "b", "c")]
    viewer = _FakeViewer(
        visuals,
        {
            "a": InMemoryImageControlsConfig(appearance=False),
            "b": InMemoryImageControlsConfig(appearance=["clim"]),
            "c": InMemoryImageControlsConfig(appearance=True),
        },
    )

    assert [t.key for t in appearance_targets(viewer)] == ["b", "c"]


def test_targets_expand_a_group_and_name_it_by_the_add():
    panels = [
        _FakeVisual(None, visual_id=key, name=f"cells_{key}")
        for key in ("xy", "xz", "yz", "vol")
    ]
    config = InMemoryImageControlsConfig(appearance=["clim"])
    viewer = _FakeViewer(
        panels, {"xy": config}, groups={"xy": ["xy", "xz", "yz", "vol"]}
    )

    (target,) = appearance_targets(viewer)

    assert target.visual is panels[0]
    assert target.visual_ids == ["xy", "xz", "yz", "vol"]
    assert target.label == "cells"


def test_targets_skip_ids_the_controller_does_not_know():
    known = _FakeVisual(None, visual_id="a")
    config = InMemoryImageControlsConfig(appearance=["clim"])
    viewer = _FakeViewer([known], {"gone": config, "a": config})

    assert [t.key for t in appearance_targets(viewer)] == ["a"]


def test_targets_are_empty_without_a_controller_or_a_config():
    visual = _FakeVisual(None, visual_id="a")
    assert appearance_targets(_FakeViewer([visual], {})) == []
    assert appearance_targets(object()) == []


def test_duplicate_names_are_numbered_in_order():
    assert unique_labels(["image", "image", "mesh", "image"]) == [
        "image",
        "image (2)",
        "mesh",
        "image (3)",
    ]


def test_a_numbered_label_does_not_collide_with_a_real_name():
    assert unique_labels(["image (2)", "image", "image"]) == [
        "image (2)",
        "image",
        "image (3)",
    ]


def test_an_empty_name_is_called_visual():
    assert unique_labels(["", ""]) == ["visual", "visual (2)"]


# ── next_selection ───────────────────────────────────────────────────────────


def _targets(*keys, configs=None):
    visuals = [_FakeVisual(None, visual_id=key, name=key) for key in keys]
    configs = configs or {
        key: InMemoryImageControlsConfig(appearance=["clim"]) for key in keys
    }
    return appearance_targets(_FakeViewer(visuals, configs))


def test_selection_starts_on_the_first_target():
    targets = _targets("a", "b")
    assert next_selection(None, targets).key == "a"


def test_selection_is_kept_while_its_visual_exists():
    before = _targets("a", "b")
    selected = before[1]
    after = _targets("a", "b", "c", configs=None)
    # A fresh resolve: new tuples, but the same key.
    assert next_selection(selected, after).key == "b"


def test_a_new_visual_does_not_take_the_selection():
    configs = {key: InMemoryImageControlsConfig(appearance=["clim"]) for key in "ab"}
    selected = _targets("a", "b", configs=configs)[0]
    configs["c"] = InMemoryImageControlsConfig(appearance=["clim"])
    assert next_selection(selected, _targets("a", "b", "c", configs=configs)).key == "a"


def test_selection_falls_back_to_the_first_when_its_visual_goes():
    selected = _targets("a", "b")[1]
    assert next_selection(selected, _targets("a")).key == "a"


def test_selection_follows_a_rekeyed_group_by_config():
    """An ortho group whose representative was removed is re-keyed, not gone."""
    config = InMemoryImageControlsConfig(appearance=["clim"])
    other = InMemoryImageControlsConfig(appearance=["clim"])
    selected = _targets("a", "xy", configs={"a": other, "xy": config})[1]

    after = _targets("a", "xz", configs={"a": other, "xz": config})

    assert next_selection(selected, after).key == "xz"


def test_no_targets_selects_nothing():
    assert next_selection(None, []) is None
    assert next_selection(_targets("a")[0], []) is None


# ── dataset_info: the three forms of the setting ─────────────────────────────


class _FakeStore:
    """A minimal store: the one method ``dataset_info=True`` calls."""

    def __init__(self, info):
        self._info = info

    def dataset_info(self):
        return self._info


def test_dataset_info_true_asks_the_store_to_describe_itself():
    """The point of the generalization: rows come from the store, not the caller.

    Before this, every caller hand-wrote the rows, which is why both example
    scripts asserted a hardcoded ``"2x isotropic"`` that no longer matched an
    anisotropic pyramid.
    """
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=True),
        _FakeStore(info),
    )
    assert kinds(result) == ["image", "aabb", "loading", "dataset_info"]
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_true_without_a_store_builds_no_block():
    """A block asserting that a store has no metadata is worse than no block."""
    config = MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=True)
    assert kinds(appearance_specs(_multiscale(), config)) == [
        "image",
        "aabb",
        "loading",
    ]


def test_dataset_info_accepts_a_prebuilt_dataset_info():
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Nodes", "3")])])
    result = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(appearance=["color_map"], dataset_info=info),
    )
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_is_available_on_every_config_class():
    """It used to live on ``MultiscaleImageControlsConfig`` alone.

    That gate meant a points or graph visual could not show the block at all,
    however well its store could describe itself.
    """
    from cellier.convenience.gui._controls_config import (
        GraphControlsConfig,
        LinesControlsConfig,
        MeshControlsConfig,
        PointsControlsConfig,
    )
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    for config_class in (
        PointsControlsConfig,
        LinesControlsConfig,
        MeshControlsConfig,
        GraphControlsConfig,
        BaseControlsConfig,
    ):
        config = config_class(appearance=["visible"], dataset_info=True)
        result = appearance_specs(_multiscale(), config, _FakeStore(info))
        assert "dataset_info" in kinds(result), config_class.__name__


def test_the_loading_indicator_is_multiscale_only_and_can_be_turned_off():
    """Only a multiscale visual loads progressively (design v3 5.13)."""
    from cellier.visuals import ImageVisual

    fields = ["color_map"]
    on = appearance_specs(
        _multiscale(), MultiscaleImageControlsConfig(appearance=fields)
    )
    assert kinds(on)[-1] == "loading"
    assert on.specs[-1].title == "Data fetch status"
    off = appearance_specs(
        _multiscale(),
        MultiscaleImageControlsConfig(appearance=fields, loading_indicator=False),
    )
    assert "loading" not in kinds(off)
    # An in-memory image under the multiscale config has nothing to load.
    in_memory = ImageVisual(name="image", data_store_id="store")
    assert "loading" not in kinds(
        appearance_specs(in_memory, MultiscaleImageControlsConfig(appearance=fields))
    )
