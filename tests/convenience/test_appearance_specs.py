"""Unit tests for the toolkit-neutral appearance decision layer.

``layout._shared.appearance_specs`` answers "which controls, in what order,
with what values" for both renderers, and is **pure** -- no controller, no
widgets, no toolkit import.  It walks the visual's ``controls=`` list and
hands each item to its builder, so the order of the list is the order of the
specs (``plans/ordered_controls_list_design.md`` section 6).

The first sections construct no widget and no viewer.  The order tests at
the end add real visuals through an offscreen ``Viewer``, so that every kind
of visual is covered with the store a dock would be given.
"""

from __future__ import annotations

import warnings

import pytest

from cellier.convenience.gui._controls_config import (
    BoundingBoxControl,
    ClippingPlanesControl,
    DatasetInfoControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingIndicator,
    MeshLodControl,
    RenderPlanesControl,
    TrailControl,
)
from cellier.convenience.layout._shared import (
    appearance_specs,
    appearance_targets,
    next_selection,
    unique_labels,
)
from cellier.visuals._base_visual import AABBParams
from tests.convenience._controls_lists import PARITY_CONTROLS
from tests.convenience._spec_parity_cases import build_case


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


def _image(*fields, **options):
    """An ``ImageControl`` naming *fields*."""
    return ImageControl(fields=list(fields), **options)


def kinds(result) -> list[str]:
    """The spec kinds, in order."""
    return [spec.kind for spec in result.specs]


# ── the field predicate matrix ───────────────────────────────────────────────


def test_every_multiscale_field_maps_to_a_control():
    result = appearance_specs(
        _multiscale(),
        [
            _image("color_map", "clim", "render_mode", "iso_threshold", "attenuation"),
            LevelOfDetailControl(),
            BoundingBoxControl(),
            LoadingIndicator(),
        ],
    )
    assert kinds(result) == ["image", "level_of_detail", "aabb", "loading"]
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
        result = appearance_specs(_multiscale(), [_image(*fields)])
        assert kinds(result) == ["image"], fields


def test_the_image_fields_choose_rows_and_do_not_order_them():
    """Design D1: the image control has its own layout."""
    result = appearance_specs(_in_memory(), [_image("clim", "visible", "color_map")])
    assert result.specs[0].values["fields"] == ["visible", "color_map", "clim"]


def test_only_what_is_listed_is_built():
    """Nothing comes along: no bounding box, no loading indicator."""
    result = appearance_specs(_multiscale(), [_image("clim")])
    assert kinds(result) == ["image"]


# A field the kind of visual has no control for -- a misspelling, or a real
# name of another kind -- does not reach this function: ``validate_controls``
# refuses it when the visual is added.  See
# ``test_controls_list_validation.py``.


def test_an_image_field_missing_from_the_visuals_model_is_skipped():
    """A multiscale field on an in-memory visual: the model is narrower."""
    result = appearance_specs(
        _in_memory(), [_image("clim", "attenuation"), BoundingBoxControl()]
    )
    assert kinds(result) == ["image", "aabb"]
    assert result.specs[0].values["fields"] == ["clim"]
    assert result.skipped == ["attenuation"]


def test_a_mesh_field_the_model_lacks_is_skipped_and_reported():
    """A flat mesh has no ``shininess``; every name is one the caller wrote."""
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    visual = _FakeVisual(MeshFlatAppearance())

    result = appearance_specs(
        visual, [FieldControls(["wireframe", "shininess", "visible", "flat_shading"])]
    )

    assert kinds(result) == ["wireframe", "visible"]
    assert result.skipped == ["flat_shading", "shininess"]


def test_an_image_control_with_no_field_the_model_carries_builds_nothing():
    result = appearance_specs(_in_memory(), [_image("attenuation")])
    assert result.specs == []
    assert result.skipped == ["attenuation"]


def test_the_level_of_detail_control_needs_a_model_with_levels():
    """An in-memory image has no settled bias: no spec, and no report."""
    result = appearance_specs(
        _in_memory(), [_image("clim"), LevelOfDetailControl(), BoundingBoxControl()]
    )
    assert kinds(result) == ["image", "aabb"]
    assert result.skipped == []


@pytest.mark.parametrize("controls", [None, []])
def test_no_list_yields_no_specs(controls):
    result = appearance_specs(_in_memory(), controls)
    assert result.specs == []
    assert result.skipped == []


def test_a_visual_with_no_appearance_gets_no_field_control():
    class _NoAppearance:
        id = "v0"
        aabb = AABBParams()

    result = appearance_specs(
        _NoAppearance(), [_image("color_map"), BoundingBoxControl()]
    )
    assert kinds(result) == ["aabb"]
    assert result.skipped == ["color_map"]


# ── the values carried on each spec ──────────────────────────────────────────


def test_clim_range_is_inferred_from_the_current_clim():
    """Widened to include 0 and 1, matching what both builders used to do."""
    result = appearance_specs(_in_memory(clim=(-5.0, 200.0)), [_image("clim")])
    (image_spec,) = result.specs
    assert image_spec.values["clim_range"] == [-5.0, 200.0]
    assert image_spec.values["single"]["clim"] == [-5.0, 200.0]


def test_a_clim_inside_the_unit_interval_is_widened_to_it():
    result = appearance_specs(_in_memory(clim=(0.25, 0.75)), [_image("clim")])
    assert result.specs[0].values["clim_range"] == [0.0, 1.0]


def test_a_configured_clim_range_wins_over_the_inferred_one():
    result = appearance_specs(
        _in_memory(clim=(0.0, 1.0)), [_image("clim", clim_range=(0.0, 4095.0))]
    )
    assert result.specs[0].values["clim_range"] == [0.0, 4095.0]


def test_the_image_spec_carries_the_configured_decimals():
    default = appearance_specs(_in_memory(clim=(0.0, 1.0)), [_image("clim")])
    configured = appearance_specs(
        _in_memory(clim=(0.0, 1.0)), [_image("clim", decimals=0)]
    )
    assert default.specs[0].values["decimals"] == 2
    assert configured.specs[0].values["decimals"] == 0


def test_the_image_spec_widens_clim_range_over_every_channel():
    """Composite channels count too, so every contrast slider fits its limits."""
    from cellier.visuals import InMemoryImageChannelAppearance

    visual = _in_memory()
    visual.channels = {0: InMemoryImageChannelAppearance(clim=(0.0, 4096.0))}
    result = appearance_specs(visual, [_image("render_mode")])
    assert result.specs[0].values["clim_range"] == [0.0, 4096.0]


def test_colormap_names_come_from_the_image_control():
    result = appearance_specs(
        _in_memory(), [_image("color_map", colormap_names=["magma", "grays"])]
    )
    assert result.specs[0].values["colormap_names"] == ["magma", "grays"]
    # Normalised through ``colormap_to_str``, which is what the widgets take.
    assert result.specs[0].values["single"]["color_map"] == "colorbrewer:greys"


def test_the_aabb_spec_is_seeded_from_the_visual_not_from_defaults():
    visual = _in_memory()
    visual.aabb = AABBParams(enabled=True, line_width=7.5, color="#ff00ff")

    result = appearance_specs(visual, [_image("color_map"), BoundingBoxControl()])
    aabb_spec = result.specs[-1]
    assert aabb_spec.kind == "aabb"
    assert aabb_spec.values == {
        "initial_enabled": True,
        "initial_line_width": 7.5,
        "initial_color": "#ff00ff",
    }


def test_dataset_info_rows_are_built_only_when_non_empty():
    result = appearance_specs(
        _multiscale(),
        [_image("color_map"), DatasetInfoControl(info=[("Scale levels", "4")])],
    )
    assert kinds(result) == ["image", "dataset_info"]
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}

    empty = appearance_specs(
        _multiscale(), [_image("color_map"), DatasetInfoControl(info=())]
    )
    assert kinds(empty) == ["image"]


def test_dataset_info_rows_are_coerced_to_strings():
    """A value read off a store is rarely already a string.

    Both front ends display what they are given verbatim, so the neutral layer
    is where an int shape or a numpy dtype becomes text -- once, rather than
    once per toolkit.
    """
    result = appearance_specs(
        _multiscale(), [DatasetInfoControl(info=[("Scale levels", 4)])]
    )
    assert result.specs[-1].values == {"rows": [("Scale levels", "4")]}


def test_titles_are_shared_by_both_front_ends():
    """One title per kind, defined once -- the Qt group box reads these."""
    result = appearance_specs(
        _multiscale(),
        [
            _image("color_map", "clim", "render_mode"),
            LevelOfDetailControl(),
            BoundingBoxControl(),
            LoadingIndicator(),
        ],
    )
    assert [spec.title for spec in result.specs] == [
        "Image",
        "Level of detail",
        "Bounding box",
        "Data fetch status",
    ]


# ── the same field name on two kinds of visual ───────────────────────────────


def test_the_same_field_name_means_different_controls_per_model():
    """``render_mode`` is not one control; the control model decides.

    The image models spell it ``mip``/``iso``/``minip`` and the labels models
    ``iso_categorical``/``flat_categorical``.  An image's fields are drawn by
    ``ImageControl`` and a labels visual's by ``FieldControls``, so a labels
    visual cannot be handed the image widget.
    """
    from cellier.visuals._label_memory import InMemoryLabelsAppearance

    labels_visual = _FakeVisual(InMemoryLabelsAppearance())
    image_result = appearance_specs(_in_memory(), [_image("render_mode")])
    labels_result = appearance_specs(labels_visual, [FieldControls(["render_mode"])])

    assert kinds(image_result) == ["image"]
    assert kinds(labels_result) == ["labels_render_mode"]
    # A single-field kind carries the field's value, plus the Literal's own
    # options where it has them -- which is how one field name serves both
    # the in-memory and the multiscale render-mode vocabularies.
    assert labels_result.specs[0].values == {
        "initial_value": "iso_categorical",
        "choices": (
            "iso_categorical",
            "flat_categorical",
            "smooth_iso",
            "smooth_geometry",
            "plane",
        ),
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
    config_a = [_image("clim")]
    config_b = [_image("color_map")]
    # Registered b before a: the order the user added them, not id or scene order.
    viewer = _FakeViewer([first, second], {"b": config_b, "a": config_a})

    targets = appearance_targets(viewer)

    assert [t.key for t in targets] == ["b", "a"]
    assert [t.visual for t in targets] == [second, first]
    assert [t.config for t in targets] == [config_b, config_a]
    assert [t.label for t in targets] == ["second", "first"]
    assert [t.visual_ids for t in targets] == [["b"], ["a"]]


def test_an_empty_list_is_not_a_target():
    visuals = [_FakeVisual(None, visual_id=key) for key in ("a", "b", "c")]
    viewer = _FakeViewer(
        visuals,
        {"a": [], "b": [_image("clim")], "c": [DatasetInfoControl()]},
    )

    assert [t.key for t in appearance_targets(viewer)] == ["b", "c"]


def test_a_list_with_no_field_control_is_a_target():
    """Design F3: no item depends on another being listed."""
    visuals = [_FakeVisual(None, visual_id=key) for key in ("a", "b", "c")]
    viewer = _FakeViewer(
        visuals,
        {
            "a": [ClippingPlanesControl()],
            "b": [RenderPlanesControl()],
            "c": [BoundingBoxControl()],
        },
    )

    assert [t.key for t in appearance_targets(viewer)] == ["a", "b", "c"]


def test_a_group_whose_items_all_draw_elsewhere_is_not_a_target():
    """Design F2: the viewer says where the two special controls draw.

    An ``OrthoViewer`` draws clipping planes controls in their own dock and
    names no render planes target for its 2D group, so a group left with
    nothing to draw is not offered: its panel would be empty.
    """
    visuals = [_FakeVisual(None, visual_id=key) for key in ("2d", "3d", "c", "d")]
    viewer = _FakeViewer(
        visuals,
        {
            "2d": [RenderPlanesControl(), ClippingPlanesControl()],
            "3d": [RenderPlanesControl(), ClippingPlanesControl()],
            "c": [ClippingPlanesControl()],
            "d": [ClippingPlanesControl(), BoundingBoxControl()],
        },
    )
    viewer._CLIPPING_DOCK_NODE = "OrthoClippingControls"
    asked = []

    def render_planes_target(visual_ids):
        asked.append(visual_ids)
        return None if visual_ids == ["2d"] else object()

    viewer._render_planes_target = render_planes_target

    assert [t.key for t in appearance_targets(viewer)] == ["3d", "d"]
    assert asked == [["2d"], ["3d"]]


def test_targets_expand_a_group_and_name_it_by_the_add():
    panels = [
        _FakeVisual(None, visual_id=key, name=f"cells_{key}")
        for key in ("xy", "xz", "yz", "vol")
    ]
    config = [_image("clim")]
    viewer = _FakeViewer(
        panels, {"xy": config}, groups={"xy": ["xy", "xz", "yz", "vol"]}
    )

    (target,) = appearance_targets(viewer)

    assert target.visual is panels[0]
    assert target.visual_ids == ["xy", "xz", "yz", "vol"]
    assert target.label == "cells"


def test_targets_skip_ids_the_controller_does_not_know():
    known = _FakeVisual(None, visual_id="a")
    config = [_image("clim")]
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
    configs = configs or {key: [_image("clim")] for key in keys}
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
    configs = {key: [_image("clim")] for key in "ab"}
    selected = _targets("a", "b", configs=configs)[0]
    configs["c"] = [_image("clim")]
    assert next_selection(selected, _targets("a", "b", "c", configs=configs)).key == "a"


def test_selection_falls_back_to_the_first_when_its_visual_goes():
    selected = _targets("a", "b")[1]
    assert next_selection(selected, _targets("a")).key == "a"


def test_selection_follows_a_rekeyed_group_by_its_list():
    """An ortho group whose representative was removed is re-keyed, not gone.

    Matched by identity: two lists that are equal are still two groups.
    """
    config = [_image("clim")]
    other = [_image("clim")]
    selected = _targets("a", "xy", configs={"a": other, "xy": config})[1]

    after = _targets("a", "xz", configs={"a": other, "xz": config})

    assert next_selection(selected, after).key == "xz"


def test_no_targets_selects_nothing():
    assert next_selection(None, []) is None
    assert next_selection(_targets("a")[0], []) is None


# ── dataset_info: the three forms of the setting ─────────────────────────────


class _FakeStore:
    """A minimal store: the one method ``DatasetInfoControl()`` calls."""

    def __init__(self, info):
        self._info = info

    def dataset_info(self):
        return self._info


def test_dataset_info_asks_the_store_to_describe_itself():
    """Rows come from the store, not the caller, so they stay correct."""
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    result = appearance_specs(
        _multiscale(), [_image("color_map"), DatasetInfoControl()], _FakeStore(info)
    )
    assert kinds(result) == ["image", "dataset_info"]
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_without_a_store_builds_no_block():
    """A block asserting that a store has no metadata is worse than no block."""
    result = appearance_specs(
        _multiscale(), [_image("color_map"), DatasetInfoControl()]
    )
    assert kinds(result) == ["image"]


def test_dataset_info_accepts_a_prebuilt_dataset_info():
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Nodes", "3")])])
    result = appearance_specs(_multiscale(), [DatasetInfoControl(info=info)])
    assert result.specs[-1].values == {"info": info}


def test_dataset_info_alone_builds_its_block():
    """Design F3: it used to build nothing unless a field was also asked for."""
    from cellier.data._dataset_info import DatasetInfo, RowSection

    info = DatasetInfo(sections=[RowSection(None, [("Points", "12")])])
    result = appearance_specs(_multiscale(), [DatasetInfoControl()], _FakeStore(info))
    assert kinds(result) == ["dataset_info"]


def test_the_loading_indicator_is_for_a_visual_that_loads():
    """Only a multiscale visual loads progressively (design v3 5.13)."""
    from cellier.visuals import ImageVisual

    on = appearance_specs(_multiscale(), [_image("color_map"), LoadingIndicator()])
    assert kinds(on)[-1] == "loading"
    assert on.specs[-1].title == "Data fetch status"
    # It is not on by default any more: only when listed.
    assert "loading" not in kinds(appearance_specs(_multiscale(), [_image("clim")]))
    # An in-memory image has nothing to load.
    in_memory = ImageVisual(name="image", data_store_id="store")
    assert "loading" not in kinds(
        appearance_specs(in_memory, [_image("color_map"), LoadingIndicator()])
    )


# ── the order of the list is the order of the specs ──────────────────────────
#
# Real visuals from here on: one per case of the parity golden, added through
# an offscreen ``Viewer``.

ORDER_CASES = sorted(name for name in PARITY_CONTROLS if name.endswith("/all"))


def _specs(case, controls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = appearance_specs(
            case.visual, controls, case.store, palette=case.palette
        )
    assert result.skipped == []
    return [(spec.kind, spec.title) for spec in result.specs]


@pytest.fixture
def parity_case(request, tmp_path):
    with warnings.catch_warnings():
        # A phong mesh in a scene with no lights warns; not what is tested.
        warnings.simplefilter("ignore")
        case = build_case(request.param.split("/")[0], tmp_path)
    yield request.param, case
    case.viewer.controller.close()


@pytest.mark.parametrize("parity_case", ORDER_CASES, indirect=True)
def test_the_specs_follow_the_list_for_every_kind(parity_case):
    """Reverse the list and the specs come out item by item in reverse."""
    name, case = parity_case
    controls = PARITY_CONTROLS[name]()
    per_item = [_specs(case, [item]) for item in controls]
    # Every item of an "all" list builds something for its visual.
    assert all(per_item), name

    assert _specs(case, controls) == [spec for specs in per_item for spec in specs]
    assert _specs(case, controls[::-1]) == [
        spec for specs in reversed(per_item) for spec in specs
    ]


@pytest.mark.parametrize("parity_case", ORDER_CASES, indirect=True)
def test_a_rotated_list_moves_the_first_control_to_the_end(parity_case):
    name, case = parity_case
    controls = PARITY_CONTROLS[name]()
    first = _specs(case, controls[:1])

    assert _specs(case, [*controls[1:], controls[0]]) == [
        *_specs(case, controls[1:]),
        *first,
    ]


@pytest.mark.parametrize("parity_case", ["points/all"], indirect=True)
def test_field_controls_expand_in_their_own_order(parity_case):
    """Design 1.2: the names are drawn in the order written."""
    _name, case = parity_case

    specs = _specs(
        case,
        [
            BoundingBoxControl(),
            FieldControls(["size", "color", "visible"]),
            ClippingPlanesControl(),
        ],
    )

    assert [kind for kind, _title in specs] == [
        "aabb",
        "size",
        "color",
        "visible",
        "clipping_planes",
    ]


@pytest.mark.parametrize("parity_case", ["mesh_flat/all"], indirect=True)
def test_an_item_the_visual_cannot_support_leaves_the_rest_in_order(parity_case):
    """An in-memory mesh has no LOD settings; no store, no clipping planes."""
    _name, case = parity_case
    controls = [
        DatasetInfoControl(info=[("Sample", "A1")]),
        MeshLodControl(),
        BoundingBoxControl(),
        ClippingPlanesControl(),
        FieldControls(["opacity", "visible"]),
    ]

    result = appearance_specs(case.visual, controls, None, palette=case.palette)

    assert kinds(result) == ["dataset_info", "aabb", "opacity", "visible"]
    assert result.skipped == []


@pytest.mark.parametrize("parity_case", ["graph/all"], indirect=True)
def test_the_trail_control_is_placed_where_it_is_listed(parity_case):
    _name, case = parity_case

    specs = _specs(
        case, [BoundingBoxControl(), TrailControl(), FieldControls(["node_visible"])]
    )

    assert [kind for kind, _title in specs] == ["aabb", "trail", "node_visible"]
