"""A MaskedImage's rows of the image control, on both toolkits (Phase 8).

Masked image design v2, section 10: the surface rows (thickness, placement,
the projection along the normal), the 2D band's rows (minimum width, rays,
cap), the threshold shown in 2D as well, no render-mode row, and the contrast
and threshold labelled with the image whose units they are in.  The rule is
decided once (:func:`cellier.gui._image_controls.row_visible`); the anywidget
front end applies it in JavaScript, which has no test runner here, so these
tests pin the traits it reads and
``scripts/masked_image/phase8_image_controls_js_check.py`` runs the JS.

``color_map`` is never written here: see ``tests/gui/test_image_controls.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from cellier.convenience import Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND, QT_BACKEND
from cellier.convenience.gui._controls_config import (
    DatasetInfoControl,
    ImageControl,
    LevelOfDetailControl,
    RenderPlanesControl,
    validate_controls,
)
from cellier.convenience.layout._shared import appearance_specs, appearance_targets
from cellier.convenience.layout._walk import build_appearance_widgets
from cellier.data import ImageMemoryStore
from cellier.data._dataset_info import DatasetInfo, RowSection
from cellier.gui._image_controls import (
    MASKED_FIELD_LABELS,
    default_threshold_range,
    image_control_values,
    mode_values,
    row_visible,
    to_model_value,
)
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    MaskedImageRenderConfig,
    MaskedImageSingleAppearance,
    MaskedImageVisual,
    MultiscaleImageSingleAppearance,
    MultiscaleImageVisual,
)
from tests._v2 import level_transforms

#: Every row a MaskedImage's image control has.
FIELDS = [
    "visible",
    "color_map",
    "clim",
    "opacity",
    "iso_threshold",
    "surface_thickness",
    "surface_placement",
    "surface_projection",
    "surface_min_width_px",
    "band_rays",
    "band_cap",
    "target_bricks",
]
#: The rows that can hide on a MaskedImage, and the views they show in.
THREE_D_ONLY = {"surface_projection", "target_bricks"}
TWO_D_ONLY = {"surface_min_width_px", "band_rays", "band_cap"}
SMALL = MaskedImageRenderConfig(
    gpu_budget_bytes=8 * 1024**2, value_gpu_budget_bytes=8 * 1024**2
)
EXAMPLE = (
    Path(__file__).parents[2]
    / "examples"
    / "masked_image"
    / "masked_image_synthetic.py"
)


def _model(**single) -> MaskedImageVisual:
    """A MaskedImage model; no controller needed to seed a control."""
    levels = level_transforms([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]])
    return MaskedImageVisual(
        name="masked",
        data_store_id="mask",
        value_store_id="value",
        level_transforms=levels,
        value_level_transforms=levels,
        single=MaskedImageSingleAppearance(**single),
    )


def _store(name: str, seed: int) -> ImageMemoryStore:
    rng = np.random.default_rng(seed)
    data = rng.random((16, 16, 16)).astype(np.float32)
    return ImageMemoryStore(data=data, name=name)


@pytest.fixture
def masked_viewer(request):
    """A 3D viewer with one MaskedImage listing every row; ``(viewer, model)``."""
    gui = getattr(request, "param", "qt")
    viewer = Viewer(
        spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"), gui=gui
    )
    model = viewer.add_masked_image(
        _store("mask", 0),
        _store("value", 1),
        single=MaskedImageSingleAppearance(iso_threshold=0.5, surface_thickness=2.0),
        render_config=SMALL,
        controls=[ImageControl(fields=FIELDS), DatasetInfoControl()],
    )
    yield viewer, model
    viewer.controller.close()


def _image_widget(viewer, backend):
    (target,) = appearance_targets(viewer)
    widgets = build_appearance_widgets(
        target.visual,
        target.config,
        viewer.controller,
        target.visual_ids,
        backend=backend,
    )
    return widgets[0]


# ---------------------------------------------------------------------------
# The rule, without a toolkit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("field", FIELDS)
def test_which_rows_a_masked_image_shows(field, n):
    """Design 10: the threshold in both views, projection 3D, band rows 2D."""
    expected = (
        (n == 3) if field in THREE_D_ONLY else (n == 2) if field in TWO_D_ONLY else True
    )
    assert row_visible(field, ["iso"], n, masked=True) is expected


def test_an_images_threshold_row_stays_3d_only():
    """The masked rule is opt-in: an image's rows are unchanged."""
    assert row_visible("iso_threshold", ["iso"], 2) is False
    assert row_visible("iso_threshold", ["iso"], 3) is True


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(4.0, (0.0, 8.0)), (0.0, (-1.0, 1.0)), (127.5, (0.0, 255.0)), (-3.0, (-6.0, 0.0))],
)
def test_the_default_threshold_window(threshold, expected):
    assert default_threshold_range(threshold) == expected


def test_a_masked_seed_carries_its_rows_ranges_and_labels():
    values = image_control_values(_model(iso_threshold=4.0), fields=FIELDS)

    assert values["masked"] is True
    assert values["labels"] == MASKED_FIELD_LABELS
    assert values["render_modes"] == ["iso"]
    # The threshold is in the mask's units, not the value's contrast range.
    assert values["clim_range"] == [0.0, 1.0]
    assert values["threshold_range"] == [0.0, 8.0]
    assert values["thickness_range"] == [0.0, 10.0]
    assert values["choices"] == {
        "surface_placement": ["inward", "centred", "outward"],
        "surface_projection": ["max", "mean", "min"],
        "target_bricks": ["visible", "surface"],
    }
    assert values["shared"]["target_bricks"] == "visible"
    assert values["int_ranges"] == {"band_rays": [1, 64], "band_cap": [1, 128]}
    assert values["single"]["band_rays"] == 16
    assert values["single"]["surface_placement"] == "inward"


def test_given_ranges_win():
    values = image_control_values(
        _model(),
        fields=FIELDS,
        threshold_range=(-12.0, 12.0),
        thickness_range=(0.0, 32.0),
    )
    assert values["threshold_range"] == [-12.0, 12.0]
    assert values["thickness_range"] == [0.0, 32.0]


def test_an_images_seed_is_unchanged():
    """No masked key on an image, and no surface field in its mode values."""
    image = MultiscaleImageVisual(
        name="ms",
        data_store_id="store",
        level_transforms=level_transforms([[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]),
    )
    values = image_control_values(image, fields=["clim", "iso_threshold"])
    for key in ("masked", "labels", "threshold_range", "thickness_range", "choices"):
        assert key not in values
    assert set(mode_values(MultiscaleImageSingleAppearance())) == {
        "color_map",
        "clim",
        "opacity",
        "render_mode",
        "iso_threshold",
    }
    # A threshold range given for an image is passed on, for the threshold row.
    given = image_control_values(image, fields=["clim"], threshold_range=(0.0, 9.0))
    assert given["threshold_range"] == [0.0, 9.0]


def test_the_integer_rows_reach_the_model_as_integers():
    assert to_model_value("band_rays", 8.0) == 8
    assert isinstance(to_model_value("band_cap", 12.0), int)


# ---------------------------------------------------------------------------
# The controls list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["render_mode", "attenuation", "slab_projection"])
def test_rows_a_masked_image_has_not_are_refused(field):
    with pytest.raises(ValueError, match="add_masked_image") as excinfo:
        validate_controls([ImageControl(fields=[field])], "masked_image", "m")
    assert "band_rays" in str(excinfo.value)


def test_render_planes_are_refused_for_a_masked_image():
    with pytest.raises(ValueError, match="RenderPlanesControl is not available"):
        validate_controls([RenderPlanesControl()], "masked_image", "m")


def test_every_masked_field_is_accepted():
    items = [ImageControl(fields=FIELDS), LevelOfDetailControl()]
    assert validate_controls(items, "masked_image", "m") == items


def test_the_viewer_records_the_list_and_refuses_a_bad_one():
    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("z", "y", "x"))
    try:
        with pytest.raises(ValueError, match="render_mode"):
            viewer.add_masked_image(
                _store("mask", 0),
                _store("value", 1),
                render_config=SMALL,
                controls=[ImageControl(fields=["render_mode"])],
            )
        assert viewer.controller._model.scenes[viewer.scene.id].visuals == []

        controls = [ImageControl(fields=FIELDS)]
        model = viewer.add_masked_image(
            _store("mask", 0),
            _store("value", 1),
            render_config=SMALL,
            controls=controls,
        )
        assert isinstance(model, MaskedImageVisual)
        assert viewer._controls_configs == {model.id: controls}
    finally:
        viewer.controller.close()


# ---------------------------------------------------------------------------
# The dataset info: both images (design 10, audit row 36)
# ---------------------------------------------------------------------------


def test_the_dataset_info_describes_the_mask_then_the_value(masked_viewer):
    viewer, model = masked_viewer
    controller = viewer.controller
    mask = controller.get_data_store(UUID(model.data_store_id))
    value = controller.get_data_store(UUID(model.value_store_id))

    specs = appearance_specs(model, [DatasetInfoControl()], mask, value_store=value)

    (spec,) = specs.specs
    labels = [section.label for section in spec.values["info"].sections]
    assert labels[0] == "Mask" and "Value" in labels
    assert labels.index("Value") > 0
    rows = dict(spec.values["info"].sections[0].rows)
    assert rows == dict(mask.dataset_info().sections[0].rows)


def test_a_given_dataset_info_is_shown_as_given(masked_viewer):
    viewer, model = masked_viewer
    controller = viewer.controller
    mask = controller.get_data_store(UUID(model.data_store_id))
    value = controller.get_data_store(UUID(model.value_store_id))
    info = DatasetInfo([RowSection(None, [("Source", "hand written")])])

    specs = appearance_specs(model, [DatasetInfoControl(info)], mask, value_store=value)

    assert specs.specs[0].values["info"] is info


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def _qt_shown(widget) -> set[str]:
    return {
        field
        for (_, _, field), (layout, control) in widget._rows.items()
        if layout.isRowVisible(control)
    }


def test_qt_rows_follow_the_view(qtbot, masked_viewer):
    viewer, _model = masked_viewer
    widget = _image_widget(viewer, QT_BACKEND)

    assert ("single", None, "render_mode") not in widget._controls
    assert _qt_shown(widget) == {"iso_threshold", *THREE_D_ONLY}
    viewer.set_displayed_dimensions(("y", "x"))
    assert _qt_shown(widget) == {"iso_threshold", *TWO_D_ONLY}
    viewer.set_displayed_dimensions(("z", "y", "x"))
    assert _qt_shown(widget) == {"iso_threshold", *THREE_D_ONLY}


def test_qt_labels_name_each_images_units(qtbot, masked_viewer):
    viewer, _model = masked_viewer
    widget = _image_widget(viewer, QT_BACKEND)

    for field, label in MASKED_FIELD_LABELS.items():
        layout, control = None, widget._controls[("single", None, field)]
        layout = control.parentWidget().layout()
        assert layout.labelForField(control).text() == label
    iso = widget._controls[("single", None, "iso_threshold")]
    assert (iso.minimum(), iso.maximum()) == (-0.5, 1.5)


@pytest.mark.parametrize(
    ("field", "drive", "expected"),
    [
        ("surface_thickness", lambda c: c.setValue(3.5), 3.5),
        ("iso_threshold", lambda c: c.setValue(0.25), 0.25),
        ("surface_placement", lambda c: c.setCurrentText("outward"), "outward"),
        ("surface_projection", lambda c: c.setCurrentText("mean"), "mean"),
        ("surface_min_width_px", lambda c: c.setValue(2.5), 2.5),
        ("band_rays", lambda c: c.setValue(8), 8),
        ("band_cap", lambda c: c.setValue(12), 12),
    ],
)
def test_qt_rows_write_the_model(qtbot, masked_viewer, field, drive, expected):
    viewer, model = masked_viewer
    widget = _image_widget(viewer, QT_BACKEND)

    drive(widget._controls[("single", None, field)])

    assert getattr(model.single, field) == expected


def test_qt_target_bricks_row_writes_and_follows_the_appearance(qtbot, masked_viewer):
    """The shared row of design 8.3: a combo on the appearance's field."""
    viewer, model = masked_viewer
    widget = _image_widget(viewer, QT_BACKEND)
    combo = widget._controls[("shared", None, "target_bricks")]

    assert combo.currentText() == "visible"
    combo.setCurrentText("surface")
    assert model.appearance.target_bricks == "surface"
    viewer.controller.update_appearance_field(model.id, "target_bricks", "visible")
    assert combo.currentText() == "visible"


def test_qt_rows_follow_the_model(qtbot, masked_viewer):
    viewer, model = masked_viewer
    widget = _image_widget(viewer, QT_BACKEND)
    controller = viewer.controller

    controller.update_single_appearance_field(model.id, "band_cap", 7)
    controller.update_single_appearance_field(model.id, "surface_placement", "centred")
    controller.update_single_appearance_field(model.id, "surface_thickness", 6.0)

    assert widget._controls[("single", None, "band_cap")].value() == 7
    placement = widget._controls[("single", None, "surface_placement")]
    assert placement.currentText() == "centred"
    assert widget._controls[("single", None, "surface_thickness")].value() == 6.0
    # Applied with the signals blocked: the model was not written back.
    assert model.single.band_cap == 7


# ---------------------------------------------------------------------------
# anywidget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked_viewer", ["anywidget"], indirect=True)
def test_anywidget_seeds_the_traits_the_front_end_reads(masked_viewer):
    viewer, _model = masked_viewer
    widget = _image_widget(viewer, ANYWIDGET_BACKEND)

    assert widget.masked is True
    assert widget.labels == MASKED_FIELD_LABELS
    assert widget.threshold_range == [-0.5, 1.5]
    assert widget.thickness_range == [0.0, 10.0]
    assert widget.int_ranges == {"band_rays": [1, 64], "band_cap": [1, 128]}
    assert widget.choices["surface_projection"] == ["max", "mean", "min"]
    assert "render_mode" not in widget.fields
    assert widget.n_displayed_dimensions == 3
    viewer.set_displayed_dimensions(("y", "x"))
    assert widget.n_displayed_dimensions == 2


@pytest.mark.parametrize("masked_viewer", ["anywidget"], indirect=True)
def test_anywidget_writes_and_follows_the_model(masked_viewer):
    viewer, model = masked_viewer
    widget = _image_widget(viewer, ANYWIDGET_BACKEND)

    widget.single = {
        **widget.single,
        "band_rays": 4,
        "surface_min_width_px": 2.5,
        "surface_projection": "min",
        "surface_thickness": 5.0,
    }
    assert model.single.band_rays == 4
    assert model.single.surface_min_width_px == 2.5
    assert model.single.surface_projection == "min"
    assert model.single.surface_thickness == 5.0

    viewer.controller.update_single_appearance_field(model.id, "band_cap", 9)
    assert widget.single["band_cap"] == 9


@pytest.mark.parametrize("masked_viewer", ["anywidget"], indirect=True)
def test_anywidget_target_bricks_writes_and_follows_the_appearance(masked_viewer):
    viewer, model = masked_viewer
    widget = _image_widget(viewer, ANYWIDGET_BACKEND)

    assert widget.choices["target_bricks"] == ["visible", "surface"]
    assert widget.shared["target_bricks"] == "visible"
    widget.shared = {**widget.shared, "target_bricks": "surface"}
    assert model.appearance.target_bricks == "surface"
    viewer.controller.update_appearance_field(model.id, "target_bricks", "visible")
    assert widget.shared["target_bricks"] == "visible"


# ---------------------------------------------------------------------------
# The example, driven by its controls (design 12.2, Phase 8 "done when")
# ---------------------------------------------------------------------------

SIDE = 64
SIZE = 192


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("masked_image_synthetic", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_viewer(example, gui, *flags):
    from tests.render._masked import load

    args = example.parse_args(["--side", str(SIDE), "--budget-mib", "16", *flags])
    viewer = example.build_masked(args, gui=gui)
    load(viewer, size=(SIZE, SIZE))
    return viewer


def _frame(viewer):
    from tests.render._affine import render

    return render(viewer, SIZE)[0][..., :3].astype(np.int64)


def _differ(a, b) -> float:
    """The fraction of pixels whose colour differs by more than 8 codes."""
    return float(np.mean(np.abs(a - b).max(axis=-1) > 8))


def _drive_qt(widget, field, value):
    control = widget._controls[("single", None, field)]
    if hasattr(control, "setCurrentText"):
        control.setCurrentText(value)
    else:
        control.setValue(value)


def _drive_anywidget(widget, field, value):
    widget.single = {**widget.single, field: value}


@pytest.mark.parametrize(
    ("gui", "backend", "drive"),
    [
        ("qt", QT_BACKEND, _drive_qt),
        ("anywidget", ANYWIDGET_BACKEND, _drive_anywidget),
    ],
    ids=["qt", "anywidget"],
)
@pytest.mark.parametrize(
    ("start", "edits", "flags"),
    [
        # A threshold across a band edge (iso 4 in band 1, 12 in band 2).
        ([], {"iso_threshold": 12.0}, ["--threshold", "12"]),
        # Max inward over a band, from iso 4: distances 4 to 12, band 2.
        ([], {"surface_thickness": 8.0}, ["--thickness", "8"]),
        # Min over the same interval is band 1: the thickness-0 surface's.
        (["--thickness", "8"], {"surface_projection": "min"}, []),
    ],
    ids=["threshold", "thickness", "projection"],
)
def test_the_controls_drive_the_example(
    qtbot, example, gui, backend, drive, start, edits, flags
):
    """The example's frame after the edits is the frame of the expected answer.

    The reference is the example started with flags that give the same
    surface.  Each edit must also change the frame, so the comparison is
    not of two unchanged frames.
    """
    viewer = _example_viewer(example, gui, *start)
    reference = _example_viewer(example, gui, *flags)
    try:
        widget = _image_widget(viewer, backend)
        before = _frame(viewer)
        for field, value in edits.items():
            drive(widget, field, value)
        after = _frame(viewer)
        expected = _frame(reference)

        assert int((before.max(axis=-1) > 8).sum()) > 1000
        assert _differ(before, after) > 0.02
        assert _differ(after, expected) < 0.002
    finally:
        viewer.controller.close()
        reference.controller.close()
