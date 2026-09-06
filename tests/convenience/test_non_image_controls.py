"""``controls=`` for the non-image visual types.

Stage 5 of ``plans/convenience_cleanup.md`` (section 11): the API half of
section 8.4.  Stage 4 built the widgets; this makes them reachable by giving
every visual family a config class and every ``add_*`` method a ``controls=``
argument.

The highest-value tests here are the end-to-end ones (section 11.5): a config
with ``appearance=True`` produces a dock containing the expected controls, per
visual type.  That is what actually proves stages 3, 4 and 5 line up -- the
vocabulary, the widgets, and the plumbing are three separate declarations and
nothing else checks that they agree.
"""

from __future__ import annotations

import pytest

from cellier.convenience import AppearanceControls, OrthoViewer, Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.gui._controls_config import (
    GraphControlsConfig,
    LabelsControlsConfig,
    LinesControlsConfig,
    MeshControlsConfig,
    MultiscaleImageControlsConfig,
    MultiscaleLabelsControlsConfig,
    PointsControlsConfig,
)
from cellier.convenience.layout._shared import (
    appearance_specs,
    select_appearance_target,
)
from cellier.convenience.layout._walk import build_appearance_widgets, render_dock
from cellier.visuals import MultiscaleImageAppearance
from cellier.visuals._mesh_memory import MeshFlatAppearance, MeshPhongAppearance

_PANELS = ("xy", "xz", "yz", "vol")


def _add(viewer, kind, stores, **kwargs):
    """Add one visual of *kind* to *viewer*, forwarding ``controls=``."""
    if kind == "labels":
        return viewer.add_labels(stores["labels"], **kwargs)
    if kind == "mesh":
        return viewer.add_mesh(
            stores["mesh"], appearance=MeshFlatAppearance(), **kwargs
        )
    if kind == "points":
        return viewer.add_points(stores["points"], **kwargs)
    if kind == "lines":
        return viewer.add_lines(stores["lines"], **kwargs)
    if kind == "graph":
        return viewer.add_graph(stores["graph"], **kwargs)
    raise AssertionError(kind)


CONFIGS = {
    "labels": LabelsControlsConfig,
    "mesh": MeshControlsConfig,
    "points": PointsControlsConfig,
    "lines": LinesControlsConfig,
    "graph": GraphControlsConfig,
}


@pytest.fixture
def stores(labels_store, mesh_store, points_store, lines_store, graph_store):
    return {
        "labels": labels_store,
        "mesh": mesh_store,
        "points": points_store,
        "lines": lines_store,
        "graph": graph_store,
    }


# ---------------------------------------------------------------------------
# The twelve new signatures record the config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_viewer_add_records_the_config(kind, stores):
    viewer = Viewer(("z", "y", "x"))
    config = CONFIGS[kind](appearance=True)

    visual = _add(viewer, kind, stores, controls=config)

    assert viewer._controls_configs == {visual.id: config}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_viewer_add_without_controls_records_nothing(kind, stores):
    viewer = Viewer(("z", "y", "x"))
    _add(viewer, kind, stores)
    assert viewer._controls_configs == {}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_ortho_add_records_the_config_and_the_panel_group(kind, stores):
    ortho = OrthoViewer(("z", "y", "x"))
    config = CONFIGS[kind](appearance=True)

    visuals = _add(ortho, kind, stores, controls=config)

    panel_ids = [visuals[key].id for key in _PANELS]
    assert ortho._controls_configs == {panel_ids[0]: config}
    assert ortho._visual_groups[panel_ids[0]] == panel_ids


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_ortho_add_without_controls_records_nothing(kind, stores):
    ortho = OrthoViewer(("z", "y", "x"))
    _add(ortho, kind, stores)
    assert ortho._controls_configs == {}
    assert ortho._visual_groups == {}


def test_add_labels_multiscale_takes_the_multiscale_config(multiscale_labels_store):
    from cellier.visuals._labels import MultiscaleLabelsAppearance

    viewer = Viewer(("z", "y", "x"))
    config = MultiscaleLabelsControlsConfig(appearance=["lod_bias"])

    visual = viewer.add_labels_multiscale(
        multiscale_labels_store,
        appearance=MultiscaleLabelsAppearance(),
        controls=config,
    )

    assert viewer._controls_configs == {visual.id: config}


# ---------------------------------------------------------------------------
# End to end: a default panel per visual type (section 11.5's key test)
# ---------------------------------------------------------------------------

EXPECTED_DEFAULT_TITLES = {
    "labels": [
        "Visible",
        "Opacity",
        "Render mode",
        "Salt",
        "Background label",
        "Bounding box",
    ],
    "mesh": [
        "Visible",
        "Opacity",
        "Color",
        "Side",
        "Wireframe",
        "Wireframe thickness",
        "Bounding box",
    ],
    "points": ["Visible", "Opacity", "Color", "Size", "Size space", "Bounding box"],
    "lines": [
        "Visible",
        "Opacity",
        "Color",
        "Thickness",
        "Thickness space",
        "Bounding box",
    ],
    "graph": [
        "Visible",
        "Opacity",
        "Nodes visible",
        "Node color",
        "Node size",
        "Node size space",
        "Edges visible",
        "Edge color",
        "Edge thickness",
        "Edge thickness space",
        "Bounding box",
    ],
}


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_true_builds_the_default_qt_dock(qtbot, kind, stores):
    """The test section 11.5 calls the highest-value one in the group.

    It fails if the vocabulary, the widget catalog and the plumbing disagree
    anywhere -- a valid field with no widget, a widget with no config entry, a
    config the renderer cannot resolve.
    """
    from tests.convenience._qt_acceptance import assert_panel_renders, control_labels

    viewer = Viewer(("z", "y", "x"), gui="qt")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=True))

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == EXPECTED_DEFAULT_TITLES[kind]
    assert_panel_renders(container)


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_true_builds_the_same_anywidget_dock(kind, stores):
    """Same names, same order, other toolkit -- section 4.2's whole point."""
    from tests.convenience._qt_acceptance import control_labels_anywidget

    viewer = Viewer(("z", "y", "x"), gui="anywidget")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=True))
    target = select_appearance_target(viewer)

    built = build_appearance_widgets(
        target.visual,
        target.config,
        viewer.controller,
        target.visual_ids,
        backend=ANYWIDGET_BACKEND,
    )

    assert control_labels_anywidget(built) == EXPECTED_DEFAULT_TITLES[kind]


def test_dataset_info_reaches_both_docks(qtbot, multiscale_image_store):
    """The block is a control like any other, on both front ends.

    It was anywidget-only for as long as the config carried pre-formatted HTML,
    which Qt has nothing to do with; rows are what both toolkits can render, so
    the parity assertion the other controls get now covers this one too.
    """
    from tests.convenience._qt_acceptance import (
        assert_panel_renders,
        control_labels,
        control_labels_anywidget,
    )

    rows = [("Scale levels", "2"), ("Data type", "float32")]
    expected = ["Colormap", "Bounding box", "Dataset info"]

    qt_viewer = Viewer(("z", "y", "x"), gui="qt")
    qt_viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
        controls=MultiscaleImageControlsConfig(
            appearance=["color_map"], dataset_info=rows
        ),
    )
    container = render_dock(AppearanceControls(), qt_viewer, QtLayoutHost(), [])
    assert control_labels(container) == expected
    assert_panel_renders(container)

    any_viewer = Viewer(("z", "y", "x"), gui="anywidget")
    any_viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(color_map="viridis", clim=(0.0, 1.0)),
        controls=MultiscaleImageControlsConfig(
            appearance=["color_map"], dataset_info=rows
        ),
    )
    target = select_appearance_target(any_viewer)
    built = build_appearance_widgets(
        target.visual,
        target.config,
        any_viewer.controller,
        target.visual_ids,
        backend=ANYWIDGET_BACKEND,
    )
    assert control_labels_anywidget(built) == expected


def test_dataset_info_rows_reach_each_front_end_as_data(qtbot):
    """The rows are displayed, not parsed, so both widgets carry them verbatim."""
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.qt import QtDatasetInfo

    rows = [("Path", "<b>a & b</b>")]

    any_widget = AnywidgetDatasetInfo(rows)
    assert any_widget.rows == [["Path", "<b>a & b</b>"]]

    from PySide6.QtWidgets import QLabel

    qt_widget = QtDatasetInfo(rows)
    form = qt_widget._inline_form
    values = [
        form.itemAt(row, form.ItemRole.FieldRole).widget().text()
        for row in range(form.rowCount())
    ]
    assert values == ["<b>a & b</b>"]
    assert isinstance(form.itemAt(0, form.ItemRole.FieldRole).widget(), QLabel)


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_appearance_false_still_hides_the_panel(qtbot, kind, stores):

    viewer = Viewer(("z", "y", "x"), gui="qt")
    _add(viewer, kind, stores, controls=CONFIGS[kind](appearance=False))

    assert render_dock(AppearanceControls(), viewer, QtLayoutHost(), []) is None


def test_a_phong_mesh_gets_its_own_fields_from_the_same_config(qtbot, mesh_store):
    """One ``MeshControlsConfig`` covers both mesh models.

    ``wireframe`` is flat-only and ``shininess`` phong-only; the panel follows
    the model the visual actually carries, so ``appearance=True`` is safe on
    either without two config classes.
    """
    from tests.convenience._qt_acceptance import control_labels

    viewer = Viewer(("z", "y", "x"), gui="qt")
    # A phong mesh warns about scene lighting, which is unrelated to controls.
    with pytest.warns(UserWarning, match="requires lights"):
        viewer.add_mesh(
            mesh_store,
            appearance=MeshPhongAppearance(),
            controls=MeshControlsConfig(appearance=True),
        )

    assert control_labels(
        render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])
    ) == [
        "Visible",
        "Opacity",
        "Color",
        "Side",
        "Shininess",
        "Flat shading",
        "Bounding box",
    ]


def test_the_labels_combo_offers_the_models_own_render_modes(qtbot, labels_store):
    """In-memory labels: two modes, and no image render modes."""
    from cellier.gui.qt.visuals import QtLabelsRenderModeCombo

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_labels(
        labels_store, controls=LabelsControlsConfig(appearance=["render_mode"])
    )

    spec = appearance_specs(visual, viewer._controls_configs[visual.id]).specs[0]
    widget = QtLabelsRenderModeCombo(visual.id, **spec.values)

    assert widget.choices == ("iso_categorical", "flat_categorical")
    assert "mip" not in widget.choices


# ---------------------------------------------------------------------------
# Editing a rendered control reaches the model
# ---------------------------------------------------------------------------


def test_a_rendered_control_writes_the_model(qtbot, points_store):
    from cellier.convenience.layout._spec import AppearanceControls

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_points(
        points_store, controls=PointsControlsConfig(appearance=["size"])
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])
    from PySide6.QtWidgets import QDoubleSpinBox

    container.findChild(QDoubleSpinBox).setValue(12.0)

    assert visual.appearance.size == pytest.approx(12.0)


def test_an_ortho_non_image_edit_reaches_all_four_panels(qtbot, points_store):
    """The stage-2 fan-out, now for a visual type stage 2 did not cover."""
    from cellier.convenience.layout._spec import AppearanceControls

    ortho = OrthoViewer(("z", "y", "x"))
    visuals = ortho.add_points(
        points_store, controls=PointsControlsConfig(appearance=["size"])
    )

    container = render_dock(AppearanceControls(), ortho, QtLayoutHost(), [])
    from PySide6.QtWidgets import QDoubleSpinBox

    container.findChild(QDoubleSpinBox).setValue(9.5)

    for key in _PANELS:
        assert visuals[key].appearance.size == pytest.approx(9.5)


def test_a_visible_toggle_reaches_the_model_on_its_own_event(qtbot, mesh_store):
    """``visible`` travels on ``VisualVisibilityChangedEvent`` (section 6.4.1).

    It is the one field routed specially, and now the first row of every
    default panel, so it is worth an end-to-end check rather than only the
    widget-level one.
    """
    from cellier.convenience.layout._spec import AppearanceControls

    viewer = Viewer(("z", "y", "x"), gui="qt")
    visual = viewer.add_mesh(
        mesh_store,
        appearance=MeshFlatAppearance(),
        controls=MeshControlsConfig(appearance=["visible"]),
    )
    assert visual.appearance.visible is True

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])
    from PySide6.QtWidgets import QCheckBox

    container.findChild(QCheckBox).setChecked(False)

    assert visual.appearance.visible is False


# ---------------------------------------------------------------------------
# Vocabulary, per new config class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_every_new_config_accepts_the_universal_fields(kind):
    assert CONFIGS[kind](appearance=["visible", "opacity"]).appearance == [
        "visible",
        "opacity",
    ]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("labels", "size"),
        ("mesh", "thickness"),
        ("points", "wireframe"),
        ("lines", "node_size"),
        ("graph", "shininess"),
    ],
)
def test_a_field_from_another_visual_family_is_rejected(kind, field):
    """The vocabulary is per family, so a mesh field is not a points field."""
    with pytest.raises(ValueError, match=field):
        CONFIGS[kind](appearance=[field])


def test_lod_bias_is_multiscale_labels_only():
    with pytest.raises(ValueError, match="lod_bias"):
        LabelsControlsConfig(appearance=["lod_bias"])
    assert MultiscaleLabelsControlsConfig(appearance=["lod_bias"]).appearance == [
        "lod_bias"
    ]


def test_colormap_mode_is_not_in_the_labels_vocabulary():
    """It is ``frozen=True``; a control for it could only raise."""
    with pytest.raises(ValueError, match="colormap_mode"):
        LabelsControlsConfig(appearance=["colormap_mode"])


def test_the_shared_widget_table_agrees_with_the_field_classes():
    """One name per control, not two that happen to match.

    ``APPEARANCE_FIELD_WIDGETS`` names each single-field control for the
    toolkit-neutral spec layer, and each widget class names itself with
    ``_label`` -- and since ``plans/label_ownership_unification.md`` the class
    is what actually gets drawn, in both toolkits.  Letting the two drift would
    mean a panel whose controls are named one thing and asserted to be named
    another.
    """
    from cellier.gui._appearance_fields import (
        APPEARANCE_FIELD_WIDGETS,
        field_widget_class,
    )

    for kind, (_stem, title) in APPEARANCE_FIELD_WIDGETS.items():
        for toolkit in ("qt", "anywidget"):
            widget_class = field_widget_class(kind, toolkit)
            assert widget_class._label == title, (
                f"{widget_class.__name__}._label != "
                f"APPEARANCE_FIELD_WIDGETS[{kind!r}] title"
            )


def test_composite_default_titles_match_the_shared_vocabulary():
    """The same check for the controls that read several fields at once.

    A composite's ``DEFAULT_TITLE`` is what it calls itself when constructed
    directly; ``_CONTROL_TITLES`` is what the renderers pass it.  They are two
    declarations of one name, so they are pinned together.
    """
    from cellier.convenience.layout._shared import _CONTROL_TITLES
    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.anywidget.render import (
        AnywidgetLabelsOutlineControls,
        AnywidgetVisualOcclusionControls,
        AnywidgetVisualOutlineControls,
        AnywidgetVisualPickingControls,
    )
    from cellier.gui.anywidget.visuals import (
        AnywidgetAABBWidget,
        AnywidgetClimRangeSlider,
        AnywidgetColormapCombo,
        AnywidgetLodBiasSlider,
        AnywidgetVolumeRenderControls,
    )
    from cellier.gui.qt import QtDatasetInfo
    from cellier.gui.qt.render import (
        QtLabelsOutlineControls,
        QtVisualOcclusionControls,
        QtVisualOutlineControls,
        QtVisualPickingControls,
    )
    from cellier.gui.qt.visuals import (
        QtAABBWidget,
        QtClimRangeSlider,
        QtColormapCombo,
        QtLodBiasSlider,
        QtVolumeRenderControls,
    )

    composites = {
        "color_map": (QtColormapCombo, AnywidgetColormapCombo),
        "clim": (QtClimRangeSlider, AnywidgetClimRangeSlider),
        "render": (QtVolumeRenderControls, AnywidgetVolumeRenderControls),
        "lod_bias": (QtLodBiasSlider, AnywidgetLodBiasSlider),
        "aabb": (QtAABBWidget, AnywidgetAABBWidget),
        "visual_outline": (QtVisualOutlineControls, AnywidgetVisualOutlineControls),
        "labels_outline": (QtLabelsOutlineControls, AnywidgetLabelsOutlineControls),
        "visual_occlusion": (
            QtVisualOcclusionControls,
            AnywidgetVisualOcclusionControls,
        ),
        "visual_picking": (QtVisualPickingControls, AnywidgetVisualPickingControls),
        "dataset_info": (QtDatasetInfo, AnywidgetDatasetInfo),
    }
    assert set(composites) == set(_CONTROL_TITLES)

    for kind, classes in composites.items():
        for widget_class in classes:
            assert widget_class.DEFAULT_TITLE == _CONTROL_TITLES[kind], (
                f"{widget_class.__name__}.DEFAULT_TITLE != _CONTROL_TITLES[{kind!r}]"
            )


def test_every_valid_field_name_has_a_widget():
    """Section 10.1's rule, checked across every config class at once.

    A name that passes validation and then renders nothing is the exact bug
    stage 3 removed; widening a vocabulary past the widgets would reintroduce
    it.
    """
    from cellier.convenience.gui import _controls_config
    from cellier.gui._appearance_fields import APPEARANCE_FIELD_WIDGETS

    bespoke = {"color_map", "clim", "render", "lod_bias"}
    config_classes = [
        value
        for value in vars(_controls_config).values()
        if isinstance(value, type)
        and issubclass(value, _controls_config.BaseControlsConfig)
    ]
    assert config_classes

    for config_class in config_classes:
        for field, kind in config_class.APPEARANCE_CONTROLS.items():
            assert kind in APPEARANCE_FIELD_WIDGETS or kind in bespoke, (
                f"{config_class.__name__}.{field} -> {kind} has no widget"
            )


# ── sectioned dataset info reaches both front ends ───────────────────────────


def _sectioned_info():
    """A DatasetInfo exercising all three block shapes."""
    import numpy as np

    from cellier.data._dataset_info import DatasetInfo, MatrixSection, RowSection

    return DatasetInfo(
        sections=[
            RowSection(None, [("Name", "store"), ("Store type", "in-memory points")]),
            RowSection("Axes", [("z", "space, um"), ("y", "space, um")]),
            MatrixSection(
                "World to data",
                np.array([[0.5, 0.0, -1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
                row_labels=["z", "y", "1"],
                col_labels=["z", "y", "1"],
            ),
            RowSection("Scale levels", [("s0", "8 x 8")], collapsed=True),
        ]
    )


def test_sections_reach_the_anywidget_front_end(qtbot):
    """The anywidget block could previously draw only a flat row list.

    A store's affine and per-level shapes were reachable from Qt alone, so a
    notebook user saw strictly less than a desktop one.
    """
    from cellier.gui.anywidget import AnywidgetDatasetInfo

    widget = AnywidgetDatasetInfo.from_info(_sectioned_info())
    kinds = [(s["kind"], s["label"]) for s in widget.sections]
    assert kinds == [
        ("rows", None),
        ("rows", "Axes"),
        ("matrix", "World to data"),
        ("rows", "Scale levels"),
    ]
    # The matrix crosses as pre-formatted text so both toolkits round alike.
    matrix = widget.sections[2]
    assert matrix["values"][0] == ["0.5", "0", "-1"]
    assert matrix["col_labels"] == ["z", "y", "1"]
    assert widget.sections[3]["collapsed"] is True


def test_sections_reach_the_qt_front_end(qtbot):
    """The Qt twin draws the same four blocks."""
    from PySide6.QtWidgets import QTableWidget, QWidget

    from cellier.gui.qt import QtDatasetInfo

    widget = QtDatasetInfo.from_info(_sectioned_info())
    root = widget.widget

    nested = [
        child._toggle_btn.text()
        for child in root.findChildren(QWidget)
        if type(child).__name__ == "QCollapsible" and child is not root
    ]
    assert nested == ["Axes", "Scale levels"]

    tables = root.findChildren(QTableWidget)
    assert len(tables) == 1
    table = tables[0]
    assert (table.rowCount(), table.columnCount()) == (3, 3)
    assert table.item(0, 0).text() == "0.5"
    assert table.item(0, 2).text() == "-1"


def test_both_front_ends_agree_on_matrix_formatting(qtbot):
    """One ``.4g`` formatting rule, not one per toolkit."""
    from PySide6.QtWidgets import QTableWidget

    from cellier.gui.anywidget import AnywidgetDatasetInfo
    from cellier.gui.qt import QtDatasetInfo

    info = _sectioned_info()
    any_values = AnywidgetDatasetInfo.from_info(info).sections[2]["values"]

    # Bind the wrapper: dropping it collects the Qt tree out from under us.
    qt_widget = QtDatasetInfo.from_info(info)
    table = qt_widget.widget.findChildren(QTableWidget)[0]
    qt_values = [
        [table.item(row, col).text() for col in range(table.columnCount())]
        for row in range(table.rowCount())
    ]
    assert qt_values == any_values
