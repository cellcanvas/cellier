"""Visual acceptance of the Qt appearance dock.

Stage 0 section 6.3 of ``plans/convenience_cleanup.md``: pins the group titles
the Qt appearance dock renders, in order, and that the panel actually paints.

**Stage 1 landed here**: the "Bounding box" group is new in every list below.
``QtAABBWidget`` already existed and the anywidget path already built one; only
the Qt renderer did not, which was drift rather than a toolkit limit
(section 7.1).  That change is the diff this module exists to show.
"""

from __future__ import annotations

import pytest

from cellier.convenience import AppearanceControls
from cellier.convenience._hosts import QtLayoutHost
from cellier.convenience.layout._walk import render_dock

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.convenience import Viewer
from cellier.convenience.gui import (
    BoundingBoxControl,
    FieldControls,
    ImageControl,
    LevelOfDetailControl,
    LoadingIndicator,
)
from cellier.convenience.gui._controls_config import validate_controls
from cellier.scene.dims import spatial_axes
from cellier.visuals import (
    InMemoryImageSingleAppearance,
    MultiscaleImageSingleAppearance,
)
from cellier.visuals._image import MultiscaleImageAppearance
from cellier.visuals._image_memory import InMemoryImageAppearance
from tests.convenience._qt_acceptance import (
    assert_panel_renders,
    control_labels,
)

# The image fields of a multiscale image.  Kept explicit so the
# expected-title lists below are read against a request for everything.
_ALL_MULTISCALE_FIELDS = [
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
]


def test_multiscale_panel_control_names_in_order(qtbot, multiscale_image_store):
    """Every field listed; the dock follows the order of the list.

    The image fields collapse into the single "Image" control.  The level of
    detail, the bounding box and the loading indicator are each their own
    item.
    """
    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=[
            ImageControl(fields=_ALL_MULTISCALE_FIELDS),
            LevelOfDetailControl(),
            BoundingBoxControl(),
            LoadingIndicator(),
        ],
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    # "Level of detail" is a group; the walk reports its labelled row too.
    assert control_labels(container) == [
        "Image",
        "Level of detail",
        "Settled bias",
        "Bounding box",
        "Data fetch status",
    ]
    assert_panel_renders(container)


def test_a_reordered_list_reorders_the_panel(qtbot, multiscale_image_store):
    """Move the lines and the dock follows."""
    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=[
            LoadingIndicator(),
            BoundingBoxControl(),
            ImageControl(fields=_ALL_MULTISCALE_FIELDS),
            LevelOfDetailControl(),
        ],
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == [
        "Data fetch status",
        "Bounding box",
        "Image",
        "Level of detail",
        "Settled bias",
    ]
    assert_panel_renders(container)


def test_a_field_the_model_lacks_warns_and_the_rest_is_built(qtbot, mesh_store):
    """The one drop ``validate_controls`` cannot catch.

    ``shininess`` is a valid mesh field and the flat appearance model does
    not carry it.  ``appearance_specs`` reports it as ``skipped`` and the
    renderer warns rather than dropping it silently.
    """
    from cellier.visuals._mesh_memory import MeshFlatAppearance

    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    viewer.add_mesh(
        mesh_store,
        appearance=MeshFlatAppearance(),
        controls=[FieldControls(["opacity", "shininess"]), BoundingBoxControl()],
    )

    with pytest.warns(UserWarning, match="shininess"):
        container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == ["Opacity", "Bounding box"]
    assert_panel_renders(container)


def test_the_image_rows_do_not_follow_the_field_order(qtbot, image_store):
    """``ImageControl.fields`` chooses rows; it does not order them (D1)."""
    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=[ImageControl(fields=["clim", "color_map"]), BoundingBoxControl()],
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == ["Image", "Bounding box"]


def test_a_misspelled_field_never_reaches_the_dock_at_all(qtbot, image_store):
    """The useless panel a misspelling used to produce cannot be built.

    ``add_image`` refuses the list before anything is added, so no dock of
    any shape exists to render.
    """
    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    misspelled = "colo" + "ur" + "_map"
    with pytest.raises(ValueError, match="color_map"):
        viewer.add_image(image_store, controls=[ImageControl(fields=[misspelled])])
    with pytest.raises(ValueError, match="color_map"):
        validate_controls([ImageControl(fields=[misspelled])], "image", "image")
    assert viewer.scene.visuals == []


def test_bounding_box_group_is_seeded_from_the_visual(qtbot, image_store):
    """The new AABB group reads its initial state off ``visual.aabb``.

    A group with the right title but a control seeded from defaults would
    pass ``control_labels`` and still be wrong, so this checks the wiring
    rather than the label.
    """
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QGroupBox

    viewer = Viewer(spatial_axes("z", "y", "x"), displayed_axes=("y", "x"), gui="qt")
    visual = viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=[ImageControl(fields=["color_map"]), BoundingBoxControl()],
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )
    visual.aabb.enabled = True
    visual.aabb.line_width = 7.5

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    groups = [
        g for g in container.findChildren(QGroupBox) if g.title() == "Bounding box"
    ]
    assert len(groups) == 1
    assert groups[0].findChild(QCheckBox).isChecked() is True
    assert groups[0].findChild(QDoubleSpinBox).value() == pytest.approx(7.5)
