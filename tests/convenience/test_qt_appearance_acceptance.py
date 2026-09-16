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
    InMemoryImageControlsConfig,
    MultiscaleImageControlsConfig,
)
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

# The full field vocabulary the appearance path accepts today.  Kept explicit
# so the expected-title lists below are read against a request for everything.
_ALL_MULTISCALE_FIELDS = [
    "color_map",
    "clim",
    "render_mode",
    "iso_threshold",
    "attenuation",
    "lod_bias",
]


def test_multiscale_panel_control_names_in_order(qtbot, multiscale_image_store):
    """Every field requested, so this is the maximal panel Qt builds today.

    ``lod_bias`` and the three render fields are multiscale-only; the render
    trio collapses into the single "Render mode" group.  The bounding box is
    not requested by name -- ``aabb`` is on ``BaseVisual`` with a default
    factory, so every configured panel gets it, on both toolkits.
    """
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image_multiscale(
        multiscale_image_store,
        appearance=MultiscaleImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=_ALL_MULTISCALE_FIELDS),
        single=MultiscaleImageSingleAppearance(color_map="viridis", render_mode="mip"),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == ["Image", "LOD bias", "Bounding box"]
    assert_panel_renders(container)


def test_in_memory_panel_control_names_in_order(qtbot, image_store):
    """The in-memory model has no ``lod_bias``/``attenuation``.

    This is the one drop stage 3's validation cannot catch, and so the reason
    ``appearance_specs`` still reports ``skipped``: a **multiscale** config is
    paired with an in-memory visual, so every name is valid for the config
    class and two of them are still absent from the model.  The renderer warns
    rather than dropping them silently (section 9.5 step 4).

    Requesting the same fields on an ``InMemoryImageControlsConfig`` now
    raises at construction -- see ``test_controls_config_validation.py``.
    """
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=MultiscaleImageControlsConfig(appearance=_ALL_MULTISCALE_FIELDS),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    with pytest.warns(UserWarning, match="attenuation"):
        container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == ["Image", "Bounding box"]
    assert_panel_renders(container)


def test_panel_order_follows_the_builder_not_the_config(qtbot, image_store):
    """Group order is the builder's fixed order, not the config's field order.

    The docstring on ``appearance`` says "in display order"; it is not.  Pinned
    so stage 1's shared spec list either preserves this or changes it visibly.
    """
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["clim", "color_map"]),
        single=InMemoryImageSingleAppearance(color_map="grays", clim=(0.0, 1.0)),
    )

    container = render_dock(AppearanceControls(), viewer, QtLayoutHost(), [])

    assert control_labels(container) == ["Image", "Bounding box"]


def test_a_typo_never_reaches_the_dock_at_all(qtbot, image_store):
    """The useless panel a typo used to produce cannot be built any more.

    The history is the point.  Before stage 1 a misspelled field gave Qt a
    dock with *zero* groups and the anywidget front end a dock holding only
    the bounding box; stage 1 made Qt match; stage 3 stops the config being
    constructed at all, so no dock of either shape exists to render.
    """
    with pytest.raises(ValueError, match="color_map"):
        InMemoryImageControlsConfig(appearance=["colour_map"])


def test_bounding_box_group_is_seeded_from_the_visual(qtbot, image_store):
    """The new AABB group reads its initial state off ``visual.aabb``.

    A group with the right title but a control seeded from defaults would
    pass ``control_labels`` and still be wrong, so this checks the wiring
    rather than the label.
    """
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QGroupBox

    viewer = Viewer(spatial_axes("z", "y", "x"), gui="qt")
    visual = viewer.add_image(
        image_store,
        appearance=InMemoryImageAppearance(),
        controls=InMemoryImageControlsConfig(appearance=["color_map"]),
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
