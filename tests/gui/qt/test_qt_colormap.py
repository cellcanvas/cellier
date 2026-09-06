"""Tests for the Qt ``QtColormapCombo`` colormap-control widget."""

from __future__ import annotations

import numpy as np
import pytest
from cmap import Colormap

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from cellier.controller import CellierController
from cellier.data.image._image_memory_store import ImageMemoryStore
from cellier.gui.qt.visuals._colormap import QtColormapCombo
from cellier.scene.dims import CoordinateSystem
from cellier.visuals._image_memory import InMemoryImageAppearance


def _make_controller_with_visual(color_map="viridis"):
    controller = CellierController()
    cs = CoordinateSystem(name="world", axis_labels=("y", "x"))
    scene = controller.add_scene(
        dim="2d", coordinate_system=cs, name="main", render_modes={"2d"}
    )
    data = np.zeros((16, 16), dtype=np.float32)
    store = ImageMemoryStore(data=data)
    appearance = InMemoryImageAppearance(color_map=color_map, clim=(0.0, 1.0))
    visual = controller.add_image(data=store, scene_id=scene.id, appearance=appearance)
    return controller, visual


def test_instantiate_smoke(qtbot):
    combo = QtColormapCombo(visual_id=object(), initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    assert "viridis" in combo._combo.currentColormap().name


def test_combo_edit_reaches_model(qtbot):
    controller, visual = _make_controller_with_visual()
    combo = QtColormapCombo(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    combo._on_combo_changed(Colormap("magma"))

    assert "magma" in visual.appearance.color_map.name


def test_model_push_updates_combo_without_reemit(qtbot):
    controller, visual = _make_controller_with_visual()
    combo = QtColormapCombo(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    emitted = []
    combo.changed.connect(emitted.append)

    controller.update_appearance_field(visual.id, "color_map", Colormap("cividis"))

    assert "cividis" in combo._combo.currentColormap().name
    assert emitted == []


def test_unrelated_field_change_ignored(qtbot):
    controller, visual = _make_controller_with_visual(color_map="viridis")
    combo = QtColormapCombo(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    controller.update_appearance_field(visual.id, "clim", (0.0, 500.0))

    assert "viridis" in combo._combo.currentColormap().name


def test_inbound_echo_filtered_by_source_id(qtbot):
    """A widget must ignore the bus echo of its own write.

    The core of the bus contract, and the one thing that stops a control and
    its model oscillating.  Covered on every anywidget module and, until this,
    on only three Qt ones -- the systematic half of D14
    (``plans/gui_backend_unification.md``).

    Driven through the handler rather than the controller, because the
    controller stamps its own ``source_id``; the case under test is an event
    carrying *this widget's* id.
    """
    from cellier.events import AppearanceChangedEvent

    controller, visual = _make_controller_with_visual(color_map="viridis")
    combo = QtColormapCombo(visual_id=visual.id, initial_colormap="viridis")
    qtbot.addWidget(combo.widget)
    controller.connect_widget(combo, subscription_specs=combo.subscription_specs())

    combo._on_visual_changed(
        AppearanceChangedEvent(
            source_id=combo._id,  # our own echo -> ignored
            visual_id=visual.id,
            field_name="color_map",
            new_value="magma",
            requires_reslice=False,
        )
    )

    assert combo.control.currentText() == "viridis"
