"""The labels outline control is seeded with the visual's ``outline_mode``.

The mode decides the panel's shape: slot swatches in ``whole_object`` /
``all_boundaries`` mode, a participation checkbox plus per-label rows in
``per_label`` mode.  The shared spec used to leave the mode out, so both front
ends fell back to ``per_label`` and misdrew every visual in another mode.
"""

from __future__ import annotations

import pytest

from cellier.convenience import Viewer
from cellier.convenience._backend import ANYWIDGET_BACKEND, QT_BACKEND
from cellier.convenience.gui._controls_config import LabelsControlsConfig
from cellier.convenience.layout._shared import appearance_specs
from cellier.convenience.layout._walk import build_appearance_widgets
from cellier.scene.dims import spatial_axes

_CONFIG = LabelsControlsConfig(appearance=["visible"], outline_controls=True)


def _labels(labels_store, mode):
    viewer = Viewer(spatial_axes("z", "y", "x"), gui="offscreen")
    visual = viewer.add_labels(labels_store, outline_mode=mode, controls=_CONFIG)
    return viewer, visual


def _outline_widget(viewer, visual, backend):
    widgets = build_appearance_widgets(
        visual, _CONFIG, viewer.controller, backend=backend
    )
    (widget,) = [w for w in widgets if getattr(w, "section", None) == "labels_outline"]
    return widget


@pytest.mark.parametrize("mode", ["per_label", "whole_object", "all_boundaries"])
def test_the_spec_carries_the_mode(labels_store, mode):
    _viewer, visual = _labels(labels_store, mode)

    (spec,) = [
        s for s in appearance_specs(visual, _CONFIG).specs if s.kind == "labels_outline"
    ]

    assert spec.values["outline_mode"] == mode


def test_the_qt_control_draws_the_visuals_mode(qtbot, labels_store):
    viewer, visual = _labels(labels_store, "all_boundaries")

    widget = _outline_widget(viewer, visual, QT_BACKEND)

    assert widget._mode() == "all_boundaries"
    # The slot is the colour in this mode, so there are no per-label rows.
    assert "outline_selected_labels" not in widget._appliers


def test_the_anywidget_control_draws_the_visuals_mode(labels_store):
    viewer, visual = _labels(labels_store, "all_boundaries")

    widget = _outline_widget(viewer, visual, ANYWIDGET_BACKEND)

    assert widget.values["outline_mode"] == "all_boundaries"
    assert "outline_selected_labels" not in {
        entry["field"] for entry in widget.controls
    }
