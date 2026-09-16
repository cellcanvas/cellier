"""Tests for the anywidget ``AnywidgetDimsPanel`` dims-controls widget."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.events import DimsChangedEvent
from cellier.gui._axis_values import ContinuousAxisValues
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel


def _make_panel(*, with_toggle=True, displayed_axes=(1, 2)):
    scene_id = uuid4()
    kwargs = {
        "scene_id": scene_id,
        "axis_values": {
            0: ContinuousAxisValues(min=0, max=9),
            1: ContinuousAxisValues(min=0, max=99),
            2: ContinuousAxisValues(min=0, max=99),
        },
        "axis_labels": {0: "z", 1: "y", 2: "x"},
        "slice_indices": {0: 0, 1: 0, 2: 0},
        "displayed_axes": displayed_axes,
    }
    if with_toggle:
        kwargs["axes_2d"] = (1, 2)
        kwargs["axes_3d"] = (0, 1, 2)
    return AnywidgetDimsPanel(**kwargs), scene_id


def _dims_changed_event(source_id, scene_id, *, displayed, slices):
    selection = AxisAlignedSelectionState(displayed_axes=displayed)
    state = DimsState(axis_labels=("z", "y", "x"), selection=selection)
    return DimsChangedEvent(
        source_id=source_id,
        scene_id=scene_id,
        dims_state=state,
        displayed_axes_changed=False,
        slice_indices=dict(slices),
    )


def test_construction_without_toggle():
    panel, _scene_id = _make_panel(with_toggle=False)
    assert panel.has_toggle is False
    assert panel.label == ""
    # dict/tuple keys and values are coerced for the JS-sync boundary.
    assert set(panel.slice_indices) == {"0", "1", "2"}
    assert panel.axis_values["0"] == {"kind": "continuous", "min": 0.0, "max": 9.0}
    assert isinstance(panel.axis_values["0"]["min"], float)


def test_construction_with_toggle_2d():
    panel, _scene_id = _make_panel(with_toggle=True, displayed_axes=(1, 2))
    assert panel.has_toggle is True
    assert panel.label == "Switch to 3D"


def test_construction_with_toggle_3d():
    panel, _scene_id = _make_panel(with_toggle=True, displayed_axes=(0, 1, 2))
    assert panel.has_toggle is True
    assert panel.label == "Switch to 2D"


def test_slice_indices_edit_emits_update_event():
    panel, scene_id = _make_panel(with_toggle=False, displayed_axes=())
    emitted = []
    panel.changed.connect(emitted.append)

    panel.slice_indices = {"0": 5, "1": 0, "2": 0}

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == panel._id
    assert event.scene_id == scene_id
    # Only the axis that moved: ``update_slice_indices`` merges.
    assert event.slice_indices == {0: 5}
    assert event.displayed_axes is None


def test_a_second_edit_reports_only_the_axis_it_moved():
    panel, _scene_id = _make_panel(with_toggle=False, displayed_axes=(1, 2))
    emitted = []
    panel.changed.connect(emitted.append)

    panel.slice_indices = {"0": 5, "1": 0, "2": 0}
    panel.slice_indices = {"0": 5, "1": 0, "2": 3}

    assert emitted[-1].slice_indices == {2: 3}


def test_toggle_click_emits_event_and_applies_optimistically():
    panel, scene_id = _make_panel(with_toggle=True, displayed_axes=(1, 2))
    emitted = []
    panel.changed.connect(emitted.append)

    panel._clicks += 1  # simulate JS click handler incrementing the counter

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == panel._id
    assert event.scene_id == scene_id
    assert event.displayed_axes == (0, 1, 2)
    # Every axis keeps its position in the model (D36): nothing to hand over.
    assert event.slice_indices is None

    # Unlike QtDimsControl, the anywidget panel applies the toggle
    # immediately instead of waiting for the echoed DimsChangedEvent.
    assert list(panel.displayed_axes) == [0, 1, 2]
    assert panel.label == "Switch to 2D"


def test_toggle_click_round_trip():
    panel, _scene_id = _make_panel(with_toggle=True, displayed_axes=(1, 2))

    panel._clicks += 1  # 2D -> 3D
    assert panel.label == "Switch to 2D"

    panel._clicks += 1  # 3D -> 2D
    assert list(panel.displayed_axes) == [1, 2]
    assert panel.label == "Switch to 3D"


def test_inbound_dims_changed_updates_traits_without_reemit():
    panel, scene_id = _make_panel(with_toggle=True, displayed_axes=(1, 2))
    emitted = []
    panel.changed.connect(emitted.append)

    event = _dims_changed_event(
        source_id=uuid4(),  # not the panel -> applied
        scene_id=scene_id,
        displayed=(0, 1, 2),
        slices={0: 7, 1: 0, 2: 0},
    )
    panel._on_dims_changed(event)

    assert panel.slice_indices["0"] == 7
    assert list(panel.displayed_axes) == [0, 1, 2]
    assert panel.label == "Switch to 2D"
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_skips_slider_values_but_applies_displayed_axes():
    panel, scene_id = _make_panel(with_toggle=True, displayed_axes=(1, 2))

    event = _dims_changed_event(
        source_id=panel._id,  # our own echo
        scene_id=scene_id,
        displayed=(0, 1, 2),
        slices={0: 999, 1: 0, 2: 0},
    )
    panel._on_dims_changed(event)

    # A drag has moved on since it sent the values, so they are skipped; the
    # displayed axes are the model's state and are applied.
    assert panel.slice_indices["0"] == 0
    assert list(panel.displayed_axes) == [0, 1, 2]
    assert panel.label == "Switch to 2D"


def test_inbound_dims_changed_without_toggle_does_not_relabel():
    panel, scene_id = _make_panel(with_toggle=False, displayed_axes=())

    event = _dims_changed_event(
        source_id=uuid4(),
        scene_id=scene_id,
        displayed=(1, 2),
        slices={0: 3, 1: 0, 2: 0},
    )
    panel._on_dims_changed(event)

    assert panel.label == ""
