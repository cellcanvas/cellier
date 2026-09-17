"""Tests for discrete axes in the anywidget ``AnywidgetDimsPanel``."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.events import DimsChangedEvent
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui.anywidget._dims_panel import AnywidgetDimsPanel

_DISCRETE = DiscreteAxisValues(values=(0.0, 1.0), labels=("mem9", "H2B"))


def _make_panel(*, slice_indices=None, displayed=(1, 2), with_toggle=False):
    return AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_values={
            0: _DISCRETE,
            1: ContinuousAxisValues(min=0.0, max=99.0),
            2: ContinuousAxisValues(min=0.0, max=99.0),
        },
        axis_labels={0: "c", 1: "y", 2: "x"},
        slice_indices=slice_indices or {0: 0.0, 1: 0.0, 2: 0.0},
        displayed_axes=displayed,
        axes_2d=(1, 2) if with_toggle else None,
        axes_3d=(0, 1, 2) if with_toggle else None,
    )


def _event(scene_id, *, slices, displayed=(1, 2)):
    selection = AxisAlignedSelectionState(displayed_axes=displayed)
    return DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id,
        dims_state=DimsState(axis_labels=("c", "y", "x"), selection=selection),
        displayed_axes_changed=False,
        slice_indices=dict(slices),
    )


def test_discrete_axis_serialises_as_json_for_the_front_end():
    panel = _make_panel()

    assert panel.axis_values["0"] == {
        "kind": "discrete",
        "values": [0.0, 1.0],
        "labels": ["mem9", "H2B"],
        "draw_ticks": False,
    }
    assert panel.axis_values["1"] == {"kind": "continuous", "min": 0.0, "max": 99.0}


def test_discrete_index_is_derived_at_construction():
    panel = _make_panel(slice_indices={0: 0.5, 1: 0.0, 2: 0.0})

    # 0.5 is the tie between the two channels; it rounds up.
    assert panel.discrete_index == {"0": 1}


def test_a_front_end_write_updates_the_index_and_emits():
    panel = _make_panel(slice_indices={0: 1.0, 1: 0.0, 2: 0.0}, displayed=(1, 2))
    emitted = []
    panel.changed.connect(emitted.append)

    panel.slice_indices = {"0": 0.0, "1": 0.0, "2": 0.0}

    assert panel.discrete_index == {"0": 0}
    assert emitted[-1].slice_indices == {0: 0.0}


def test_inbound_positions_show_the_nearest_value_without_emitting():
    panel = _make_panel()
    emitted = []
    panel.changed.connect(emitted.append)

    panel._on_dims_changed(_event(panel._scene_id, slices={0: 0.7}))

    assert panel.discrete_index == {"0": 1}
    # The raw position is kept: the renderer resolves it the same way.
    assert panel.slice_indices["0"] == 0.7
    assert emitted == []


def test_toggle_sends_only_the_displayed_axes():
    """The model keeps every position (D36), so the toggle hands over none."""
    panel = _make_panel(displayed=(0, 1, 2), with_toggle=True)
    panel._on_dims_changed(
        _event(panel._scene_id, slices={0: 0.7}, displayed=(0, 1, 2))
    )
    emitted = []
    panel.changed.connect(emitted.append)

    panel._clicks += 1

    assert emitted[-1].slice_indices is None
    assert emitted[-1].displayed_axes == (1, 2)


def test_bare_pairs_are_rejected():
    with pytest.raises(TypeError, match="ContinuousAxisValues"):
        AnywidgetDimsPanel(
            scene_id=uuid4(),
            axis_values={0: (0.0, 1.0)},
            axis_labels={0: "z"},
            slice_indices={0: 0.0},
        )


# ---------------------------------------------------------------------------
# draw_ticks
#
# The datalist itself is built in dims_panel.js and is not reachable from
# pytest; what is testable here is that the flag reaches the front end.
# ---------------------------------------------------------------------------


def test_draw_ticks_reaches_the_front_end():
    panel = _make_panel()

    assert panel.axis_values["0"]["draw_ticks"] is False

    ticked = AnywidgetDimsPanel(
        scene_id=uuid4(),
        axis_values={
            0: DiscreteAxisValues(values=(0.0, 1.0), draw_ticks=True),
            1: ContinuousAxisValues(min=0.0, max=99.0),
            2: ContinuousAxisValues(min=0.0, max=99.0),
        },
        axis_labels={0: "c", 1: "y", 2: "x"},
        slice_indices={0: 0.0, 1: 0.0, 2: 0.0},
        displayed_axes=(1, 2),
    )

    assert ticked.axis_values["0"]["draw_ticks"] is True
