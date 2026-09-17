"""Tests for discrete axes in the Qt ``QtDimsControl``."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from qtpy.QtWidgets import QSlider

from cellier._state import AxisAlignedSelectionState, DimsState
from cellier.events import DimsChangedEvent
from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui.qt._scene import QtDimsControl

_DISCRETE = DiscreteAxisValues(values=(0.0, 2.5, 10.0), labels=("a", "b", "c"))


def _make_control(qtbot, *, initial=2.5, with_toggle=False, displayed=(1, 2)):
    control = QtDimsControl(
        scene_id=uuid4(),
        axis_values={
            0: _DISCRETE,
            1: ContinuousAxisValues(min=0.0, max=99.0),
            2: ContinuousAxisValues(min=0.0, max=99.0),
        },
        axis_labels={0: "c", 1: "y", 2: "x"},
        initial_slice_indices={0: initial, 1: 0.0, 2: 0.0},
        initial_displayed_axes=displayed,
        axes_2d=(1, 2) if with_toggle else None,
        axes_3d=(0, 1, 2) if with_toggle else None,
    )
    qtbot.addWidget(control.widget)
    return control


def _event(scene_id, *, slices, displayed=(1, 2)):
    selection = AxisAlignedSelectionState(displayed_axes=displayed)
    return DimsChangedEvent(
        source_id=uuid4(),
        scene_id=scene_id,
        dims_state=DimsState(axis_labels=("c", "y", "x"), selection=selection),
        displayed_axes_changed=False,
        slice_indices=dict(slices),
    )


def test_discrete_axis_gets_an_integer_slider_over_positions(qtbot):
    control = _make_control(qtbot)

    slider = control._sliders[0]
    assert type(slider) is QSlider
    assert (slider.minimum(), slider.maximum()) == (0, 2)
    assert slider.value() == 1
    assert control._readouts[0].text() == "b"
    assert control.current_index()[0] == 2.5


def test_moving_a_discrete_slider_emits_the_world_value(qtbot):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    control._sliders[0].setValue(2)

    assert emitted[-1].slice_indices[0] == 10.0
    assert control._readouts[0].text() == "c"


def test_readout_shows_the_value_when_there_are_no_labels(qtbot):
    control = QtDimsControl(
        scene_id=uuid4(),
        axis_values={0: DiscreteAxisValues(values=(0.0, 0.25))},
        axis_labels={0: "c"},
        initial_slice_indices={0: 0.25},
    )
    qtbot.addWidget(control.widget)

    assert control._readouts[0].text() == "0.25"


@pytest.mark.parametrize(("position", "expected"), [(6.2, 1), (6.25, 2), (-4.0, 0)])
def test_inbound_positions_show_the_nearest_value_without_writing_back(
    qtbot, position, expected
):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    control._on_dims_changed(_event(control._scene_id, slices={0: position}))

    assert control._sliders[0].value() == expected
    assert control._readouts[0].text() == _DISCRETE.labels[expected]
    assert emitted == []


def test_toggle_sends_only_the_displayed_axes(qtbot):
    """The model keeps every position (D36), so the toggle hands over none."""
    control = _make_control(qtbot, with_toggle=True, displayed=(0, 1, 2))
    control._on_dims_changed(
        _event(control._scene_id, slices={0: 3.0}, displayed=(0, 1, 2))
    )
    emitted = []
    control.changed.connect(emitted.append)

    control._on_toggle_click()

    assert emitted[-1].slice_indices is None
    assert emitted[-1].displayed_axes == (1, 2)


def test_bare_pairs_are_rejected(qtbot):
    with pytest.raises(TypeError, match="ContinuousAxisValues"):
        QtDimsControl(
            scene_id=uuid4(), axis_values={0: (0.0, 1.0)}, axis_labels={0: "z"}
        )


# ---------------------------------------------------------------------------
# draw_ticks
# ---------------------------------------------------------------------------


def test_a_discrete_slider_draws_no_ticks_by_default(qtbot):
    control = _make_control(qtbot)

    slider = control._sliders[0]

    assert slider.tickPosition() == QSlider.TickPosition.NoTicks


def test_draw_ticks_marks_every_value(qtbot):
    control = QtDimsControl(
        scene_id=uuid4(),
        axis_values={
            0: DiscreteAxisValues(values=(0.0, 2.5, 10.0), draw_ticks=True),
            1: ContinuousAxisValues(min=0.0, max=99.0),
            2: ContinuousAxisValues(min=0.0, max=99.0),
        },
        axis_labels={0: "c", 1: "y", 2: "x"},
        initial_slice_indices={0: 0.0, 1: 0.0, 2: 0.0},
        initial_displayed_axes=(1, 2),
    )
    qtbot.addWidget(control.widget)

    slider = control._sliders[0]

    assert slider.tickPosition() == QSlider.TickPosition.TicksBelow
    assert slider.tickInterval() == 1
