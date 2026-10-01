"""Tests for keyboard stepping of continuous axes in the Qt ``QtDimsControl``."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("superqt")

from qtpy.QtCore import QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication, QSlider

from cellier.gui._axis_values import ContinuousAxisValues, DiscreteAxisValues
from cellier.gui.qt._scene import QtDimsControl

_AXIS = 1


def _make_control(qtbot, *, initial=50.0, debounce_ms=None, step_size=2.0):
    control = QtDimsControl(
        scene_id=uuid4(),
        axis_values={
            0: DiscreteAxisValues(values=(0.0, 2.5, 10.0)),
            _AXIS: ContinuousAxisValues(min=0.0, max=200.0, step_size=step_size),
        },
        axis_labels={0: "c", _AXIS: "z"},
        initial_slice_indices={0: 0.0, _AXIS: initial},
        initial_displayed_axes=(),
        debounce_ms=debounce_ms,
    )
    qtbot.addWidget(control.widget)
    return control


def _focus_target(control, axis=_AXIS) -> QSlider:
    """The widget that takes key presses for *axis*'s row."""
    slider = control._sliders[axis]
    return slider if isinstance(slider, QSlider) else slider.findChild(QSlider)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (Qt.Key.Key_Right, 52.0),
        (Qt.Key.Key_Up, 52.0),
        (Qt.Key.Key_Left, 48.0),
        (Qt.Key.Key_Down, 48.0),
        (Qt.Key.Key_PageUp, 70.0),
        (Qt.Key.Key_PageDown, 30.0),
        (Qt.Key.Key_Home, 0.0),
        (Qt.Key.Key_End, 200.0),
    ],
)
def test_key_steps_a_continuous_axis_and_submits_it(qtbot, key, expected):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    qtbot.keyClick(_focus_target(control), key)

    assert control._sliders[_AXIS].value() == pytest.approx(expected)
    assert len(emitted) == 1
    assert emitted[0].slice_indices == {_AXIS: pytest.approx(expected)}
    assert emitted[0].displayed_axes is None


@pytest.mark.parametrize(
    ("initial", "key"),
    [(200.0, Qt.Key.Key_Right), (0.0, Qt.Key.Key_PageDown), (0.0, Qt.Key.Key_Home)],
)
def test_key_at_the_end_of_the_range_submits_nothing(qtbot, initial, key):
    control = _make_control(qtbot, initial=initial)
    emitted = []
    control.changed.connect(emitted.append)

    qtbot.keyClick(_focus_target(control), key)

    assert control._sliders[_AXIS].value() == initial
    assert emitted == []


def test_key_step_clamps_to_the_range(qtbot):
    control = _make_control(qtbot, initial=199.0)

    qtbot.keyClick(_focus_target(control), Qt.Key.Key_PageUp)

    assert control._sliders[_AXIS].value() == 200.0


def test_repeated_keys_go_through_the_rate_limit(qtbot):
    """Like a drag: the first step submits at once, the rest on the tick."""
    control = _make_control(qtbot, debounce_ms=50)
    emitted = []
    control.changed.connect(emitted.append)
    target = _focus_target(control)

    for _ in range(3):
        qtbot.keyClick(target, Qt.Key.Key_Right)

    assert [e.slice_indices[_AXIS] for e in emitted] == [pytest.approx(52.0)]
    qtbot.waitUntil(lambda: len(emitted) == 2, timeout=2000)
    assert emitted[-1].slice_indices == {_AXIS: pytest.approx(56.0)}


def test_other_keys_are_left_alone(qtbot):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    qtbot.keyClick(_focus_target(control), Qt.Key.Key_A)

    assert control._sliders[_AXIS].value() == 50.0
    assert emitted == []


def test_discrete_axis_still_steps_by_one_value(qtbot):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    qtbot.keyClick(_focus_target(control, 0), Qt.Key.Key_Right)

    assert emitted[-1].slice_indices == {0: 2.5}


def test_arrow_key_moves_one_world_unit_by_default(qtbot):
    control = _make_control(qtbot, step_size=1.0)

    qtbot.keyClick(_focus_target(control), Qt.Key.Key_Right)

    assert control._sliders[_AXIS].value() == pytest.approx(51.0)


def _wheel(widget, angle_delta_y, modifiers=Qt.KeyboardModifier.NoModifier):
    center = widget.rect().center()
    event = QWheelEvent(
        QPointF(center),
        QPointF(widget.mapToGlobal(center)),
        QPoint(0, 0),
        QPoint(0, angle_delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


@pytest.mark.parametrize(("angle_delta", "expected"), [(120, 52.0), (-120, 48.0)])
def test_wheel_notch_moves_a_continuous_axis_one_step(qtbot, angle_delta, expected):
    control = _make_control(qtbot)
    emitted = []
    control.changed.connect(emitted.append)

    _wheel(_focus_target(control), angle_delta)

    assert control._sliders[_AXIS].value() == pytest.approx(expected)
    assert emitted[-1].slice_indices == {_AXIS: pytest.approx(expected)}
