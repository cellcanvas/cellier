"""Tests for paint command history primitives."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from cellier.paint._history import (
    ActiveStroke,
    CommandHistory,
    PaintStrokeCommand,
)


def _make_command(value: int) -> PaintStrokeCommand:
    """Build a tiny command stamped with *value* in voxel_indices."""
    stroke = ActiveStroke(visual_id=uuid4(), data_store_id=uuid4())
    stroke.record(
        np.array([[value, 0]], dtype=np.int64),
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    return stroke.finalise()


def test_push_three_commands_can_undo_not_redo():
    history = CommandHistory(max_depth=10)
    for i in range(3):
        history.push(_make_command(i))
    assert history.can_undo is True
    assert history.can_redo is False


def test_undo_returns_last_command_and_enables_redo():
    history = CommandHistory(max_depth=10)
    for i in range(3):
        history.push(_make_command(i))

    command = history.undo()

    assert command is not None
    assert int(command.voxel_indices[0, 0]) == 2
    assert history.can_redo is True


def test_redo_replays_undone_command():
    history = CommandHistory(max_depth=10)
    for i in range(3):
        history.push(_make_command(i))
    history.undo()

    command = history.redo()

    assert command is not None
    assert int(command.voxel_indices[0, 0]) == 2


def test_push_after_undo_clears_redo_stack():
    history = CommandHistory(max_depth=10)
    for i in range(3):
        history.push(_make_command(i))
    history.undo()
    assert history.can_redo is True

    history.push(_make_command(99))

    assert history.can_redo is False


def test_max_depth_bounds_undo_stack():
    max_depth = 4
    history = CommandHistory(max_depth=max_depth)

    for i in range(max_depth + 3):
        history.push(_make_command(i))

    assert len(history._undo_stack) == max_depth


def test_undo_on_empty_history_returns_none():
    history = CommandHistory(max_depth=10)
    assert history.undo() is None


def test_redo_on_empty_history_returns_none():
    history = CommandHistory(max_depth=10)
    assert history.redo() is None


def test_active_stroke_finalise_with_no_records_returns_empty():
    stroke = ActiveStroke(visual_id=uuid4(), data_store_id=uuid4())

    command = stroke.finalise()

    assert command.voxel_indices.shape[0] == 0
    assert command.old_values.shape[0] == 0
    assert command.new_values.shape[0] == 0


def test_active_stroke_concatenates_records():
    stroke = ActiveStroke(visual_id=uuid4(), data_store_id=uuid4())
    stroke.record(
        np.array([[0, 0], [1, 1]], dtype=np.int64),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    )
    stroke.record(
        np.array([[2, 2]], dtype=np.int64),
        np.array([0.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )

    command = stroke.finalise()

    assert command.voxel_indices.shape == (3, 2)
    assert command.old_values.shape == (3,)
    assert command.new_values.shape == (3,)


def test_active_stroke_keeps_the_store_dtype():
    """Label ids above 2**24 survive a stroke: float32 would round them."""
    big = 2**24 + 1
    stroke = ActiveStroke(visual_id=uuid4(), data_store_id=uuid4(), dtype=np.int32)
    stroke.record(
        np.array([[0, 0]], dtype=np.int64),
        np.array([big], dtype=np.int32),
        np.array([big + 2], dtype=np.int32),
    )

    command = stroke.finalise()

    assert command.old_values.dtype == np.int32
    assert command.old_values[0] == big
    assert command.new_values[0] == big + 2


def test_an_empty_stroke_has_the_store_dtype():
    stroke = ActiveStroke(visual_id=uuid4(), data_store_id=uuid4(), dtype=np.uint16)

    command = stroke.finalise()

    assert command.old_values.dtype == np.uint16
    assert command.new_values.dtype == np.uint16
