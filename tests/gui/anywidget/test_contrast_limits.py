"""Tests for the anywidget ``AnywidgetClimRangeSlider`` contrast-limits widget."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.events import AppearanceChangedEvent
from cellier.gui.anywidget.visuals._contrast_limits import (
    AnywidgetClimRangeSlider,
)


def _make_widget(**kwargs):
    visual_id = uuid4()
    return AnywidgetClimRangeSlider(visual_id, **kwargs), visual_id


def test_instantiate_smoke():
    widget, _visual_id = _make_widget(clim_range=(0, 255), initial_clim=(10, 200))
    assert widget.clim == [10.0, 200.0]
    assert widget.clim_range == [0.0, 255.0]
    assert isinstance(widget.clim[0], float)


def test_clim_edit_emits_tuple():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget.clim = [10.0, 200.0]

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == "clim"
    assert event.value == (10.0, 200.0)
    assert isinstance(event.value, tuple)


def test_inbound_change_updates_trait_without_reemit():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),  # not the widget -> applied
            visual_id=visual_id,
            field_name="clim",
            new_value=(5, 9),
            requires_reslice=False,
        )
    )

    assert widget.clim == [5.0, 9.0]
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    widget, visual_id = _make_widget(initial_clim=(0.0, 1.0))

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual_id,
            field_name="clim",
            new_value=(5, 9),
            requires_reslice=False,
        )
    )

    assert widget.clim == [0.0, 1.0]  # unchanged


def test_inbound_unrelated_field_ignored():
    widget, visual_id = _make_widget(initial_clim=(0.0, 1.0))

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual_id,
            field_name="color_map",
            new_value="magma",
            requires_reslice=False,
        )
    )

    assert widget.clim == [0.0, 1.0]  # unchanged
