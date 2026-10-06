"""Tests for the anywidget ``AnywidgetLodBiasSlider`` LOD-bias-control widget."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("anywidget")

from cellier.events import AppearanceChangedEvent
from cellier.gui.anywidget.visuals._lod_bias import AnywidgetLodBiasSlider


def _make_widget(**kwargs):
    visual_id = uuid4()
    return AnywidgetLodBiasSlider(visual_id, **kwargs), visual_id


def test_instantiate_smoke():
    widget, _visual_id = _make_widget(initial_lod_bias=2.5)
    assert widget.settled_lod_bias == pytest.approx(2.5)


def test_trait_edit_emits_update_event():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget.settled_lod_bias = 3.5

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == "settled_lod_bias"
    assert event.value == pytest.approx(3.5)


def test_inbound_change_updates_trait_without_reemit():
    widget, visual_id = _make_widget()
    emitted = []
    widget.changed.connect(emitted.append)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),  # not the widget -> applied
            visual_id=visual_id,
            field_name="settled_lod_bias",
            new_value=4.0,
            requires_reslice=True,
        )
    )

    assert widget.settled_lod_bias == pytest.approx(4.0)
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    widget, visual_id = _make_widget(initial_lod_bias=1.0)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual_id,
            field_name="settled_lod_bias",
            new_value=9.0,
            requires_reslice=True,
        )
    )

    assert widget.settled_lod_bias == pytest.approx(1.0)  # unchanged


def test_inbound_unrelated_field_ignored():
    widget, visual_id = _make_widget(initial_lod_bias=1.0)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual_id,
            field_name="clim",
            new_value=(0.0, 500.0),
            requires_reslice=True,
        )
    )

    assert widget.settled_lod_bias == pytest.approx(1.0)  # unchanged
