"""Tests for the anywidget ``AnywidgetColormapCombo`` colormap-control widget."""

from __future__ import annotations

from uuid import uuid4

import pytest
from cmap import Colormap

pytest.importorskip("anywidget")

from cellier.events import AppearanceChangedEvent
from cellier.gui.anywidget.visuals._colormap import (
    AnywidgetColormapCombo,
)


def _make_widget(**kwargs):
    visual_id = uuid4()
    return AnywidgetColormapCombo(visual_id, **kwargs), visual_id


def test_instantiate_smoke():
    widget, _visual_id = _make_widget(initial_colormap="viridis")
    assert widget.color_map == "viridis"
    assert "viridis" in widget.colormap_names


def test_instantiate_seeded_from_colormap_object():
    # Seeded from a cmap.Colormap; stored as its canonical name string.
    widget, _visual_id = _make_widget(initial_colormap=Colormap("magma"))
    assert isinstance(widget.color_map, str)
    assert "magma" in widget.color_map


def test_custom_colormap_names():
    widget, _visual_id = _make_widget(colormap_names=["viridis", "magma"])
    assert widget.colormap_names == ["viridis", "magma"]


def test_trait_edit_emits_update_event():
    widget, visual_id = _make_widget(initial_colormap="viridis")
    emitted = []
    widget.changed.connect(emitted.append)

    widget.color_map = "magma"

    assert len(emitted) == 1
    event = emitted[0]
    assert event.source_id == widget._id
    assert event.visual_id == visual_id
    assert event.field == "color_map"
    assert event.value == "magma"


def test_inbound_change_updates_trait_without_reemit():
    widget, visual_id = _make_widget(initial_colormap="viridis")
    emitted = []
    widget.changed.connect(emitted.append)

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),  # not the widget -> applied
            visual_id=visual_id,
            field_name="color_map",
            new_value=Colormap("magma"),
            requires_reslice=False,
        )
    )

    assert isinstance(widget.color_map, str)
    assert "magma" in widget.color_map
    assert emitted == []  # applied under _applying, no echo


def test_inbound_echo_filtered_by_source_id():
    widget, visual_id = _make_widget(initial_colormap="viridis")

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=widget._id,  # our own echo -> ignored
            visual_id=visual_id,
            field_name="color_map",
            new_value=Colormap("magma"),
            requires_reslice=False,
        )
    )

    assert "viridis" in widget.color_map  # unchanged


def test_inbound_unrelated_field_ignored():
    widget, visual_id = _make_widget(initial_colormap="viridis")

    widget._on_appearance_changed(
        AppearanceChangedEvent(
            source_id=uuid4(),
            visual_id=visual_id,
            field_name="clim",
            new_value=(0.0, 500.0),
            requires_reslice=False,
        )
    )

    assert "viridis" in widget.color_map  # unchanged
