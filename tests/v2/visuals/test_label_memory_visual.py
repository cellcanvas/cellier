"""Tests for InMemoryLabelsAppearance and LabelMemoryVisual model layer."""

from __future__ import annotations

import pytest

from cellier.visuals import InMemoryLabelsAppearance, LabelMemoryVisual

# ── InMemoryLabelsAppearance ────────────────────────────────────────────────────


def test_appearance_defaults():
    a = InMemoryLabelsAppearance()
    assert a.colormap_mode == "random"
    assert a.background_label == 0
    assert a.salt == 0
    assert a.color_dict == {}
    assert a.render_mode == "iso_categorical"
    assert a.opacity == 1.0


def test_appearance_json_roundtrip():
    a = InMemoryLabelsAppearance(
        colormap_mode="direct",
        background_label=-1,
        salt=42,
        color_dict={1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)},
        render_mode="flat_categorical",
        opacity=0.7,
    )
    json_str = a.model_dump_json()
    b = InMemoryLabelsAppearance.model_validate_json(json_str)
    assert b.colormap_mode == "direct"
    assert b.background_label == -1
    assert b.salt == 42
    assert b.color_dict == {1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)}
    assert b.render_mode == "flat_categorical"
    assert abs(b.opacity - 0.7) < 1e-6


def test_appearance_mutation_fires_event():
    a = InMemoryLabelsAppearance()
    received = []
    a.events.background_label.connect(lambda v: received.append(v))
    a.background_label = 5
    assert received == [5]


def test_appearance_salt_mutation_fires_event():
    a = InMemoryLabelsAppearance()
    received = []
    a.events.salt.connect(lambda v: received.append(v))
    a.salt = 123
    assert received == [123]


def test_colormap_mode_is_frozen():
    from pydantic import ValidationError

    a = InMemoryLabelsAppearance(colormap_mode="random")
    with pytest.raises(ValidationError):
        a.colormap_mode = "direct"


def test_render_mode_mutation_fires_event():
    a = InMemoryLabelsAppearance(render_mode="iso_categorical")
    received = []
    a.events.render_mode.connect(lambda v: received.append(v))
    a.render_mode = "flat_categorical"
    assert received == ["flat_categorical"]


def test_smooth_render_modes_are_accepted():
    for mode in ("smooth_iso", "smooth_geometry"):
        assert InMemoryLabelsAppearance(render_mode=mode).render_mode == mode


# ── LabelMemoryVisual ────────────────────────────────────────────────────────


def test_visual_json_roundtrip():
    from uuid import uuid4

    store_id = str(uuid4())
    a = InMemoryLabelsAppearance(
        colormap_mode="random",
        background_label=0,
        color_dict={7: (0.1, 0.2, 0.3, 1.0)},
    )
    v = LabelMemoryVisual(
        name="test",
        data_store_id=store_id,
        appearance=a,
    )
    json_str = v.model_dump_json()
    v2 = LabelMemoryVisual.model_validate_json(json_str)
    assert v2.data_store_id == store_id
    assert v2.appearance.colormap_mode == "random"
    assert v2.appearance.color_dict == {7: (0.1, 0.2, 0.3, 1.0)}


def test_visual_type_discriminator():
    from uuid import uuid4

    v = LabelMemoryVisual(
        name="x",
        data_store_id=str(uuid4()),
        appearance=InMemoryLabelsAppearance(),
    )
    assert v.visual_type == "label_memory"


def test_visual_requires_camera_reslice_is_false():
    from uuid import uuid4

    v = LabelMemoryVisual(
        name="x",
        data_store_id=str(uuid4()),
        appearance=InMemoryLabelsAppearance(),
    )
    assert v.requires_camera_reslice is False
