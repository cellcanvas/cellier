"""Tests for LinesMemoryAppearance and LinesVisual model layer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellier.visuals import LinesMemoryAppearance, LinesVisual


def test_appearance_defaults():
    a = LinesMemoryAppearance()
    assert a.color == (1.0, 1.0, 1.0, 1.0)
    assert a.thickness == 2.0
    assert a.thickness_space == "screen"
    assert a.color_mode == "uniform"
    assert a.opacity == 1.0


def test_appearance_mutation_fires_event():
    a = LinesMemoryAppearance()
    received = []
    a.events.thickness.connect(lambda v: received.append(v))
    a.thickness = 4.0
    assert received == [4.0]


def test_appearance_json_roundtrip(tmp_path):
    a = LinesMemoryAppearance(color=(0.5, 0.5, 0.5, 1.0), thickness=3.0)
    path = tmp_path / "appearance.json"
    path.write_text(a.model_dump_json())
    b = LinesMemoryAppearance.model_validate_json(path.read_text())
    assert a.model_dump_json() == b.model_dump_json()


def test_visual_requires_camera_reslice_is_false():
    from uuid import uuid4

    v = LinesVisual(
        name="lines",
        data_store_id=str(uuid4()),
        appearance=LinesMemoryAppearance(),
    )
    assert v.requires_camera_reslice is False


def test_visual_requires_camera_reslice_is_frozen():
    from uuid import uuid4

    v = LinesVisual(
        name="lines",
        data_store_id=str(uuid4()),
        appearance=LinesMemoryAppearance(),
    )
    with pytest.raises(ValidationError):
        v.requires_camera_reslice = True


def test_visual_json_roundtrip(tmp_path):
    from uuid import uuid4

    store_id = str(uuid4())
    v = LinesVisual(
        name="segments",
        data_store_id=store_id,
        appearance=LinesMemoryAppearance(color=(1.0, 0.0, 0.0, 1.0), thickness=5.0),
    )
    path = tmp_path / "visual.json"
    path.write_text(v.model_dump_json())
    v2 = LinesVisual.model_validate_json(path.read_text())
    assert v2.data_store_id == store_id
    assert v2.appearance.thickness == 5.0
    assert v2.name == "segments"


# ── visual_type rename + migration validator ──────────────────────────────────


def test_visual_type_is_lines_memory():
    from uuid import uuid4

    v = LinesVisual(name="l", data_store_id=str(uuid4()))
    assert v.visual_type == "lines_memory"


def test_visual_type_migration():
    """Serialised JSON with old 'lines' discriminator should still load."""
    import json
    from uuid import uuid4

    v = LinesVisual(name="l", data_store_id=str(uuid4()))
    d = json.loads(v.model_dump_json())
    d["visual_type"] = "lines"  # simulate old serialised data
    v2 = LinesVisual.model_validate(d)
    assert v2.visual_type == "lines_memory"


def test_appearance_type_field_removed():
    """LinesMemoryAppearance must not have appearance_type field."""
    a = LinesMemoryAppearance()
    assert not hasattr(a, "appearance_type")
