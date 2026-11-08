"""Tests for PointsMarkerAppearance and PointsVisual model layer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual


def test_appearance_defaults():
    a = PointsMarkerAppearance()
    assert a.color == (1.0, 1.0, 1.0, 1.0)
    assert a.size == 5.0
    assert a.size_space == "screen"
    assert a.color_mode == "uniform"
    assert a.opacity == 1.0


def test_appearance_mutation_fires_event():
    a = PointsMarkerAppearance()
    received = []
    a.events.size.connect(lambda v: received.append(v))
    a.size = 10.0
    assert received == [10.0]


def test_appearance_json_roundtrip(tmp_path):
    a = PointsMarkerAppearance(color=(0.5, 0.5, 0.5, 1.0), size=8.0)
    path = tmp_path / "appearance.json"
    path.write_text(a.model_dump_json())
    b = PointsMarkerAppearance.model_validate_json(path.read_text())
    assert a.model_dump_json() == b.model_dump_json()


def test_visual_requires_camera_reslice_is_false():
    from uuid import uuid4

    v = PointsVisual(
        name="points",
        data_store_id=str(uuid4()),
        appearance=PointsMarkerAppearance(),
    )
    assert v.requires_camera_reslice is False


def test_visual_requires_camera_reslice_is_frozen():
    from uuid import uuid4

    v = PointsVisual(
        name="points",
        data_store_id=str(uuid4()),
        appearance=PointsMarkerAppearance(),
    )
    with pytest.raises(ValidationError):
        v.requires_camera_reslice = True


def test_visual_json_roundtrip(tmp_path):
    from uuid import uuid4

    store_id = str(uuid4())
    v = PointsVisual(
        name="cloud",
        data_store_id=store_id,
        appearance=PointsMarkerAppearance(color=(0.0, 1.0, 0.0, 1.0), size=12.0),
    )
    path = tmp_path / "visual.json"
    path.write_text(v.model_dump_json())
    v2 = PointsVisual.model_validate_json(path.read_text())
    assert v2.data_store_id == store_id
    assert v2.appearance.size == 12.0
    assert v2.name == "cloud"


# ── visual_type rename + migration validator ──────────────────────────────────


def test_visual_type_is_points_memory():
    from uuid import uuid4

    v = PointsVisual(name="p", data_store_id=str(uuid4()))
    assert v.visual_type == "points_memory"


def test_visual_type_migration():
    """Serialised JSON with old 'points' discriminator should still load."""
    import json
    from uuid import uuid4

    v = PointsVisual(name="p", data_store_id=str(uuid4()))
    d = json.loads(v.model_dump_json())
    d["visual_type"] = "points"  # simulate old serialised data
    v2 = PointsVisual.model_validate(d)
    assert v2.visual_type == "points_memory"


def test_appearance_type_field_removed():
    """PointsMarkerAppearance must not have appearance_type field."""
    a = PointsMarkerAppearance()
    assert not hasattr(a, "appearance_type")
