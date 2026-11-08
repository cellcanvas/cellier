"""Tests for BaseVisual field normalisation."""

from __future__ import annotations

import uuid

from pydantic import Field

from cellier.visuals._base_visual import BaseAppearance, BaseVisual


class _MinimalVisual(BaseVisual):
    """Minimal concrete subclass for testing BaseVisual fields."""

    appearance: BaseAppearance = Field(default_factory=BaseAppearance)
    visual_type: str = "test"


def test_aabb_defaults_on_base_visual():
    v = _MinimalVisual(name="x", data_store_id=str(uuid.uuid4()))
    assert v.aabb.enabled is False
    assert v.aabb.color == "#ffffff"
    assert v.aabb.line_width == 2.0


def test_data_store_id_on_base_visual():
    sid = str(uuid.uuid4())
    v = _MinimalVisual(name="x", data_store_id=sid)
    assert v.data_store_id == sid


def test_mesh_visual_inherits_aabb():
    from cellier.visuals import MeshFlatAppearance, MeshVisual

    v = MeshVisual(
        name="mesh",
        data_store_id=str(uuid.uuid4()),
        appearance=MeshFlatAppearance(),
    )
    assert hasattr(v, "aabb")
    assert v.aabb.enabled is False


def test_points_visual_inherits_aabb():
    from cellier.visuals._points_memory import PointsMarkerAppearance, PointsVisual

    v = PointsVisual(
        name="pts",
        data_store_id=str(uuid.uuid4()),
        appearance=PointsMarkerAppearance(),
    )
    assert hasattr(v, "aabb")
    assert v.aabb.enabled is False


def test_lines_visual_inherits_aabb():
    from cellier.visuals import LinesMemoryAppearance, LinesVisual

    v = LinesVisual(
        name="lines",
        data_store_id=str(uuid.uuid4()),
        appearance=LinesMemoryAppearance(),
    )
    assert hasattr(v, "aabb")
    assert v.aabb.enabled is False
