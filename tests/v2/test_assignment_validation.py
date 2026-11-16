"""Appearance and overlay models validate assignments, as at construction."""

import pytest
from pydantic import ValidationError

from cellier.scene._background import BackgroundAppearance
from cellier.visuals._canvas_overlay import CenteredAxes2D, CenteredAxes2DAppearance
from cellier.visuals._graph_memory import TrailConfig
from cellier.visuals._label_memory import InMemoryLabelsAppearance
from cellier.visuals._labels import MultiscaleLabelsAppearance
from cellier.visuals._lines_memory import LinesMemoryAppearance
from cellier.visuals._points_memory import PointsMarkerAppearance


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (InMemoryLabelsAppearance(), "opacity", 5.0),
        (InMemoryLabelsAppearance(), "render_mode", "bogus"),
        (MultiscaleLabelsAppearance(), "opacity", 5.0),
        (PointsMarkerAppearance(), "opacity", 3),
        (LinesMemoryAppearance(), "opacity", -1.0),
        (CenteredAxes2DAppearance(), "corner", "middle"),
        (BackgroundAppearance(), "mode", "bogus"),
        (TrailConfig(), "before", "sometimes"),
    ],
)
def test_an_invalid_assignment_raises(model, field, value):
    before = getattr(model, field)
    with pytest.raises(ValidationError):
        setattr(model, field, value)
    assert getattr(model, field) == before


def test_overlay_assignment_is_validated():
    overlay = CenteredAxes2D(name="axes")
    with pytest.raises(ValidationError):
        overlay.visible = "sometimes"
